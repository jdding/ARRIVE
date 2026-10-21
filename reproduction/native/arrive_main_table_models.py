#!/usr/bin/env python3
"""Thin model adapters for the ARRIVE paper-facing baseline matrix.

All adapters consume the same batch contract and expose the same sampled
set-retrieval objective.  The training controller, data split and evaluator
live in ``run_arrive_main_table_baseline.py``; this file contains model logic
only.  This keeps adding a dataset or a seed independent of model code.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lfm_timeconcat_control import FIRST_ITEM, PAD


class MainTableModel(nn.Module):
    """Shared tied-head set retrieval contract used by every adapter."""

    def __init__(self, vocab_size: int, noise_probabilities: np.ndarray, embed_dim: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.item_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        probabilities = torch.as_tensor(noise_probabilities, dtype=torch.float32)
        if len(probabilities) != vocab_size or probabilities[:FIRST_ITEM].sum() != 0:
            raise ValueError("invalid noise distribution")
        self.register_buffer("noise_probabilities", probabilities, persistent=False)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        with torch.no_grad():
            self.item_embedding.weight[PAD].zero_()

    @staticmethod
    def gather_states(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, positions]

    def item_vectors(self, indices: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(indices)

    def encode_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def score_items(
        self,
        states: torch.Tensor,
        item_ids: torch.Tensor,
        *,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        cached = None if batch is None else batch.get("_catalog_vectors")
        vectors = self.item_vectors(item_ids) if cached is None else cached.index_select(0, item_ids)
        return states @ vectors.T

    def sampled_loss(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        sessions: torch.Tensor,
        weights: torch.Tensor,
        negatives: torch.Tensor,
        *,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        positive = self.score_items(states, targets, batch=batch).diagonal().unsqueeze(1)
        negative = self.score_items(states, negatives, batch=batch)
        count = float(negatives.numel())
        positive -= torch.log(count * self.noise_probabilities[targets].unsqueeze(1)).clamp_min(-30)
        negative -= torch.log(count * self.noise_probabilities[negatives]).clamp_min(-30).unsqueeze(0)
        negative = negative.masked_fill(negatives.unsqueeze(0).eq(targets.unsqueeze(1)), float("-inf"))
        losses = F.cross_entropy(
            torch.cat((positive, negative), dim=1),
            torch.zeros(len(targets), dtype=torch.long, device=states.device),
            reduction="none",
        )
        per_session = torch.zeros(int(sessions.max()) + 1, device=states.device)
        per_session.scatter_add_(0, sessions, losses * weights)
        return per_session.mean()

    def auxiliary_loss(self, batch: dict[str, torch.Tensor], hidden: torch.Tensor) -> torch.Tensor:
        del batch
        return hidden.sum() * 0.0

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        negative_samples: int,
        objective: str,
        negatives: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if objective != "oneshot_only":
            raise ValueError("main-table adapters require oneshot_only")
        hidden = self.encode_batch(batch)
        initial = self.gather_states(hidden, batch["initial_positions"])
        states = initial[batch["set_sessions"]]
        if negatives is None:
            negatives = torch.multinomial(self.noise_probabilities, negative_samples, replacement=True)
        set_loss = self.sampled_loss(
            states,
            batch["set_targets"],
            batch["set_sessions"],
            batch["set_weights"],
            negatives,
            batch=batch,
        )
        auxiliary = self.auxiliary_loss(batch, hidden)
        total = set_loss + auxiliary
        return total, {
            "set_loss": float(set_loss.detach()),
            "ar_loss": 0.0,
            "auxiliary_loss": float(auxiliary.detach()),
        }


class CausalTransformer(MainTableModel):
    def __init__(
        self,
        vocab_size: int,
        noise_probabilities: np.ndarray,
        *,
        embed_dim: int,
        max_sequence: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(vocab_size, noise_probabilities, embed_dim)
        self.max_sequence = int(max_sequence)
        self.position_embedding = nn.Embedding(max_sequence, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def token_vectors(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.item_vectors(sequence)

    def encode_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        sequence = batch["sequence"]
        positions = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
        hidden = self.dropout(self.token_vectors(sequence) + self.position_embedding(positions))
        hidden = hidden.masked_fill(sequence.eq(PAD).unsqueeze(-1), 0.0)
        causal = torch.triu(
            torch.full((sequence.shape[1], sequence.shape[1]), float("-inf"), device=sequence.device),
            diagonal=1,
        )
        return self.norm(self.encoder(hidden, mask=causal)).masked_fill(
            sequence.eq(PAD).unsqueeze(-1), 0.0
        )


class TiSASRecBlock(nn.Module):
    """Vectorized temporal key/value attention reused from the RCCT runner."""

    def __init__(self, dim: int, heads: int, dropout: float, max_len: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.heads, self.head_dim = heads, dim // heads
        self.attention_norm = nn.LayerNorm(dim, eps=1e-8)
        self.query, self.key, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(dim, eps=1e-8)
        self.feedforward_1, self.feedforward_2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.ones(max_len, max_len, dtype=torch.bool).triu(1), persistent=False)

    def heads_view(self, values: torch.Tensor) -> torch.Tensor:
        b, l, _ = values.shape
        return values.view(b, l, self.heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden, padding, absolute_key, absolute_value, temporal_key, temporal_value):
        b, l, _ = hidden.shape
        q_input = self.attention_norm(hidden)
        query, key, value = self.heads_view(self.query(q_input)), self.heads_view(self.key(hidden)), self.heads_view(self.value(hidden))
        absolute_key = absolute_key.view(l, self.heads, self.head_dim).permute(1, 0, 2)
        absolute_value = absolute_value.view(l, self.heads, self.head_dim).permute(1, 0, 2)
        temporal_key = temporal_key.view(b, l, l, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        temporal_value = temporal_value.view(b, l, l, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        attention = torch.einsum("bhid,bhjd->bhij", query, key)
        attention += torch.einsum("bhid,hjd->bhij", query, absolute_key)
        attention += torch.einsum("bhid,bhijd->bhij", query, temporal_key)
        attention = (attention / math.sqrt(self.head_dim)).masked_fill(
            padding[:, None, None, :] | self.causal[None, None, :l, :l], torch.finfo(attention.dtype).min
        )
        attention = self.attention_dropout(F.softmax(attention, dim=-1)).masked_fill(padding[:, None, :, None], 0.0)
        output = torch.einsum("bhij,bhjd->bhid", attention, value)
        output += torch.einsum("bhij,hjd->bhid", attention, absolute_value)
        output += torch.einsum("bhij,bhijd->bhid", attention, temporal_value)
        output = output.transpose(1, 2).contiguous().view(b, l, -1)
        hidden = q_input + output
        ff_input = self.feedforward_norm(hidden)
        ff = self.feedforward_2(self.feedforward_dropout(F.relu(self.feedforward_1(ff_input))))
        return (ff_input + self.feedforward_dropout(ff)).masked_fill(padding.unsqueeze(-1), 0.0)


class TiSASRecAdapter(MainTableModel):
    def __init__(self, vocab_size, noise_probabilities, *, embed_dim, max_sequence, num_heads, num_layers, dropout, time_span=3650):
        super().__init__(vocab_size, noise_probabilities, embed_dim)
        self.max_sequence, self.time_span = max_sequence, time_span
        self.absolute_key, self.absolute_value = nn.Embedding(max_sequence, embed_dim), nn.Embedding(max_sequence, embed_dim)
        self.temporal_key, self.temporal_value = nn.Embedding(time_span + 1, embed_dim), nn.Embedding(time_span + 1, embed_dim)
        self.blocks = nn.ModuleList([TiSASRecBlock(embed_dim, num_heads, dropout, max_sequence) for _ in range(num_layers)])
        self.norm, self.dropout = nn.LayerNorm(embed_dim, eps=1e-8), nn.Dropout(dropout)

    def encode_batch(self, batch):
        sequence, times = batch["sequence"], batch["event_times"]
        padding = sequence.eq(PAD)
        relative = (times[:, :, None] - times[:, None, :]).abs().div(86400, rounding_mode="floor")
        relative = relative.clamp(0, self.time_span).long()
        hidden = self.dropout(self.item_embedding(sequence) * math.sqrt(self.embed_dim)).masked_fill(padding.unsqueeze(-1), 0.0)
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        absolute_key, absolute_value = self.dropout(self.absolute_key(positions)), self.dropout(self.absolute_value(positions))
        temporal_key, temporal_value = self.dropout(self.temporal_key(relative)), self.dropout(self.temporal_value(relative))
        for block in self.blocks:
            hidden = block(hidden, padding, absolute_key, absolute_value, temporal_key, temporal_value)
        return self.norm(hidden).masked_fill(padding.unsqueeze(-1), 0.0)


class CoSeRNNAdapter(MainTableModel):
    """Session-preference sequence with available temporal context.

    This is a PyTorch task adaptation of CoSeRNN's defining operator: items are
    pooled within sessions, session representations are combined with context,
    and an LSTM evolves the preference across sessions.  Device and skip/listen
    fields absent from Music4All are not fabricated.
    """

    def __init__(self, vocab_size, noise_probabilities, *, embed_dim, max_sequence, num_layers, dropout, **_: Any):
        super().__init__(vocab_size, noise_probabilities, embed_dim)
        self.max_sequence = max_sequence
        self.context_projection = nn.Sequential(
            nn.Linear(5, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        self.context_rnn = nn.LSTM(
            embed_dim, embed_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.preference = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def encode_batch(self, batch):
        sequence = batch["sequence"]
        embedded = self.item_embedding(sequence)
        valid = sequence.ne(PAD)
        slots = sequence.shape[1]
        row_offset = torch.arange(len(sequence), device=sequence.device).unsqueeze(1) * slots
        groups = (batch["event_session_index"] + row_offset).reshape(-1)
        session_sum = torch.zeros(len(sequence) * slots, self.embed_dim, device=sequence.device)
        session_count = torch.zeros(len(sequence) * slots, 1, device=sequence.device)
        session_sum.scatter_add_(0, groups.unsqueeze(1).expand(-1, self.embed_dim),
                                 (embedded * valid.unsqueeze(-1)).reshape(-1, self.embed_dim))
        session_count.scatter_add_(0, groups.unsqueeze(1), valid.reshape(-1, 1).float())
        session_mean = (session_sum / session_count.clamp_min(1.0)).view(len(sequence), slots, -1)
        contextual = session_mean + self.context_projection(batch["session_features"])
        recurrent, _ = self.context_rnn(contextual)
        position = batch["session_positions"]
        rows = torch.arange(len(sequence), device=sequence.device)
        state = self.norm(self.preference(torch.cat((session_mean[rows, position], recurrent[rows, position]), -1)))
        output = torch.zeros_like(embedded)
        output[rows, batch["initial_positions"]] = state
        return output


class MUSEAdapter(MainTableModel):
    """SR-GNN plus transition augmentation and fine-grained VICReg matching."""

    def __init__(
        self, vocab_size, noise_probabilities, *, embed_dim, max_sequence,
        num_layers, dropout, augmentation_weight=0.1, **_: Any,
    ):
        super().__init__(vocab_size, noise_probabilities, embed_dim)
        self.max_sequence = max_sequence
        self.steps = num_layers
        self.in_edge = nn.Linear(embed_dim, embed_dim)
        self.out_edge = nn.Linear(embed_dim, embed_dim)
        self.graph_cell = nn.GRUCell(embed_dim * 2, embed_dim)
        self.query, self.key, self.attention = nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, 1)
        self.output = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.augmentation_weight = augmentation_weight

    def encode_sequence(self, sequence):
        hidden = self.item_embedding(sequence)
        valid = sequence.ne(PAD)
        for _ in range(self.steps):
            incoming = torch.zeros_like(hidden)
            outgoing = torch.zeros_like(hidden)
            incoming[:, 1:] = self.in_edge(hidden[:, :-1])
            outgoing[:, :-1] = self.out_edge(hidden[:, 1:])
            edge_valid = valid[:, 1:] & valid[:, :-1]
            incoming[:, 1:] = incoming[:, 1:].masked_fill(~edge_valid.unsqueeze(-1), 0.0)
            outgoing[:, :-1] = outgoing[:, :-1].masked_fill(~edge_valid.unsqueeze(-1), 0.0)
            updated = self.graph_cell(
                torch.cat((incoming, outgoing), dim=-1).reshape(-1, self.embed_dim * 2),
                hidden.reshape(-1, self.embed_dim),
            ).view_as(hidden)
            hidden = updated.masked_fill(~valid.unsqueeze(-1), 0.0)
        return hidden

    def gather_states(self, hidden, positions):
        # Token representations remain available to local VICReg; session
        # pooling is only the retrieval state, not a replacement for tokens.
        valid = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0) <= positions.unsqueeze(1)
        last = hidden[torch.arange(len(hidden), device=hidden.device), positions]
        weights = self.attention(torch.sigmoid(self.query(last).unsqueeze(1) + self.key(hidden))).squeeze(-1)
        weights = weights.masked_fill(~valid, float("-inf")).softmax(-1)
        pooled = torch.einsum("bl,bld->bd", weights, hidden)
        return self.output(torch.cat((pooled, last), dim=-1))

    def encode_batch(self, batch):
        return self.encode_sequence(batch["sequence"])

    @staticmethod
    def _off_diagonal(matrix):
        size = matrix.shape[0]
        return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()

    def _vicreg_pair(self, first, second):
        if first.numel() == 0:
            return first.sum() * 0.0
        invariance = F.mse_loss(first, second)
        centered_first, centered_second = first - first.mean(0), second - second.mean(0)
        variance = (
            F.relu(1.0 - torch.sqrt(centered_first.var(0, unbiased=False) + 1e-4)).mean()
            + F.relu(1.0 - torch.sqrt(centered_second.var(0, unbiased=False) + 1e-4)).mean()
        ) / 2
        denominator = max(len(first) - 1, 1)
        covariance = (
            self._off_diagonal(centered_first.T @ centered_first / denominator).pow(2).mean()
            + self._off_diagonal(centered_second.T @ centered_second / denominator).pow(2).mean()
        ) / 2
        return invariance + variance + 10.0 * covariance

    def auxiliary_loss(self, batch, hidden):
        augmented = batch.get("augmented_sequence", batch["sequence"])
        other = self.encode_sequence(augmented)
        original_state = self.gather_states(hidden, batch["initial_positions"])
        augmented_positions = batch.get("augmented_positions")
        if augmented_positions is None:
            augmented_state = original_state
            return self.augmentation_weight * self._vicreg_pair(original_state, augmented_state)
        augmented_lengths = augmented.ne(PAD).sum(1).clamp_min(1)
        augmented_state = self.gather_states(other, augmented_lengths - 1)

        # Similarity matching: every valid original token is paired with its
        # nearest valid token representation in the transition-augmented view.
        distances = torch.cdist(hidden, other)
        distances = distances.masked_fill(~augmented.ne(PAD).unsqueeze(1), float("inf"))
        nearest = distances.argmin(-1)
        matched = other.gather(1, nearest.unsqueeze(-1).expand_as(hidden))
        valid_original = batch["sequence"].ne(PAD)
        local_item = self._vicreg_pair(hidden[valid_original], matched[valid_original])

        # Position matching preserves representations of original items after
        # bridge insertion; -1 marks the inserted transition item.
        aligned = augmented_positions.ge(0) & augmented.ne(PAD)
        source = hidden.gather(
            1, augmented_positions.clamp_min(0).unsqueeze(-1).expand_as(other)
        )
        local_position = self._vicreg_pair(source[aligned], other[aligned])
        global_pair = self._vicreg_pair(original_state, augmented_state)
        return self.augmentation_weight * (0.2 * global_pair + 0.4 * local_item + 0.4 * local_position)


class MEGANAdapter(CausalTransformer):
    """Task-aligned heterogeneous multi-view graph-attention model."""

    def __init__(
        self, *args, content_features: np.ndarray,
        transition_items: np.ndarray, transition_weights: np.ndarray, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.register_buffer("content_features", torch.as_tensor(content_features, dtype=torch.float32), persistent=True)
        self.register_buffer("transition_items", torch.as_tensor(transition_items, dtype=torch.long), persistent=True)
        self.register_buffer("transition_weights", torch.as_tensor(transition_weights, dtype=torch.float32), persistent=True)
        self.content_projection = nn.Sequential(nn.Linear(content_features.shape[1], self.embed_dim), nn.GELU(), nn.Linear(self.embed_dim, self.embed_dim))
        self.item_view_attention = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.Tanh(), nn.Linear(self.embed_dim, 1))
        self.user_view_attention = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.Tanh(), nn.Linear(self.embed_dim, 1))

    def item_vectors(self, indices):
        identity = self.item_embedding(indices)
        content = self.content_projection(self.content_features[indices])
        neighbors = self.transition_items[indices]
        valid = neighbors.ge(FIRST_ITEM)
        weights = self.transition_weights[indices].masked_fill(~valid, float("-inf"))
        weights = torch.where(valid.any(-1, keepdim=True), weights.softmax(-1), torch.zeros_like(weights))
        transition = torch.einsum(
            "...k,...kd->...d", weights,
            self.item_embedding(neighbors.clamp_min(0)),
        )
        views = torch.stack((identity, content, transition), dim=-2)
        attention = self.item_view_attention(views).squeeze(-1).softmax(-1)
        return torch.einsum("...v,...vd->...d", attention, views)

    def encode_batch(self, batch):
        hidden = super().encode_batch(batch)
        sequence = batch["sequence"]
        valid = sequence.ne(PAD)
        token_views = self.item_vectors(sequence)
        behavior = (token_views * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        current_session = batch["event_session_index"].eq(batch["session_positions"].unsqueeze(1)) & valid
        session = (token_views * current_session.unsqueeze(-1)).sum(1) / current_session.sum(1, keepdim=True).clamp_min(1)
        rows = torch.arange(len(sequence), device=sequence.device)
        short = hidden[rows, batch["initial_positions"]]
        views = torch.stack((behavior, session, short), dim=1)
        attention = self.user_view_attention(views).squeeze(-1).softmax(-1)
        state = torch.einsum("bv,bvd->bd", attention, views)
        output = hidden.clone()
        output[rows, batch["initial_positions"]] = state
        return output


class FactUBRAdapter(CausalTransformer):
    """Paper-faithful RCB policy plus differentiable temporal channels."""

    def __init__(self, *args, content_features: np.ndarray, retrieval_k=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrieval_k = int(retrieval_k)
        self.register_buffer(
            "content_features", torch.as_tensor(content_features, dtype=torch.float32),
            persistent=True,
        )
        policy_dim = min(self.embed_dim, 64)
        self.history_policy = nn.Linear(content_features.shape[1], policy_dim)
        self.candidate_policy = nn.Linear(content_features.shape[1], policy_dim)
        self.state_policy = nn.Linear(self.embed_dim, policy_dim)
        self.retrieval_scale = nn.Parameter(torch.tensor(0.1))
        self.temporal_scale = nn.Parameter(torch.tensor(0.1))
        self.policy_weight = 0.01

    def _retrieval_terms(self, states, sequence, event_times, item_ids, *, training):
        history = self.item_embedding(sequence)
        candidates = self.item_vectors(item_ids)
        padding = sequence.eq(PAD)
        history_content = self.content_features[sequence]
        candidate_content = self.content_features[item_ids]
        history_key = self.history_policy(history_content)
        query = self.candidate_policy(candidate_content)
        state_key = self.state_policy(states)
        policy = torch.einsum("bld,cd->bcl", history_key, query)
        policy += torch.einsum("bld,bd->bl", history_key, state_key).unsqueeze(1)
        policy = policy.masked_fill(padding.unsqueeze(1), float("-inf"))
        evidence = torch.einsum("bld,cd->bcl", history, candidates)

        perturbed_policy = policy
        if training:
            uniform = torch.rand_like(policy).clamp_(1e-6, 1 - 1e-6)
            perturbed_policy = policy - torch.log(-torch.log(uniform))
        # Retain a tensorized width while masking per-row unavailable slots.
        # A user's retrieval budget must not depend on their batch neighbors.
        selected = perturbed_policy.topk(
            max(1, min(self.retrieval_k, sequence.shape[1])), dim=-1
        ).indices
        selected_valid = (~padding).unsqueeze(1).expand_as(policy).gather(-1, selected)
        selected_evidence = (
            evidence.gather(-1, selected).masked_fill(~selected_valid, 0.0).sum(-1)
            / selected_valid.sum(-1).clamp_min(1)
        )

        current = event_times.gather(1, sequence.ne(PAD).sum(1, keepdim=True).clamp_min(1) - 1)
        age = (current - event_times).clamp_min(0).float() / 86400.0
        channel_masks = (
            age.le(1),
            age.gt(1) & age.le(7),
            age.gt(7) & age.le(30),
            age.gt(30),
        )
        temporal = []
        for mask in channel_masks:
            valid = mask & ~padding
            channel = evidence.masked_fill(~valid.unsqueeze(1), float("-inf"))
            if training:
                uniform = torch.rand_like(channel).clamp_(1e-6, 1 - 1e-6)
                channel = channel - 0.1 * torch.log(-torch.log(uniform))
            maximum = channel.max(-1).values
            temporal.append(torch.where(valid.any(-1, keepdim=True), maximum, torch.zeros_like(maximum)))
        temporal_score = torch.stack(temporal, -1).mean(-1)
        return selected_evidence, temporal_score, policy, selected

    def score_items(self, states, item_ids, *, batch=None):
        base = super().score_items(states, item_ids, batch=batch)
        if batch is None or "candidate_sessions" not in batch:
            return base
        sessions = batch["candidate_sessions"]
        sequence = batch["sequence"].index_select(0, sessions)
        cached = batch.get("_catalog_vectors")
        candidates = self.item_vectors(item_ids) if cached is None else cached.index_select(0, item_ids)
        similarity = torch.einsum("nd,cd->nc", states, candidates)
        selected, temporal, _, _ = self._retrieval_terms(
            states, sequence, batch["event_times"].index_select(0, sessions),
            item_ids, training=False,
        )
        return similarity + self.retrieval_scale * selected + self.temporal_scale * temporal

    def _session_candidate_scores(self, states, sequence, event_times, item_ids):
        candidates = self.item_vectors(item_ids)
        scores = states @ candidates.T
        selected, temporal, policy, chosen = self._retrieval_terms(
            states, sequence, event_times, item_ids, training=self.training,
        )
        return scores + self.retrieval_scale * selected + self.temporal_scale * temporal, policy, chosen

    def loss(self, batch, *, negative_samples, objective, negatives=None):
        if objective != "oneshot_only":
            raise ValueError("main-table adapters require oneshot_only")
        hidden = self.encode_batch(batch)
        states = self.gather_states(hidden, batch["initial_positions"])
        targets, sessions, weights = batch["set_targets"], batch["set_sessions"], batch["set_weights"]
        if negatives is None:
            negatives = torch.multinomial(self.noise_probabilities, negative_samples, replacement=True)
        candidate_ids = torch.cat((targets, negatives))
        scores, policy, chosen = self._session_candidate_scores(
            states, batch["sequence"], batch["event_times"], candidate_ids
        )
        positive = scores[sessions, torch.arange(len(targets), device=targets.device)].unsqueeze(1)
        negative = scores.index_select(0, sessions)[:, len(targets) :]
        count = float(negatives.numel())
        positive -= torch.log(
            count * self.noise_probabilities[targets].unsqueeze(1)
        ).clamp_min(-30)
        negative -= torch.log(count * self.noise_probabilities[negatives]).clamp_min(-30).unsqueeze(0)
        negative = negative.masked_fill(negatives.unsqueeze(0).eq(targets.unsqueeze(1)), float("-inf"))
        losses = F.cross_entropy(
            torch.cat((positive, negative), dim=1),
            torch.zeros(len(targets), dtype=torch.long, device=states.device),
            reduction="none",
        )
        per_session = torch.zeros(int(sessions.max()) + 1, device=states.device)
        per_session.scatter_add_(0, sessions, losses * weights)
        set_loss = per_session.mean()
        positive_policy = policy[sessions, torch.arange(len(targets), device=targets.device)]
        positive_choice = chosen[sessions, torch.arange(len(targets), device=targets.device)]
        chosen_valid = batch["sequence"][sessions].ne(PAD).gather(1, positive_choice)
        log_probability = (
            positive_policy.log_softmax(-1).gather(1, positive_choice).masked_fill(~chosen_valid, 0.0).sum(1)
            / chosen_valid.sum(1).clamp_min(1)
        )
        selected_content = self.content_features[batch["sequence"][sessions]].gather(
            1, positive_choice.unsqueeze(-1).expand(-1, -1, self.content_features.shape[1])
        )
        normalized = F.normalize(selected_content, dim=-1)
        pair_valid = chosen_valid.unsqueeze(1) & chosen_valid.unsqueeze(2)
        diversity = 1.0 - (
            torch.einsum("bkd,bjd->bkj", normalized, normalized).masked_fill(~pair_valid, 0.0).sum((1, 2))
            / pair_valid.sum((1, 2)).clamp_min(1)
        )
        reward = positive.squeeze(1).detach() + 0.1 * diversity.detach()
        policy_loss = -(reward * log_probability).mean()
        total = set_loss + self.policy_weight * policy_loss
        return total, {
            "set_loss": float(set_loss.detach()), "ar_loss": 0.0,
            "auxiliary_loss": float((self.policy_weight * policy_loss).detach()),
        }


def load_official_ssd_block(source: Path):
    """Load the pinned official BiSSDLayer without importing RecBole's model shell."""
    source = Path(source)
    text = source.read_text()
    start = text.index("class BiSSDLayer")
    namespace: dict[str, Any] = {"torch": torch, "nn": nn}
    from mamba_ssm import Mamba2  # type: ignore
    namespace["Mamba2"] = Mamba2
    exec(text[start:], namespace)
    return namespace["BiSSDLayer"]


class SSD4RecAdapter(MainTableModel):
    """Packed official BiSSD/Mamba2 carrier with a common tied retrieval head."""

    def __init__(self, vocab_size, noise_probabilities, *, embed_dim, max_sequence, num_layers, dropout, official_source, **_: Any):
        super().__init__(vocab_size, noise_probabilities, embed_dim)
        block = load_official_ssd_block(Path(official_source))
        self.max_sequence = max_sequence
        self.blocks = nn.ModuleList([block(0.1, embed_dim, 64, 4, 2, dropout, num_layers, 16) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._official_init_weights)

    @staticmethod
    def _official_init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @staticmethod
    def registers(valid):
        lengths = valid.sum(1)
        groups = torch.repeat_interleave(torch.arange(valid.shape[0], device=valid.device), lengths)
        starts = lengths.cumsum(0) - lengths
        positions = torch.arange(len(groups), device=valid.device)
        flip = 2 * starts.index_select(0, groups) + lengths.index_select(0, groups) - 1 - positions
        return groups.to(torch.int32), flip

    def encode_batch(self, batch):
        sequence = batch["sequence"]
        valid = sequence.ne(PAD)
        groups, flip = self.registers(valid)
        packed = self.item_embedding(sequence[valid]).unsqueeze(0)
        register = groups.unsqueeze(0)
        for block in self.blocks:
            packed = block(packed, register, flip)
        output = torch.zeros((*sequence.shape, self.embed_dim), device=sequence.device, dtype=packed.dtype)
        output[valid] = packed.squeeze(0)
        return self.norm(output).masked_fill(sequence.eq(PAD).unsqueeze(-1), 0.0)


MODEL_CLASSES = {
    "tisasrec": TiSASRecAdapter,
    "ssd4rec": SSD4RecAdapter,
    "cosernn": CoSeRNNAdapter,
    "muse": MUSEAdapter,
    "megan": MEGANAdapter,
    "factubr": FactUBRAdapter,
}
