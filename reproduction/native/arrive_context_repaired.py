#!/usr/bin/env python3
"""Source-grounded context baselines; deliberately not registered in the runner.

CAR follows the public TensorFlow implementation at bc61c20b4a8b3bd4db67974b4ed1578409e79822
(model.py and modules.py), including its three-way gate and absolute consistency
input, which differ from the two-way gate and signed difference in the paper.
The set objective is the native BPR loss averaged over distinct positives, then
episodes. CAR's optional creator branch is off unless real creator IDs are supplied.

CoSeRNNPlayedOnlyAdapter is NOT full CoSeRNN. It implements the paper's played
session pathway, recency-weighted long-term embedding, learned attention fusion,
and native cosine target loss using an explicitly supplied frozen track space.
Unavailable skip/device/source/age signals are omitted, never synthesized. The
observed return anchor is a completed singleton pseudo-session; the suffix is the
next pseudo-session. This is an explicit task adaptation, not the original task.

Primary sources and integration limits: refine-logs/arrive_repair_20260912/CONTEXT_FIDELITY.md.
No class here reads a dataset, chooses a checkpoint, or accesses Test.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from arrive_main_table_models import MainTableModel
from lfm_timeconcat_control import FIRST_ITEM, PAD


def _observed_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    sequence = batch["sequence"]
    positions = batch["initial_positions"]
    if sequence.ndim != 2 or positions.shape != (len(sequence),):
        raise ValueError("invalid observed-prefix shapes")
    if bool(((positions < 0) | (positions >= sequence.shape[1])).any()):
        raise ValueError("initial_positions out of bounds")
    valid = sequence.ne(PAD) & (torch.arange(sequence.shape[1], device=sequence.device)[None] <= positions[:, None])
    if not bool(valid[torch.arange(len(sequence), device=sequence.device), positions].all()):
        raise ValueError("observed anchor must be non-padding")
    return valid


class _CARAttention(nn.Module):
    """Official projected keys, raw values, scaled attention, without output projection."""

    def __init__(self, dim: int, heads: int, dropout: float, *, project_query: bool):
        super().__init__()
        if dim % heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.heads = heads
        self.query = nn.Linear(dim, dim) if project_query else nn.Identity()
        self.key = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, length, dim = values.shape
        def split(tensor):
            return tensor.reshape(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        q, k, v = split(self.query(queries)), split(self.key(values)), split(values)
        logits = q @ k.transpose(-1, -2) / math.sqrt(dim // self.heads)
        causal = torch.ones(length, length, device=values.device, dtype=torch.bool).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        # Finite sentinel plus explicit zeroing avoids all-padding softmax NaNs.
        weights = F.softmax(logits.masked_fill(~allowed, torch.finfo(logits.dtype).min), dim=-1)
        weights = weights.masked_fill(~allowed, 0.0)
        output = self.dropout(weights) @ v
        return output.transpose(1, 2).reshape(batch, length, dim).masked_fill(~valid[..., None], 0.0)


class _CARBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.current_norm = nn.LayerNorm(dim, eps=1e-8)
        self.current_attention = _CARAttention(dim, heads, dropout, project_query=True)
        self.general_attention = _CARAttention(dim, heads, dropout, project_query=False)
        self.gating_attention = _CARAttention(dim, heads, dropout, project_query=False)
        self.general_query = nn.Parameter(torch.empty(dim))
        self.gating_query = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.general_query, std=dim ** -0.5)
        nn.init.normal_(self.gating_query, std=dim ** -0.5)
        self.gate = nn.Linear(dim * 2, 3)
        self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-8)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(dim, dim), nn.Dropout(dropout))

    def components(self, values: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized = self.current_norm(values)
        current = self.current_attention(normalized, values, valid)
        general = self.general_attention(self.general_query.expand_as(values), values, valid)
        list_input = self.gating_attention(self.gating_query.expand_as(values), values, valid)
        centroid = (values * valid[..., None]).cumsum(1) / valid.cumsum(1).clamp_min(1)[..., None]
        consistency = (values - centroid).abs()
        gate = F.softmax(self.gate(self.dropout(torch.cat((list_input, consistency), -1))), dim=-1)
        return dict(current=current, general=general, residual=normalized,
                    consistency=consistency, list_input=list_input, gate=gate)

    def forward(self, values, valid, creator=None):
        parts = self.components(values, valid)
        mixed = sum(parts[name] * parts["gate"][..., slot:slot + 1]
                    for slot, name in enumerate(("current", "general", "residual")))
        if creator is not None:
            mixed = mixed + self.dropout(creator[:, None, :])
        normalized = self.ff_norm(mixed)
        return (normalized + self.ff(normalized)).masked_fill(~valid[..., None], 0.0)


class CARRepairedAdapter(MainTableModel):
    """Official-operator CAR, shared observed-prefix / set-positive task adapter.

    Default report label: CAR-no-user (source-operator task adaptation).
    This is not a drop-in checkpoint replacement for the former CARControl.
    """

    def __init__(self, vocab_size, noise_probabilities, *, embed_dim, max_sequence,
                 num_heads=1, num_layers=1, dropout=0.0, num_users=None,
                 use_user_embedding=False, query_l2=0.1, embedding_l2=0.0, **_: Any):
        super().__init__(vocab_size, noise_probabilities, embed_dim)
        self.max_sequence = int(max_sequence)
        self.position_embedding = nn.Embedding(max_sequence, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(_CARBlock(embed_dim, num_heads, dropout) for _ in range(num_layers))
        self.output_norm = nn.LayerNorm(embed_dim, eps=1e-8)
        self.query_l2, self.embedding_l2 = float(query_l2), float(embedding_l2)
        if use_user_embedding and (num_users is None or num_users < 1):
            raise ValueError("CAR creator branch requires a real Train user vocabulary")
        self.creator_embedding = nn.Embedding(num_users, embed_dim) if use_user_embedding else None

    def encode_batch(self, batch):
        valid = _observed_mask(batch)
        sequence = batch["sequence"]
        if sequence.shape[1] > self.max_sequence:
            raise ValueError("context exceeds frozen max_sequence")
        # Official input uses left padding. Align the right-padded shared batch's
        # positions to that same fixed window; padding/batch size cannot shift them.
        counts = valid.sum(1, keepdim=True)
        positions = (self.max_sequence - counts + valid.cumsum(1) - 1).clamp(0, self.max_sequence - 1)
        hidden = self.dropout(self.item_embedding(sequence) * math.sqrt(self.embed_dim)
                              + self.position_embedding(positions))
        hidden = hidden.masked_fill(~valid[..., None], 0.0)
        creator = None
        if self.creator_embedding is not None:
            if "user_ids" not in batch:
                raise ValueError("CAR creator branch requires observed user_ids")
            creator = self.creator_embedding(batch["user_ids"]) * math.sqrt(self.embed_dim)
        for block in self.blocks:
            hidden = block(hidden, valid, creator)
        return self.output_norm(hidden).masked_fill(~valid[..., None], 0.0)

    def auxiliary_loss(self, batch, hidden):
        del batch
        # tf.contrib.layers.l2_regularizer(scale) uses scale * sum(x^2) / 2.
        regularizer = sum((block.general_query.square().sum() + block.gating_query.square().sum())
                          * (self.query_l2 / 2) for block in self.blocks)
        matrices = [self.item_embedding.weight, self.position_embedding.weight]
        if self.creator_embedding is not None:
            matrices.append(self.creator_embedding.weight)
        regularizer = regularizer + sum(matrix.square().sum() for matrix in matrices) * (self.embedding_l2 / 2)
        return hidden.sum() * 0 + regularizer

    def sampled_loss(self, states, targets, sessions, weights, negatives, *, batch):
        if negatives.ndim != 1 or not len(negatives) or bool((negatives < FIRST_ITEM).any()):
            raise ValueError("CAR BPR requires real catalog negative IDs")
        positive = (states * self.item_vectors(targets)).sum(-1, keepdim=True)
        negative = self.score_items(states, negatives, batch=batch)
        # All suffix positives of the same episode are relevant, not negatives.
        positive_matches = batch["set_targets"][:, None].eq(negatives[None, :])
        per_episode_matches = torch.zeros(len(batch["sequence"]), len(negatives),
                                          device=states.device, dtype=torch.long)
        per_episode_matches.index_add_(0, batch["set_sessions"], positive_matches.long())
        observed = _observed_mask(batch)
        history_matches = torch.cat([
            (batch["sequence"][:, :, None].eq(chunk[None, None, :]) & observed[:, :, None]).any(1)
            for chunk in negatives.split(128)
        ], dim=1)
        excluded = (per_episode_matches.bool() | history_matches)[sessions]
        count = (~excluded).sum(1)
        if bool((count == 0).any()):
            raise ValueError("CAR BPR negative pool contains no irrelevant item for an episode")
        pairwise = F.softplus(negative - positive).masked_fill(excluded, 0.0).sum(1) / count
        per_episode = states.new_zeros(len(batch["sequence"]))
        per_episode.scatter_add_(0, sessions, pairwise * weights)
        return per_episode.mean()

    def loss(self, batch, *, negative_samples, objective, negatives=None):
        if negatives is None:
            negatives = torch.randint(FIRST_ITEM, self.vocab_size, (negative_samples,), device=batch["sequence"].device)
        return super().loss(batch, negative_samples=negative_samples, objective=objective, negatives=negatives)


class CoSeRNNPlayedOnlyAdapter(MainTableModel):
    """Restricted CoSeRNN-derived played/time-only model, never full CoSeRNN.

    Requires frozen Train-produced embeddings with provenance. Session suffix
    targets use cosine distance to the unit-normalized mean of distinct known
    track embeddings, not sampled softmax. No false skip labels are constructed.
    """

    def __init__(self, vocab_size, noise_probabilities, *, embed_dim, max_sequence,
                 fixed_item_embeddings=None, embedding_provenance=None,
                 fc_dim=200, hidden_dim=400, num_layers=1, dropout=0.0, **_: Any):
        if fixed_item_embeddings is None or not embedding_provenance:
            raise ValueError("CoSeRNN played-only requires frozen Train track embeddings and provenance")
        if embedding_provenance.get("training_split") != "train" or not embedding_provenance.get("sha256"):
            raise ValueError("fixed track space needs training_split=train and source sha256")
        super().__init__(vocab_size, noise_probabilities, embed_dim)
        fixed = torch.tensor(fixed_item_embeddings, dtype=torch.float32)
        if fixed.shape != (vocab_size, embed_dim) or not torch.isfinite(fixed).all():
            raise ValueError("invalid fixed track embedding matrix")
        if bool((fixed[FIRST_ITEM:].norm(dim=1) == 0).any()):
            raise ValueError("every catalog track needs a nonzero frozen embedding")
        fixed = F.normalize(fixed, dim=-1)
        fixed[:FIRST_ITEM] = 0.0
        self.item_embedding = nn.Embedding.from_pretrained(fixed, freeze=True, padding_idx=PAD)
        self.embedding_provenance = dict(embedding_provenance)
        self.max_sequence = int(max_sequence)
        fc_dim, hidden_dim = int(fc_dim), int(hidden_dim)
        if fc_dim < 1 or hidden_dim < 1:
            raise ValueError("CoSeRNN fc_dim and hidden_dim must be positive")
        # Current day/hour one-hot; previous session embedding/count; current gap.
        self.input_layers = nn.Sequential(nn.Linear(embed_dim + 7 + 24 + 2, fc_dim), nn.ReLU(),
                                          nn.Linear(fc_dim, fc_dim), nn.ReLU(), nn.Dropout(dropout))
        self.recurrent = nn.LSTM(fc_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0.0)
        self.combine = nn.Sequential(nn.Linear(hidden_dim, fc_dim), nn.ReLU(),
                                     nn.Linear(fc_dim, fc_dim), nn.ReLU(), nn.Dropout(dropout))
        self.offset = nn.Linear(fc_dim, embed_dim)
        self.fusion = nn.Linear(fc_dim, 2)

    def get_extra_state(self):
        return {"embedding_provenance": self.embedding_provenance,
                "method_label": "CoSeRNN-played-time-only-prefix-adaptation"}

    def set_extra_state(self, state):
        if state.get("method_label") != "CoSeRNN-played-time-only-prefix-adaptation":
            raise ValueError("wrong CoSeRNN checkpoint method identity")
        self.embedding_provenance = dict(state["embedding_provenance"])

    def session_inputs(self, batch):
        """Return observed-only shifted inputs plus long-term vectors for audit/tests."""
        valid = _observed_mask(batch)
        sequence, times = batch["sequence"], batch["event_times"]
        if sequence.shape[1] > self.max_sequence:
            raise ValueError("context exceeds frozen max_sequence")
        batch_size, width = sequence.shape
        order = torch.arange(width, device=sequence.device)[None]
        if bool(((times[:, 1:] < times[:, :-1]) & valid[:, 1:]).any()):
            raise ValueError("observed timestamps must be nondecreasing")

        # Final observed anchor is explicitly split, even if a malformed
        # session index merges it with history.  The normal prepared rows
        # already satisfy this; the assignment below preserves fail-safe parity.
        groups = batch["event_session_index"].clone()
        rows = torch.arange(batch_size, device=sequence.device)
        anchors = batch["initial_positions"]
        has_prior = anchors.gt(0)
        prior_group = groups[rows, (anchors - 1).clamp_min(0)]
        anchor_group = groups[rows, anchors]
        merged = has_prior & anchor_group.eq(prior_group)
        groups[rows[merged], anchors[merged]] = prior_group[merged] + 1

        lengths_tensor = (groups.masked_fill(~valid, -1).amax(1) + 1).long()
        if bool((lengths_tensor < 1).any()):
            raise ValueError("empty observed CoSeRNN context")
        max_sessions = int(lengths_tensor.max())
        flat_groups = rows[:, None] * max_sessions + groups.clamp_min(0)
        flat_keys = flat_groups[valid]
        vectors = self.item_embedding(sequence.clamp_min(0))
        sums = vectors.new_zeros(batch_size * max_sessions, self.embed_dim)
        sums.index_add_(0, flat_keys, vectors[valid])
        counts = vectors.new_zeros(batch_size * max_sessions)
        counts.index_add_(0, flat_keys, torch.ones_like(flat_keys, dtype=vectors.dtype))
        means = F.normalize(
            sums.reshape(batch_size, max_sessions, self.embed_dim)
            / counts.reshape(batch_size, max_sessions, 1).clamp_min(1), dim=-1,
        )

        sentinel = torch.iinfo(times.dtype).max
        starts = torch.full(
            (batch_size * max_sessions,), sentinel, dtype=times.dtype, device=times.device
        )
        ends = torch.zeros(batch_size * max_sessions, dtype=times.dtype, device=times.device)
        starts.scatter_reduce_(0, flat_keys, times[valid], reduce="amin", include_self=True)
        ends.scatter_reduce_(0, flat_keys, times[valid], reduce="amax", include_self=True)
        starts = starts.reshape(batch_size, max_sessions)
        ends = ends.reshape(batch_size, max_sessions)
        session_slots = torch.arange(max_sessions, device=sequence.device)[None]
        current_slots = torch.minimum(session_slots + 1, lengths_tensor[:, None] - 1)
        current_times = starts.gather(1, current_slots)
        day = ((current_times // 86400 + 3) % 7).long()
        hour = ((current_times % 86400) // 3600).long()
        context = torch.cat((F.one_hot(day, 7), F.one_hot(hour, 24)), -1).float()
        elapsed = (current_times - ends).clamp_min(0).float().unsqueeze(-1) / 86400.0
        inputs = torch.cat((means, context, counts.reshape(batch_size, max_sessions, 1), elapsed), -1)
        session_valid = session_slots < lengths_tensor[:, None]
        inputs = inputs.masked_fill(~session_valid[..., None], 0)
        recency = (session_slots + 1).to(means.dtype)[..., None]
        long_term = F.normalize((means * recency * session_valid[..., None]).sum(1), dim=-1)
        return inputs, long_term, lengths_tensor.cpu().tolist()

    def encode_batch(self, batch):
        inputs, long_term, lengths = self.session_inputs(batch)
        contextual = self.input_layers(inputs)
        packed = nn.utils.rnn.pack_padded_sequence(contextual, lengths, batch_first=True, enforce_sorted=False)
        recurrent, _ = self.recurrent(packed)
        recurrent, _ = nn.utils.rnn.pad_packed_sequence(recurrent, batch_first=True)
        rows = torch.arange(len(inputs), device=inputs.device)
        final = recurrent[rows, torch.as_tensor(lengths, device=inputs.device) - 1]
        combined = self.combine(final)
        offset, beta = self.offset(combined), F.softmax(self.fusion(combined), -1)
        state = F.normalize(beta[:, :1] * offset + beta[:, 1:] * long_term, dim=-1)
        hidden = state.new_zeros((*batch["sequence"].shape, self.embed_dim))
        hidden[rows, batch["initial_positions"]] = state
        return hidden

    def loss(self, batch, *, negative_samples, objective, negatives=None):
        del negative_samples, negatives
        if objective != "oneshot_only":
            raise ValueError("CoSeRNN played-only requires oneshot_only set targets")
        hidden = self.encode_batch(batch)
        states = self.gather_states(hidden, batch["initial_positions"])
        targets = states.new_zeros(states.shape)
        targets.index_add_(0, batch["set_sessions"],
                           self.item_vectors(batch["set_targets"]) * batch["set_weights"][:, None])
        if bool((targets.norm(dim=-1) == 0).any()):
            raise ValueError("cosine session target needs a nonzero known-target mean")
        target_unit = F.normalize(targets, dim=-1)
        loss = (1.0 - (states * target_unit).sum(-1)).mean()
        return loss, {"set_loss": float(loss.detach()), "ar_loss": 0.0, "auxiliary_loss": 0.0}


class FactUBRUnavailable:
    """Fail closed instead of substituting a candidate-conditioned heuristic."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FactUBR is not admitted: full primary method/source and exact sequential RCB "
            "plus perturbed-max DTC verification are unavailable; no invented fallback."
        )
