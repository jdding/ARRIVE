"""VISTA-inspired fixed-history adaptation, not an official implementation.

Source: arXiv:2510.22049v3, sections 3.2--3.3. Standard softmax attention
is used at L=128; QLA, HSTU, quantization and production delivery are not ported.
All three arms share initialization, candidate reader and ranking head. Only
the memory surface and auxiliary reconstruction objective differ.
"""
from __future__ import annotations

import math
import torch
from torch import nn

ARMS = ("vista_full", "vista_summary", "vista_reconstruct")
VERSION = "arrive.vista.fixed128.v1"


class VistaScorer(nn.Module):
    def __init__(self, input_dim, embedding_dim, arm, hidden=64, seeds=128,
                 heads=4, reconstruction_weight=1.0, reconstruction_discrimination_weight=0.0,
                 history_grouping="none"):
        super().__init__()
        if (arm not in ARMS or heads < 1 or hidden < 1 or hidden % heads or seeds < 1
                or not math.isfinite(reconstruction_weight) or reconstruction_weight < 0
                or not math.isfinite(reconstruction_discrimination_weight)
                or reconstruction_discrimination_weight < 0):
            raise ValueError("invalid_vista_configuration")
        self.arm, self.embedding_dim, self.hidden = arm, embedding_dim, hidden
        if history_grouping not in ("none", "cd_sbrs") or (history_grouping != "none" and arm != "vista_reconstruct"):
            raise ValueError("invalid_vista_history_grouping")
        self.history_grouping, self.heads = history_grouping, heads
        self.reconstruction_weight = reconstruction_weight
        self.reconstruction_discrimination_weight = reconstruction_discrimination_weight
        self.reconstruction_metrics = {}
        self.seed_count = seeds
        self.encoder = nn.Linear(input_dim, hidden)
        self.position = nn.Embedding(128, hidden)
        self.seeds = nn.Parameter(torch.empty(seeds, hidden))
        nn.init.normal_(self.seeds, std=.02)
        # No dropout: matches the deterministic old head's resume contract.
        self.history_block = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=0., activation="gelu", batch_first=True)
        self.item_query = nn.Linear(embedding_dim, hidden, bias=False)
        self.target_attention = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.target_norm = nn.LayerNorm(hidden)
        self.target_ff = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(),
                                       nn.Linear(hidden * 4, hidden))
        self.target_ff_norm = nn.LayerNorm(hidden)
        self.scorer = nn.Sequential(nn.Linear(31 + hidden, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.decoder_input = nn.Linear(embedding_dim, hidden)
        self.decoder_position = nn.Embedding(128, hidden)
        self.decoder = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=0., activation="gelu", batch_first=True)
        self.decoder_output = nn.Linear(hidden, embedding_dim)
        self.register_buffer("mean", torch.zeros(31))
        self.register_buffer("scale", torch.ones(31))

    def encode_history(self, history, items, groups=None):
        """No candidate, cue, future target, or 31-feature input reaches this step."""
        mask = history >= 3
        h = self.encoder(items[history]) + self.position(
            torch.arange(history.shape[1], device=history.device))[None]
        has_history = mask.any(-1)
        if self.arm == "vista_full":
            # One zero placeholder avoids undefined all-masked attention.
            safe_mask = mask.clone()
            safe_mask[:, 0] |= ~has_history
            h = h.masked_fill(~mask[..., None], 0.)
            memory = self.history_block(h, src_key_padding_mask=~safe_mask)
            memory = memory.masked_fill(~mask[..., None], 0.)
            return memory, safe_mask, has_history
        seed = self.seeds[None].expand(len(history), -1, -1)
        joined = torch.cat((h, seed), 1)
        joined_mask = torch.cat((mask, torch.ones_like(seed[..., 0], dtype=torch.bool)), 1)
        attention_mask = None
        if self.history_grouping == "cd_sbrs":
            from arrive_vista_yeti import segment_attention_mask
            attention_mask = segment_attention_mask(groups, mask, self.seed_count, self.heads)
        encoded = self.history_block(joined, src_mask=attention_mask, src_key_padding_mask=~joined_mask)
        memory = encoded[:, -self.seed_count:] * has_history[:, None, None]
        return memory, torch.ones_like(memory[..., 0], dtype=torch.bool), has_history

    def read_memory(self, ids, items, memory, memory_mask, has_history):
        # Each candidate reads history independently; no candidate-candidate edges.
        q = self.item_query(items[ids, :self.embedding_dim])
        context = self.target_attention(q, memory, memory, key_padding_mask=~memory_mask,
                                        need_weights=False)[0]
        context = context * has_history[:, None, None]
        read = self.target_norm(q + context)
        read = self.target_ff_norm(read + self.target_ff(read))
        return read

    def score_memory(self, batch, items, memory, memory_mask, has_history):
        read = self.read_memory(batch["ids"], items, memory, memory_mask, has_history)
        x = (batch["x"] - self.mean) / self.scale
        return self.scorer(torch.cat((x, read), -1)).squeeze(-1)

    def reconstruct_embeddings(self, memory, history_embeddings, mask):
        """Decoder v_i sees summary seeds and direct u_0..u_i, never direct u_(i+1).

        Summaries encode the complete PRE-REQUEST history, intentionally. This
        is sequence reconstruction, not a forecast claiming unseen past targets.
        """
        length = history_embeddings.shape[1]
        h = self.decoder_input(history_embeddings.detach()) + self.decoder_position(
            torch.arange(length, device=memory.device))[None]
        joined = torch.cat((memory, h), 1)
        k = memory.shape[1]
        valid = torch.cat((torch.ones_like(memory[..., 0], dtype=torch.bool), mask), 1)
        causal = torch.ones(joined.shape[1], joined.shape[1], dtype=torch.bool,
                            device=memory.device).triu(1)
        decoded = self.decoder(joined, src_mask=causal, src_key_padding_mask=~valid)
        return self.decoder_output(decoded[:, k:])

    def reconstruction_loss(self, memory, history, items):
        mask = history >= 3
        target = items[history, :self.embedding_dim].detach()
        predictions = self.reconstruct_embeddings(memory, target, mask)
        pair_mask = mask[:, :-1] & mask[:, 1:]
        error = (predictions[:, :-1] - target[:, 1:]).square().mean(-1)
        # Normalize per valid history transition AND embedding coordinate.
        # Explicit task adaptation of the paper's unnormalized squared-error sum.
        count = pair_mask.sum()
        loss = (error * pair_mask).sum() / count.clamp_min(1)
        discrimination = loss.new_zeros(())
        if self.reconstruction_discrimination_weight:
            discrimination = self.historical_discrimination_loss(
                predictions[:, :-1], history[:, 1:], pair_mask, items[:, :self.embedding_dim])
        self.reconstruction_metrics = {"mse": loss.detach(), "discrimination_ce": discrimination.detach()}
        loss = loss + self.reconstruction_discrimination_weight * discrimination
        return loss, count

    @staticmethod
    def historical_discrimination_loss(predictions, target_ids, valid, embeddings):
        """Train-only identity reconstruction with frozen, in-batch song alternatives.

        Keep squared-distance geometry of MSE. The positive is always present;
        repeated IDs are never negatives. Up to256 distinct targets from this
        fit minibatch supply deterministic shared alternatives. No new RNG,
        data input, serving parameter or serving operation is introduced.
        """
        if not valid.any():
            return predictions.sum() * 0.
        truth = target_ids[valid]
        unique = torch.unique(truth, sorted=True)
        if len(unique) < 2:
            return predictions.sum() * 0.
        if len(unique) > 256:
            unique = unique[torch.linspace(0, len(unique)-1, 256, device=unique.device).long()]
        frozen = embeddings.detach()
        # Fixed within-batch scale from target dispersion, shared by all logits.
        alternatives = frozen[unique]
        scale = (alternatives - alternatives.mean(0)).square().mean().clamp_min(1e-6)
        query = predictions[valid]
        target = frozen[truth]
        positive_distance = (query-target).square().mean(-1)
        distance = (query.square().sum(-1, keepdim=True) + alternatives.square().sum(-1)[None]
                    - 2 * query @ alternatives.T) / query.shape[-1]
        relative_logits = (positive_distance[:, None]-distance) / scale
        relative_logits = relative_logits.masked_fill(truth[:, None] == unique[None], -torch.inf)
        # log(1 + sum exp(d_positive-d_negative)): candidate0 is the true ID.
        return torch.logsumexp(torch.cat((torch.zeros_like(positive_distance[:, None]), relative_logits), 1), 1).mean()

    def forward(self, batch, items, read_mode=None):
        if read_mode is not None and read_mode != self.arm:
            raise ValueError("vista_read_mode_switch_not_supported")
        memory, mask, present = self.encode_history(batch["history"], items, batch.get("cd_groups"))
        return self.score_memory(batch, items, memory, mask, present)

    def training_outputs(self, batch, items):
        # One history encoding per batch, shared by ranking and reconstruction.
        memory, mask, present = self.encode_history(batch["history"], items, batch.get("cd_groups"))
        scores = self.score_memory(batch, items, memory, mask, present)
        if self.arm == "vista_reconstruct":
            recon, count = self.reconstruction_loss(memory, batch["history"], items)
        else:
            recon, count = scores.new_zeros(()), torch.zeros((), device=scores.device, dtype=torch.long)
        return scores, recon, count
