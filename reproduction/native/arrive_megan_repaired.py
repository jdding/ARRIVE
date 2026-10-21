"""Source-grounded MEGAN operators for the ARRIVE task adaptation.

The seven HMG views follow Wang et al. (TOIS 2024, DOI 10.1145/3592853).
The short-term attention and long/short/dynamic fusion are additionally
cross-checked against the same-authors disclosure CN114969547B, claims 6--7.
Node embeddings are initialized from a shared Train-only LINE fit over the
frozen heterogeneous music graph, matching the initialization family used in
the reported experiments.  The shared set-retrieval target and evaluator are
the explicit ARRIVE task adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from arrive_main_table_models import MainTableModel
from lfm_timeconcat_control import FIRST_ITEM


@dataclass(frozen=True)
class NeighborTable:
    """Pre-sampled typed neighbors: [source nodes, samples], -1 for absence.

    Weights are the graph edge weights, NOT log weights or normalized sampling
    probabilities. The exact frequency-to-edge-weight mapping still requires
    the readable primary equation; the data builder must not guess it.
    """

    ids: torch.Tensor
    weights: torch.Tensor


def weighted_sample_neighbors(
    ids: torch.Tensor,
    weights: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator,
) -> NeighborTable:
    """Draw with replacement from complete weighted neighbor rows (Sec.4.2).

    This utility does not construct graph weights, cap adjacency with TopK, or
    consume labels. Empty rows stay empty. Sampling is explicit so callers can
    freeze an independently seeded inference realization; no batch-dependent
    graph resampling occurs inside the representation operators.
    """
    if ids.ndim != 2 or weights.shape != ids.shape or count < 1:
        raise ValueError("expected matching [nodes, degree] tensors and positive count")
    if ids.dtype != torch.long or ids.device != weights.device:
        raise ValueError("neighbor ids must be int64 on the weights device")
    valid = ids.ge(0)
    if not torch.isfinite(weights).all() or (weights[valid] <= 0).any():
        raise ValueError("valid graph edges require finite positive weights")
    if (weights[~valid] != 0).any() or (ids < -1).any():
        raise ValueError("absent edges must have id -1 and weight zero")
    sampled_ids = ids.new_full((len(ids), count), -1)
    sampled_weights = weights.new_zeros((len(ids), count))
    nonempty = valid.any(-1)
    if nonempty.any():
        positions = torch.multinomial(weights[nonempty], count, replacement=True, generator=generator)
        sampled_ids[nonempty] = ids[nonempty].gather(1, positions)
        sampled_weights[nonempty] = weights[nonempty].gather(1, positions)
    return NeighborTable(sampled_ids, sampled_weights)


def _masked_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Empty neighborhoods return exact zeros without NaN forward/backward."""
    logits = logits.masked_fill(~valid, -torch.inf)
    logits = torch.where(valid.any(-1, keepdim=True), logits, torch.zeros_like(logits))
    return logits.softmax(-1).masked_fill(~valid, 0.0)


class WeightedNeighborAttention(nn.Module):
    """Eqs.17--20; Eq.1 aggregation and its explicitly analogous typed views."""

    def __init__(self, dim: int, hidden_dim: int, *, negative_slope: float = 0.2):
        super().__init__()
        self.transform = nn.Linear(dim, hidden_dim)
        self.attention = nn.Linear(2 * hidden_dim, 1, bias=False)
        # The paper specifies LeakyReLU but the accessible text does not expose
        # its slope; 0.2 is an explicit engineering parameter, not a paper fact.
        self.negative_slope = negative_slope

    def forward(self, target, neighbors, edge_weights, valid):
        target_hidden = self.transform(target).tanh().unsqueeze(-2).expand(*neighbors.shape[:-1], -1)
        neighbor_hidden = self.transform(neighbors).tanh()
        scores = self.attention(torch.cat((target_hidden, neighbor_hidden), -1)).squeeze(-1)
        scores = F.leaky_relu(scores, self.negative_slope)
        # Eq.17 adds the graph weight itself, not its logarithm.
        attention = _masked_softmax(edge_weights + scores, valid)
        aggregate = (attention.unsqueeze(-1) * neighbors).sum(-2)
        return aggregate, attention


class MetadataCNN(nn.Module):
    """Feature-word lookup is external; Conv1d -> tanh -> valid mean pooling."""

    def __init__(self, feature_dim: int, dim: int, kernel_size: int):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = kernel_size
        self.convolution = nn.Conv1d(feature_dim, dim, kernel_size)

    def forward(self, embeddings, valid):
        shape = embeddings.shape[:-2]
        width, feature_dim = embeddings.shape[-2:]
        embeddings = embeddings.reshape(-1, width, feature_dim)
        valid = valid.reshape(-1, width)
        lengths = valid.sum(-1)
        expected = torch.arange(width, device=valid.device).unsqueeze(0) < lengths.unsqueeze(1)
        if not torch.equal(valid, expected):
            raise ValueError("metadata tokens must be right padded")
        embedded = embeddings.masked_fill(~valid.unsqueeze(-1), 0).transpose(1, 2)
        if width < self.kernel_size:
            embedded = F.pad(embedded, (0, self.kernel_size - width))
        outputs = self.convolution(embedded).tanh()
        # Explicit short-document adaptation: one zero-padded window if 0<L<k.
        window_count = (lengths - self.kernel_size + 1).clamp_min(1) * lengths.gt(0)
        mask = torch.arange(outputs.shape[-1], device=valid.device)[None] < window_count[:, None]
        pooled = (outputs * mask.unsqueeze(1)).sum(-1) / window_count.clamp_min(1).unsqueeze(1)
        return pooled.reshape(*shape, -1)


class ViewAttention(nn.Module):
    """Eqs.11--13 and27--29 semantic-view attention."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.transform = nn.Linear(dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, views, valid):
        logits = self.query(self.transform(views).tanh()).squeeze(-1)
        attention = _masked_softmax(logits, valid)
        return (attention.unsqueeze(-1) * views).sum(-2), attention


class MEGANVerifiedRepresentation(MainTableModel):
    """Seven-view HMG representation and long/short/dynamic preference fusion.

    user IDs and session IDs are zero-based; feature 0 is padding; music uses
    ARRIVE's existing vocabulary with FIRST_ITEM as the first real music ID.
    All graph buffers must be built from permitted Train interactions only.
    This class cannot certify that provenance; it only validates typed shapes.
    """

    fidelity_status = "SOURCE_GROUNDED_ARRIVE_TASK_ADAPTATION"
    user_view_order = ("profile", "behavior", "session")
    music_view_order = ("attribute", "interaction", "transition", "context")
    def __init__(
        self, vocab_size: int, noise_probabilities: np.ndarray, *, embed_dim: int,
        user_tokens: torch.Tensor, music_tokens: torch.Tensor,
        feature_count: int, session_count: int,
        relations: Mapping[str, NeighborTable],
        line_embeddings: np.ndarray,
        line_offsets: Mapping[str, int],
        feature_dim: int = 32, hidden_dim: int = 32, kernel_size: int = 3,
        short_history: int = 20,
    ):
        super().__init__(vocab_size, noise_probabilities, embed_dim)
        user_tokens = torch.as_tensor(user_tokens, dtype=torch.long).clone()
        music_tokens = torch.as_tensor(music_tokens, dtype=torch.long).clone()
        for name, tokens in (("user", user_tokens), ("music", music_tokens)):
            if tokens.ndim != 2 or tokens.shape[1] < 1 or (tokens < 0).any() or (tokens >= feature_count).any():
                raise ValueError(f"invalid {name} feature token table")
        if len(music_tokens) != vocab_size or len(user_tokens) < 1 or session_count < 1:
            raise ValueError("invalid typed node cardinalities")
        self.register_buffer("user_tokens", user_tokens)
        self.register_buffer("music_tokens", music_tokens)
        self.user_embedding = nn.Embedding(len(user_tokens), embed_dim)
        self.session_embedding = nn.Embedding(session_count, embed_dim)
        self.feature_embedding = nn.Embedding(feature_count, feature_dim, padding_idx=0)
        line = torch.as_tensor(line_embeddings, dtype=torch.float32)
        if line.ndim != 2 or line.shape[1] != embed_dim or feature_dim != embed_dim:
            raise ValueError("LINE embeddings must match MEGAN embedding dimensions")
        expected_offsets = {
            "user": 0,
            "music": len(user_tokens),
            "session": len(user_tokens) + vocab_size,
            "feature": len(user_tokens) + vocab_size + session_count,
        }
        if {key: int(line_offsets[key]) for key in expected_offsets} != expected_offsets:
            raise ValueError("LINE node offsets do not match typed HMG cardinalities")
        if len(line) != expected_offsets["feature"] + feature_count:
            raise ValueError("LINE node count does not match typed HMG cardinalities")
        with torch.no_grad():
            self.user_embedding.weight.copy_(line[
                expected_offsets["user"]:expected_offsets["music"]
            ])
            self.item_embedding.weight.copy_(line[
                expected_offsets["music"]:expected_offsets["session"]
            ])
            self.session_embedding.weight.copy_(line[
                expected_offsets["session"]:expected_offsets["feature"]
            ])
            self.feature_embedding.weight.copy_(line[expected_offsets["feature"]:])
            self.item_embedding.weight[:FIRST_ITEM].zero_()
            self.feature_embedding.weight[0].zero_()
        self.profile_cnn = MetadataCNN(feature_dim, embed_dim, kernel_size)
        self.attribute_cnn = MetadataCNN(feature_dim, embed_dim, kernel_size)
        schema = {
            "user_music": (len(user_tokens), vocab_size, FIRST_ITEM),
            "user_session": (len(user_tokens), session_count, 0),
            "music_user": (vocab_size, len(user_tokens), 0),
            "music_music": (vocab_size, vocab_size, FIRST_ITEM),
            "music_session": (vocab_size, session_count, 0),
        }
        if set(relations) != set(schema):
            raise ValueError("all five typed HMG neighborhood relations are required")
        for name, (rows, target_count, minimum_id) in schema.items():
            table = relations[name]
            ids = torch.as_tensor(table.ids, dtype=torch.long).clone()
            weights = torch.as_tensor(table.weights, dtype=torch.float32).clone()
            if ids.ndim != 2 or ids.shape[0] != rows or ids.shape[1] < 1 or weights.shape != ids.shape:
                raise ValueError(f"invalid {name} table shape")
            valid = ids.ne(-1)
            if ((ids[valid] < minimum_id) | (ids[valid] >= target_count)).any():
                raise ValueError(f"{name} contains an out-of-domain node ID")
            if not torch.isfinite(weights).all() or (weights[valid] <= 0).any() or (weights[~valid] != 0).any():
                raise ValueError(f"{name} contains invalid edge weights")
            self.register_buffer(name + "_ids", ids)
            self.register_buffer(name + "_weights", weights)
        self.relation_attention = nn.ModuleDict({name: WeightedNeighborAttention(embed_dim, hidden_dim) for name in schema})
        self.user_view_attention = ViewAttention(embed_dim, hidden_dim)
        self.music_view_attention = ViewAttention(embed_dim, hidden_dim)
        self.short_history = int(short_history)
        if self.short_history < 1:
            raise ValueError("short_history must be positive")
        self.short_user = nn.Linear(embed_dim, embed_dim, bias=False)
        self.short_item = nn.Linear(embed_dim, embed_dim, bias=False)
        self.short_last = nn.Linear(embed_dim, embed_dim, bias=True)
        self.short_query = nn.Linear(embed_dim, 1, bias=False)
        self.preference_fusion = nn.Linear(3 * embed_dim, 3)

    def _relation(self, name, source_ids, targets, embedding):
        neighbors = getattr(self, name + "_ids")[source_ids]
        weights = getattr(self, name + "_weights")[source_ids]
        valid = neighbors.ne(-1)
        values = embedding(neighbors.clamp_min(0))
        aggregate, _ = self.relation_attention[name](targets, values, weights, valid)
        return aggregate, valid.any(-1)

    def user_views(self, user_ids):
        target = self.user_embedding(user_ids)
        tokens = self.user_tokens[user_ids]
        profile = self.profile_cnn(self.feature_embedding(tokens), tokens.ne(0))
        behavior, b_valid = self._relation("user_music", user_ids, target, self.item_embedding)
        session, s_valid = self._relation("user_session", user_ids, target, self.session_embedding)
        return torch.stack((profile, behavior, session), -2), torch.stack((tokens.ne(0).any(-1), b_valid, s_valid), -1)

    def music_views(self, music_ids):
        target = self.item_embedding(music_ids)
        tokens = self.music_tokens[music_ids]
        attribute = self.attribute_cnn(self.feature_embedding(tokens), tokens.ne(0))
        interaction, i_valid = self._relation("music_user", music_ids, target, self.user_embedding)
        transition, t_valid = self._relation("music_music", music_ids, target, self.item_embedding)
        context, c_valid = self._relation("music_session", music_ids, target, self.session_embedding)
        return torch.stack((attribute, interaction, transition, context), -2), torch.stack((tokens.ne(0).any(-1), i_valid, t_valid, c_valid), -1)

    def aggregate_user_views(self, user_ids):
        return self.user_view_attention(*self.user_views(user_ids))

    def aggregate_music_views(self, music_ids):
        return self.music_view_attention(*self.music_views(music_ids))

    def encode_batch(self, batch):
        if "user_ids" not in batch:
            raise ValueError("MEGAN requires globally aligned observed user_ids")
        sequence, positions = batch["sequence"], batch["initial_positions"]
        users = batch["user_ids"]
        if sequence.ndim != 2 or users.shape != positions.shape:
            raise ValueError("invalid MEGAN batch shapes")
        if bool(((users < 0) | (users >= len(self.user_tokens))).any()):
            raise ValueError("MEGAN user id outside frozen HMG")
        valid = sequence.ne(0)
        observed = valid & (
            torch.arange(sequence.shape[1], device=sequence.device)[None]
            <= positions[:, None]
        )
        recent = torch.arange(sequence.shape[1], device=sequence.device)[None]
        recent = observed & recent.ge((positions - self.short_history + 1)[:, None])
        long_term, _ = self.aggregate_user_views(users)
        item_vectors = self.item_vectors(sequence)
        rows = torch.arange(len(sequence), device=sequence.device)
        last = item_vectors[rows, positions]
        # Preserve the disclosed linear compatibility equation exactly.  Its
        # user/last terms are position-constant and therefore cancel inside
        # the following softmax; this source-level limitation is documented.
        logits = self.short_query(
            self.short_user(long_term)[:, None]
            + self.short_item(item_vectors)
            + self.short_last(last)[:, None]
        ).squeeze(-1)
        attention = _masked_softmax(logits, recent)
        short_term = (attention[..., None] * item_vectors).sum(1)
        weights = torch.sigmoid(self.preference_fusion(
            torch.cat((long_term, short_term, last), dim=-1)
        )).softmax(-1)
        state = (
            weights[:, :1] * long_term
            + weights[:, 1:2] * short_term
            + weights[:, 2:] * last
        )
        hidden = state.new_zeros((*sequence.shape, self.embed_dim))
        hidden[rows, positions] = state
        return hidden

    def item_vectors(self, indices):
        shape = indices.shape
        flat = indices.reshape(-1)
        if flat.numel() == 0:
            return self.item_embedding.weight.new_empty(*shape, self.embed_dim)
        # The shared sampled-set objective reuses the same catalog IDs many
        # times across a batch.  MEGAN's graph representation is item-local,
        # so compute each distinct item once and gather it back.  Autograd
        # accumulates through the inverse gather, preserving the exact loss.
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        if unique.numel() > 300000:
            # Full-catalog evaluation materializes every distinct item in one
            # aggregate_music_views call; the per-view attention buffers then
            # exceed a 24 GiB card alongside the 453M-parameter training
            # state.  Aggregation is item-local, so slicing the distinct
            # items changes no values and only lowers the peak footprint.
            parts = []
            for start in range(0, unique.numel(), 8192):
                part, _ = self.aggregate_music_views(unique[start:start + 8192])
                parts.append(part)
                if flat.is_cuda:
                    # Eval-only path: return each chunk's transient buffers
                    # to the driver so the caching allocator does not hold
                    # the whole catalog build alongside the training state.
                    torch.cuda.empty_cache()
            vectors = torch.cat(parts, dim=0)
        else:
            vectors, _ = self.aggregate_music_views(unique)
        return vectors.index_select(0, inverse).reshape(*shape, self.embed_dim)
