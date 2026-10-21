"""MUSE operators adapted to ARRIVE's observed-prefix / session-set contract.

Primary implementation: yunhak0/MUSE@6fdca43d3de1f9489464d80be359c146e71b48d6.
This is a task adaptation, not a reproduction of its MSSD experiment. See
refine-logs/arrive_repair_20260912/MUSE_FIDELITY.md for operator-level mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from arrive_main_table_models import MainTableModel
from lfm_timeconcat_control import FIRST_ITEM, PAD


UPSTREAM_COMMIT = "6fdca43d3de1f9489464d80be359c146e71b48d6"


@dataclass
class SessionGraph:
    items: torch.Tensor
    aliases: torch.Tensor
    incoming: torch.Tensor
    outgoing: torch.Tensor
    valid: torch.Tensor


def session_graph(sequence: torch.Tensor) -> SessionGraph:
    """Binary item-node graph with SR-GNN's directed degree normalization."""
    if sequence.ndim != 2 or not sequence.shape[0] or not sequence.shape[1]:
        raise ValueError("MUSE requires a nonempty [batch, length] sequence")
    valid = sequence.ne(PAD)
    if not bool(valid.any(1).all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("MUSE requires nonempty right-padded prefixes")
    if bool(sequence.lt(0).any()):
        raise ValueError("negative item ID")
    # Like the pinned implementation, topology is built from IDs on the CPU;
    # all learned message passing remains batched and differentiable.
    values = sequence.detach().cpu().numpy()
    batch, width = values.shape
    nodes = np.zeros_like(values)
    aliases = np.zeros_like(values)
    incoming = np.zeros((batch, width, width), dtype=np.float32)
    outgoing = np.zeros_like(incoming)
    for row, prefix in enumerate(values):
        unique, inverse = np.unique(prefix, return_inverse=True)
        nodes[row, :len(unique)] = unique
        aliases[row] = inverse
        adjacency = np.zeros((width, width), dtype=np.float32)
        length = int(np.count_nonzero(prefix != PAD))
        adjacency[inverse[:length-1], inverse[1:length]] = 1.0
        incoming[row] = (adjacency / np.maximum(adjacency.sum(0), 1)).T
        outgoing[row] = adjacency / np.maximum(adjacency.sum(1, keepdims=True), 1)
    device = sequence.device
    return SessionGraph(
        torch.as_tensor(nodes, device=device), torch.as_tensor(aliases, device=device),
        torch.as_tensor(incoming, device=device), torch.as_tensor(outgoing, device=device), valid,
    )


class GatedSessionGraph(nn.Module):
    """SR-GNN edge transforms, channel biases, and GRU node update."""

    def __init__(self, dimension: int, steps: int):
        super().__init__()
        self.steps = steps
        self.edge_in = nn.Linear(dimension, dimension)
        self.edge_out = nn.Linear(dimension, dimension)
        self.bias_in = nn.Parameter(torch.empty(dimension))
        self.bias_out = nn.Parameter(torch.empty(dimension))
        self.cell = nn.GRUCell(2 * dimension, dimension)

    def forward(self, graph: SessionGraph, hidden: torch.Tensor) -> torch.Tensor:
        for _ in range(self.steps):
            incoming = graph.incoming.to(hidden.dtype) @ self.edge_in(hidden) + self.bias_in
            outgoing = graph.outgoing.to(hidden.dtype) @ self.edge_out(hidden) + self.bias_out
            messages = torch.cat((incoming, outgoing), -1)
            hidden = self.cell(messages.flatten(0, 1), hidden.flatten(0, 1)).view_as(hidden)
            hidden = hidden.masked_fill(graph.items.eq(PAD).unsqueeze(-1), 0.0)
        return hidden


def nearest_pairs(first, second, first_valid, second_valid, count):
    """Per-episode top-k nearest pairs in one direction; no padding matches."""
    distance = torch.cdist(first, second).square()
    eligible = first_valid.unsqueeze(2) & second_valid.unsqueeze(1)
    distance = distance.masked_fill(~eligible, float("inf"))
    nearest_distance, nearest = distance.min(-1)
    k = min(int(count), first.shape[1])
    # Stable ties and original-source order match the pinned matching operator.
    source = torch.argsort(nearest_distance, dim=1, stable=True)[:, :k]
    source = source.sort(1).values
    destination = nearest.gather(1, source)
    valid = first_valid.gather(1, source) & second_valid.gather(1, destination)
    left = first.gather(1, source.unsqueeze(-1).expand(-1, -1, first.shape[-1]))
    right = second.gather(1, destination.unsqueeze(-1).expand(-1, -1, second.shape[-1]))
    return left, right, valid


def identity_pairs(first, second, first_ids, second_ids):
    """One pair per shared item ID, as in the paper's item-set definition."""
    source_valid, target_valid = first_ids.ne(PAD), second_ids.ne(PAD)
    equal_source = first_ids.unsqueeze(2).eq(first_ids.unsqueeze(1))
    before = torch.ones(first_ids.shape[1], first_ids.shape[1], dtype=torch.bool,
                        device=first_ids.device).tril(-1)
    first_occurrence = ~(equal_source & before).any(-1)
    matches = first_ids.unsqueeze(2).eq(second_ids.unsqueeze(1)) & target_valid.unsqueeze(1)
    valid = source_valid & first_occurrence & matches.any(-1)
    destination = matches.long().argmax(-1)
    paired = second.gather(1, destination.unsqueeze(-1).expand(-1, -1, second.shape[-1]))
    return first, paired, valid


def masked_vicreg(first, second, valid, *, invariance=1.0, variance=1.0, covariance=10.0):
    """VICReg across batch per pair-slot, excluding unavailable matches.

    Inputs are [B, slots, D]. A singleton slot uses zero sample variance and
    covariance rather than the upstream undefined n-1 division. This keeps
    a real final batch of size one finite without treating padding as data.
    """
    mask = valid.unsqueeze(-1).to(first.dtype)
    dimension = first.shape[-1]
    pair_count = valid.sum()
    inv = ((first - second).square() * mask).sum() / (pair_count * dimension).clamp_min(1)
    counts = valid.sum(0).to(first.dtype)
    occupied = counts.gt(0)
    var_sum = first.sum() * 0.0
    cov_sum = first.sum() * 0.0
    for values in (first, second):
        mean = (values * mask).sum(0) / counts.clamp_min(1).unsqueeze(-1)
        centered = (values - mean.unsqueeze(0)) * mask
        denominator = (counts - 1).clamp_min(1)
        sample_variance = centered.square().sum(0) / denominator.unsqueeze(-1)
        penalty = F.relu(1.0 - (sample_variance + 1e-4).sqrt()).mean(-1)
        var_sum = var_sum + (penalty * occupied).sum() / occupied.sum().clamp_min(1) / 2
        cov = torch.einsum("bkd,bke->kde", centered, centered) / denominator[:, None, None]
        off_diagonal = ~torch.eye(dimension, dtype=torch.bool, device=first.device)
        penalty = cov[:, off_diagonal].square().sum(-1) / dimension
        cov_sum = cov_sum + (penalty * occupied).sum() / occupied.sum().clamp_min(1) / 2
    return invariance * inv + variance * var_sum + covariance * cov_sum


class MUSERepairedAdapter(MainTableModel):
    """Paper/source-mapped MUSE model with an explicitly changed ARRIVE task."""

    def __init__(self, vocab_size, noise_probabilities, *, embed_dim, max_sequence,
                 num_layers=1, dropout=0.0, alpha=0.2, num_matches=(5, 5),
                 warmup_epochs=1, inv_coeff=1.0, var_coeff=1.0, cov_coeff=10.0,
                 max_augmented_sequence=None, **_: Any):
        super().__init__(vocab_size, noise_probabilities, embed_dim)
        if not 0 <= alpha <= 1 or num_layers < 1 or max_sequence < 1 or warmup_epochs < 0:
            raise ValueError("invalid MUSE architecture or matching configuration")
        if len(num_matches) != 2 or any(int(value) < 1 for value in num_matches):
            raise ValueError("num_matches requires two positive directional counts")
        if dropout != 0:
            raise ValueError("pinned MUSE SRGNN has no dropout; pass dropout=0 explicitly")
        self.max_sequence, self.alpha = int(max_sequence), float(alpha)
        self.max_augmented_sequence = (self.max_sequence if max_augmented_sequence is None
                                       else int(max_augmented_sequence))
        if self.max_augmented_sequence < self.max_sequence:
            raise ValueError("augmented capacity must preserve the entire observed prefix")
        self.num_matches, self.warmup_epochs = tuple(map(int, num_matches)), int(warmup_epochs)
        self.vicreg_coefficients = dict(invariance=inv_coeff, variance=var_coeff, covariance=cov_coeff)
        self.gnn = GatedSessionGraph(embed_dim, num_layers)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        # The official SRGNN uses a vector gate (D outputs), without softmax.
        self.linear3 = nn.Linear(embed_dim, embed_dim)
        self.linear_transform = nn.Linear(2 * embed_dim, embed_dim)
        self.register_buffer("training_epoch", torch.tensor(0, dtype=torch.long))
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -1 / math.sqrt(embed_dim), 1 / math.sqrt(embed_dim))
        with torch.no_grad():
            self.item_embedding.weight[PAD].zero_()

    def set_epoch(self, epoch: int) -> None:
        """One-based epoch supplied by the shared controller, also on resume."""
        if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 1:
            raise ValueError("MUSE epoch must be a positive one-based integer")
        self.training_epoch.fill_(int(epoch))

    def encode_sequence(self, sequence):
        if sequence.shape[1] > self.max_sequence:
            raise ValueError("MUSE observed prefix exceeds configured width")
        return self._graph_tokens(sequence)

    def _graph_tokens(self, sequence):
        graph = session_graph(sequence)
        nodes = self.gnn(graph, self.item_embedding(graph.items))
        tokens = nodes.gather(1, graph.aliases.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        return tokens.masked_fill(~graph.valid.unsqueeze(-1), 0.0)

    def encode_batch(self, batch):
        sequence = batch["sequence"]
        if not torch.equal(batch["initial_positions"], sequence.ne(PAD).sum(1) - 1):
            raise ValueError("MUSE batch must contain observed prefix only, no later targets")
        return self.encode_sequence(sequence)

    def gather_states(self, hidden, positions):
        rows = torch.arange(len(hidden), device=hidden.device)
        last = hidden[rows, positions]
        valid = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0) <= positions.unsqueeze(1)
        gate = self.linear3(torch.sigmoid(self.linear1(last).unsqueeze(1) + self.linear2(hidden)))
        pooled = (gate * hidden * valid.unsqueeze(-1)).sum(1)
        return self.linear_transform(torch.cat((pooled, last), -1))

    def matching_components(self, batch, hidden):
        epoch = int(self.training_epoch.item())
        if epoch < 1:
            raise ValueError("call MUSE.set_epoch(one_based_epoch) before computing training loss")
        if "augmented_sequence" not in batch:
            raise ValueError("MUSE requires an explicit observed-prefix augmented view")
        augmented = batch["augmented_sequence"]
        if augmented.shape[1] > self.max_augmented_sequence:
            raise ValueError("MUSE augmented view exceeds its explicitly configured capacity")
        other = self._graph_tokens(augmented)
        original_valid, augmented_valid = batch["sequence"].ne(PAD), augmented.ne(PAD)
        state = self.gather_states(hidden, batch["initial_positions"])
        augmented_state = self.gather_states(other, augmented_valid.sum(1) - 1)
        all_rows = torch.ones(len(hidden), 1, dtype=torch.bool, device=hidden.device)
        global_loss = masked_vicreg(state[:, None], augmented_state[:, None], all_rows,
                                    **self.vicreg_coefficients)
        pairs = identity_pairs(hidden, other, batch["sequence"], augmented)
        item_loss = masked_vicreg(*pairs, **self.vicreg_coefficients)
        similarity_loss = hidden.sum() * 0.0
        if epoch > self.warmup_epochs:
            forward = nearest_pairs(hidden, other, original_valid, augmented_valid, self.num_matches[0])
            reverse = nearest_pairs(other, hidden, augmented_valid, original_valid, self.num_matches[1])
            similarity_loss = 0.5 * (masked_vicreg(*forward, **self.vicreg_coefficients)
                                     + masked_vicreg(*reverse, **self.vicreg_coefficients))
        return {"global": global_loss, "item": item_loss, "similarity": similarity_loss}

    def auxiliary_loss(self, batch, hidden):
        parts = self.matching_components(batch, hidden)
        return (1 - self.alpha) * parts["global"] + self.alpha * (parts["item"] + parts["similarity"])


def train_transition_counts(events, times, sessions, vocab_size, *, session_gap_seconds=1800):
    """Build full transition counts from split-0 observed timelines only.

    Prepared return episodes repeat users' earlier histories, so timestamp/item
    pairs are deduplicated per user before adjacent within-session transitions
    are counted. This matches the existing ARRIVE transition-cache boundary.
    """
    train = sessions[sessions[:, 5] == 0]
    train = train[np.argsort(train[:, 0], kind="stable")]
    user_boundaries = np.r_[0, np.flatnonzero(np.diff(train[:, 0])) + 1, len(train)]
    counts: dict[tuple[int, int], int] = {}
    # Retain only one user's repeated timeline, not every user's event dict.
    for start, stop in zip(user_boundaries, user_boundaries[1:]):
        timeline: set[tuple[int, int]] = set()
        for _, begin, _, end, _, _ in train[start:stop]:
            for item, timestamp in zip(events[int(begin):int(end)], times[int(begin):int(end)]):
                if FIRST_ITEM <= int(item) < vocab_size:
                    timeline.add((int(timestamp), int(item)))
        ordered = sorted(timeline)
        for left, right in zip(ordered, ordered[1:]):
            if right[0] - left[0] <= session_gap_seconds:
                key = (left[1], right[1])
                counts[key] = counts.get(key, 0) + 1
    if not counts:
        return sparse.csr_matrix((vocab_size, vocab_size), dtype=np.float64)
    edges = np.asarray(list(counts), dtype=np.int64)
    return sparse.csr_matrix((np.asarray(list(counts.values()), dtype=np.float64),
                              (edges[:, 0], edges[:, 1])), shape=(vocab_size, vocab_size))


class MUSETransitionAugmenter:
    """Stochastic MUSE bridge insertion on observed within-session edges.

    Music4All has no observed shuffle flag, so all eligible observed edges use
    the transition branch. We do not infer a shuffle label or reorder the
    first-return anchor. This explicit task adaptation is not MSSD parity.
    """

    def __init__(self, transition_counts, *, max_sequence, session_gap_seconds=1800,
                 transition_r=1.0, proposal_cache_size=131072):
        counts = sparse.csr_matrix(transition_counts, dtype=np.float64, copy=True)
        counts.sum_duplicates()
        if counts.shape[0] != counts.shape[1] or max_sequence < 1 or session_gap_seconds < 0:
            raise ValueError("invalid transition catalog or maximum length")
        if not 0 <= transition_r <= 1:
            raise ValueError("transition_r must lie in [0,1]")
        if isinstance(proposal_cache_size, bool) or int(proposal_cache_size) != proposal_cache_size \
                or proposal_cache_size < 0:
            raise ValueError("proposal_cache_size must be a nonnegative integer")
        if (not np.isfinite(counts.data).all() or (counts.data < 0).any()
                or (counts.data != np.floor(counts.data)).any()):
            raise ValueError("transition counts must be raw nonnegative integers, not log-score caches")
        # Pinned embedder.py filters cnt > 1, then applies log(cnt).
        counts = counts.tocoo()
        keep = (counts.row >= FIRST_ITEM) & (counts.col >= FIRST_ITEM) & (counts.data > 1)
        counts = sparse.csr_matrix((np.log(counts.data[keep]),
                                    (counts.row[keep], counts.col[keep])), shape=counts.shape)
        counts.eliminate_zeros()
        self.source = self._normalize(counts)
        self.target = self._normalize(counts.T.tocsr())
        self.max_sequence, self.session_gap_seconds = int(max_sequence), int(session_gap_seconds)
        self.transition_r = float(transition_r)
        # Bridge distributions depend only on a pair of observed item IDs and
        # the frozen transition-count matrix.  The former implementation built
        # two temporary sparse rows plus a sparse product for every occurrence
        # of every observed edge.  Direct CSR intersection preserves the exact
        # candidate order and probabilities; the bounded FIFO cache only avoids
        # recomputing deterministic distributions and never stores RNG draws.
        self.proposal_cache_size = int(proposal_cache_size)
        self._proposal_cache = {}
        self.proposal_cache_hits = 0
        self.proposal_cache_misses = 0

    @classmethod
    def from_counts(cls, transition_counts, *, max_sequence, session_gap_seconds=1800,
                    transition_r=1.0):
        """Explicit raw-count constructor; top-k/log-score caches are invalid."""
        return cls(transition_counts, max_sequence=max_sequence,
                   session_gap_seconds=session_gap_seconds, transition_r=transition_r)

    @staticmethod
    def _normalize(matrix):
        total = np.asarray(matrix.sum(1)).ravel()
        inverse = np.divide(1.0, total, out=np.zeros_like(total), where=total > 0)
        normalized = (sparse.diags(inverse) @ matrix).tocsr()
        normalized.sort_indices()
        return normalized

    def _compute_bridge_distribution(self, source: int, target: int):
        """Exact sparse-row product without constructing temporary matrices."""
        left_start, left_stop = self.source.indptr[source : source + 2]
        right_start, right_stop = self.target.indptr[target : target + 2]
        left_indices = self.source.indices[left_start:left_stop]
        right_indices = self.target.indices[right_start:right_stop]
        if not len(left_indices) or not len(right_indices):
            return None
        left_values = self.source.data[left_start:left_stop]
        right_values = self.target.data[right_start:right_stop]
        # Both CSR rows have sorted unique indices.  Probe the shorter row into
        # the longer one instead of sorting concatenated rows in intersect1d.
        # The surviving candidates remain ascending, matching the legacy CSR
        # product's order exactly.
        if len(left_indices) <= len(right_indices):
            locations = np.searchsorted(right_indices, left_indices)
            valid = locations < len(right_indices)
            matched = valid.copy()
            matched[valid] = right_indices[locations[valid]] == left_indices[valid]
            candidates = left_indices[matched]
            probabilities = left_values[matched] * right_values[locations[matched]]
        else:
            locations = np.searchsorted(left_indices, right_indices)
            valid = locations < len(left_indices)
            matched = valid.copy()
            matched[valid] = left_indices[locations[valid]] == right_indices[valid]
            candidates = right_indices[matched]
            probabilities = left_values[locations[matched]] * right_values[matched]
        if not len(candidates):
            return None
        logits = probabilities - probabilities.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        return candidates, weights

    def _bridge_distribution(self, source: int, target: int):
        key = (int(source), int(target))
        if key in self._proposal_cache:
            self.proposal_cache_hits += 1
            return self._proposal_cache[key]
        self.proposal_cache_misses += 1
        distribution = self._compute_bridge_distribution(*key)
        if distribution is None:
            result = None
        else:
            candidates, weights = distribution
            cumulative = weights.cumsum()
            # Generator.choice(p=...) normalizes the cumulative probability
            # before applying searchsorted to one uniform draw.  Retaining that
            # exact operation once in the deterministic cache avoids invoking
            # the comparatively expensive general-purpose choice path for
            # every observed edge.
            cumulative /= cumulative[-1]
            result = candidates, cumulative
        if self.proposal_cache_size:
            if len(self._proposal_cache) >= self.proposal_cache_size:
                self._proposal_cache.pop(next(iter(self._proposal_cache)))
            self._proposal_cache[key] = result
        return result

    @staticmethod
    def _draw_candidate(candidates, cumulative, rng: np.random.Generator) -> int:
        chosen = np.searchsorted(cumulative, rng.random(), side="right")
        return int(candidates[chosen])

    def __call__(self, items, times, rng: np.random.Generator):
        if len(items) == 0 or len(items) != len(times) or len(items) > self.max_sequence:
            raise ValueError("invalid observed augmentation prefix")
        if np.any(np.diff(times) < 0):
            raise ValueError("observed augmentation timestamps must be nondecreasing")
        if any(int(item) < 0 or int(item) >= self.source.shape[0] for item in items):
            raise ValueError("observed item outside transition catalog")
        available = self.max_sequence - len(items)
        if available == 0:
            return list(items), list(range(len(items)))
        proposals = []
        for at, (source, target) in enumerate(zip(items, items[1:])):
            if source < FIRST_ITEM or target < FIRST_ITEM:
                continue
            if times[at + 1] - times[at] > self.session_gap_seconds:
                continue
            distribution = self._bridge_distribution(source, target)
            if distribution is None:
                continue
            candidates, cumulative = distribution
            proposals.append((at, self._draw_candidate(candidates, cumulative, rng)))
        if len(proposals) > available:
            selected = rng.choice(len(proposals), available, replace=False)
            proposals = [proposals[int(at)] for at in selected]
        else:
            # Pinned Transition first samples one candidate per supported edge;
            # with sufficient room it then applies independent Bernoulli gates.
            selected = rng.binomial(1, self.transition_r, size=len(proposals)).astype(bool)
            proposals = [proposal for proposal, keep in zip(proposals, selected) if keep]
        inserted = dict(proposals)
        augmented, positions = [], []
        for at, item in enumerate(items):
            augmented.append(int(item))
            positions.append(at)
            if at in inserted:
                augmented.append(inserted[at])
                positions.append(-1)
        return augmented, positions
