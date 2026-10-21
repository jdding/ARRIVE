#!/usr/bin/env python3
"""True TimeConcat-SASRec control for LFM return-session continuation.

The model supports matched multi-positive, teacher-forced continuation, and
joint objectives.  The same checkpoint supports one-shot and autoregressive
decoding, which isolates rollout from backbone capacity.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from music_sequence_core import set_metrics


PAD, UNK, EOS, FIRST_ITEM = 0, 1, 2, 3


class LFMTimeConcatData:
    """Hash-bound prepared LFM arrays with no metric identities in training."""

    def __init__(self, root: Path, *, max_context: int = 129, horizon: int = 64) -> None:
        self.root = Path(root)
        self.max_context = int(max_context)
        self.horizon = int(horizon)
        self.events = np.load(self.root / "events.npy", mmap_mode="r")
        self.sessions = np.load(self.root / "sessions.npy", mmap_mode="r")
        self.frequencies = np.load(self.root / "train_frequencies.npy", mmap_mode="r")
        self.vocab_size = len(self.frequencies)
        if self.max_context < 1 or self.horizon < 1:
            raise ValueError("invalid context or horizon")

    def row_ids(self, split: int, *, returns_only: bool = False) -> np.ndarray:
        selected = self.sessions[:, 5] == split
        if returns_only:
            selected &= self.sessions[:, 4] >= 30 * 86400
        ids = np.flatnonzero(selected)
        if split == 0:
            # A training row must expose at least one train-vocabulary target.
            keep = [bool(self.known_suffix(int(index))) for index in ids]
            ids = ids[np.asarray(keep, dtype=bool)]
        return ids

    def context(self, row_id: int) -> list[int]:
        _, first_user, start, _, _, _ = map(int, self.sessions[row_id])
        begin = max(first_user, start - self.max_context + 1)
        values = list(map(int, self.events[begin : start + 1]))
        if not values:
            raise ValueError("empty context")
        return values

    def known_suffix(self, row_id: int) -> list[int]:
        _, _, start, end, _, _ = map(int, self.sessions[row_id])
        values = map(int, self.events[start + 1 : min(end, start + 1 + self.horizon)])
        return [value for value in values if value >= FIRST_ITEM]

    def gap_days(self, row_id: int) -> float:
        return max(float(self.sessions[row_id, 4]) / 86400.0, 0.0)

    def training_batch(
        self,
        row_ids: Iterable[int],
        device: torch.device,
        *,
        target_order: str = "observed",
        shuffle_seed: int = 0,
    ) -> dict[str, Any]:
        if target_order not in {"observed", "shuffled"}:
            raise ValueError("unsupported target order")
        contexts, targets, gaps = [], [], []
        for row_id in row_ids:
            context = self.context(int(row_id))
            suffix = self.known_suffix(int(row_id))
            if not suffix:
                raise ValueError("training row has no known target")
            if target_order == "shuffled" and len(suffix) > 1:
                rng = np.random.default_rng(np.random.SeedSequence([shuffle_seed, int(row_id)]))
                permutation = rng.permutation(len(suffix))
                if np.array_equal(permutation, np.arange(len(suffix))):
                    permutation = np.roll(permutation, 1)
                suffix = [suffix[int(index)] for index in permutation]
            contexts.append(context)
            targets.append(suffix)
            gaps.append(self.gap_days(int(row_id)))
        return build_training_batch(
            contexts,
            targets,
            gaps,
            max_sequence=self.max_context + self.horizon - 1,
            device=device,
        )

    def serving_batch(
        self,
        contexts: list[list[int]],
        gaps: list[float],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        max_sequence = self.max_context + self.horizon - 1
        sequence = torch.full((len(contexts), max_sequence), PAD, dtype=torch.long)
        final_positions = torch.empty(len(contexts), dtype=torch.long)
        for row, values in enumerate(contexts):
            values = values[-max_sequence:]
            sequence[row, : len(values)] = torch.as_tensor(values, dtype=torch.long)
            final_positions[row] = len(values) - 1
        return {
            "sequence": sequence.to(device),
            "state_positions": final_positions.to(device),
            "gap_days": torch.as_tensor(gaps, dtype=torch.float32, device=device),
        }


def build_training_batch(
    contexts: list[list[int]],
    targets: list[list[int]],
    gaps: list[float],
    *,
    max_sequence: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build fixed-width causal states for both matched retrieval objectives."""

    if not contexts or not (len(contexts) == len(targets) == len(gaps)):
        raise ValueError("misaligned training batch")
    sequence = torch.full((len(contexts), max_sequence), PAD, dtype=torch.long)
    initial_positions = []
    ar_positions, ar_targets, ar_sessions, ar_weights = [], [], [], []
    set_targets, set_sessions, set_weights = [], [], []
    for session, (context, suffix) in enumerate(zip(contexts, targets)):
        if not context or not suffix or any(target < FIRST_ITEM for target in suffix):
            raise ValueError("invalid context or target")
        combined = (context + suffix[:-1])[-max_sequence:]
        sequence[session, : len(combined)] = torch.as_tensor(combined, dtype=torch.long)
        initial = len(context) - 1
        initial_positions.append(initial)
        for position, target in enumerate(suffix):
            ar_positions.append((session, initial + position))
            ar_targets.append(target)
            ar_sessions.append(session)
            ar_weights.append(1.0 / len(suffix))
        unique = list(dict.fromkeys(suffix))
        for target in unique:
            set_targets.append(target)
            set_sessions.append(session)
            set_weights.append(1.0 / len(unique))
    if max(position for _, position in ar_positions) >= max_sequence:
        raise ValueError("training target state exceeds sequence")
    return {
        "sequence": sequence.to(device),
        "gap_days": torch.as_tensor(gaps, dtype=torch.float32, device=device),
        "initial_positions": torch.as_tensor(initial_positions, dtype=torch.long, device=device),
        "ar_positions": torch.as_tensor(ar_positions, dtype=torch.long, device=device),
        "ar_targets": torch.as_tensor(ar_targets, dtype=torch.long, device=device),
        "ar_sessions": torch.as_tensor(ar_sessions, dtype=torch.long, device=device),
        "ar_weights": torch.as_tensor(ar_weights, dtype=torch.float32, device=device),
        "set_targets": torch.as_tensor(set_targets, dtype=torch.long, device=device),
        "set_sessions": torch.as_tensor(set_sessions, dtype=torch.long, device=device),
        "set_weights": torch.as_tensor(set_weights, dtype=torch.float32, device=device),
    }


class LFMTimeConcatSASRec(nn.Module):
    """TimeConcat-SASRec with tied item retrieval and shared rollout states."""

    def __init__(
        self,
        vocab_size: int,
        noise_probabilities: np.ndarray,
        *,
        embed_dim: int = 128,
        max_sequence: int = 192,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if vocab_size <= FIRST_ITEM or len(noise_probabilities) != vocab_size:
            raise ValueError("invalid vocabulary")
        self.vocab_size = int(vocab_size)
        self.max_sequence = int(max_sequence)
        self.item_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.position_embedding = nn.Embedding(max_sequence, embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        probabilities = torch.as_tensor(noise_probabilities, dtype=torch.float32)
        if probabilities[:FIRST_ITEM].sum() != 0 or not torch.isclose(probabilities.sum(), torch.tensor(1.0)):
            raise ValueError("noise probabilities must cover train catalog items only")
        self.register_buffer("noise_probabilities", probabilities, persistent=False)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.item_embedding.weight[PAD].zero_()

    def encode_all(self, sequence: torch.Tensor, gap_days: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2 or sequence.shape[1] != self.max_sequence:
            raise ValueError("sequence shape does not match frozen width")
        positions = torch.arange(self.max_sequence, device=sequence.device).unsqueeze(0)
        time_shift = self.time_mlp(torch.log1p(gap_days.clamp_min(0)).unsqueeze(-1)).unsqueeze(1)
        hidden = self.item_embedding(sequence) + self.position_embedding(positions) + time_shift
        hidden = self.dropout(hidden)
        hidden = hidden.masked_fill(sequence.eq(PAD).unsqueeze(-1), 0.0)
        causal = torch.triu(
            torch.full((self.max_sequence, self.max_sequence), float("-inf"), device=sequence.device),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal)
        return self.layer_norm(hidden)

    @staticmethod
    def gather_states(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, positions]

    def sampled_loss(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        sessions: torch.Tensor,
        weights: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        if states.shape[0] != targets.numel() or targets.numel() != weights.numel():
            raise ValueError("sampled-loss tensors are misaligned")
        positive = (states * self.item_embedding(targets)).sum(-1, keepdim=True)
        negative = states @ self.item_embedding(negatives).T
        count = float(negatives.numel())
        positive -= torch.log(count * self.noise_probabilities[targets].unsqueeze(1)).clamp_min(-30)
        negative -= torch.log(count * self.noise_probabilities[negatives]).clamp_min(-30).unsqueeze(0)
        negative = negative.masked_fill(negatives.unsqueeze(0).eq(targets.unsqueeze(1)), float("-inf"))
        logits = torch.cat([positive, negative], dim=1)
        losses = torch.nn.functional.cross_entropy(
            logits, torch.zeros(states.shape[0], dtype=torch.long, device=states.device), reduction="none"
        )
        per_session = torch.zeros(int(sessions.max()) + 1, device=states.device)
        per_session.scatter_add_(0, sessions, losses * weights)
        return per_session.mean()

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        negative_samples: int,
        objective: str = "shared",
        negatives: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if objective not in {
            "shared",
            "oneshot_only",
            "continuation_only",
            "shared_shuffled",
            "shuffled_continuation_only",
        }:
            raise ValueError("unsupported objective")
        hidden = self.encode_all(batch["sequence"], batch["gap_days"])
        if negatives is None:
            negatives = torch.multinomial(self.noise_probabilities, negative_samples, replacement=True)
        elif negatives.numel() != negative_samples:
            raise ValueError("negative sample count mismatch")
        initial = self.gather_states(hidden, batch["initial_positions"])
        set_states = initial[batch["set_sessions"]]
        set_loss = self.sampled_loss(
            set_states,
            batch["set_targets"],
            batch["set_sessions"],
            batch["set_weights"],
            negatives,
        )
        if objective in {
            "shared",
            "continuation_only",
            "shared_shuffled",
            "shuffled_continuation_only",
        }:
            ar_states = hidden[batch["ar_positions"][:, 0], batch["ar_positions"][:, 1]]
            ar_loss = self.sampled_loss(
                ar_states,
                batch["ar_targets"],
                batch["ar_sessions"],
                batch["ar_weights"],
                negatives,
            )
            total = (
                ar_loss
                if objective in {"continuation_only", "shuffled_continuation_only"}
                else 0.5 * set_loss + 0.5 * ar_loss
            )
        else:
            ar_loss = set_loss.detach() * 0.0
            total = set_loss
        return total, {"set_loss": float(set_loss.detach()), "ar_loss": float(ar_loss.detach())}


def popularity_noise(frequencies: np.ndarray, power: float = 0.75) -> np.ndarray:
    values = np.asarray(frequencies, dtype=np.float64).copy()
    values[:FIRST_ITEM] = 0
    values = np.power(np.maximum(values, 0), power)
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("empty popularity noise distribution")
    return values / total


@torch.no_grad()
def exact_catalog_topk(
    model: LFMTimeConcatSASRec,
    states: torch.Tensor,
    *,
    k: int,
    chunk_size: int = 50000,
    exclude: list[set[int]] | None = None,
) -> torch.Tensor:
    """Exact tied-head Top-K over all known catalog items, in bounded memory."""

    best_scores = torch.full((states.shape[0], k), float("-inf"), device=states.device)
    best_items = torch.full((states.shape[0], k), -1, dtype=torch.long, device=states.device)
    for start in range(FIRST_ITEM, model.vocab_size, chunk_size):
        stop = min(start + chunk_size, model.vocab_size)
        scores = states @ model.item_embedding.weight[start:stop].T
        if exclude:
            for row, items in enumerate(exclude):
                local = [item - start for item in items if start <= item < stop]
                if local:
                    scores[row, torch.as_tensor(local, device=scores.device)] = float("-inf")
        local_k = min(k, stop - start)
        local_scores, local_items = scores.topk(local_k, dim=1)
        local_items += start
        joined_scores = torch.cat([best_scores, local_scores], dim=1)
        joined_items = torch.cat([best_items, local_items], dim=1)
        best_scores, chosen = joined_scores.topk(k, dim=1)
        best_items = joined_items.gather(1, chosen)
    if (best_items < FIRST_ITEM).any():
        raise ValueError("catalog Top-K was not filled")
    return best_items


def gap_bucket(days: float) -> str:
    if days < 90:
        return "30-90d"
    if days < 180:
        return "90-180d"
    if days <= 365:
        return "180-365d"
    return ">365d"


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for decoder in ("oneshot", "autoregressive"):
        for scope in ("overall", "30-90d", "90-180d", "180-365d", ">365d"):
            selected = rows if scope == "overall" else [row for row in rows if row["gap_bucket"] == scope]
            by_user: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"recall": [], "ndcg": []})
            for row in selected:
                recall = row[f"{decoder}_recall10"]
                ndcg = row[f"{decoder}_ndcg10"]
                # Empty continuation labels are retained in the protocol row
                # count, but they do not define Recall/NDCG observations.
                if recall is None or ndcg is None:
                    if recall is not None or ndcg is not None:
                        raise ValueError("partial_null_metric_pair")
                    continue
                by_user[row["user"]]["recall"].append(float(recall))
                by_user[row["user"]]["ndcg"].append(float(ndcg))
            metric_rows = sum(len(value["recall"]) for value in by_user.values())
            output[f"{decoder}:{scope}"] = {
                "rows": len(selected),
                "metric_rows": metric_rows,
                "users": len(by_user),
                "recall_at_10": float(np.mean([np.mean(value["recall"]) for value in by_user.values()])) if by_user else None,
                "ndcg_at_10": float(np.mean([np.mean(value["ndcg"]) for value in by_user.values()])) if by_user else None,
            }
    return output


@torch.no_grad()
def evaluate_shared_checkpoint(
    model: LFMTimeConcatSASRec,
    data: LFMTimeConcatData,
    *,
    split: int,
    device: torch.device,
    batch_size: int = 16,
    limit: int = 0,
) -> dict[str, Any]:
    """Evaluate one-shot and free-running AR outputs from one checkpoint."""

    model.eval()
    row_ids = data.row_ids(split, returns_only=True)
    if limit:
        row_ids = row_ids[:limit]
    identities = np.load(data.root / "metric_labels.npz", allow_pickle=False)
    lookup = {int(row_id): index for index, row_id in enumerate(identities["row_ids"])}
    rows: list[dict[str, Any]] = []
    for at in range(0, len(row_ids), batch_size):
        ids = row_ids[at : at + batch_size]
        contexts = [data.context(int(row_id)) for row_id in ids]
        gaps = [data.gap_days(int(row_id)) for row_id in ids]
        serving = data.serving_batch(contexts, gaps, device)
        hidden = model.encode_all(serving["sequence"], serving["gap_days"])
        state = model.gather_states(hidden, serving["state_positions"])
        one_shot = exact_catalog_topk(model, state, k=10)

        generated: list[list[int]] = [[] for _ in ids]
        for _ in range(10):
            rollout_contexts = [context + generated[row] for row, context in enumerate(contexts)]
            serving = data.serving_batch(rollout_contexts, gaps, device)
            hidden = model.encode_all(serving["sequence"], serving["gap_days"])
            state = model.gather_states(hidden, serving["state_positions"])
            chosen = exact_catalog_topk(
                model, state, k=1, exclude=[set(values) for values in generated]
            )[:, 0].cpu().tolist()
            for row, item in enumerate(chosen):
                generated[row].append(int(item))

        for offset, row_id in enumerate(ids):
            identity_index = lookup[int(row_id)]
            truth = identities["values"][
                identities["offsets"][identity_index] : identities["offsets"][identity_index + 1]
            ].tolist()
            one = one_shot[offset].cpu().tolist()
            autoregressive = generated[offset]
            one_recall, one_ndcg = set_metrics(one, truth)
            ar_recall, ar_ndcg = set_metrics(autoregressive, truth)
            user, _, start, _, gap, _ = map(int, data.sessions[row_id])
            rows.append(
                {
                    "row_id": int(row_id),
                    "user": user,
                    "gap_days": gap / 86400.0,
                    "gap_bucket": gap_bucket(gap / 86400.0),
                    "prefix_known": int(data.events[start]) >= FIRST_ITEM,
                    "suffix_length": len(truth),
                    "known_suffix_events": sum(value >= FIRST_ITEM for value in truth),
                    "oneshot_prediction": one,
                    "autoregressive_prediction": autoregressive,
                    "oneshot_recall10": one_recall,
                    "oneshot_ndcg10": one_ndcg,
                    "autoregressive_recall10": ar_recall,
                    "autoregressive_ndcg10": ar_ndcg,
                    "top10_changed": one != autoregressive,
                    "prediction_overlap": len(set(one) & set(autoregressive)) / 10.0,
                    "autoregressive_duplicate_fraction": 0.0,
                }
            )
        if at % 128 == 0:
            print({"event": "exact_evaluation", "rows": min(at + len(ids), len(row_ids)), "total": len(row_ids)}, flush=True)
    return {
        "schema": "genrec.music_lfm1b_timeconcat_evaluation.v1",
        "split": split,
        "rows": rows,
        "metrics": aggregate_rows(rows),
        "future_labels_enter_serving": False,
        "same_checkpoint_decoders": ["oneshot", "autoregressive"],
        "autoregressive_without_replacement": True,
        "engineering_limit": limit,
    }
