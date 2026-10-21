#!/usr/bin/env python3
"""Train-only OOF gate for fixed-support Top-10 conversion.

The exact A1+B Top-200 candidate set and score are frozen.  The experiment
compares a candidate-independent value model with a candidate-set contextual
model under identical features, labels, optimization, and calibration.  It
never opens Validation or Test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from lfm_timeconcat_control import LFMTimeConcatData
from run_music4all_candidate_support_a1 import atomic_json, atomic_npz, atomic_torch, digest, set_seed, stable_topk
from run_music4all_component_c_oof import PREDECESSOR_SCHEMA, context_states


SCHEMA = "genrec.music4all_onion.component_c2_conversion_oof.v1"
VARIANTS = ("pointwise", "set_contextual", "shuffled_context")
SCOPES = ("overall", "30-90d", "90-180d", "180-365d", ">365d")


def deterministic_partition(users: np.ndarray, fold: int, modulus: int = 5) -> np.ndarray:
    values = []
    for user in users:
        payload = f"c2-cal:{fold}:{int(user)}".encode()
        values.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulus)
    return np.asarray(values, dtype=np.int16)


def deterministic_shuffle(indices: np.ndarray, seed: int, label: str) -> np.ndarray:
    keys = np.asarray(
        [int.from_bytes(hashlib.sha256(f"{label}:{seed}:{int(i)}".encode()).digest()[:8], "big") for i in indices],
        dtype=np.uint64,
    )
    return indices[np.argsort(keys, kind="stable")]


def user_disjoint_derangement(indices: np.ndarray, users: np.ndarray, seed: int, label: str) -> np.ndarray:
    """Return source rows aligned to recipients, with no row/user preserved."""

    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2 or len(np.unique(users[indices])) < 2:
        raise ValueError("user_derangement_impossible")
    rng_seed = int.from_bytes(hashlib.sha256(f"{label}:{seed}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(rng_seed)
    for _ in range(1000):
        sources = rng.permutation(indices)
        if np.all(sources != indices) and np.all(users[sources] != users[indices]):
            return sources
    raise ValueError("user_derangement_not_found")


def target_mask(candidates: np.ndarray, targets: np.ndarray, counts: np.ndarray) -> np.ndarray:
    output = np.zeros(candidates.shape, dtype=bool)
    for row in range(len(candidates)):
        truth = set(map(int, targets[row, : int(counts[row])]))
        output[row] = np.asarray([int(item) in truth for item in candidates[row]], dtype=bool)
    return output


def token_features(
    candidates: torch.Tensor,
    base_scores: torch.Tensor,
    states: torch.Tensor,
    gaps: torch.Tensor,
    item_embeddings: torch.Tensor,
) -> torch.Tensor:
    items = item_embeddings[candidates]
    item_unit = torch.nn.functional.normalize(items, dim=-1)
    state_unit = torch.nn.functional.normalize(states, dim=-1).unsqueeze(1)
    interaction = item_unit * state_unit
    cosine = interaction.sum(dim=-1, keepdim=True)
    score_mean = base_scores.mean(dim=1, keepdim=True)
    score_std = base_scores.std(dim=1, keepdim=True).clamp_min(1e-6)
    score_z = ((base_scores - score_mean) / score_std).unsqueeze(-1)
    order = torch.argsort(base_scores, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(1, base_scores.shape[1] + 1, device=base_scores.device).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    reciprocal_rank = (1.0 / torch.log2(ranks.to(base_scores.dtype) + 1.0)).unsqueeze(-1)
    tenth = base_scores.topk(10, dim=1).values[:, -1:]
    boundary_margin = (base_scores - tenth).unsqueeze(-1)
    gap = (torch.log1p(gaps) / 8.0).view(-1, 1, 1).expand(-1, candidates.shape[1], 1)
    return torch.cat((item_unit, interaction, cosine, score_z, reciprocal_rank, boundary_margin, gap), dim=-1)


class TokenwiseMixer(nn.Module):
    """Parameter-matched replacement for MHA without cross-candidate access."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        mixed = (self.query(hidden) + self.key(hidden) + self.value(hidden)) / math.sqrt(3.0)
        return self.output(mixed)


class ConversionRanker(nn.Module):
    def __init__(self, item_width: int, hidden: int, contextual: bool) -> None:
        super().__init__()
        self.contextual = contextual
        self.input = nn.Sequential(nn.Linear(item_width * 2 + 5, hidden), nn.GELU(), nn.LayerNorm(hidden))
        if contextual:
            self.attention = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        else:
            self.attention = TokenwiseMixer(hidden)
        self.norm = nn.LayerNorm(hidden)
        self.feed_forward = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
        self.feed_norm = nn.LayerNorm(hidden)
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input(features)
        if self.contextual:
            attended, _ = self.attention(hidden, hidden, hidden, need_weights=False)
        else:
            attended = self.attention(hidden)
        hidden = self.norm(hidden + attended)
        hidden = self.feed_norm(hidden + self.feed_forward(hidden))
        return self.output(hidden).squeeze(-1)


def conversion_loss(scores: torch.Tensor, positives: torch.Tensor, target_counts: torch.Tensor, margin: float) -> torch.Tensor:
    valid = positives.any(dim=1)
    if not valid.any():
        raise ValueError("batch_has_no_supported_positive")
    scores, positives = scores[valid], positives[valid]
    denominator = target_counts[valid].to(scores.dtype).clamp_min(1.0)
    log_prob = torch.log_softmax(scores, dim=1)
    proper = -(log_prob * positives.to(scores.dtype)).sum(dim=1) / denominator
    positive_scores = scores.masked_fill(~positives, float("-inf")).sort(dim=1, descending=True).values[:, :10]
    negative_scores = scores.masked_fill(positives, float("-inf")).topk(10, dim=1).values
    supported = positives.sum(dim=1).clamp(max=10)
    quota = torch.zeros_like(proper)
    for rank in range(10):
        active = supported > rank
        threshold = negative_scores[:, 9 - rank]
        term = torch.nn.functional.softplus(margin + threshold - positive_scores[:, rank]) / denominator
        quota = quota + torch.where(active, term, torch.zeros_like(term))
    return (proper + quota).mean()


@torch.no_grad()
def predict_residual(
    model: nn.Module,
    candidates: np.ndarray,
    scores: np.ndarray,
    states: np.ndarray,
    gaps: np.ndarray,
    embeddings: torch.Tensor,
    device: str,
    batch_size: int,
) -> np.ndarray:
    output = np.empty(scores.shape, dtype=np.float32)
    model.eval()
    for start in range(0, len(scores), batch_size):
        stop = min(start + batch_size, len(scores))
        c = torch.as_tensor(candidates[start:stop], dtype=torch.long, device=device)
        s = torch.as_tensor(scores[start:stop], device=device)
        h = torch.as_tensor(states[start:stop], device=device)
        g = torch.as_tensor(gaps[start:stop], device=device)
        output[start:stop] = model(token_features(c, s, h, g, embeddings)).cpu().numpy()
    return output


def row_metrics(predictions: np.ndarray, targets: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    recall = np.zeros(len(predictions), dtype=np.float64)
    ndcg = np.zeros(len(predictions), dtype=np.float64)
    for row, prediction in enumerate(predictions):
        truth = set(map(int, targets[row, : int(counts[row])]))
        hits = [int(int(item) in truth) for item in prediction]
        recall[row] = sum(hits) / max(int(counts[row]), 1)
        ideal_count = min(int(counts[row]), 10)
        ideal = sum(1.0 / math.log2(rank + 2.0) for rank in range(ideal_count))
        ndcg[row] = sum(hit / math.log2(rank + 2.0) for rank, hit in enumerate(hits)) / ideal if ideal else 0.0
    return recall, ndcg


def user_macro(values: np.ndarray, users: np.ndarray, selected: np.ndarray) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for value, user in zip(values[selected], users[selected]):
        grouped[int(user)].append(float(value))
    return float(np.mean([np.mean(group) for group in grouped.values()])) if grouped else 0.0


def paired_bootstrap(delta: np.ndarray, users: np.ndarray, selected: np.ndarray, seed: int) -> dict[str, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for value, user in zip(delta[selected], users[selected]):
        grouped[int(user)].append(float(value))
    unit = np.asarray([np.mean(group) for group in grouped.values()], dtype=np.float64)
    if len(unit) == 0:
        return {
            "delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "users": 0,
        }
    rng = np.random.default_rng(seed)
    samples = np.asarray([rng.choice(unit, len(unit), replace=True).mean() for _ in range(2000)])
    return {
        "delta": float(unit.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "users": int(len(unit)),
    }


def selection_score(recall: np.ndarray, ndcg: np.ndarray, users: np.ndarray, buckets: np.ndarray, selected: np.ndarray) -> float:
    represented = buckets[selected]
    if not len(represented):
        raise ValueError("selection_score_requires_rows")
    # Use the longest hiatus stratum represented on the current dataset.
    # Music4All-Onion reaches bucket 3 (>365d); corrected Yambda spans only
    # about 301 days and therefore legitimately tops out at bucket 2.
    long = selected & (buckets == int(represented.max()))
    return user_macro(recall, users, selected) + user_macro(ndcg, users, selected) + user_macro(recall, users, long) + user_macro(ndcg, users, long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--predecessor", type=Path, default=Path("runs/music4all_component_c/oof_predecessor/oof_predecessor.npz"))
    parser.add_argument("--predecessor-manifest", type=Path, default=Path("runs/music4all_component_c/oof_predecessor/PREDECESSOR_MANIFEST.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("runs/music4all_candidate_support_a1/crossfit_sasrec_s2024/final_model.pt"))
    parser.add_argument("--base-training-manifest", type=Path, default=Path("runs/music4all_candidate_support_a1/crossfit_sasrec_s2024/training_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--beta-grid", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0, 2.0))
    parser.add_argument("--max-context", type=int, default=129)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.output.exists():
        raise FileExistsError("component_c2_output_exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    set_seed(args.seed)
    predecessor_manifest = json.loads(args.predecessor_manifest.read_text())
    if predecessor_manifest.get("schema") != PREDECESSOR_SCHEMA or predecessor_manifest.get("status") != "pass" or predecessor_manifest.get("artifact_sha256") != digest(args.predecessor):
        raise ValueError("oof_predecessor_manifest_invalid")
    packed = dict(np.load(args.predecessor, allow_pickle=False))
    rows = packed["row_ids"].astype(np.int64)
    users = packed["users"].astype(np.int64)
    folds = packed["fold_ids"].astype(np.int16)
    candidates = packed["candidates"].astype(np.int64)
    base_scores = packed["scores"].astype(np.float32)
    targets = packed["targets"].astype(np.int64)
    target_counts = packed["target_counts"].astype(np.int64)
    gaps = packed["gaps"].astype(np.float32)
    buckets = packed["bucket_ids"].astype(np.int16)
    positives = target_mask(candidates, targets, target_counts)
    data = LFMTimeConcatData(args.data_dir, max_context=args.max_context, horizon=args.horizon)
    states, item_array, base_manifest = context_states(args, data, rows)
    item_embeddings = torch.as_tensor(item_array, device=args.device)
    predictions = packed["top10"].astype(np.int32).copy()
    fold_records: list[dict[str, Any]] = []
    args.output.mkdir(parents=True)
    started = time.monotonic()
    updates = 0
    for fold in range(args.folds):
        outer_train = folds != fold
        eval_rows = np.flatnonzero(folds == fold)
        partition = deterministic_partition(users, fold)
        fit_rows = np.flatnonzero(outer_train & (partition != 0) & positives.any(axis=1))
        calibration_rows = np.flatnonzero(outer_train & (partition == 0))
        if not len(fit_rows) or not len(calibration_rows) or not len(eval_rows):
            raise ValueError("empty_oof_partition")
        contextual = args.variant != "pointwise"
        model = ConversionRanker(item_array.shape[1], args.hidden_dim, contextual=contextual).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        context_source = np.arange(len(rows), dtype=np.int64)
        if args.variant == "shuffled_context":
            context_source[fit_rows] = user_disjoint_derangement(fit_rows, users, args.seed + 10000 * fold, "c2-fit-context")
            context_source[calibration_rows] = user_disjoint_derangement(calibration_rows, users, args.seed + 20000 * fold, "c2-cal-context")
            context_source[eval_rows] = user_disjoint_derangement(eval_rows, users, args.seed + 30000 * fold, "c2-eval-context")
            for subset in (fit_rows, calibration_rows, eval_rows):
                assert np.all(context_source[subset] != subset)
                assert np.all(users[context_source[subset]] != users[subset])
        best: dict[str, Any] | None = None
        epochs: list[dict[str, Any]] = []
        for epoch in range(1, args.epochs + 1):
            order = deterministic_shuffle(fit_rows, args.seed + epoch + 1000 * fold, "c2-fit")
            model.train()
            losses = []
            for start in range(0, len(order), args.batch_size):
                index = order[start : start + args.batch_size]
                state_index = context_source[index]
                c = torch.as_tensor(candidates[index], dtype=torch.long, device=args.device)
                s = torch.as_tensor(base_scores[index], device=args.device)
                h = torch.as_tensor(states[state_index], device=args.device)
                g = torch.as_tensor(gaps[state_index], device=args.device)
                residual = model(token_features(c, s, h, g, item_embeddings))
                loss = conversion_loss(s + residual, torch.as_tensor(positives[index], device=args.device), torch.as_tensor(target_counts[index], device=args.device), args.margin)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite_component_c2_loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                updates += 1
                if args.stop_after and updates >= args.stop_after:
                    atomic_json(args.output / "SMOKE.json", {"schema": SCHEMA, "status": "engineering_pass", "variant": args.variant, "updates": updates, "validation_accessed": False, "test_accessed": False})
                    return
            calibration_states = states[calibration_rows]
            calibration_gaps = gaps[calibration_rows]
            if args.variant == "shuffled_context":
                source = context_source[calibration_rows]
                calibration_states, calibration_gaps = states[source], gaps[source]
            residual = predict_residual(model, candidates[calibration_rows], base_scores[calibration_rows], calibration_states, calibration_gaps, item_embeddings, args.device, args.eval_batch_size)
            epoch_best = None
            for beta in args.beta_grid:
                pred = stable_topk(base_scores[calibration_rows] + float(beta) * residual, candidates[calibration_rows])
                rec, ndcg = row_metrics(pred, targets[calibration_rows], target_counts[calibration_rows])
                score = selection_score(rec, ndcg, users[calibration_rows], buckets[calibration_rows], np.ones(len(calibration_rows), dtype=bool))
                candidate = {"epoch": epoch, "beta": float(beta), "selection_score": score}
                if epoch_best is None or (candidate["selection_score"], -candidate["beta"]) > (epoch_best["selection_score"], -epoch_best["beta"]):
                    epoch_best = candidate
            assert epoch_best is not None
            epochs.append({"epoch": epoch, "mean_loss": float(np.mean(losses)), **epoch_best})
            if best is None or (epoch_best["selection_score"], -epoch_best["beta"], -epoch) > (best["selection_score"], -best["beta"], -best["epoch"]):
                best = {**epoch_best, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
            print(json.dumps({"event": "component_c2_epoch", "variant": args.variant, "fold": fold, **epochs[-1]}), flush=True)
        assert best is not None
        model.load_state_dict(best.pop("model"), strict=True)
        eval_states, eval_gaps = states[eval_rows], gaps[eval_rows]
        if args.variant == "shuffled_context":
            source = context_source[eval_rows]
            eval_states, eval_gaps = states[source], gaps[source]
        residual = predict_residual(model, candidates[eval_rows], base_scores[eval_rows], eval_states, eval_gaps, item_embeddings, args.device, args.eval_batch_size)
        predictions[eval_rows] = stable_topk(base_scores[eval_rows] + float(best["beta"]) * residual, candidates[eval_rows])
        checkpoint = args.output / f"fold_{fold}.pt"
        atomic_torch(checkpoint, {"schema": SCHEMA, "variant": args.variant, "fold": fold, "selection": best, "model": model.state_dict()})
        fold_records.append({"fold": fold, "fit_rows": int(len(fit_rows)), "calibration_rows": int(len(calibration_rows)), "eval_rows": int(len(eval_rows)), "selection": best, "epochs": epochs, "checkpoint_sha256": digest(checkpoint)})
    base_recall, base_ndcg = row_metrics(packed["top10"], targets, target_counts)
    method_recall, method_ndcg = row_metrics(predictions, targets, target_counts)
    metrics = {}
    for scope_index, scope in enumerate(SCOPES):
        selected = np.ones(len(rows), dtype=bool) if scope == "overall" else buckets == scope_index - 1
        metrics[scope] = {
            "rows": int(selected.sum()),
            "users": int(len(np.unique(users[selected]))),
            "base_recall_at_10": user_macro(base_recall, users, selected),
            "method_recall_at_10": user_macro(method_recall, users, selected),
            "recall_paired": paired_bootstrap(method_recall - base_recall, users, selected, args.seed + scope_index),
            "base_ndcg_at_10": user_macro(base_ndcg, users, selected),
            "method_ndcg_at_10": user_macro(method_ndcg, users, selected),
            "ndcg_paired": paired_bootstrap(method_ndcg - base_ndcg, users, selected, args.seed + 100 + scope_index),
        }
    rescued_rows = int(((method_recall > base_recall) & (base_recall == 0)).sum())
    displaced_rows = int(((method_recall < base_recall) & (method_recall == 0)).sum())
    better_rows = int((method_recall > base_recall).sum())
    worse_rows = int((method_recall < base_recall).sum())
    supported_count = positives.sum(axis=1)
    oracle_recall = np.minimum(supported_count, 10) / np.maximum(target_counts, 1)
    oracle_residual = user_macro(oracle_recall - base_recall, users, np.ones(len(rows), dtype=bool))
    realized = metrics["overall"]["recall_paired"]["delta"]
    artifact = args.output / "oof_predictions.npz"
    atomic_npz(artifact, row_ids=rows, users=users, bucket_ids=buckets, fold_ids=folds, base_top10=packed["top10"], method_top10=predictions, base_recall=base_recall, method_recall=method_recall, base_ndcg=base_ndcg, method_ndcg=method_ndcg)
    result = {
        "schema": SCHEMA,
        "status": "pass",
        "variant": args.variant,
        "surface": "Train-only user-disjoint OOF",
        "metrics": metrics,
        "benefit_harm": {"recall_better_rows": better_rows, "recall_worse_rows": worse_rows, "rescued_zero_to_hit_rows": rescued_rows, "displaced_hit_to_zero_rows": displaced_rows},
        "opportunity": {"within_support_recall_residual": oracle_residual, "realized_recovery": realized / oracle_residual if oracle_residual > 0 else 0.0, "status": "fixed_support_oracle_only"},
        "fold_records": fold_records,
        "binding": {
            "predecessor_sha256": predecessor_manifest["artifact_sha256"],
            "base_checkpoint_sha256": digest(args.base_checkpoint),
            "base_manifest_sha256": digest(args.base_training_manifest),
            "base_architecture": base_manifest["binding"]["architecture_contract"],
            "producer_sha256": digest(Path(__file__)),
            "seed": args.seed,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "validation_accessed": False,
            "test_accessed": False,
        },
        "predictions_sha256": digest(artifact),
        "seconds": time.monotonic() - started,
    }
    atomic_json(args.output / "OOF_CONVERSION_RESULT.json", result)
    print(json.dumps({"event": "component_c2_complete", "variant": args.variant, "overall": metrics["overall"], "long": metrics[">365d"], "benefit_harm": result["benefit_harm"], "recovery": result["opportunity"]["realized_recovery"]}), flush=True)


if __name__ == "__main__":
    main()
