#!/usr/bin/env python3
"""Train-only OOF attainability gate for Music4All component C.

The runner keeps the frozen A1+B support contract and asks whether multiple
residual queries can recover Train-seen continuation targets outside that
support.  It never opens Validation or Test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from lfm_timeconcat_control import LFMTimeConcatData
from run_music4all_artist_component_b import (
    ProposalResidual,
    candidate_view,
    metric_table as artist_metric_table,
    proposal_loss,
    proposal_numeric,
    residual_scores as proposal_residual_scores,
)
from run_music4all_candidate_support_a1 import (
    SCOPES,
    SetwiseResidual,
    atomic_json,
    atomic_npz,
    atomic_torch,
    digest,
    load_standard_base,
    encode_base_states,
    load_base_data,
    calibration_score,
    metric_table as a1_metric_table,
    numeric_features,
    predict_residual,
    set_seed,
    setwise_loss,
    stable_topk,
)


SCHEMA = "genrec.music4all_onion.component_c_oof_support.v1"
PREDECESSOR_SCHEMA = "genrec.music4all_onion.component_c_oof_predecessor.v1"
VARIANTS = {
    "base": 0,
    "single_query": 1,
    "multi_query": 4,
    "shuffled_assignment": 4,
    "balanced_slots": 4,
}


def fold_ids(users: np.ndarray, folds: int) -> np.ndarray:
    if folds < 2:
        raise ValueError("oof_requires_at_least_two_folds")
    return np.asarray(
        [int.from_bytes(hashlib.sha256(f"c-oof:{int(user)}".encode()).digest()[:8], "big") % folds for user in users],
        dtype=np.int16,
    )


def sinkhorn_assignment(scores: torch.Tensor, epsilon: float, column_mass: torch.Tensor | None = None, iterations: int = 30) -> torch.Tensor:
    """Entropic assignment with uniform target and slot marginals."""

    if scores.ndim != 2 or not scores.numel() or epsilon <= 0:
        raise ValueError("invalid_sinkhorn_input")
    rows, columns = scores.shape
    kernel = torch.exp((scores - scores.max()) / epsilon).clamp_min(1e-12)
    row_mass = torch.full((rows,), 1.0 / rows, device=scores.device, dtype=scores.dtype)
    if column_mass is None:
        column_mass = torch.full((columns,), 1.0 / columns, device=scores.device, dtype=scores.dtype)
    else:
        column_mass = column_mass.to(device=scores.device, dtype=scores.dtype)
        column_mass = column_mass / column_mass.sum().clamp_min(1e-12)
    u = torch.ones_like(row_mass)
    v = torch.ones_like(column_mass)
    for _ in range(iterations):
        u = row_mass / (kernel @ v).clamp_min(1e-12)
        v = column_mass / (kernel.T @ u).clamp_min(1e-12)
    plan = u[:, None] * kernel * v[None, :]
    return plan / plan.sum().clamp_min(1e-12)


class ResidualQueries(nn.Module):
    def __init__(self, state_width: int, item_width: int, slots: int, hidden: int) -> None:
        super().__init__()
        self.slots = slots
        self.item_width = item_width
        self.network = nn.Sequential(nn.Linear(state_width + 1, hidden), nn.GELU(), nn.Linear(hidden, slots * item_width))
        nn.init.normal_(self.network[-1].weight, std=0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, states: torch.Tensor, gaps: torch.Tensor) -> torch.Tensor:
        gap_feature = torch.log1p(gaps).unsqueeze(-1) / 8.0
        output = self.network(torch.cat((states, gap_feature), dim=-1))
        return output.reshape(len(states), self.slots, self.item_width)


def shuffled_slot(row_id: int, target: int, slots: int, seed: int) -> int:
    payload = f"c-shuffle:{seed}:{row_id}:{target}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % slots


def residual_loss(
    queries: torch.Tensor,
    item_embeddings: torch.Tensor,
    row_targets: list[list[int]],
    row_ids: np.ndarray,
    *,
    variant: str,
    epsilon: float,
    seed: int,
) -> torch.Tensor:
    logits = torch.einsum("bmd,vd->bmv", queries, item_embeddings)
    logits[:, :, :3] = -1e9
    log_normalizer = torch.logsumexp(logits, dim=-1)
    losses = []
    balanced_log_prob = []
    for at, targets in enumerate(row_targets):
        target_tensor = torch.as_tensor(targets, dtype=torch.long, device=queries.device)
        target_scores = logits[at, :, target_tensor].T
        log_prob = target_scores - log_normalizer[at][None, :]
        if variant == "single_query":
            losses.append(-log_prob[:, 0])
        elif variant == "multi_query":
            losses.append(-(torch.logsumexp(log_prob, dim=1) - math.log(queries.shape[1])))
        elif variant == "shuffled_assignment":
            assignment = torch.as_tensor(
                [shuffled_slot(int(row_ids[at]), int(target), queries.shape[1], seed) for target in targets],
                dtype=torch.long,
                device=queries.device,
            )
            losses.append(-log_prob.gather(1, assignment[:, None]).squeeze(1))
        elif variant == "balanced_slots":
            balanced_log_prob.append(log_prob)
        else:
            raise ValueError(f"unsupported_variant:{variant}")
    if variant == "balanced_slots":
        joined = torch.cat(balanced_log_prob, dim=0)
        with torch.no_grad():
            plan = sinkhorn_assignment(joined, epsilon)
        return -(plan * joined).sum()
    return torch.cat(losses).mean()


def load_train_surface(args: argparse.Namespace, data: LFMTimeConcatData):
    arrays = dict(np.load(args.train_features, allow_pickle=False))
    manifest = json.loads(args.train_feature_manifest.read_text())
    if (
        manifest.get("status") != "pass"
        or manifest.get("surface") != "train_cv"
        or manifest.get("feature_sha256") != digest(args.train_features)
        or manifest.get("test_accessed") is not False
    ):
        raise ValueError("train_feature_manifest_invalid")
    rows = arrays["row_ids"].astype(np.int64)
    all_targets: list[list[int]] = []
    for row in map(int, rows):
        truth = list(dict.fromkeys(data.known_suffix(row)))
        all_targets.append(truth)
    return arrays, rows, all_targets, manifest


def inner_calibration(users: np.ndarray, eligible: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        [int.from_bytes(hashlib.sha256(f"c-pred-cal:{fold}:{int(user)}".encode()).digest()[:8], "big") for user in users],
        dtype=np.uint64,
    )
    calibration = eligible & (values >= np.uint64(int(.8 * (1 << 64))))
    fit = eligible & ~calibration
    if not fit.any() or not calibration.any() or set(users[fit]) & set(users[calibration]):
        raise ValueError("predecessor_inner_partition_invalid")
    return fit, calibration


def fit_a1_fold(args: argparse.Namespace, arrays: dict[str, np.ndarray], fit: np.ndarray, calibration: np.ndarray):
    features, relation = numeric_features(arrays, "full")
    model = SetwiseResidual(features.shape[-1], args.predecessor_hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.predecessor_learning_rate, weight_decay=args.weight_decay)
    fit_rows = np.flatnonzero(fit & arrays["target_mask"].any(axis=1))
    base_predictions = arrays["candidates"][calibration, :10].copy()
    cal_arrays = {key: value[calibration] for key, value in arrays.items()}
    base_metrics, _, _ = a1_metric_table(base_predictions, cal_arrays)
    best = None; best_state = None
    for epoch in range(1, args.predecessor_epochs + 1):
        order = np.random.default_rng(args.seed + 17000 + epoch).permutation(fit_rows)
        model.train()
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            residual = model(torch.as_tensor(features[index], device=args.device), torch.as_tensor(relation[index], device=args.device))
            loss = setwise_loss(torch.as_tensor(arrays["base_scores"][index], device=args.device) + residual, torch.as_tensor(arrays["target_mask"][index].astype(bool), device=args.device), .25)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        cal_residual = predict_residual(model, features[calibration], relation[calibration], args.device, args.eval_batch_size)
        for alpha in (0.0, .25, .5, 1.0, 2.0):
            prediction = base_predictions if alpha == 0 else stable_topk(arrays["base_scores"][calibration] + alpha * cal_residual, arrays["candidates"][calibration])
            metrics, _, _ = a1_metric_table(prediction, cal_arrays)
            score = calibration_score(metrics, base_metrics)
            key = (score, -abs(alpha), -alpha)
            if best is None or key > best:
                best = key; best_state = (float(alpha), {name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
    if best_state is None:
        raise RuntimeError("oof_a1_selection_failed")
    model.load_state_dict(best_state[1], strict=True); model.eval()
    return model, best_state[0], features, relation


def fit_artist_fold(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    view: dict[str, np.ndarray],
    fit: np.ndarray,
    calibration: np.ndarray,
):
    model = ProposalResidual(view["features"].shape[-1], args.predecessor_hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.predecessor_learning_rate, weight_decay=args.weight_decay)
    fit_rows = np.flatnonzero(fit & view["positives"].any(axis=1))
    cal_view = {key: value[calibration] for key, value in view.items()}
    cal_arrays = {key: value[calibration] for key, value in arrays.items()}
    a1_top10 = stable_topk(cal_view["scores"], cal_view["candidates"])
    base_metrics, _, _ = artist_metric_table(a1_top10, cal_view, cal_arrays)
    best = None; best_state = None
    for epoch in range(1, args.predecessor_epochs + 1):
        order = np.random.default_rng(args.seed + 29000 + epoch).permutation(fit_rows)
        model.train()
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            residual = model(torch.as_tensor(view["features"][index], device=args.device), torch.as_tensor(view["mask"][index], device=args.device))
            scores = torch.as_tensor(view["scores"][index], device=args.device) + residual
            loss = proposal_loss(scores, torch.as_tensor(view["positives"][index].astype(bool), device=args.device), torch.as_tensor(view["mask"][index], device=args.device), .25, 1.0)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        cal_residual = proposal_residual_scores(model, cal_view["features"], cal_view["mask"], args.device, args.eval_batch_size)
        for alpha in (0.0, .25, .5, 1.0, 2.0):
            prediction = a1_top10 if alpha == 0 else stable_topk(cal_view["scores"] + alpha * cal_residual, cal_view["candidates"])
            metrics, _, _ = artist_metric_table(prediction, cal_view, cal_arrays)
            score = sum(metrics[scope][name] for scope in ("overall", ">365d") for name in ("recall_at_10", "ndcg_at_10"))
            key = (score, -abs(alpha), -alpha)
            if best is None or key > best:
                best = key; best_state = (float(alpha), {name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
    if best_state is None:
        raise RuntimeError("oof_artist_selection_failed")
    model.load_state_dict(best_state[1], strict=True); model.eval()
    return model, best_state[0]


def build_predecessor(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError("component_c_predecessor_output_exists")
    set_seed(args.seed)
    data = LFMTimeConcatData(args.data_dir, max_context=args.max_context, horizon=args.horizon)
    arrays, rows, targets, manifest = load_train_surface(args, data)
    users = arrays["users"].astype(np.int64); folds = fold_ids(users, args.folds)
    candidates = np.empty((len(rows), 200), dtype=np.int32)
    scores = np.empty((len(rows), 200), dtype=np.float32)
    records = []
    for fold in range(args.folds):
        eligible = folds != fold
        fit, calibration = inner_calibration(users, eligible, fold)
        a1_model, a1_alpha, a1_features, a1_relation = fit_a1_fold(args, arrays, fit, calibration)
        original_residual = predict_residual(a1_model, a1_features, a1_relation, args.device, args.eval_batch_size)
        original_scores = arrays["base_scores"] + a1_alpha * original_residual
        proposal_features, proposal_mask = proposal_numeric(arrays, 0)
        proposal_proxy = {
            "base_scores": arrays["proposal_base_scores"][:, 0],
            "base_ranks": arrays["proposal_base_ranks"][:, 0],
            "x_ranks": arrays["proposal_x_ranks"][:, 0],
            "h_ranks": arrays["proposal_h_ranks"][:, 0],
            "log_popularity": arrays["proposal_log_popularity"][:, 0],
            "gaps": arrays["gaps"],
        }
        proposal_a1_features, proposal_a1_mask = numeric_features(proposal_proxy, "full")
        proposal_residual = predict_residual(a1_model, proposal_a1_features, proposal_a1_mask, args.device, args.eval_batch_size)
        proposal_scores = arrays["proposal_base_scores"][:, 0] + a1_alpha * proposal_residual
        a1_top10 = stable_topk(original_scores, arrays["candidates"])
        view = candidate_view(arrays, original_scores, proposal_scores, proposal_features, 0, a1_top10)
        artist_model, artist_alpha = fit_artist_fold(args, arrays, view, fit, calibration)
        artist_residual = proposal_residual_scores(artist_model, view["features"], view["mask"], args.device, args.eval_batch_size)
        combined_scores = view["scores"] + artist_alpha * artist_residual
        eval_rows = np.flatnonzero(folds == fold)
        candidates[eval_rows] = view["candidates"][eval_rows]
        scores[eval_rows] = combined_scores[eval_rows]
        records.append({"fold": fold, "fit_rows": int(fit.sum()), "calibration_rows": int(calibration.sum()), "eval_rows": len(eval_rows), "a1_alpha": a1_alpha, "artist_alpha": artist_alpha})
        print(json.dumps({"event": "component_c_predecessor_fold", **records[-1]}), flush=True)
    top10 = stable_topk(scores, candidates)
    residual_values = np.full((len(rows), args.horizon), -1, dtype=np.int32)
    residual_counts = np.zeros(len(rows), dtype=np.uint16)
    target_values = np.full((len(rows), args.horizon), -1, dtype=np.int32)
    target_counts = np.zeros(len(rows), dtype=np.uint16)
    for at, truth in enumerate(targets):
        support = set(map(int, candidates[at]))
        residual = [item for item in truth if data.frequencies[item] > 0 and item not in support]
        target_values[at, : len(truth)] = truth; target_counts[at] = len(truth)
        residual_values[at, : len(residual)] = residual; residual_counts[at] = len(residual)
    args.output.mkdir(parents=True)
    artifact = args.output / "oof_predecessor.npz"
    atomic_npz(artifact, row_ids=rows, users=users, gaps=arrays["gaps"], bucket_ids=arrays["bucket_ids"], fold_ids=folds, candidates=candidates, scores=scores, top10=top10, targets=target_values, target_counts=target_counts, residual_targets=residual_values, residual_counts=residual_counts)
    atomic_json(args.output / "PREDECESSOR_MANIFEST.json", {"schema": PREDECESSOR_SCHEMA, "status": "pass", "surface": "Train-only user-disjoint OOF", "rows": len(rows), "folds": args.folds, "records": records, "artifact_sha256": digest(artifact), "train_feature_sha256": manifest["feature_sha256"], "producer_sha256": digest(Path(__file__)), "validation_accessed": False, "test_accessed": False})


@torch.no_grad()
def context_states(args: argparse.Namespace, data: LFMTimeConcatData, rows: np.ndarray):
    loader_args = argparse.Namespace(**vars(args))
    data = load_base_data(loader_args, args.base_training_manifest)
    model, manifest = load_standard_base(
        loader_args, data, checkpoint_path=args.base_checkpoint, manifest_path=args.base_training_manifest
    )
    states = []
    for start in range(0, len(rows), args.eval_batch_size):
        batch_rows = rows[start : start + args.eval_batch_size]
        batch_states, _ = encode_base_states(loader_args, data, model, batch_rows)
        states.append(batch_states.cpu().numpy())
    embeddings = model.item_embedding.weight.detach().cpu().numpy().astype(np.float32)
    return np.concatenate(states).astype(np.float32), embeddings, manifest


def compile_support(base_candidates: np.ndarray, base_scores: np.ndarray, proposals: np.ndarray, quota: int) -> np.ndarray:
    if quota == 0:
        return base_candidates.copy()
    output = np.empty_like(base_candidates)
    for row in range(len(base_candidates)):
        order = np.lexsort((base_candidates[row], -base_scores[row]))
        protected = list(map(int, base_candidates[row, order[:10]]))
        additions = []
        seen = set(protected)
        for item in map(int, proposals[row].reshape(-1)):
            if item >= 3 and item not in seen and item not in set(map(int, base_candidates[row])):
                additions.append(item); seen.add(item)
                if len(additions) == quota:
                    break
        keep_count = 200 - len(additions)
        retained = [int(base_candidates[row, index]) for index in order if int(base_candidates[row, index]) not in seen]
        merged = protected + retained[: keep_count - len(protected)] + additions
        if len(merged) != 200 or len(set(merged)) != 200:
            raise ValueError("compiled_support_invalid")
        output[row] = merged
    return output


def per_row_recall(support: np.ndarray, targets: list[list[int]]) -> np.ndarray:
    values = np.zeros(len(support), dtype=np.float64)
    for row, truth in enumerate(targets):
        relevant = set(truth)
        values[row] = len(relevant.intersection(map(int, support[row]))) / max(len(relevant), 1)
    return values


def macro(values: np.ndarray, users: np.ndarray, selected: np.ndarray) -> float:
    totals: dict[int, list[float]] = {}
    for value, user in zip(values[selected], users[selected]):
        totals.setdefault(int(user), []).append(float(value))
    return float(np.mean([np.mean(row) for row in totals.values()])) if totals else 0.0


def paired_bootstrap(delta: np.ndarray, users: np.ndarray, selected: np.ndarray, seed: int) -> dict[str, float]:
    grouped: dict[int, list[float]] = {}
    for value, user in zip(delta[selected], users[selected]):
        grouped.setdefault(int(user), []).append(float(value))
    unit = np.asarray([np.mean(values) for values in grouped.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.asarray([rng.choice(unit, len(unit), replace=True).mean() for _ in range(2000)])
    return {"delta": float(unit.mean()), "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975)), "users": len(unit)}


@torch.no_grad()
def retrieve(model: ResidualQueries, states: np.ndarray, gaps: np.ndarray, items: torch.Tensor, args: argparse.Namespace) -> np.ndarray:
    per_slot = args.proposal_quota if model.slots == 1 else args.per_slot
    output = np.empty((len(states), model.slots, per_slot), dtype=np.int32)
    model.eval()
    for start in range(0, len(states), args.eval_batch_size):
        stop = min(start + args.eval_batch_size, len(states))
        query = model(
            torch.as_tensor(states[start:stop], device=args.device),
            torch.as_tensor(gaps[start:stop], device=args.device),
        )
        scores = torch.einsum("bmd,vd->bmv", query, items)
        scores[:, :, :3] = -1e9
        output[start:stop] = scores.topk(per_slot, dim=-1).indices.cpu().numpy()
    return output


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError("component_c_oof_output_exists")
    if args.variant not in VARIANTS:
        raise ValueError("unsupported_variant")
    set_seed(args.seed)
    data = LFMTimeConcatData(args.data_dir, max_context=args.max_context, horizon=args.horizon)
    predecessor_manifest = json.loads(args.predecessor_manifest.read_text())
    if predecessor_manifest.get("schema") != PREDECESSOR_SCHEMA or predecessor_manifest.get("status") != "pass" or predecessor_manifest.get("artifact_sha256") != digest(args.predecessor):
        raise ValueError("oof_predecessor_manifest_invalid")
    packed = dict(np.load(args.predecessor, allow_pickle=False))
    rows = packed["row_ids"].astype(np.int64); users = packed["users"].astype(np.int64); folds = packed["fold_ids"]
    targets = [list(map(int, packed["targets"][at, : int(packed["target_counts"][at])])) for at in range(len(rows))]
    residual = [list(map(int, packed["residual_targets"][at, : int(packed["residual_counts"][at])])) for at in range(len(rows))]
    states, item_array, base_manifest = context_states(args, data, rows)
    base_support = packed["candidates"].astype(np.int32)
    base_scores = packed["scores"].astype(np.float32)
    base_recall = per_row_recall(base_support, targets)
    method_support = base_support.copy()
    records = []
    args.output.mkdir(parents=True)
    started = time.monotonic()
    if args.variant != "base":
        item_tensor = torch.as_tensor(item_array, device=args.device)
        for fold in range(args.folds):
            train_rows = np.flatnonzero((folds != fold) & np.asarray([bool(x) for x in residual]))
            eval_rows = np.flatnonzero(folds == fold)
            model = ResidualQueries(states.shape[1], item_array.shape[1], VARIANTS[args.variant], args.hidden_dim).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            fold_records = []
            updates = 0
            for epoch in range(1, args.epochs + 1):
                order = np.random.default_rng(args.seed + 1000 * fold + epoch).permutation(train_rows)
                losses = []
                model.train()
                for start in range(0, len(order), args.batch_size):
                    index = order[start : start + args.batch_size]
                    query = model(
                        torch.as_tensor(states[index], device=args.device),
                        torch.as_tensor(packed["gaps"][index], device=args.device),
                    )
                    loss = residual_loss(
                        query, item_tensor, [residual[int(i)] for i in index], rows[index],
                        variant=args.variant, epsilon=args.epsilon, seed=args.seed,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite_component_c_loss")
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step(); losses.append(float(loss.detach())); updates += 1
                    if args.stop_after and updates >= args.stop_after:
                        atomic_json(args.output / "SMOKE.json", {"schema": SCHEMA, "status": "engineering_pass", "scientific_result": False, "variant": args.variant, "updates": updates, "test_accessed": False})
                        return
                fold_records.append({"epoch": epoch, "mean_loss": float(np.mean(losses)), "updates": updates})
                print(json.dumps({"event": "component_c_oof_epoch", "variant": args.variant, "fold": fold, **fold_records[-1]}), flush=True)
            proposals = retrieve(model, states[eval_rows], packed["gaps"][eval_rows], item_tensor, args)
            method_support[eval_rows] = compile_support(
                base_support[eval_rows], base_scores[eval_rows], proposals, args.proposal_quota
            )
            checkpoint = args.output / f"fold_{fold}.pt"
            atomic_torch(checkpoint, {"schema": SCHEMA, "variant": args.variant, "fold": fold, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}})
            records.append({"fold": fold, "train_rows": len(train_rows), "eval_rows": len(eval_rows), "epochs": fold_records, "checkpoint_sha256": digest(checkpoint)})
    method_recall = per_row_recall(method_support, targets)
    delta = method_recall - base_recall
    residual_total = sum(len(values) for values in residual)
    rescued_targets = 0; targeted_rescued = 0; displaced_targets = 0
    for at, truth in enumerate(targets):
        old = set(map(int, base_support[at])); new = set(map(int, method_support[at]))
        rescued_targets += len((set(truth) & new) - old)
        targeted_rescued += len(set(residual[at]) & (new - old))
        displaced_targets += len((set(truth) & old) - new)
    metrics = {}
    for scope_index, scope in enumerate(SCOPES):
        selected = np.ones(len(rows), dtype=bool) if scope == "overall" else packed["bucket_ids"] == scope_index - 1
        metrics[scope] = {
            "rows": int(selected.sum()), "users": int(len(np.unique(users[selected]))),
            "base_recall_at_200": macro(base_recall, users, selected),
            "method_recall_at_200": macro(method_recall, users, selected),
            "paired": paired_bootstrap(delta, users, selected, args.seed + scope_index),
        }
    predictions = args.output / "oof_support.npz"
    atomic_npz(predictions, row_ids=rows, users=users, bucket_ids=packed["bucket_ids"], fold_ids=folds, base_recall=base_recall, method_recall=method_recall)
    binding = {
        "schema": SCHEMA, "variant": args.variant, "seed": args.seed, "folds": args.folds,
        "epochs": args.epochs, "slots": VARIANTS[args.variant], "per_slot": args.per_slot,
        "proposal_quota": args.proposal_quota, "epsilon": args.epsilon,
        "predecessor_sha256": predecessor_manifest["artifact_sha256"],
        "base_checkpoint_sha256": digest(args.base_checkpoint),
        "base_manifest_sha256": digest(args.base_training_manifest), "producer_sha256": digest(Path(__file__)),
        "base_architecture": base_manifest["binding"]["architecture_contract"], "validation_accessed": False, "test_accessed": False,
    }
    result = {
        "schema": SCHEMA, "status": "pass", "variant": args.variant, "metrics": metrics,
        "residual_train_seen_targets": residual_total, "rescued_targets": rescued_targets,
        "targeted_rescued_targets": targeted_rescued,
        "displaced_targets": displaced_targets,
        "targeted_residual_recovery": targeted_rescued / residual_total if residual_total else 0.0,
        "long_point_lift": metrics[">365d"]["paired"]["delta"],
        "binding": binding, "fold_records": records, "predictions_sha256": digest(predictions),
        "seconds": time.monotonic() - started,
    }
    atomic_json(args.output / "OOF_RESULT.json", result)
    print(json.dumps({"event": "component_c_oof_complete", "variant": args.variant, "overall": metrics["overall"], "targeted_recovery": result["targeted_residual_recovery"]}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build-predecessor", "run"), default="run")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="balanced_slots")
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--train-features", type=Path, default=Path("runs/music4all_artist_b/features/train_cv_features.npz"))
    parser.add_argument("--train-feature-manifest", type=Path, default=Path("runs/music4all_artist_b/features/train_cv_features_manifest.json"))
    parser.add_argument("--predecessor", type=Path, default=Path("runs/music4all_component_c/oof_predecessor/oof_predecessor.npz"))
    parser.add_argument("--predecessor-manifest", type=Path, default=Path("runs/music4all_component_c/oof_predecessor/PREDECESSOR_MANIFEST.json"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("runs/music4all_candidate_support_a1/crossfit_sasrec_s2024/final_model.pt"))
    parser.add_argument("--base-training-manifest", type=Path, default=Path("runs/music4all_candidate_support_a1/crossfit_sasrec_s2024/training_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--predecessor-hidden-dim", type=int, default=64)
    parser.add_argument("--predecessor-epochs", type=int, default=30)
    parser.add_argument("--predecessor-learning-rate", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=.1)
    parser.add_argument("--per-slot", type=int, default=10)
    parser.add_argument("--proposal-quota", type=int, default=40)
    parser.add_argument("--max-context", type=int, default=129)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(); torch.set_num_threads(args.threads)
    return args


if __name__ == "__main__":
    parsed = parse_args()
    build_predecessor(parsed) if parsed.mode == "build-predecessor" else run(parsed)
