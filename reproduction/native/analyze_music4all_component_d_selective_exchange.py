#!/usr/bin/env python3
"""Train-only OOF falsifier for risk-calibrated Top-10 exchange.

Component C2 improves the frozen A1+B Top-10 on average, but it also displaces
some correct predecessor items.  This diagnostic learns candidate value from
OOF rows only and accepts/reverts C2 boundary exchanges only when the estimated
pairwise net utility clears a calibrated uncertainty margin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np


SCHEMA = "genrec.music4all_onion.component_d_selective_exchange_oof.v1"
SCOPES = ("overall", "30-90d", "90-180d", "180-365d", ">365d")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.close()
        os.replace(temporary, path)
    finally:
        try:
            handle.close()
        except Exception:
            pass
        temporary.unlink(missing_ok=True)


def user_fold(users: np.ndarray, folds: int) -> np.ndarray:
    return np.asarray(
        [int.from_bytes(hashlib.sha256(f"component-d-exchange:{int(user)}".encode()).digest()[:8], "big") % folds for user in users],
        dtype=np.int16,
    )


def calibration_mask(users: np.ndarray, outer_fold: int) -> np.ndarray:
    return np.asarray(
        [int.from_bytes(hashlib.sha256(f"component-d-cal:{outer_fold}:{int(user)}".encode()).digest()[:8], "big") % 5 == 0 for user in users],
        dtype=bool,
    )


def candidate_lookup(features: dict[str, np.ndarray], row: int, item: int) -> tuple[float, float, float, float]:
    base = np.flatnonzero(features["candidates"][row] == item)
    if len(base):
        at = int(base[0])
        return (
            float(features["base_ranks"][row, at]),
            float(features["x_ranks"][row, at]),
            float(features["h_ranks"][row, at]),
            float(features["log_popularity"][row, at]),
        )
    proposal = np.flatnonzero(features["proposal_candidates"][row, 0] == item)
    if len(proposal):
        at = int(proposal[0])
        return (
            float(features["proposal_base_ranks"][row, 0, at]),
            float(features["proposal_x_ranks"][row, 0, at]),
            float(features["proposal_h_ranks"][row, 0, at]),
            float(features["proposal_log_popularity"][row, 0, at]),
        )
    raise ValueError("candidate_feature_lookup_failed")


def build_rows(
    predecessor: dict[str, np.ndarray],
    c2: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    pools: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for row, (base_top, method_top) in enumerate(zip(c2["base_top10"], c2["method_top10"])):
        base_list = list(map(int, base_top))
        method_list = list(map(int, method_top))
        pool = np.asarray(list(dict.fromkeys(method_list + base_list)), dtype=np.int32)
        score_map = {int(item): float(score) for item, score in zip(predecessor["candidates"][row], predecessor["scores"][row])}
        values = np.asarray(list(score_map.values()), dtype=np.float32)
        mean, scale = float(values.mean()), max(float(values.std()), 1e-6)
        truth = set(map(int, predecessor["targets"][row, : int(predecessor["target_counts"][row])]))
        matrix = np.zeros((len(pool), 17), dtype=np.float32)
        for at, item in enumerate(pool):
            base_rank, x_rank, h_rank, popularity = candidate_lookup(features, row, int(item))
            predecessor_rank = int(np.flatnonzero(predecessor["candidates"][row] == item)[0]) + 1
            b_rank = base_list.index(int(item)) + 1 if int(item) in base_list else 11
            m_rank = method_list.index(int(item)) + 1 if int(item) in method_list else 11
            score = score_map[int(item)]
            matrix[at] = (
                (score - mean) / scale,
                1.0 / math.log2(predecessor_rank + 1.0),
                1.0 / math.log2(base_rank + 2.0),
                1.0 / math.log2(x_rank + 2.0),
                1.0 / math.log2(h_rank + 2.0),
                popularity,
                math.log1p(float(predecessor["gaps"][row])) / 8.0,
                float(predecessor["bucket_ids"][row] == 0),
                float(predecessor["bucket_ids"][row] == 1),
                float(predecessor["bucket_ids"][row] == 2),
                float(predecessor["bucket_ids"][row] == 3),
                float(int(item) in base_list),
                float(int(item) in method_list),
                1.0 / b_rank,
                1.0 / m_rank,
                float(m_rank - b_rank) / 10.0,
                float(len(set(base_list) ^ set(method_list))) / 20.0,
            )
        pools.append(pool)
        matrices.append(matrix)
        labels.append(np.asarray([int(int(item) in truth) for item in pool], dtype=np.uint8))
    return pools, matrices, labels


def training_table(matrices: list[np.ndarray], labels: list[np.ndarray], selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.flatnonzero(selected)
    x = np.concatenate([matrices[row] for row in rows])
    y = np.concatenate([labels[row] for row in rows])
    weights = np.concatenate([np.full(len(labels[row]), 1.0 / len(labels[row]), dtype=np.float32) for row in rows])
    positive = int(y.sum())
    weights[y == 1] *= min(max((len(y) - positive) / max(positive, 1), 1.0), 30.0)
    return x, y, weights


def fit_ensemble(x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int, members: int) -> list[lgb.LGBMClassifier]:
    models = []
    for member in range(members):
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=160, num_leaves=31, learning_rate=0.04,
            min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.85,
            reg_lambda=1.0, random_state=seed + member * 997, n_jobs=8, verbosity=-1,
        )
        model.fit(x, y, sample_weight=weights)
        models.append(model)
    return models


def predict(models: list[lgb.LGBMClassifier], matrices: list[np.ndarray], selected: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    means = [np.empty(0, dtype=np.float32) for _ in matrices]
    stds = [np.empty(0, dtype=np.float32) for _ in matrices]
    for row in np.flatnonzero(selected):
        estimates = np.stack([model.booster_.predict(matrices[row], num_threads=8) for model in models])
        means[row] = estimates.mean(axis=0).astype(np.float32)
        stds[row] = estimates.std(axis=0).astype(np.float32)
    return means, stds


def selective_prediction(
    pools: list[np.ndarray], means: list[np.ndarray], stds: list[np.ndarray],
    base_top10: np.ndarray, method_top10: np.ndarray, selected: np.ndarray,
    margin: float, kappa: float,
) -> np.ndarray:
    output = method_top10.copy()
    for row in np.flatnonzero(selected):
        value = {int(item): (float(means[row][at]), float(stds[row][at])) for at, item in enumerate(pools[row])}
        current = list(map(int, method_top10[row]))
        incoming = [int(item) for item in base_top10[row] if int(item) not in current]
        outgoing = [int(item) for item in current if int(item) not in set(map(int, base_top10[row]))]
        incoming.sort(key=lambda item: (-(value[item][0] - kappa * value[item][1]), item))
        outgoing.sort(key=lambda item: (value[item][0] + kappa * value[item][1], item))
        for restore, remove in zip(incoming, outgoing):
            lower_delta = value[restore][0] - value[remove][0] - kappa * (value[restore][1] + value[remove][1])
            if lower_delta <= margin:
                continue
            position = current.index(remove)
            current[position] = restore
        output[row] = np.asarray(current, dtype=np.int32)
    return output


def row_metrics(predictions: np.ndarray, targets: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    recall = np.zeros(len(predictions), dtype=np.float64)
    ndcg = np.zeros(len(predictions), dtype=np.float64)
    for row, prediction in enumerate(predictions):
        truth = set(map(int, targets[row, : int(counts[row])]))
        hits = np.asarray([int(int(item) in truth) for item in prediction], dtype=np.float64)
        recall[row] = hits.sum() / max(int(counts[row]), 1)
        ideal = sum(1.0 / math.log2(rank + 2.0) for rank in range(min(int(counts[row]), 10)))
        ndcg[row] = sum(float(hit) / math.log2(rank + 2.0) for rank, hit in enumerate(hits)) / ideal if ideal else 0.0
    return recall, ndcg


def user_macro(values: np.ndarray, users: np.ndarray, selected: np.ndarray) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for value, user in zip(values[selected], users[selected]):
        grouped[int(user)].append(float(value))
    return float(np.mean([np.mean(group) for group in grouped.values()])) if grouped else 0.0


def paired_ci(delta: np.ndarray, users: np.ndarray, selected: np.ndarray, seed: int) -> dict[str, float | int]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for value, user in zip(delta[selected], users[selected]):
        grouped[int(user)].append(float(value))
    units = np.asarray([np.mean(group) for group in grouped.values()])
    rng = np.random.default_rng(seed)
    samples = np.asarray([rng.choice(units, len(units), replace=True).mean() for _ in range(3000)])
    return {"delta": float(units.mean()), "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975)), "users": int(len(units))}


def score(prediction: np.ndarray, base_recall: np.ndarray, base_ndcg: np.ndarray, predecessor: dict[str, np.ndarray], selected: np.ndarray) -> tuple[float, dict[str, float]]:
    recall, ndcg = row_metrics(prediction, predecessor["targets"], predecessor["target_counts"])
    users, buckets = predecessor["users"], predecessor["bucket_ids"]
    long = selected & (buckets == 3)
    values = {
        "overall_recall_delta": user_macro(recall - base_recall, users, selected),
        "overall_ndcg_delta": user_macro(ndcg - base_ndcg, users, selected),
        "long_recall_delta": user_macro(recall - base_recall, users, long),
        "long_ndcg_delta": user_macro(ndcg - base_ndcg, users, long),
    }
    return sum(values.values()), values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor", type=Path, default=Path("runs/music4all_component_c/oof_predecessor/oof_predecessor.npz"))
    parser.add_argument("--features", type=Path, default=Path("runs/music4all_artist_b/features/train_cv_features.npz"))
    parser.add_argument("--c2", type=Path, default=Path("refine-logs/music4all_onion/component_c2/remote_evidence/runs/music4all_component_c2/oof_set_contextual_formal_s2024/oof_predictions.npz"))
    parser.add_argument("--output", type=Path, default=Path("refine-logs/music4all_onion/component_d/SELECTIVE_EXCHANGE_OOF_RESULT_V2.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--members", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("selective_exchange_output_exists")
    predecessor = dict(np.load(args.predecessor, allow_pickle=False))
    features = dict(np.load(args.features, allow_pickle=False))
    c2 = dict(np.load(args.c2, allow_pickle=False))
    if not np.array_equal(predecessor["row_ids"], features["row_ids"]) or not np.array_equal(predecessor["row_ids"], c2["row_ids"]):
        raise ValueError("row_identity_mismatch")
    if not np.array_equal(c2["base_top10"], predecessor["top10"]):
        raise ValueError("c2_predecessor_top10_mismatch")
    pools, matrices, labels = build_rows(predecessor, c2, features)
    users = predecessor["users"]
    folds = user_fold(users, args.folds)
    outputs = {
        "point_estimate": c2["method_top10"].copy(),
        "risk_calibrated": c2["method_top10"].copy(),
    }
    fold_records = []
    # The very large margin is the exact no-op option and makes the long-gap
    # safety constraint feasible in every fold.
    grid = [(margin, kappa) for margin in (-.05, 0.0, .02, .05, .10, .20, 1e6) for kappa in (0.0, .5, 1.0, 2.0)]
    for fold in range(args.folds):
        evaluation = folds == fold
        candidate_cal = calibration_mask(users, fold)
        calibration = (~evaluation) & candidate_cal
        fit = (~evaluation) & (~candidate_cal)
        train_x, train_y, train_w = training_table(matrices, labels, fit)
        models = fit_ensemble(train_x, train_y, train_w, 2024 + fold * 100, args.members)
        cal_mean, cal_std = predict(models, matrices, calibration)
        base_recall, base_ndcg = row_metrics(c2["method_top10"], predecessor["targets"], predecessor["target_counts"])
        choices = []
        for margin, kappa in grid:
            prediction = selective_prediction(pools, cal_mean, cal_std, c2["base_top10"], c2["method_top10"], calibration, margin, kappa)
            objective, values = score(prediction, base_recall, base_ndcg, predecessor, calibration)
            choices.append({"margin": margin, "kappa": kappa, "objective": objective, **values})
        safe = [row for row in choices if row["long_recall_delta"] >= 0 and row["long_ndcg_delta"] >= 0]
        best = max(safe, key=lambda row: (row["objective"], row["kappa"], row["margin"]))
        point_safe = [row for row in safe if row["kappa"] == 0.0]
        point_best = max(point_safe, key=lambda row: (row["objective"], row["margin"]))
        eval_mean, eval_std = predict(models, matrices, evaluation)
        for arm, choice in (("point_estimate", point_best), ("risk_calibrated", best)):
            prediction = selective_prediction(pools, eval_mean, eval_std, c2["base_top10"], c2["method_top10"], evaluation, float(choice["margin"]), float(choice["kappa"]))
            outputs[arm][evaluation] = prediction[evaluation]
        fold_records.append({"fold": fold, "fit_rows": int(fit.sum()), "calibration_rows": int(calibration.sum()), "evaluation_rows": int(evaluation.sum()), "point_selection": point_best, "risk_selection": best})
        print(json.dumps({"event": "component_d_fold", **fold_records[-1]}), flush=True)
    base_recall, base_ndcg = row_metrics(c2["method_top10"], predecessor["targets"], predecessor["target_counts"])
    arms: dict[str, Any] = {}
    for arm_index, (arm, output) in enumerate(outputs.items()):
        method_recall, method_ndcg = row_metrics(output, predecessor["targets"], predecessor["target_counts"])
        metrics: dict[str, Any] = {}
        for scope_index, scope in enumerate(SCOPES):
            selected = np.ones(len(users), dtype=bool) if scope == "overall" else predecessor["bucket_ids"] == scope_index - 1
            metrics[scope] = {
                "rows": int(selected.sum()),
                "c2_recall_at_10": user_macro(base_recall, users, selected),
                "method_recall_at_10": user_macro(method_recall, users, selected),
                "recall_paired": paired_ci(method_recall - base_recall, users, selected, 3000 + 100 * arm_index + scope_index),
                "c2_ndcg_at_10": user_macro(base_ndcg, users, selected),
                "method_ndcg_at_10": user_macro(method_ndcg, users, selected),
                "ndcg_paired": paired_ci(method_ndcg - base_ndcg, users, selected, 4000 + 100 * arm_index + scope_index),
            }
        changed = np.any(output != c2["method_top10"], axis=1)
        better = method_recall > base_recall
        worse = method_recall < base_recall
        arms[arm] = {"metrics": metrics, "coverage": {"changed_rows": int(changed.sum()), "recall_better_rows": int(better.sum()), "recall_worse_rows": int(worse.sum())}}
    metrics = arms["risk_calibrated"]["metrics"]
    coverage = arms["risk_calibrated"]["coverage"]
    point_metrics = arms["point_estimate"]["metrics"]
    pass_gate = metrics["overall"]["recall_paired"]["ci95_low"] > 0 and metrics["overall"]["ndcg_paired"]["ci95_low"] > 0 and metrics[">365d"]["recall_paired"]["delta"] >= 0 and metrics[">365d"]["ndcg_paired"]["delta"] >= 0 and coverage["recall_better_rows"] > coverage["recall_worse_rows"]
    risk_specific = metrics["overall"]["recall_paired"]["delta"] > point_metrics["overall"]["recall_paired"]["delta"] or metrics["overall"]["ndcg_paired"]["delta"] > point_metrics["overall"]["ndcg_paired"]["delta"]
    result = {
        "schema": SCHEMA,
        "status": "pass",
        "decision": "SELECTIVE_EXCHANGE_OOF_PASS" if pass_gate else "SELECTIVE_EXCHANGE_OOF_STOP",
        "evidence_purpose": "attainability",
        "surface": "Train-only nested user-disjoint OOF",
        "method": "ensemble candidate-value lower-confidence-bound reversion of C2 boundary exchanges",
        "arms": arms,
        "risk_specific_increment_present": risk_specific,
        "fold_records": fold_records,
        "bindings": {"predecessor": str(args.predecessor), "c2": str(args.c2), "features": str(args.features), "test_accessed": False, "validation_accessed": False},
    }
    atomic_json(args.output, result)
    print(json.dumps({"decision": result["decision"], "risk_specific_increment_present": risk_specific, "arms": arms}), flush=True)


if __name__ == "__main__":
    main()
