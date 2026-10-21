#!/usr/bin/env python3
"""Seed-matched cumulative D: listwise content residual on frozen C2 scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from scipy import sparse

# Load PyTorch's OpenMP runtime before LightGBM on macOS.  Reversing this
# order can terminate the process before Python can emit a traceback.
import lightgbm as lgb

from analyze_music4all_component_d_content_conversion import build_features, derangement, source_contexts, split_calibration
from analyze_music4all_component_d_content_validation import aggregate, metric_table, packed_prediction
from analyze_music4all_component_d_selective_exchange import atomic_json, row_metrics, user_macro
from lfm_timeconcat_control import LFMTimeConcatData
from run_music4all_artist_component_b import load_artist_features
from run_music4all_candidate_support_a1 import stable_topk
from run_music4all_component_c_oof import context_states
from run_music4all_component_c2_conversion_oof import ConversionRanker, predict_residual
from run_music4all_component_c2_validation import frozen_predecessor


SCHEMA = "genrec.music4all_onion.component_d_joint_content_validation.v1"


def fit_ranker(matrices: list[np.ndarray], labels: list[np.ndarray], selected: np.ndarray, columns, seed: int):
    rows = np.flatnonzero(selected)
    x = np.concatenate([matrices[row][:, columns] for row in rows])
    y = np.concatenate([labels[row] for row in rows])
    groups = np.asarray([len(labels[row]) for row in rows], dtype=np.int32)
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", eval_at=(10,), label_gain=(0, 1),
        n_estimators=220, num_leaves=31, learning_rate=.035, min_child_samples=80,
        subsample=.8, subsample_freq=1, colsample_bytree=.9, reg_lambda=1.0,
        random_state=seed, n_jobs=8, verbosity=-1,
    )
    model.fit(x, y, group=groups)
    return model


def ranker_values(model, matrices: list[np.ndarray], selected: np.ndarray, columns):
    output = [np.empty(0, dtype=np.float32) for _ in matrices]
    for row in np.flatnonzero(selected):
        value = model.booster_.predict(matrices[row][:, columns], num_threads=8).astype(np.float32)
        output[row] = (value - value.mean()) / max(float(value.std()), 1e-6)
    return output


def blended_top10(pools, matrices, values, selected: np.ndarray, alpha: float, base_top10: np.ndarray):
    output = base_top10.copy()
    if alpha == 0:
        return output
    for row in np.flatnonzero(selected):
        score = matrices[row][:, 0] + alpha * values[row]
        order = np.lexsort((pools[row], -score))[:10]
        output[row] = pools[row][order]
    return output


def choose_alpha(model, pools, matrices, columns, c2, pred, calibration):
    values = ranker_values(model, matrices, calibration, columns)
    base_r, base_n = row_metrics(c2["method_top10"], pred["targets"], pred["target_counts"])
    choices = []
    for alpha in (0.0, .05, .1, .2, .5, 1.0, 2.0):
        output = blended_top10(pools, matrices, values, calibration, alpha, c2["method_top10"])
        r, n = row_metrics(output, pred["targets"], pred["target_counts"])
        long = calibration & (pred["bucket_ids"] == 3)
        record = {
            "alpha": alpha,
            "overall_recall_delta": user_macro(r - base_r, pred["users"], calibration),
            "overall_ndcg_delta": user_macro(n - base_n, pred["users"], calibration),
            "long_recall_delta": user_macro(r - base_r, pred["users"], long),
            "long_ndcg_delta": user_macro(n - base_n, pred["users"], long),
        }
        record["objective"] = sum(record[key] for key in record if key.endswith("_delta"))
        choices.append(record)
    safe = [row for row in choices if row["long_recall_delta"] >= 0 and row["long_ndcg_delta"] >= 0]
    return max(safe, key=lambda row: (row["objective"], -row["alpha"])), choices


@torch.no_grad()
def c2_scores(checkpoint_path: Path, candidates, scores, states, gaps, embeddings):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ConversionRanker(128, 64, contextual=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    residual = predict_residual(model, candidates, scores, states, gaps, embeddings, "cpu", 256)
    return scores + float(checkpoint["selection"]["beta"]) * residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--features-root", type=Path, default=Path("runs/music4all_artist_b/features"))
    parser.add_argument("--content-dir", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1"))
    parser.add_argument("--c2-root", type=Path, default=Path("refine-logs/music4all_onion/component_c2/validation_remote_evidence"))
    parser.add_argument("--a1-root", type=Path, default=Path("refine-logs/music4all_onion/candidate_support_a1/remote_evidence"))
    parser.add_argument("--artist-root", type=Path, default=Path("refine-logs/music4all_onion/component_b2/artist_b_remote_evidence"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("runs/music4all_candidate_support_a1/crossfit_sasrec_s2024/final_model.pt"))
    parser.add_argument("--base-training-manifest", type=Path, default=Path("runs/music4all_candidate_support_a1/crossfit_sasrec_s2024/training_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("refine-logs/music4all_onion/component_d/JOINT_CONTENT_VALIDATION_RESULT.json"))
    parser.add_argument("--row-metrics-output", type=Path)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--max-context", type=int, default=129)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("joint_content_validation_output_exists")
    torch.set_num_threads(4)

    train_features, _ = load_artist_features(args.features_root / "train_cv_features.npz", args.features_root / "train_cv_features_manifest.json", "train_cv")
    validation_features, _ = load_artist_features(args.features_root / "validation_features.npz", args.features_root / "validation_features_manifest.json", "validation")
    data = LFMTimeConcatData(args.data_dir, max_context=args.max_context, horizon=args.horizon)
    train_states, item_array, _ = context_states(args, data, train_features["row_ids"])
    validation_states, validation_items, _ = context_states(args, data, validation_features["row_ids"])
    if not np.array_equal(item_array, validation_items):
        raise ValueError("item_embedding_drift")
    embeddings = torch.as_tensor(item_array)

    seed_inputs = {}
    for seed in (2024, 2025, 2026):
        run_args = SimpleNamespace(**vars(args), seed=seed)
        train_candidates, train_base_scores, _, _, _, _ = frozen_predecessor(run_args, train_features, "train_cv")
        validation_candidates, validation_base_scores, _, _, _, _ = frozen_predecessor(run_args, validation_features, "validation")
        directory = args.c2_root / f"validation_set_contextual_s{seed}_formal"
        c2 = dict(np.load(directory / "validation_predictions.npz", allow_pickle=False))
        checkpoint = directory / "component_c2_checkpoint.pt"
        train_scores = c2_scores(checkpoint, train_candidates, train_base_scores, train_states, train_features["gaps"], embeddings)
        validation_scores = c2_scores(checkpoint, validation_candidates, validation_base_scores, validation_states, validation_features["gaps"], embeddings)
        reconstructed_top10 = stable_topk(validation_scores, validation_candidates)
        mismatch_rows = int(np.any(reconstructed_top10 != c2["method_top10"], axis=1).sum())
        if mismatch_rows:
            print(json.dumps({
                "event": "c2_score_reconstruction_mismatch",
                "seed": seed,
                "rows": mismatch_rows,
                "cells": int((reconstructed_top10 != c2["method_top10"]).sum()),
                "first_row": int(np.flatnonzero(np.any(reconstructed_top10 != c2["method_top10"], axis=1))[0]),
            }), flush=True)
            if mismatch_rows / len(reconstructed_top10) > 0.002:
                raise ValueError("c2_score_reconstruction_failed")
        train_pred = packed_prediction(train_features, train_candidates, train_scores)
        validation_pred = packed_prediction(validation_features, validation_candidates, validation_scores)
        train_c2 = {"method_top10": stable_topk(train_scores, train_candidates)}
        seed_inputs[seed] = (train_pred, train_c2, validation_pred, c2)
        print(json.dumps({"event": "joint_content_c2_reconstructed", "seed": seed}), flush=True)

    audio = np.load(args.content_dir / "musicnn_normalized.npy", mmap_mode="r")
    genre = sparse.load_npz(args.content_dir / "genre_normalized.npz").tocsr()
    train_contexts = source_contexts(next(iter(seed_inputs.values()))[0], args.data_dir, audio, genre)
    validation_contexts = source_contexts(next(iter(seed_inputs.values()))[2], args.data_dir, audio, genre)
    train_source = np.arange(len(train_features["row_ids"]), dtype=np.int64)
    validation_source = np.arange(len(validation_features["row_ids"]), dtype=np.int64)
    train_shuffle = derangement(len(train_source))
    validation_shuffle = derangement(len(validation_source))
    calibration = split_calibration(train_features["users"], 99)
    fit = ~calibration

    all_arrays = {arm: {} for arm in ("behavior_only", "real_content", "shuffled_content")}
    seed_results: dict[str, Any] = {}
    selection_records: dict[str, Any] = {}
    for seed, (train_pred, train_c2, validation_pred, validation_c2) in seed_inputs.items():
        train_real = build_features(train_pred, train_c2, train_features, audio, genre, train_contexts, train_source, args.depth)
        train_shuffled = build_features(train_pred, train_c2, train_features, audio, genre, train_contexts, train_shuffle, args.depth)
        validation_real = build_features(validation_pred, validation_c2, validation_features, audio, genre, validation_contexts, validation_source, args.depth)
        validation_shuffled = build_features(validation_pred, validation_c2, validation_features, audio, genre, validation_contexts, validation_shuffle, args.depth)
        arms = {
            "behavior_only": (train_real, validation_real, list(range(13))),
            "real_content": (train_real, validation_real, slice(None)),
            "shuffled_content": (train_shuffled, validation_shuffled, slice(None)),
        }
        seed_result = {}
        for arm_index, (arm, (training, validation, columns)) in enumerate(arms.items()):
            train_pools, train_matrices, train_labels = training
            validation_pools, validation_matrices, _ = validation
            calibration_model = fit_ranker(train_matrices, train_labels, fit, columns, 40000 + seed + arm_index)
            selection, grid = choose_alpha(calibration_model, train_pools, train_matrices, columns, train_c2, train_pred, calibration)
            model = fit_ranker(train_matrices, train_labels, np.ones(len(train_matrices), dtype=bool), columns, 41000 + seed + arm_index)
            selected = np.ones(len(validation_matrices), dtype=bool)
            values = ranker_values(model, validation_matrices, selected, columns)
            output = blended_top10(validation_pools, validation_matrices, values, selected, float(selection["alpha"]), validation_c2["method_top10"])
            metrics, arrays = metric_table(output, validation_c2["method_top10"], validation_pred, 42000 + seed + arm_index * 100)
            seed_result[arm] = {
                "metrics": metrics,
                "coverage": {
                    "changed_rows": int(np.any(output != validation_c2["method_top10"], axis=1).sum()),
                    "recall_better_rows": int((arrays["method_r"] > arrays["base_r"]).sum()),
                    "recall_worse_rows": int((arrays["method_r"] < arrays["base_r"]).sum()),
                },
            }
            all_arrays[arm][str(seed)] = arrays
            selection_records[f"{seed}:{arm}"] = {"selected": selection, "grid": grid}
        seed_results[str(seed)] = seed_result
        print(json.dumps({"event": "joint_content_seed", "seed": seed, "results": seed_result}), flush=True)

    aggregated = aggregate(all_arrays, validation_features["users"], validation_features["bucket_ids"])
    real, behavior, shuffled = aggregated["real_content"], aggregated["behavior_only"], aggregated["shuffled_content"]
    overall, long = real["overall"], real[">365d"]
    content_increment = all(
        overall[metric]["delta"] > behavior["overall"][metric]["delta"]
        and overall[metric]["delta"] > shuffled["overall"][metric]["delta"]
        for metric in ("recall_paired", "ndcg_paired")
    )
    seed_positive = all(
        seed_results[str(seed)]["real_content"]["metrics"]["overall"]["recall_paired"]["delta"] > 0
        and seed_results[str(seed)]["real_content"]["metrics"]["overall"]["ndcg_paired"]["delta"] > 0
        for seed in (2024, 2025, 2026)
    )
    better = sum(seed_results[str(seed)]["real_content"]["coverage"]["recall_better_rows"] for seed in (2024, 2025, 2026))
    worse = sum(seed_results[str(seed)]["real_content"]["coverage"]["recall_worse_rows"] for seed in (2024, 2025, 2026))
    passed = (
        overall["recall_paired"]["ci95_low"] > 0
        and overall["ndcg_paired"]["ci95_low"] > 0
        and long["recall_paired"]["delta"] >= 0
        and long["ndcg_paired"]["delta"] >= 0
        and content_increment and seed_positive and better > worse
    )
    result = {
        "schema": SCHEMA,
        "status": "pass",
        "decision": "JOINT_CONTENT_VALIDATION_PASS" if passed else "JOINT_CONTENT_VALIDATION_STOP",
        "method": "seed-matched listwise content residual over frozen A1+B+C2 scores",
        "selections": selection_records,
        "seeds": seed_results,
        "aggregate": aggregated,
        "gate": {
            "content_increment_over_controls": content_increment,
            "all_seed_overall_recall_and_ndcg_positive": seed_positive,
            "recall_better_rows": better,
            "recall_worse_rows": worse,
        },
        "bindings": {"validation_accessed": True, "test_accessed": False, "predecessor_retrained": False},
    }
    if args.row_metrics_output is not None:
        args.row_metrics_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "users": validation_features["users"],
            "bucket_ids": validation_features["bucket_ids"],
        }
        for seed, arrays in all_arrays["real_content"].items():
            payload[f"recall_s{seed}"] = arrays["method_r"]
            payload[f"ndcg_s{seed}"] = arrays["method_n"]
        np.savez_compressed(args.row_metrics_output, **payload)
    atomic_json(args.output, result)
    print(json.dumps({"decision": result["decision"], "aggregate": aggregated, "gate": result["gate"]}), flush=True)


if __name__ == "__main__":
    main()
