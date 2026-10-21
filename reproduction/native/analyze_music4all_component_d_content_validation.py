#!/usr/bin/env python3
"""Three-seed Validation transfer for return-context content conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import sparse

from analyze_music4all_component_d_content_conversion import (
    build_features,
    choose,
    derangement,
    exchange,
    probabilities,
    source_contexts,
    split_calibration,
    table,
    train,
)
from analyze_music4all_component_d_selective_exchange import (
    SCOPES,
    atomic_json,
    paired_ci,
    row_metrics,
    user_macro,
)
from run_music4all_artist_component_b import load_artist_features
from run_music4all_component_c2_validation import frozen_predecessor


SCHEMA = "genrec.music4all_onion.component_d_content_conversion_validation.v1"


def supported_targets(features: dict[str, np.ndarray]) -> np.ndarray:
    width = int(features["target_counts"].max())
    output = np.full((len(features["row_ids"]), width), -1, dtype=np.int32)
    for row in range(len(output)):
        truth = list(map(int, features["candidates"][row][features["target_mask"][row].astype(bool)]))
        proposal = features["proposal_candidates"][row, 0]
        proposal_mask = features["proposal_target_mask"][row, 0].astype(bool)
        for item in map(int, proposal[proposal_mask]):
            if item not in truth:
                truth.append(item)
        output[row, : len(truth)] = truth
    return output


def packed_prediction(
    features: dict[str, np.ndarray], candidates: np.ndarray, scores: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "row_ids": features["row_ids"],
        "users": features["users"],
        "gaps": features["gaps"],
        "bucket_ids": features["bucket_ids"],
        "candidates": candidates,
        "scores": scores,
        "targets": supported_targets(features),
        "target_counts": features["target_counts"],
    }


def metric_table(
    prediction: np.ndarray,
    base: np.ndarray,
    packed: dict[str, np.ndarray],
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    base_r, base_n = row_metrics(base, packed["targets"], packed["target_counts"])
    method_r, method_n = row_metrics(prediction, packed["targets"], packed["target_counts"])
    metrics: dict[str, Any] = {}
    for scope_index, scope in enumerate(SCOPES):
        selected = np.ones(len(base_r), dtype=bool) if scope == "overall" else packed["bucket_ids"] == scope_index - 1
        metrics[scope] = {
            "predecessor_recall_at_10": user_macro(base_r, packed["users"], selected),
            "method_recall_at_10": user_macro(method_r, packed["users"], selected),
            "recall_paired": paired_ci(method_r - base_r, packed["users"], selected, seed + scope_index),
            "predecessor_ndcg_at_10": user_macro(base_n, packed["users"], selected),
            "method_ndcg_at_10": user_macro(method_n, packed["users"], selected),
            "ndcg_paired": paired_ci(method_n - base_n, packed["users"], selected, seed + 100 + scope_index),
        }
    return metrics, {
        "base_r": base_r,
        "base_n": base_n,
        "method_r": method_r,
        "method_n": method_n,
    }


def aggregate(
    arrays: dict[str, dict[str, dict[str, np.ndarray]]],
    users: np.ndarray,
    buckets: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm_index, (arm, seed_arrays) in enumerate(arrays.items()):
        recall_delta = np.mean([row["method_r"] - row["base_r"] for row in seed_arrays.values()], axis=0)
        ndcg_delta = np.mean([row["method_n"] - row["base_n"] for row in seed_arrays.values()], axis=0)
        arm_result = {}
        for scope_index, scope in enumerate(SCOPES):
            selected = np.ones(len(users), dtype=bool) if scope == "overall" else buckets == scope_index - 1
            arm_result[scope] = {
                "recall_paired": paired_ci(recall_delta, users, selected, 30000 + arm_index * 100 + scope_index),
                "ndcg_paired": paired_ci(ndcg_delta, users, selected, 31000 + arm_index * 100 + scope_index),
            }
        result[arm] = arm_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor", type=Path, default=Path("runs/music4all_component_c/oof_predecessor/oof_predecessor.npz"))
    parser.add_argument("--oof-c2", type=Path, default=Path("refine-logs/music4all_onion/component_c2/remote_evidence/runs/music4all_component_c2/oof_set_contextual_formal_s2024/oof_predictions.npz"))
    parser.add_argument("--train-features", type=Path, default=Path("runs/music4all_artist_b/features/train_cv_features.npz"))
    parser.add_argument("--validation-features-root", type=Path, default=Path("runs/music4all_artist_b/features"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--content-dir", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1"))
    parser.add_argument("--validation-c2-root", type=Path, default=Path("refine-logs/music4all_onion/component_c2/validation_remote_evidence"))
    parser.add_argument("--a1-root", type=Path, default=Path("refine-logs/music4all_onion/candidate_support_a1/remote_evidence"))
    parser.add_argument("--artist-root", type=Path, default=Path("refine-logs/music4all_onion/component_b2/artist_b_remote_evidence"))
    parser.add_argument("--output", type=Path, default=Path("refine-logs/music4all_onion/component_d/CONTENT_CONVERSION_VALIDATION_RESULT.json"))
    parser.add_argument("--depth", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("content_conversion_validation_output_exists")

    train_pred = dict(np.load(args.predecessor, allow_pickle=False))
    train_c2 = dict(np.load(args.oof_c2, allow_pickle=False))
    train_raw = dict(np.load(args.train_features, allow_pickle=False))
    validation_features, _ = load_artist_features(
        args.validation_features_root / "validation_features.npz",
        args.validation_features_root / "validation_features_manifest.json",
        "validation",
    )

    # Reconstruct every PyTorch predecessor before starting LightGBM workers.
    validation_inputs: dict[int, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    for seed in (2024, 2025, 2026):
        c2 = dict(np.load(args.validation_c2_root / f"validation_set_contextual_s{seed}_formal" / "validation_predictions.npz", allow_pickle=False))
        run_args = SimpleNamespace(
            seed=seed,
            features_root=args.validation_features_root,
            a1_root=args.a1_root,
            artist_root=args.artist_root,
            device="cpu",
            eval_batch_size=256,
        )
        candidates, scores, top10, _, _, _ = frozen_predecessor(run_args, validation_features, "validation")
        if not np.array_equal(top10, c2["base_top10"]):
            raise ValueError("validation_predecessor_reconstruction_failed")
        validation_inputs[seed] = (c2, packed_prediction(validation_features, candidates, scores))
        print(json.dumps({"event": "content_validation_predecessor", "seed": seed}), flush=True)

    audio = np.load(args.content_dir / "musicnn_normalized.npy", mmap_mode="r")
    genre = sparse.load_npz(args.content_dir / "genre_normalized.npz").tocsr()
    train_contexts = source_contexts(train_pred, args.data_dir, audio, genre)
    train_real = build_features(
        train_pred, train_c2, train_raw, audio, genre, train_contexts,
        np.arange(len(train_pred["row_ids"]), dtype=np.int64), args.depth,
    )
    train_shuffled = build_features(
        train_pred, train_c2, train_raw, audio, genre, train_contexts,
        derangement(len(train_pred["row_ids"])), args.depth,
    )

    validation_real: dict[int, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]] = {}
    validation_shuffled: dict[int, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]] = {}
    validation_source = np.arange(len(validation_features["row_ids"]), dtype=np.int64)
    shuffled_source = derangement(len(validation_source))
    for seed, (c2, pred) in validation_inputs.items():
        contexts = source_contexts(pred, args.data_dir, audio, genre)
        validation_real[seed] = build_features(pred, c2, validation_features, audio, genre, contexts, validation_source, args.depth)
        validation_shuffled[seed] = build_features(pred, c2, validation_features, audio, genre, contexts, shuffled_source, args.depth)

    arm_inputs = {
        "behavior_only": (train_real, list(range(13))),
        "real_content": (train_real, slice(None)),
        "shuffled_content": (train_shuffled, slice(None)),
    }
    calibration = split_calibration(train_pred["users"], 99)
    fit = ~calibration
    models = {}
    selections = {}
    for arm_index, (arm, ((pools, matrices, labels), columns)) in enumerate(arm_inputs.items()):
        x, y, weights = table(matrices, labels, fit, columns)
        calibration_model = train(x, y, weights, 20000 + arm_index)
        selections[arm] = choose(calibration_model, matrices, pools, train_c2, train_pred, calibration, columns)
        x_all, y_all, weights_all = table(matrices, labels, np.ones(len(matrices), dtype=bool), columns)
        models[arm] = train(x_all, y_all, weights_all, 21000 + arm_index)
        print(json.dumps({"event": "content_validation_arm_frozen", "arm": arm, "selection": selections[arm]}), flush=True)

    seed_results: dict[str, Any] = {}
    all_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {arm: {} for arm in arm_inputs}
    for seed, (c2, pred) in validation_inputs.items():
        seed_result: dict[str, Any] = {}
        for arm, (_, columns) in arm_inputs.items():
            pools, matrices, _ = validation_shuffled[seed] if arm == "shuffled_content" else validation_real[seed]
            selected = np.ones(len(matrices), dtype=bool)
            probs = probabilities(models[arm], matrices, selected, columns)
            choice = selections[arm]
            output = exchange(pools, probs, c2["method_top10"], selected, int(choice["quota"]), float(choice["margin"]))
            metrics, arrays = metric_table(output, c2["method_top10"], pred, 22000 + seed + len(seed_result) * 100)
            seed_result[arm] = {
                "metrics": metrics,
                "coverage": {
                    "changed_rows": int(np.any(output != c2["method_top10"], axis=1).sum()),
                    "recall_better_rows": int((arrays["method_r"] > arrays["base_r"]).sum()),
                    "recall_worse_rows": int((arrays["method_r"] < arrays["base_r"]).sum()),
                },
            }
            all_arrays[arm][str(seed)] = arrays
        seed_results[str(seed)] = seed_result
        print(json.dumps({"event": "content_validation_seed", "seed": seed, "results": seed_result}), flush=True)

    aggregate_metrics = aggregate(all_arrays, validation_features["users"], validation_features["bucket_ids"])
    real = aggregate_metrics["real_content"]
    behavior = aggregate_metrics["behavior_only"]
    shuffled = aggregate_metrics["shuffled_content"]
    real_overall = real["overall"]
    real_long = real[">365d"]
    three_seed_positive = all(
        seed_results[str(seed)]["real_content"]["metrics"]["overall"]["recall_paired"]["delta"] > 0
        for seed in (2024, 2025, 2026)
    )
    better = sum(seed_results[str(seed)]["real_content"]["coverage"]["recall_better_rows"] for seed in (2024, 2025, 2026))
    worse = sum(seed_results[str(seed)]["real_content"]["coverage"]["recall_worse_rows"] for seed in (2024, 2025, 2026))
    content_increment = (
        real_overall["recall_paired"]["delta"] > behavior["overall"]["recall_paired"]["delta"]
        and real_overall["recall_paired"]["delta"] > shuffled["overall"]["recall_paired"]["delta"]
        and real_overall["ndcg_paired"]["delta"] > behavior["overall"]["ndcg_paired"]["delta"]
        and real_overall["ndcg_paired"]["delta"] > shuffled["overall"]["ndcg_paired"]["delta"]
    )
    passed = (
        real_overall["recall_paired"]["ci95_low"] > 0
        and real_overall["ndcg_paired"]["ci95_low"] > 0
        and real_long["recall_paired"]["delta"] >= 0
        and real_long["ndcg_paired"]["delta"] >= 0
        and content_increment
        and three_seed_positive
        and better > worse
    )
    result = {
        "schema": SCHEMA,
        "status": "pass",
        "decision": "CONTENT_CONVERSION_VALIDATION_PASS" if passed else "CONTENT_CONVERSION_VALIDATION_STOP",
        "method": "return-context content conversion over frozen A1+B+C2 Top-50 boundary pool",
        "selections": selections,
        "seeds": seed_results,
        "aggregate": aggregate_metrics,
        "gate": {
            "content_increment_over_controls": content_increment,
            "three_seed_overall_recall_positive": three_seed_positive,
            "real_content_recall_better_rows": better,
            "real_content_recall_worse_rows": worse,
        },
        "bindings": {
            "validation_accessed": True,
            "test_accessed": False,
            "predecessor_retrained": False,
            "oof_c2": str(args.oof_c2),
        },
    }
    atomic_json(args.output, result)
    print(json.dumps({"decision": result["decision"], "aggregate": aggregate_metrics, "gate": result["gate"]}), flush=True)


if __name__ == "__main__":
    main()
