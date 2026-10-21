#!/usr/bin/env python3
"""Apply the frozen cumulative ARRIVE method once on sealed Music4All Test."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import sparse
import torch

from analyze_music4all_component_d_content_conversion import build_features, source_contexts
from analyze_music4all_component_d_content_validation import packed_prediction
from lfm_timeconcat_control import FIRST_ITEM, LFMTimeConcatData, gap_bucket
from music_sequence_core import set_metrics
from run_music4all_artist_component_b import load_artist_features
from run_music4all_component_c_oof import context_states
from run_music4all_component_c2_validation import frozen_predecessor
from run_music4all_component_d_joint_content_validation import (
    blended_top10,
    c2_scores,
    ranker_values,
)
from run_music4all_candidate_support_a1 import stable_topk


SCOPES = ("overall", "30-90d", "90-180d", "180-365d", ">365d")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def metric_table(values: np.ndarray, users: np.ndarray, buckets: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope in SCOPES:
        selected = np.ones(len(values), dtype=bool) if scope == "overall" else buckets == scope
        grouped: dict[int, list[float]] = defaultdict(list)
        for value, user in zip(values[selected], users[selected]):
            grouped[int(user)].append(float(value))
        result[scope] = {
            "rows": int(selected.sum()),
            "users": len(grouped),
            "value": float(np.mean([np.mean(row) for row in grouped.values()])) if grouped else None,
        }
    return result


def validate_test(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.test_manifest.read_text())
    if manifest.get("schema") != "genrec.music4all_onion.sealed_test.v1" or manifest.get("status") != "pass":
        raise ValueError("sealed_test_manifest_invalid")
    if manifest.get("test_accessed") is not True or manifest.get("selection_or_tuning_permitted") is not False:
        raise ValueError("sealed_test_policy_invalid")
    for name, entry in manifest["artifacts"].items():
        if digest(args.test_data / name) != entry["sha256"]:
            raise ValueError(f"sealed_test_artifact_mismatch:{name}")
    return manifest


def score_rows(predictions: np.ndarray, data: LFMTimeConcatData, labels: Any, row_ids: np.ndarray):
    if not np.array_equal(row_ids, labels["row_ids"]) or len(predictions) != len(row_ids):
        raise ValueError("prediction_label_row_identity_mismatch")
    users = data.sessions[row_ids, 0].astype(np.int64)
    buckets = np.asarray([gap_bucket(data.gap_days(int(row))) for row in row_ids])
    recall = np.zeros(len(row_ids), dtype=np.float64)
    ndcg = np.zeros(len(row_ids), dtype=np.float64)
    seen_recall = np.zeros(len(row_ids), dtype=np.float64)
    seen_ndcg = np.zeros(len(row_ids), dtype=np.float64)
    seen_rows = np.zeros(len(row_ids), dtype=bool)
    for at in range(len(row_ids)):
        truth = labels["values"][labels["offsets"][at] : labels["offsets"][at + 1]].tolist()
        recall[at], ndcg[at] = set_metrics(predictions[at].tolist(), truth)
        known = [int(item) for item in truth if int(item) >= FIRST_ITEM]
        if known:
            seen_rows[at] = True
            seen_recall[at], seen_ndcg[at] = set_metrics(predictions[at].tolist(), known)
    metrics = {
        "all_targets": {
            "recall_at_10": metric_table(recall, users, buckets),
            "ndcg_at_10": metric_table(ndcg, users, buckets),
        },
        "train_seen_targets": {
            "rows_with_seen_target": int(seen_rows.sum()),
            "recall_at_10": metric_table(seen_recall[seen_rows], users[seen_rows], buckets[seen_rows]),
            "ndcg_at_10": metric_table(seen_ndcg[seen_rows], users[seen_rows], buckets[seen_rows]),
        },
    }
    arrays = {
        "users": users,
        "gap_buckets": buckets,
        "recall": recall,
        "ndcg": ndcg,
        "seen_recall": seen_recall,
        "seen_ndcg": seen_ndcg,
        "seen_rows": seen_rows,
    }
    return metrics, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(2024, 2025, 2026), required=True)
    parser.add_argument("--train-data", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, default=Path("runs/music4all_artist_b/features"))
    parser.add_argument("--content-dir", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1"))
    parser.add_argument("--c2-root", type=Path, default=Path("runs/music4all_component_c2"))
    parser.add_argument("--a1-root", type=Path, default=Path("runs/music4all_candidate_support_a1"))
    parser.add_argument("--artist-root", type=Path, default=Path("runs/music4all_artist_b"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("runs/music4all_candidate_support_a1/crossfit_sasrec_s2024/final_model.pt"))
    parser.add_argument("--base-training-manifest", type=Path, default=Path("runs/music4all_candidate_support_a1/crossfit_sasrec_s2024/training_manifest.json"))
    parser.add_argument("--validation-decision", type=Path, default=Path("refine-logs/music4all_onion/component_d/JOINT_CONTENT_VALIDATION_RESULT.json"))
    parser.add_argument("--content-model", type=Path, required=True,
                        help="Saved and Validation-reproduced D predictor; no fitting is allowed here")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--max-context", type=int, default=129)
    parser.add_argument("--horizon", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("sealed_arrive_test_output_exists")
    torch.set_num_threads(4)
    test_manifest = validate_test(args)
    decision = json.loads(args.validation_decision.read_text())
    if decision.get("decision") != "JOINT_CONTENT_VALIDATION_PASS" or decision.get("bindings", {}).get("test_accessed") is not False:
        raise ValueError("validation_decision_not_frozen")

    from freeze_arrive_content_model import load_frozen_content
    model, content_manifest = load_frozen_content(
        args.content_model, seed=args.seed, decision=args.validation_decision,
    )
    test, test_features_manifest = load_artist_features(
        args.features_root / "test_features.npz",
        args.features_root / "test_features_manifest.json",
        "test",
    )
    test_data = LFMTimeConcatData(args.test_data, max_context=args.max_context, horizon=args.horizon)
    test_states, item_array, _ = context_states(args, test_data, test["row_ids"])
    embeddings = torch.as_tensor(item_array)

    run_args = SimpleNamespace(**vars(args))
    test_candidates, test_base_scores, _, _, _, test_predecessor_bindings = frozen_predecessor(run_args, test, "test")
    frozen_predecessor_bindings = content_manifest["predecessor_bindings"]["train_cv"]
    for key in ("a1_result_sha256", "a1_checkpoint_sha256", "artist_b_result_sha256", "artist_b_checkpoint_sha256"):
        if test_predecessor_bindings.get(key) != frozen_predecessor_bindings[key]:
            raise ValueError(f"content_model_predecessor_binding_mismatch:{key}")
    if (args.max_context, args.horizon) != (content_manifest["max_context"], content_manifest["horizon"]):
        raise ValueError("content_model_context_boundary_mismatch")
    if content_manifest["feature_builder_sha256"] != digest(Path(__file__).with_name("analyze_music4all_component_d_content_conversion.py")):
        raise ValueError("content_model_feature_builder_changed")
    c2_directory = args.c2_root / f"validation_set_contextual_s{args.seed}_formal"
    c2_checkpoint = c2_directory / "component_c2_checkpoint.pt"
    for key, path in (
        ("c2_checkpoint_sha256", c2_checkpoint), ("base_checkpoint_sha256", args.base_checkpoint),
        ("audio_sha256", args.content_dir / "musicnn_normalized.npy"),
        ("genre_sha256", args.content_dir / "genre_normalized.npz"),
    ):
        if content_manifest[key] != digest(path):
            raise ValueError(f"content_model_input_binding_mismatch:{key}")
    test_scores = c2_scores(c2_checkpoint, test_candidates, test_base_scores, test_states, test["gaps"], embeddings)
    test_c2 = {"method_top10": stable_topk(test_scores, test_candidates)}
    test_pred = packed_prediction(test, test_candidates, test_scores)

    audio = np.load(args.content_dir / "musicnn_normalized.npy", mmap_mode="r")
    genre = sparse.load_npz(args.content_dir / "genre_normalized.npz").tocsr()
    test_context = source_contexts(test_pred, args.test_data, audio, genre)
    test_source = np.arange(len(test["row_ids"]), dtype=np.int64)
    test_pools, test_matrices, _ = build_features(
        test_pred, test_c2, test, audio, genre, test_context, test_source, args.depth
    )
    columns = slice(None)
    if args.depth != content_manifest["depth"]:
        raise ValueError("content_model_depth_mismatch")
    selected = np.ones(len(test_matrices), dtype=bool)
    values = ranker_values(model, test_matrices, selected, columns)
    selection = decision["selections"][f"{args.seed}:real_content"]["selected"]
    predictions = blended_top10(
        test_pools,
        test_matrices,
        values,
        selected,
        float(selection["alpha"]),
        test_c2["method_top10"],
    )

    labels = np.load(args.test_data / "metric_labels.npz", allow_pickle=False)
    row_ids = test["row_ids"].astype(np.int64)
    if not np.array_equal(row_ids, labels["row_ids"]):
        raise ValueError("sealed_arrive_row_identity_mismatch")
    metrics, arrays = score_rows(predictions, test_data, labels, row_ids)
    metrics["target_coverage"] = test_manifest["counts"]["train_seen_target_coverage"]
    args.output.mkdir(parents=True, exist_ok=False)
    prediction_path = args.output / "test_predictions.npz"
    atomic_npz(prediction_path, row_ids=row_ids, top10=predictions, **arrays)
    result = {
        "schema": "genrec.arrive.sealed_test_result.v1",
        "status": "pass",
        "method": "arrive",
        "seed": args.seed,
        "inference_only": True,
        "rows": len(row_ids),
        "metrics": metrics,
        "selection": selection,
        "validation_decision_sha256": digest(args.validation_decision),
        "test_manifest_sha256": digest(args.test_manifest),
        "test_features_manifest_sha256": digest(args.features_root / "test_features_manifest.json"),
        "c2_checkpoint_sha256": digest(c2_checkpoint),
        "predecessor_bindings": content_manifest["predecessor_bindings"]["train_cv"],
        "content_model_sha256": content_manifest["model_sha256"],
        "content_model_manifest_sha256": digest(args.content_model / "MODEL_MANIFEST.json"),
        "audit_correction": True,
        "new_untouched_test_claim": False,
        "test_predecessor_bindings": test_predecessor_bindings,
        "predictions_sha256": digest(prediction_path),
        "test_accessed": True,
        "selection_or_tuning_performed": False,
    }
    atomic_json(args.output / "TEST_RESULT.json", result)
    print(json.dumps({"event": "sealed_arrive_test_complete", "seed": args.seed, "metrics": metrics["all_targets"]}), flush=True)


if __name__ == "__main__":
    main()
