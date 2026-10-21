#!/usr/bin/env python3
"""Inference-only evaluator for frozen ARRIVE main-table checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from lfm_timeconcat_control import FIRST_ITEM, LFMTimeConcatData, exact_catalog_topk, gap_bucket
from music_sequence_core import set_metrics
import run_music4all_baselines as sasrec_runner
import run_music4all_arrive_confirmation_baselines as confirmation_runner
from run_arrive_main_table_baseline import MainTableData, create_model as create_main_model, exact_topk


METHODS = ("sasrec", "car", "tisasrec", "ssd4rec", "cosernn", "muse", "megan", "factubr", "deltagate")
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
        if not np.any(selected):
            result[scope] = {"rows": 0, "users": 0, "value": None}
            continue
        by_user: dict[int, list[float]] = defaultdict(list)
        for value, user in zip(values[selected], users[selected]):
            by_user[int(user)].append(float(value))
        result[scope] = {
            "rows": int(selected.sum()),
            "users": len(by_user),
            "value": float(np.mean([np.mean(row) for row in by_user.values()])),
        }
    return result


REGISTERED_SEALED_TEST = {
    "genrec.music4all_onion.sealed_test.v1": {"rows": 8701, "gap_buckets": [5964, 1184, 623, 930]},
    "genrec.yambda500m.sealed_test.v1": {"rows": 55788, "gap_buckets": [41637, 8246, 3370, 2535]},
    "genrec.yambda500m.sealed_test.v2": {"rows": 8213, "gap_buckets": [5906, 1599, 708, 0]},
}


def validate_test(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.test_manifest.read_text())
    # Both dataset suites share this evaluator; each registered sealed-test
    # contract carries its own frozen denominators (test opening is
    # one-shot, so the expected counts are pinned per schema at registration
    # time, not hardcoded to one dataset).
    registered = REGISTERED_SEALED_TEST.get(manifest.get("schema"))
    if registered is None or manifest.get("status") != "pass":
        raise ValueError("sealed_test_manifest_invalid")
    if manifest.get("test_accessed") is not True or manifest.get("selection_or_tuning_permitted") is not False:
        raise ValueError("sealed_test_policy_invalid")
    for name, entry in manifest["artifacts"].items():
        if digest(args.test_data / name) != entry["sha256"]:
            raise ValueError(f"sealed_test_artifact_mismatch:{name}")
    if (manifest["counts"]["rows"] != registered["rows"]
            or manifest["counts"]["gap_buckets"] != registered["gap_buckets"]):
        raise ValueError("sealed_test_denominator_mismatch")
    return manifest


def adapter_identity(method: str, training: dict[str, Any]) -> tuple[str, str]:
    """Return the serving implementation family and an honest report label."""

    revision = training.get("binding", {}).get("adapter_revision", "original")
    if method == "car" and revision == "audit20260912":
        return "main_table", "CAR-no-user (task-adapted)"
    if method == "cosernn" and revision == "faithful20260912":
        return "main_table", "CoSeRNN-Plays-Time (task-adapted)"
    if method == "muse" and revision == "audit20260912":
        return "main_table", "MUSE (observed-prefix task-adapted)"
    if method == "megan" and revision == "faithful20260912":
        return "main_table", "MEGAN (task-adapted)"
    if method == "deltagate" and revision == "faithful20260917":
        return "main_table", "DeltaGate (observed-return task-adapted)"
    if method == "deltagate" and revision == "warmhead20260917":
        return "main_table", "DeltaGate (warm head-only on SASRec backbone)"
    if method == "deltagate" and revision == "warmfull20260917":
        return "main_table", "DeltaGate (warm full fine-tune on SASRec backbone)"
    if method == "sasrec":
        return "sasrec", "SASRec"
    if method == "car":
        return "legacy_car", "CAR historical control"
    return "main_table", method


def load_model(args: argparse.Namespace, data: LFMTimeConcatData, training: dict[str, Any]):
    checkpoint_path = args.checkpoint_dir / "best_model.pt"
    if training.get("status") != "pass" or training.get("checkpoint_strict_reload") is not True:
        raise ValueError("frozen_training_manifest_invalid")
    if training.get("test_accessed") is not False or digest(checkpoint_path) != training.get("best_model_sha256"):
        raise ValueError("frozen_checkpoint_binding_invalid")
    configuration = dict(training["binding"]["configuration"])
    configuration.update({
        "method": args.method,
        "audio_features": args.audio_features,
        "genre_features": args.genre_features,
        "transition_cache": args.transition_cache,
        "ssd4rec_source": args.ssd4rec_source,
        "faithful_asset_manifest": (
            args.megan_asset_manifest if args.method == "megan" else args.faithful_asset_manifest
        ),
        "cosernn_embeddings": args.cosernn_embeddings,
        "megan_assets": args.megan_assets,
        "megan_line_manifest": args.megan_line_manifest,
        "megan_line_embeddings": args.megan_line_embeddings,
    })
    configuration["adapter_revision"] = training.get("binding", {}).get("adapter_revision", "original")
    backbone = configuration.get("deltagate_backbone")
    if args.method == "deltagate" and backbone:
        expected = training.get("binding", {}).get("inputs", {}).get("deltagate_backbone")
        if not expected or digest(Path(backbone)) != expected:
            raise ValueError("deltagate_backbone_binding_invalid")
    model_args = SimpleNamespace(**configuration)
    family, _ = adapter_identity(args.method, training)
    if family == "sasrec":
        model_args.baseline = "sasrec_standard"
        model = sasrec_runner.create_model(model_args, data)
    elif family == "legacy_car":
        model_args.variant = "car"
        model = confirmation_runner.create_model(model_args, data)
    else:
        model_args.method = args.method
        model = create_main_model(model_args, data)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("binding") != training.get("binding"):
        raise ValueError("frozen_checkpoint_manifest_binding_mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(args.device).eval(), training


@torch.no_grad()
def predict(
    args: argparse.Namespace,
    model,
    data: LFMTimeConcatData,
    row_ids: np.ndarray,
    family: str,
) -> np.ndarray:
    output = np.empty((len(row_ids), 10), dtype=np.int32)
    catalog_vectors = None
    if family == "main_table":
        catalog_vectors = model.item_vectors(torch.arange(model.vocab_size, device=args.device)).detach()
    for start in range(0, len(row_ids), args.batch_size):
        stop = min(start + args.batch_size, len(row_ids))
        ids = row_ids[start:stop]
        if family != "main_table":
            batch = data.serving_batch(
                [data.context(int(row)) for row in ids],
                [data.gap_days(int(row)) for row in ids],
                torch.device(args.device),
            )
            hidden = model.encode_all(batch["sequence"], batch["gap_days"])
            states = model.gather_states(hidden, batch["state_positions"])
            top = exact_catalog_topk(model, states, k=10)
        else:
            batch = data.serving_rows(ids, torch.device(args.device))
            batch["_catalog_vectors"] = catalog_vectors
            hidden = model.encode_batch(batch)
            states = model.gather_states(hidden, batch["initial_positions"])
            top = exact_topk(model, states, batch, k=10, chunk_size=4096)
        output[start:stop] = top.cpu().numpy()
        print(json.dumps({"event": "test_inference", "method": args.method, "rows": stop, "total": len(row_ids)}), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--audio-features", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1/musicnn_normalized.npy"))
    parser.add_argument("--genre-features", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1/genre_normalized.npz"))
    parser.add_argument("--transition-cache", type=Path, default=Path("data/music4all_onion/main_table_shared_v1/transition_topk.npz"))
    parser.add_argument("--ssd4rec-source", type=Path, default=Path("vendor/ssd4rec_official/ssd4rec.py"))
    parser.add_argument("--faithful-asset-manifest", type=Path, default=Path("data/music4all_onion/faithful_baseline_assets_v1/COSERNN_ASSETS.json"))
    parser.add_argument("--cosernn-embeddings", type=Path, default=Path("data/music4all_onion/faithful_baseline_assets_v1/cosernn_track_space.npy"))
    parser.add_argument("--megan-asset-manifest", type=Path, default=Path("data/music4all_onion/faithful_baseline_assets_v1/MEGAN_ASSETS.json"))
    parser.add_argument("--megan-assets", type=Path, default=Path("data/music4all_onion/faithful_baseline_assets_v1/megan_hmg_assets.npz"))
    parser.add_argument("--megan-line-manifest", type=Path, default=Path("data/music4all_onion/faithful_baseline_assets_v1/megan_line_v2/MEGAN_LINE.json"))
    parser.add_argument("--megan-line-embeddings", type=Path, default=Path("data/music4all_onion/faithful_baseline_assets_v1/megan_line_v2/megan_line_embeddings.npy"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("sealed_test_output_exists")
    torch.set_num_threads(4)
    test_manifest = validate_test(args)
    training = json.loads((args.checkpoint_dir / "training_manifest.json").read_text())
    family, method_variant = adapter_identity(args.method, training)
    revision = training.get("binding", {}).get("adapter_revision", "original")
    if family == "main_table":
        data = MainTableData(
            args.test_data,
            event_times=args.test_data / "event_times.npy",
            max_context=129,
            horizon=64,
            transition_cache=(
                args.transition_cache
                if args.method == "muse" and revision != "audit20260912"
                else None
            ),
        )
    else:
        data = LFMTimeConcatData(args.test_data, max_context=129, horizon=64)
    row_ids = data.row_ids(2, returns_only=True)
    labels = np.load(args.test_data / "metric_labels.npz", allow_pickle=False)
    if not np.array_equal(row_ids, labels["row_ids"]):
        raise ValueError("sealed_test_row_identity_mismatch")
    model, training = load_model(args, data, training)
    predictions = predict(args, model, data, row_ids, family)

    users = data.sessions[row_ids, 0].astype(np.int64)
    buckets = np.asarray([gap_bucket(data.gap_days(int(row))) for row in row_ids])
    recall = np.zeros(len(row_ids), dtype=np.float64)
    ndcg = np.zeros(len(row_ids), dtype=np.float64)
    seen_recall = np.zeros(len(row_ids), dtype=np.float64)
    seen_ndcg = np.zeros(len(row_ids), dtype=np.float64)
    seen_rows = np.zeros(len(row_ids), dtype=bool)
    for at in range(len(row_ids)):
        truth = labels["values"][labels["offsets"][at]:labels["offsets"][at + 1]].tolist()
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
        "target_coverage": test_manifest["counts"]["train_seen_target_coverage"],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    prediction_path = args.output / "test_predictions.npz"
    atomic_npz(
        prediction_path, row_ids=row_ids, users=users, gap_buckets=buckets,
        top10=predictions, recall=recall, ndcg=ndcg,
        seen_recall=seen_recall, seen_ndcg=seen_ndcg, seen_rows=seen_rows,
    )
    result = {
        "schema": "genrec.arrive_main_table.sealed_test_result.v1",
        "status": "pass",
        "method": args.method,
        "method_variant": method_variant,
        "seed": args.seed,
        "inference_only": True,
        "rows": len(row_ids),
        "metrics": metrics,
        "training_manifest_sha256": digest(args.checkpoint_dir / "training_manifest.json"),
        "checkpoint_sha256": digest(args.checkpoint_dir / "best_model.pt"),
        "selected_validation_epoch": training["best_epoch"],
        "test_manifest_sha256": digest(args.test_manifest),
        "predictions_sha256": digest(prediction_path),
        "test_accessed": True,
        "selection_or_tuning_performed": False,
    }
    atomic_json(args.output / "TEST_RESULT.json", result)
    print(json.dumps({"event": "sealed_test_complete", "method": args.method, "seed": args.seed, "metrics": metrics["all_targets"]}), flush=True)


if __name__ == "__main__":
    main()
