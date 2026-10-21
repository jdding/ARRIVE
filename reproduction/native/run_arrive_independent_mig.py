#!/usr/bin/env python3
"""Fit the unchanged MIG recipe on a seed-owned STC, then load for Validation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch  # Load this OpenMP runtime before LightGBM on macOS.
import lightgbm as lgb
from scipy import sparse

from run_music4all_candidate_support_a1 import atomic_json, atomic_npz, digest, stable_topk
from run_music4all_artist_component_b import load_artist_features
from run_music4all_component_d_joint_content_validation import fit_ranker, choose_alpha, blended_top10
from analyze_music4all_component_d_content_conversion import build_features, source_contexts, split_calibration
from analyze_music4all_component_d_content_validation import metric_table


def batched_values(model, matrices):
    """Same per-query normalization as MIG, one batched tree traversal."""
    sizes = [len(x) for x in matrices]
    flat = np.concatenate(matrices)
    values = model.booster_.predict(flat, num_threads=8).astype(np.float32)
    result = []
    at = 0
    for size in sizes:
        row = values[at:at + size]
        result.append((row - row.mean()) / max(float(row.std()), 1e-6))
        at += size
    return result


def ranking_inputs(args, surface, stc_result, audio, genre):
    path = args.stc_dir / f"{surface}_ranking.npz"
    if digest(path) != stc_result["ranking_sha256"][surface]:
        raise ValueError("stc_ranking_hash_mismatch")
    pred = dict(np.load(path, allow_pickle=False))
    raw, manifest = load_artist_features(
        args.rcce_dir / "artist_features" / f"{surface}_features.npz",
        args.rcce_dir / "artist_features" / f"{surface}_features_manifest.json", surface,
    )
    key = "train_feature_sha256" if surface == "train_cv" else "validation_feature_sha256"
    if manifest["feature_sha256"] != stc_result["binding"][key]:
        raise ValueError("rcce_stc_feature_binding_mismatch")
    for key in ("row_ids", "users", "bucket_ids", "target_counts"):
        if not np.array_equal(raw[key], pred[key]):
            raise ValueError(f"rcce_stc_row_mismatch:{key}")
    if not np.array_equal(stable_topk(pred["scores"], pred["candidates"]), pred["top10"]):
        raise ValueError("stc_top10_reconstruction_failed")
    sessions = np.load(args.data_dir / "sessions.npy", mmap_mode="r")
    if not np.all(sessions[pred["row_ids"], 5] == (0 if surface == "train_cv" else 1)):
        raise ValueError("ranking_split_mismatch")
    top = {"method_top10": pred["top10"]}
    contexts = source_contexts(pred, args.data_dir, audio, genre)
    pools, matrices, labels = build_features(pred, top, raw, audio, genre, contexts, np.arange(len(pred["row_ids"])), 50)
    print(json.dumps({"event": "mig_inputs", "surface": surface, "rows": len(pools), "seed": args.seed}), flush=True)
    return pools, matrices, labels, pred, top


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(2024, 2025, 2026))
    parser.add_argument("--rcce-dir", type=Path, required=True)
    parser.add_argument("--stc-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--content-dir", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("mig_output_exists")
    started = time.monotonic()
    stc_path = args.stc_dir / "VALIDATION_RESULT.json"
    stc = json.loads(stc_path.read_text())
    if (stc["status"] != "pass" or stc["seed"] != args.seed
            or stc["variant"] != "set_contextual" or stc["test_accessed"] is not False):
        raise ValueError("stc_not_admitted")
    if digest(args.stc_dir / "component_c2_checkpoint.pt") != stc["checkpoint_sha256"]:
        raise ValueError("stc_checkpoint_mismatch")
    audio = np.load(args.content_dir / "musicnn_normalized.npy", mmap_mode="r")
    genre = sparse.load_npz(args.content_dir / "genre_normalized.npz").tocsr()
    pools, matrices, labels, pred, top = ranking_inputs(args, "train_cv", stc, audio, genre)
    calibration = split_calibration(pred["users"], 99)
    selection_model = fit_ranker(matrices, labels, ~calibration, slice(None), 40001 + args.seed)
    # Selection uses only the existing Train-held-out recipe.
    selection, grid = choose_alpha(selection_model, pools, matrices, slice(None), top, pred, calibration)
    model = fit_ranker(matrices, labels, np.ones(len(matrices), bool), slice(None), 41001 + args.seed)
    args.output.mkdir(parents=True)
    model_path = args.output / "model.txt"
    temp = args.output / "model.txt.tmp"
    model.booster_.save_model(str(temp))
    temp.replace(model_path)
    restored = SimpleNamespace(booster_=lgb.Booster(model_file=str(model_path)))
    probe = np.concatenate(matrices[:8])
    parity = float(np.max(np.abs(model.booster_.predict(probe) - restored.booster_.predict(probe))))
    if parity != 0:
        raise ValueError("mig_reload_mismatch")
    manifest = {
        "schema": "arrive.independent_mig.v1", "status": "frozen_before_validation",
        "seed": args.seed, "selection": selection, "selection_grid": grid,
        "model_sha256": digest(model_path), "stc_result_sha256": digest(stc_path),
        "stc_checkpoint_sha256": stc["checkpoint_sha256"], "fitted_on": "train_cv_only",
        "fit_seed": 41001 + args.seed, "depth": 50, "feature_count": 17,
        "ranking_sha256": stc["ranking_sha256"], "test_accessed": False,
        "audio_sha256": digest(args.content_dir / "musicnn_normalized.npy"),
        "genre_sha256": digest(args.content_dir / "genre_normalized.npz"),
        "producer_sha256": digest(Path(__file__)), "model_reload_max_abs_error": parity,
        "feature_builder_sha256": digest(Path(__file__).with_name("analyze_music4all_component_d_content_conversion.py")),
        "fitter_sha256": digest(Path(__file__).with_name("run_music4all_component_d_joint_content_validation.py")),
        "lightgbm_version": lgb.__version__,
    }
    atomic_json(args.output / "MODEL_MANIFEST.json", manifest)
    print(json.dumps({"event": "mig_model_frozen", "seed": args.seed, "selection": selection}), flush=True)
    # Only after saving the selected predictor do we construct Validation inputs.
    pools, matrices, _, pred, top = ranking_inputs(args, "validation", stc, audio, genre)
    stored = np.load(args.stc_dir / "validation_predictions.npz", allow_pickle=False)
    if not np.array_equal(top["method_top10"], stored["method_top10"]):
        raise ValueError("validation_stc_cache_mismatch")
    values = batched_values(restored, matrices)
    for row in (0, len(matrices) - 1):
        raw = restored.booster_.predict(matrices[row], num_threads=8).astype(np.float32)
        normalized = (raw - raw.mean()) / max(float(raw.std()), 1e-6)
        if not np.array_equal(normalized, values[row]):
            raise ValueError("batched_mig_parity_failed")
    output = blended_top10(pools, matrices, values, np.ones(len(pools), bool), float(selection["alpha"]), top["method_top10"])
    metrics, arrays = metric_table(output, top["method_top10"], pred, 42100 + args.seed)
    path = args.output / "validation_predictions.npz"
    atomic_npz(path, row_ids=pred["row_ids"], users=pred["users"], bucket_ids=pred["bucket_ids"], top10=output, base_top10=top["method_top10"], targets=pred["targets"], target_counts=pred["target_counts"], **arrays)
    result = {
        "schema": "arrive.independent_mig_result.v1", "status": "pass", "seed": args.seed,
        "metrics": metrics, "selection": selection, "model_manifest_sha256": digest(args.output / "MODEL_MANIFEST.json"),
        "predictions_sha256": digest(path), "seconds": time.monotonic() - started,
        "test_accessed": False, "inference_uses_saved_model": True,
        "benefit_harm": {"rescued": int(((arrays["method_r"] > 0) & (arrays["base_r"] == 0)).sum()),
                         "displaced": int(((arrays["method_r"] == 0) & (arrays["base_r"] > 0)).sum())},
    }
    atomic_json(args.output / "VALIDATION_RESULT.json", result)
    print(json.dumps({"event": "mig_complete", "seed": args.seed, "seconds": result["seconds"], "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
