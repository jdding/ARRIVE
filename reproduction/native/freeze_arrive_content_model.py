#!/usr/bin/env python3
"""Materialize the existing D recipe; no reselection, Test access, or new claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch  # Keep before LightGBM (macOS OpenMP runtime).
import lightgbm as lgb
from scipy import sparse

from evaluate_music4all_arrive_test import atomic_json, atomic_npz, digest
from run_music4all_component_d_joint_content_validation import (
    blended_top10, c2_scores, fit_ranker, ranker_values,
)
from analyze_music4all_component_d_content_conversion import build_features, source_contexts
from analyze_music4all_component_d_content_validation import packed_prediction
from analyze_music4all_component_d_selective_exchange import row_metrics, user_macro
from lfm_timeconcat_control import LFMTimeConcatData
from run_music4all_artist_component_b import load_artist_features
from run_music4all_candidate_support_a1 import stable_topk
from run_music4all_component_c_oof import context_states
from run_music4all_component_c2_validation import frozen_predecessor


def load_frozen_content(directory: Path, *, seed: int, decision: Path):
    manifest = json.loads((directory / "MODEL_MANIFEST.json").read_text())
    if manifest.get("schema") != "genrec.arrive.content_predictor.v1" or manifest.get("status") != "pass":
        raise ValueError("content_predictor_not_admitted")
    if manifest["seed"] != seed or manifest["decision_sha256"] != digest(decision):
        raise ValueError("content_predictor_selection_binding_mismatch")
    model_path = directory / "model.txt"
    if digest(model_path) != manifest["model_sha256"]:
        raise ValueError("content_predictor_hash_mismatch")
    return SimpleNamespace(booster_=lgb.Booster(model_file=str(model_path))), manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, choices=(2024, 2025, 2026), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    p.add_argument("--features-root", type=Path, default=Path("runs/music4all_artist_b/features"))
    p.add_argument("--content-dir", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1"))
    p.add_argument("--c2-root", type=Path, default=Path("refine-logs/music4all_onion/component_c2/validation_remote_evidence"))
    p.add_argument("--a1-root", type=Path, default=Path("refine-logs/music4all_onion/candidate_support_a1/remote_evidence"))
    p.add_argument("--artist-root", type=Path, default=Path("refine-logs/music4all_onion/component_b2/artist_b_remote_evidence"))
    p.add_argument("--base-checkpoint", type=Path, default=Path("runs/music4all_candidate_support_a1/crossfit_sasrec_s2024/final_model.pt"))
    p.add_argument("--base-training-manifest", type=Path, default=Path("runs/music4all_candidate_support_a1/crossfit_sasrec_s2024/training_manifest.json"))
    p.add_argument("--decision", type=Path, default=Path("refine-logs/music4all_onion/component_d/JOINT_CONTENT_VALIDATION_RESULT.json"))
    p.add_argument("--reference-rows", type=Path)
    p.add_argument("--depth", type=int, default=50, choices=(50,))
    p.add_argument("--max-context", type=int, default=129)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("content_model_output_exists")
    decision = json.loads(a.decision.read_text())
    if decision["decision"] != "JOINT_CONTENT_VALIDATION_PASS":
        raise ValueError("original_selection_not_frozen")
    selection = decision["selections"][f"{a.seed}:real_content"]["selected"]
    torch.set_num_threads(4)
    data = LFMTimeConcatData(a.data_dir, max_context=a.max_context, horizon=a.horizon)
    audio = np.load(a.content_dir / "musicnn_normalized.npy", mmap_mode="r")
    genre = sparse.load_npz(a.content_dir / "genre_normalized.npz").tocsr()
    c2 = a.c2_root / f"validation_set_contextual_s{a.seed}_formal/component_c2_checkpoint.pt"
    parts, inputs, predecessors = {}, {}, {}
    for split in ("train_cv", "validation"):
        path = a.features_root / f"{split}_features.npz"
        receipt = a.features_root / f"{split}_features_manifest.json"
        f, _ = load_artist_features(path, receipt, split)
        if not np.all(data.sessions[f["row_ids"], 5] == (0 if split == "train_cv" else 1)):
            raise ValueError("feature_split_mismatch")
        states, item_array, _ = context_states(a, data, f["row_ids"])
        candidates, scores, _, _, _, binding = frozen_predecessor(a, f, split)
        scores = c2_scores(c2, candidates, scores, states, f["gaps"], torch.as_tensor(item_array))
        pred = packed_prediction(f, candidates, scores)
        top = {"method_top10": stable_topk(scores, candidates)}
        if split == "validation":
            stored = np.load(c2.parent / "validation_predictions.npz", allow_pickle=False)
            # Preserve exactly the original D validation base fallback.
            top = {"method_top10": stored["method_top10"]}
        contexts = source_contexts(pred, a.data_dir, audio, genre)
        pools, matrices, labels = build_features(pred, top, f, audio, genre, contexts, np.arange(len(f["row_ids"])), a.depth)
        parts[split] = (pools, matrices, labels, pred, top)
        inputs[split] = {"features": digest(path), "manifest": digest(receipt)}
        predecessors[split] = binding
        print(json.dumps({"event": "content_inputs", "seed": a.seed, "split": split, "rows": len(matrices)}), flush=True)
    _, train_x, train_y, _, _ = parts["train_cv"]
    pools, val_x, _, pred, top = parts["validation"]
    model = fit_ranker(train_x, train_y, np.ones(len(train_x), bool), slice(None), 41001 + a.seed)
    a.output.mkdir(parents=True, exist_ok=False)
    model_path = a.output / "model.txt"
    model.booster_.save_model(str(model_path))
    restored = SimpleNamespace(booster_=lgb.Booster(model_file=str(model_path)))
    x = np.concatenate(val_x)
    parity = float(np.max(np.abs(model.booster_.predict(x, num_threads=4) - restored.booster_.predict(x, num_threads=4))))
    if parity != 0:
        raise ValueError("booster_reload_prediction_mismatch")
    mask = np.ones(len(val_x), bool)
    values = ranker_values(restored, val_x, mask, slice(None))
    output = blended_top10(pools, val_x, values, mask, float(selection["alpha"]), top["method_top10"])
    r, n = row_metrics(output, pred["targets"], pred["target_counts"])
    differences = {}
    for scope, selected in (("overall", mask), (">365d", pred["bucket_ids"] == 3)):
        old = decision["seeds"][str(a.seed)]["real_content"]["metrics"][scope]
        for metric, array in (("recall", r), ("ndcg", n)):
            differences[f"{scope}:{metric}"] = user_macro(array, pred["users"], selected) - old[f"method_{metric}_at_10"]
    row_parity = None
    if a.reference_rows is not None:
        ref = np.load(a.reference_rows, allow_pickle=False)
        if not np.array_equal(pred["users"], ref["users"]) or not np.array_equal(pred["bucket_ids"], ref["bucket_ids"]):
            raise ValueError("reference_row_identity_mismatch")
        row_parity = max(float(np.max(np.abs(r-ref[f"recall_s{a.seed}"]))), float(np.max(np.abs(n-ref[f"ndcg_s{a.seed}"]))))
    passed = max(map(abs, differences.values())) < 1e-12 and row_parity in (None, 0.0)
    atomic_npz(a.output / "validation_predictions.npz", row_ids=pred["row_ids"], top10=output, recall=r, ndcg=n)
    manifest = {
        "schema": "genrec.arrive.content_predictor.v1", "status": "pass" if passed else "validation_reproduction_mismatch",
        "seed": a.seed, "fitted_on": "train_cv_only", "fit_seed": 41001+a.seed,
        "model_sha256": digest(model_path), "decision_sha256": digest(a.decision),
        "selection": selection, "depth": a.depth, "feature_count": int(x.shape[1]),
        "max_context": a.max_context, "horizon": a.horizon,
        "feature_schema": "arrive_d_joint_content_17_columns_v1",
        "feature_builder_sha256": digest(Path(__file__).with_name("analyze_music4all_component_d_content_conversion.py")),
        "inputs": inputs, "predecessor_bindings": predecessors,
        "c2_checkpoint_sha256": digest(c2), "base_checkpoint_sha256": digest(a.base_checkpoint),
        "audio_sha256": digest(a.content_dir / "musicnn_normalized.npy"), "genre_sha256": digest(a.content_dir / "genre_normalized.npz"),
        "booster_reload_max_abs_error": parity, "old_validation_metric_differences": differences,
        "old_validation_row_metric_max_error": row_parity,
        "lightgbm_version": lgb.__version__, "test_accessed": False,
        "created_after_original_test": True, "retroactive_pretest_freeze_claim": False,
    }
    atomic_json(a.output / "MODEL_MANIFEST.json", manifest)
    print(json.dumps(manifest), flush=True)
    if not passed:
        raise ValueError("original_validation_not_reproduced_do_not_promote")


if __name__ == "__main__":
    main()
