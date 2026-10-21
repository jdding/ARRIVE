"""ILMART backend for the shared ARRIVE ranker runner.

This uses the authors' Ilmart class and modified LightGBM in an isolated
Python environment. Candidate exports, Train/calibration users, and evaluator
are identical to the completed LambdaMART/YetiRank comparison.
"""
from __future__ import annotations

import json
import hashlib
import inspect
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from ilmart import Ilmart


ILMART_SOURCE_COMMIT = "a7a589c0026c18461b993cbb5772a8b02c7b3d54"
LIGHTGBM_FORK_COMMIT = "07280fff07b9690715b74fa0bd14dee8c5a6d683"
ILMART_MODULE_SHA256 = "c8fd9618c185d7eee8d6408d63f3a562634285d1361b6b17f664d16eda2ec4c8"
LIGHTGBM_ENGINE_SHA256 = "27a2b9f260d7f6e7bb3e3f6482e86a7b69c9e5628da8ba5f348808035c904564"
MAIN_EFFECTS = 28
MAX_INTERACTIONS = 50


def source_provenance() -> dict:
    """Bind imported author code and the installed fork to pinned sources."""
    author_path = Path(inspect.getfile(Ilmart)).resolve()
    engine_path = Path(lgb.engine.__file__).resolve()
    library_path = Path(lgb.basic._LIB._name).resolve()
    author_hash = hashlib.sha256(author_path.read_bytes()).hexdigest()
    engine_hash = hashlib.sha256(engine_path.read_bytes()).hexdigest()
    if (lgb.__version__ != "4.3.0.99" or author_hash != ILMART_MODULE_SHA256
            or engine_hash != LIGHTGBM_ENGINE_SHA256 or not library_path.is_file()):
        raise ValueError("ilmart_author_or_fork_source_mismatch")
    return {"author_module_path": str(author_path), "author_module_sha256": author_hash,
            "lightgbm_engine_path": str(engine_path), "lightgbm_engine_sha256": engine_hash,
            "lightgbm_library_path": str(library_path),
            "lightgbm_library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest()}


@dataclass(frozen=True)
class GroupedView:
    """Only the three fields consumed by the author's Ilmart.fit methods."""

    X: np.ndarray
    y: np.ndarray
    sizes: np.ndarray

    def get_query_sizes(self) -> np.ndarray:
        return self.sizes


def _split_features(node: dict) -> set[int]:
    if "split_feature" not in node:
        return set()
    return ({int(node["split_feature"])} |
            _split_features(node["left_child"]) |
            _split_features(node["right_child"]))


def audit_structure(main: lgb.Booster, full: lgb.Booster) -> dict:
    """Reject a silently substituted stock LightGBM implementation."""
    main_trees = main.dump_model()["tree_info"]
    full_trees = full.dump_model()["tree_info"]
    if len(full_trees) <= len(main_trees):
        raise ValueError("ilmart_interaction_stage_missing")
    main_sets = [_split_features(t["tree_structure"]) for t in main_trees]
    if any(len(features) > 1 for features in main_sets):
        raise ValueError("ilmart_main_effect_constraint_broken")
    main_used = set().union(*main_sets)
    interaction_sets = [_split_features(t["tree_structure"]) for t in full_trees[len(main_trees):]]
    if any(len(features) > 2 or not features.issubset(main_used)
           for features in interaction_sets):
        raise ValueError("ilmart_interaction_constraint_broken")
    pairs = {tuple(sorted(features)) for features in interaction_sets if len(features) == 2}
    if not pairs:
        raise ValueError("ilmart_no_two_feature_interactions")
    if len(pairs) > MAX_INTERACTIONS:
        raise ValueError("ilmart_pair_budget_broken")
    return {"main_trees": len(main_trees), "interaction_trees": len(interaction_sets),
            "main_features_used": len(main_used), "interaction_pairs_used": len(pairs)}


def fit_ilmart(fit: GroupedView, calibration: GroupedView, seed: int,
               total_trees: int, leaves: int, threads: int) -> tuple[lgb.Booster, lgb.Booster, dict]:
    if total_trees not in (220, 440) or leaves not in (31, 63):
        raise ValueError("unfrozen_reference_capacity")
    if fit.X.shape[1] != MAIN_EFFECTS or calibration.X.shape[1] != MAIN_EFFECTS:
        raise ValueError("feature_contract_mismatch")
    if (fit.X.shape[0] != len(fit.y) or fit.X.shape[0] != int(fit.sizes.sum())
            or calibration.X.shape[0] != int(calibration.sizes.sum())
            or not np.isin(fit.y, (0, 1)).all()):
        raise ValueError("group_or_label_contract_mismatch")
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
              "label_gain": [0, 1], "lambdarank_truncation_level": 13,
              "learning_rate": 0.035, "num_leaves": leaves,
              "min_data_in_leaf": 80, "bagging_fraction": 0.8,
              "bagging_freq": 1, "feature_fraction": 0.9, "lambda_l2": 1.0,
              "seed": seed + 1000, "num_threads": threads,
              "deterministic": True, "force_col_wise": True, "verbosity": -1}
    model = Ilmart(params, verbose=False, main_strategy="greedy", inter_strategy="greedy")
    stage_rounds = total_trees // 2
    began = time.monotonic()
    model.fit_main_effects(stage_rounds, fit, calibration, n_main_effects=MAIN_EFFECTS,
                           early_stopping_rounds=min(100, stage_rounds - 1))
    main = model.get_model()
    model.fit_inter_effects(stage_rounds, fit, calibration, num_interactions=MAX_INTERACTIONS,
                            early_stopping_rounds=min(100, stage_rounds - 1))
    full = model.get_model()
    structure = audit_structure(main, full)
    structure.update({"fit_seconds": time.monotonic() - began,
                      "reference_total_trees": total_trees, "reference_leaves": leaves,
                      "main_best_iteration": int(main.best_iteration),
                      "full_best_iteration": int(full.best_iteration)})
    return main, full, structure


def smoke() -> None:
    """Local author-code smoke: grouped fit, validation, structure, and reload."""
    from tempfile import TemporaryDirectory

    provenance = source_provenance()
    rng = np.random.default_rng(19)
    x = rng.normal(size=(20 * 12, MAIN_EFFECTS)).astype(np.float32)
    y = np.zeros(len(x), dtype=np.int32)
    y[np.arange(0, len(x), 12)] = 1
    fit = GroupedView(x[:144], y[:144], np.full(12, 12, dtype=np.int32))
    cal = GroupedView(x[144:], y[144:], np.full(8, 12, dtype=np.int32))
    # A small two-stage author fit, with the same constraints as the formal arm.
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
              "label_gain": [0, 1], "num_leaves": 7, "min_data_in_leaf": 5,
              "learning_rate": 0.1, "num_threads": 2, "seed": 2024,
              "deterministic": True, "force_col_wise": True, "verbosity": -1}
    model = Ilmart(params, verbose=False, main_strategy="greedy", inter_strategy="greedy")
    model.fit_main_effects(8, fit, cal, n_main_effects=MAIN_EFFECTS, early_stopping_rounds=7)
    main = model.get_model()
    model.fit_inter_effects(8, fit, cal, num_interactions=MAX_INTERACTIONS,
                            early_stopping_rounds=7)
    full = model.get_model()
    structure = audit_structure(main, full)
    scores = np.asarray(full.predict(cal.X, num_threads=2))
    if not np.isfinite(scores).all() or len(scores) != len(cal.y):
        raise ValueError("ilmart_smoke_prediction_invalid")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "ilmart.lgbm"
        full.save_model(str(path))
        reloaded = lgb.Booster(model_file=str(path))
        if not np.allclose(scores, reloaded.predict(cal.X, num_threads=2), rtol=0, atol=1e-12):
            raise ValueError("ilmart_smoke_reload_mismatch")
    print(json.dumps({"event": "ilmart_local_smoke_pass", "structure": structure,
                      "lightgbm_version": lgb.__version__,
                      "author_module_sha256": provenance["author_module_sha256"],
                      "lightgbm_engine_sha256": provenance["lightgbm_engine_sha256"]}), flush=True)


def train(args) -> None:
    """Fit one seed on the frozen Onion export, without opening Test."""
    from run_arrive_full_rcce_control import load_export
    from run_music4all_candidate_support_a1 import (atomic_json, atomic_npz,
                                                      digest, stable_topk, user_partition)
    from run_music4all_component_c2_conversion_oof import row_metrics, user_macro

    provenance = source_provenance()
    if args.seed not in (2024, 2025, 2026) or args.surface != "validation":
        raise ValueError("only_frozen_onion_validation_supported")
    destination = args.output / f"s{args.seed}"
    if destination.exists():
        raise FileExistsError(f"existing_ranker_output:{destination}")
    parent = args.project / "runs/arrive_full_rcce_control_20260924/onion" / f"s{args.seed}"
    x, _, y, train_meta, train_manifest = load_export(parent / "train_cv", args.seed, "train_cv")
    vx, vids, _, val_meta, val_manifest = load_export(parent / "validation", args.seed, "validation")
    if train_manifest.get("limit") != 0 or val_manifest.get("limit") != 0:
        raise ValueError("limited_parent_export_forbidden")
    reference_path = parent / "training_manifest.json"
    reference = json.loads(reference_path.read_text())
    if (reference.get("status") != "pass" or reference.get("seed") != args.seed
            or reference.get("test_accessed") is not False
            or reference.get("train_export_sha256") != digest(parent / "train_cv/EXPORT_MANIFEST.json")
            or reference.get("validation_export_sha256") != digest(parent / "validation/EXPORT_MANIFEST.json")):
        raise ValueError("reference_manifest_binding_invalid")
    fit_partition = user_partition(train_meta["users"])
    positive = np.add.reduceat(np.asarray(y, dtype=np.int32), train_meta["offsets"][:-1]) > 0
    fit_rows = np.flatnonzero(fit_partition & positive)
    calibration_rows = np.flatnonzero(~fit_partition)
    if args.canary_rows:
        fit_rows = fit_rows[:args.canary_rows]
        calibration_rows = calibration_rows[:min(args.canary_rows, len(calibration_rows))]
    if (not len(fit_rows) or not len(calibration_rows)
            or set(train_meta["users"][fit_rows].tolist()) &
            set(train_meta["users"][calibration_rows].tolist())):
        raise ValueError("fit_calibration_user_contract_invalid")

    offsets = train_meta["offsets"]
    def view(rows: np.ndarray) -> GroupedView:
        positions = np.concatenate([np.arange(int(offsets[r]), int(offsets[r + 1]),
                                               dtype=np.int64) for r in rows])
        return GroupedView(np.asarray(x[positions], dtype=np.float32),
                           np.asarray(y[positions], dtype=np.int32),
                           np.diff(offsets)[rows].astype(np.int32))

    fit, calibration = view(fit_rows), view(calibration_rows)
    capacity = reference["selected_config"]
    main_model, model, structure = fit_ilmart(fit, calibration, args.seed,
                                              int(capacity["trees"]), int(capacity["leaves"]), args.threads)
    del fit, calibration

    destination.mkdir(parents=True)
    model_path = destination / "ilmart_model.txt"
    # Default serialization follows LightGBM's best_iteration, as does
    # Booster.predict() after early stopping. Score from this saved artifact.
    model.save_model(str(model_path))
    reloaded = lgb.Booster(model_file=str(model_path))
    saved_structure = audit_structure(main_model, reloaded)
    structure.update({"saved_main_trees": saved_structure["main_trees"],
                      "saved_interaction_trees": saved_structure["interaction_trees"],
                      "saved_interaction_pairs_used": saved_structure["interaction_pairs_used"]})

    rows = (np.arange(min(args.canary_rows, len(val_meta["row_ids"]))) if args.canary_rows
            else np.arange(len(val_meta["row_ids"])))
    source = val_meta["rrf_indices"][rows]
    flat = (val_meta["offsets"][rows, None] + source).reshape(-1)
    began = time.monotonic()
    candidate_ids = np.asarray(vids[flat], dtype=np.int32).reshape(len(rows), -1)
    scores = np.asarray(reloaded.predict(np.asarray(vx[flat]), num_threads=args.threads),
                        dtype=np.float64).reshape(len(rows), -1)
    ranked = stable_topk(scores, candidate_ids, 10)
    score_seconds = time.monotonic() - began
    recall, ndcg = row_metrics(ranked, val_meta["targets"][rows],
                               val_meta["target_counts"][rows])
    probe = np.asarray(vx[flat[:min(len(flat), 256)]])
    if not np.allclose(model.predict(probe, num_threads=args.threads),
                       reloaded.predict(probe, num_threads=args.threads), rtol=0, atol=1e-10):
        raise ValueError("ilmart_saved_model_prediction_mismatch")
    atomic_npz(destination / "validation_predictions.npz",
               row_ids=val_meta["row_ids"][rows], users=val_meta["users"][rows],
               bucket_ids=val_meta["bucket_ids"][rows],
               target_counts=val_meta["target_counts"][rows],
               targets=val_meta["targets"][rows], top10=ranked,
               recall=recall, ndcg=ndcg)
    atomic_json(destination / "training_manifest.json", {
        "schema": "arrive.recent_ranker.ilmart.v1",
        "status": "canary" if args.canary_rows else "pass",
        "method": "ILMART_author_greedy_main_plus_interactions",
        "seed": args.seed, "test_accessed": False,
        "fit_rows": len(fit_rows), "calibration_rows": len(calibration_rows),
        "validation_rows": len(rows), "fit_seconds": structure["fit_seconds"],
        "score_seconds": score_seconds, "structure": structure,
        "main_effect_budget": MAIN_EFFECTS, "interaction_budget": MAX_INTERACTIONS,
        "reference_capacity": {"trees": int(capacity["trees"]),
                               "leaves": int(capacity["leaves"])},
        "lightgbm_version": lgb.__version__,
        "ilmart_source_commit": ILMART_SOURCE_COMMIT,
        "lightgbm_fork_commit": LIGHTGBM_FORK_COMMIT,
        "imported_source": provenance,
        "model_bytes": model_path.stat().st_size,
        "train_export_sha256": digest(parent / "train_cv/EXPORT_MANIFEST.json"),
        "validation_export_sha256": digest(parent / "validation/EXPORT_MANIFEST.json"),
        "reference_manifest_sha256": digest(reference_path),
        "model_sha256": digest(model_path),
        "prediction_sha256": digest(destination / "validation_predictions.npz"),
        "overall_recall": user_macro(recall, val_meta["users"][rows],
                                      np.ones(len(rows), bool)),
        "overall_ndcg": user_macro(ndcg, val_meta["users"][rows],
                                    np.ones(len(rows), bool)),
        "producer_sha256": digest(Path(__file__)),
    })
    print(json.dumps({"event": "ilmart_ranker_complete", "seed": args.seed,
                      "fit_seconds": structure["fit_seconds"],
                      "score_seconds": score_seconds}), flush=True)
