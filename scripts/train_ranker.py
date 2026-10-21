#!/usr/bin/env python3
"""Fit and Train-calibrate the native statistics31 ranker on supplied exports."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import numpy as np
from export_io import (digest, load_cue_treatment, load_export, require_fresh_output,
                       require_same_run, write_json)
from arrive_framework_datasets import feature_schema
from run_arrive_recent_ranker import ITERATIONS, LOSS, fit_pool, top10
from run_music4all_candidate_support_a1 import user_partition
from run_music4all_component_c2_conversion_oof import row_metrics, user_macro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-export", type=Path, required=True)
    parser.add_argument("--validation-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cue-match-mode", choices=("true", "none", "wrong"), default="true")
    parser.add_argument("--catalog", type=Path, help="Required for wrong: Train-only catalog assets and manifests.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Synthetic exports only; select among 5/10 trees.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    output = require_fresh_output(args.output)
    train = load_export(args.train_export, "train_cv")
    validation = load_export(args.validation_export, "validation")
    require_same_run(train, validation)
    if args.smoke and not train.manifest["synthetic"]:
        raise ValueError("smoke_requires_declared_synthetic_exports")
    if args.cue_match_mode == "wrong":
        if train.manifest.get("catalog_manifest_sha256") != validation.manifest.get("catalog_manifest_sha256"):
            raise ValueError("train_validation_catalog_mismatch")
        train_treatment = load_cue_treatment(train, args.catalog)
        validation_treatment = load_cue_treatment(validation, args.catalog)
    else:
        if args.catalog is not None:
            parser.error("--catalog is only used with --cue-match-mode wrong")
        train_treatment = validation_treatment = None
    partition = user_partition(train.meta["users"])
    positive = np.add.reduceat(np.asarray(train.labels, dtype=np.int32), train.meta["offsets"][:-1]) > 0
    fit_rows = np.flatnonzero(partition & positive)
    calibration_rows = np.flatnonzero(~partition)
    if not len(fit_rows) or not len(calibration_rows):
        raise ValueError("empty_fit_or_calibration")
    if set(train.meta["users"][fit_rows]) & set(train.meta["users"][calibration_rows]):
        raise ValueError("fit_calibration_user_overlap")
    import catboost
    from catboost import CatBoostRanker
    iterations = (5, 10) if args.smoke else ITERATIONS
    seed = train.manifest["seed"]
    model = CatBoostRanker(iterations=max(iterations), depth=6, learning_rate=0.035,
        loss_function=LOSS, random_seed=seed + 1000, thread_count=args.threads,
        task_type="CPU", verbose=False, allow_writing_files=False)
    model.fit(fit_pool(train.features, train.labels, train.meta, fit_rows, train.context,
        "statistics", args.threads, cue_match_mode=args.cue_match_mode,
        cue_treatment=train_treatment, candidate_ids=train.ids))
    calibration = []
    for trees in iterations:
        ranked = top10(model, train.features, train.ids, train.meta, calibration_rows, trees,
            train.context, "statistics", threads=args.threads,
            cue_match_mode=args.cue_match_mode, cue_treatment=train_treatment)
        _, ndcg = row_metrics(ranked, train.meta["targets"][calibration_rows],
                              train.meta["target_counts"][calibration_rows])
        score = user_macro(ndcg, train.meta["users"][calibration_rows], np.ones(len(calibration_rows), bool))
        calibration.append({"trees": int(trees), "train_calibration_rrf_ndcg": float(score)})
    selected = max(calibration, key=lambda record: (record["train_calibration_rrf_ndcg"], -record["trees"]))["trees"]
    ranked = top10(model, validation.features, validation.ids, validation.meta, validation.rows,
        selected, validation.context, "statistics", threads=args.threads,
        cue_match_mode=args.cue_match_mode, cue_treatment=validation_treatment)
    recall, ndcg = row_metrics(ranked, validation.meta["targets"], validation.meta["target_counts"])
    output.mkdir(parents=True, exist_ok=False)
    model.save_model(str(output / "yeti_ndcg.cbm"))
    np.savez_compressed(output / "validation_predictions.npz", row_ids=validation.meta["row_ids"],
        users=validation.meta["users"], targets=validation.meta["targets"],
        target_counts=validation.meta["target_counts"], bucket_ids=validation.meta["bucket_ids"],
        top10=ranked, recall=recall, ndcg=ndcg)
    report = {"schema": "arrive.reviewer_ranker.v1", "status": "pass", "dataset": train.manifest["dataset"],
        "seed": seed, "synthetic": train.manifest["synthetic"], "smoke": args.smoke,
        "paper_metric_reproduction": False, "scope": "supplied_export_ranker_only",
        "test_accessed": False, "feature_mode": "statistics", "feature_schema": feature_schema(),
        "feature_count": 31, "loss_function": LOSS, "depth": 6, "learning_rate": 0.035,
        "random_seed": seed + 1000, "threads": args.threads, "catboost_version": catboost.__version__,
        "tree_options": list(iterations), "selected_trees": selected, "calibration": calibration,
        "fit_rows": len(fit_rows), "calibration_rows": len(calibration_rows),
        "zero_positive_fit_partition_rows": int((partition & ~positive).sum()),
        "cue_match_mode": args.cue_match_mode, "catalog_manifest_sha256": train.manifest.get("catalog_manifest_sha256")
        if args.cue_match_mode == "wrong" else None,
        "wrong_cue_scope": "supplied_cues_and_catalog_checked_with_native_selector; raw_event_producer_omitted"
        if args.cue_match_mode == "wrong" else None,
        "train_export_sha256": train.fingerprint, "validation_export_sha256": validation.fingerprint,
        "model_sha256": digest(output / "yeti_ndcg.cbm"),
        "validation_predictions_sha256": digest(output / "validation_predictions.npz"),
        "validation_recall": user_macro(recall, validation.meta["users"], np.ones(len(recall), bool)),
        "validation_ndcg": user_macro(ndcg, validation.meta["users"], np.ones(len(ndcg), bool))}
    write_json(output / "training_manifest.json", report)
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
