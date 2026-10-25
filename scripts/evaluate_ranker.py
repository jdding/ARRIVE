#!/usr/bin/env python3
"""Evaluate a saved reviewer ranker on a supplied Validation or Test export."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import numpy as np
from feature_schema import feature_schema
from export_io import (digest, load_cue_treatment, load_export, read_json,
                       require_fresh_output, write_json)
from ranking import LOSS, top10
from metrics import row_metrics, user_macro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True, dest="export_root")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, help="Required when the saved model uses wrong-cue matching.")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    output = require_fresh_output(args.output)
    export = load_export(args.export_root)
    if export.manifest["surface"] not in ("validation", "test"):
        raise ValueError("evaluation_requires_validation_or_test_export")
    manifest_path = args.model_dir / "training_manifest.json"
    manifest = read_json(manifest_path)
    model_path = args.model_dir / "yeti_ndcg.cbm"
    if (manifest.get("schema") != "arrive.reviewer_ranker.v1" or manifest.get("status") != "pass"
            or manifest.get("feature_mode") != "statistics" or manifest.get("feature_count") != 31
            or manifest.get("feature_schema") != feature_schema() or manifest.get("loss_function") != LOSS
            or manifest.get("cue_match_mode") not in ("true", "none", "wrong")
            or type(manifest.get("selected_trees")) is not int or manifest["selected_trees"] <= 0
            or type(manifest.get("smoke")) is not bool
            or manifest.get("tree_options") != ([5, 10] if manifest.get("smoke") else [220, 440])
            or manifest["selected_trees"] not in manifest.get("tree_options", [])
            or manifest.get("random_seed") != export.manifest["seed"] + 1000
            or manifest.get("test_accessed") is not False
            or manifest.get("depth") != 6 or manifest.get("learning_rate") != 0.035
            or digest(model_path) != manifest.get("model_sha256")
            or any(manifest.get(key) != export.manifest[key] for key in ("dataset", "seed", "synthetic"))):
        raise ValueError("saved_ranker_contract_or_identity")
    mode = manifest["cue_match_mode"]
    if mode == "wrong":
        if manifest.get("catalog_manifest_sha256") != export.manifest.get("catalog_manifest_sha256"):
            raise ValueError("saved_ranker_catalog_mismatch")
        treatment = load_cue_treatment(export, args.catalog)
    else:
        if args.catalog is not None:
            parser.error("--catalog is only used for a saved wrong-cue model")
        treatment = None
    from catboost import CatBoostRanker
    model = CatBoostRanker()
    model.load_model(str(model_path))
    selected = manifest["selected_trees"]
    if selected > model.tree_count_ or len(model.feature_names_) != 31:
        raise ValueError("saved_model_tree_or_feature_count")
    ranked = top10(model, export.features, export.ids, export.meta, export.rows, selected,
        export.context, "statistics", threads=args.threads, cue_match_mode=mode, cue_treatment=treatment)
    recall, ndcg = row_metrics(ranked, export.meta["targets"], export.meta["target_counts"])
    report = {"schema": "arrive.reviewer_evaluation.v1", "status": "pass",
        "dataset": export.manifest["dataset"], "seed": export.manifest["seed"],
        "surface": export.manifest["surface"], "synthetic": export.manifest["synthetic"],
        "paper_metric_reproduction": False, "scope": "supplied_export_ranker_only",
        "selection_or_tuning_performed": False, "new_untouched_test_claim": False,
        "selected_trees": selected, "cue_match_mode": mode, "rows": len(ranked),
        "recall": user_macro(recall, export.meta["users"], np.ones(len(recall), bool)),
        "ndcg": user_macro(ndcg, export.meta["users"], np.ones(len(ndcg), bool)),
        "export_manifest_sha256": export.fingerprint, "model_sha256": digest(model_path),
        "training_manifest_sha256": digest(manifest_path)}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "predictions.npz", row_ids=export.meta["row_ids"], users=export.meta["users"],
        targets=export.meta["targets"], target_counts=export.meta["target_counts"],
        bucket_ids=export.meta["bucket_ids"], top10=ranked, recall=recall, ndcg=ndcg)
    report["predictions_sha256"] = digest(output / "predictions.npz")
    write_json(output / "evaluation.json", report)
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
