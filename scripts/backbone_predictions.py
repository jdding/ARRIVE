#!/usr/bin/env python3
"""Evaluate the same frozen backbone Top-10 saved in complete source exports."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import numpy as np
from export_io import digest, load_export, load_sources, require_fresh_output, write_json
from metrics import row_metrics, user_macro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True, dest="export_root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = require_fresh_output(args.output)
    export = load_export(args.export_root)
    if export.manifest["surface"] != "test":
        raise ValueError("backbone_report_requires_test_export")
    sources = load_sources(export)
    top = np.asarray([row[0][:10] for row in sources], dtype=np.int32)
    recall, ndcg = row_metrics(top, export.meta["targets"], export.meta["target_counts"])
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "predictions.npz", **{key: export.meta[key] for key in (
        "row_ids", "users", "targets", "target_counts", "bucket_ids")}, top10=top, recall=recall, ndcg=ndcg)
    report = dict(schema="arrive.reviewer_backbone.v1", dataset=export.manifest["dataset"],
        seed=export.manifest["seed"], surface="test", synthetic=export.manifest["synthetic"],
        recall=user_macro(recall, export.meta["users"], np.ones(len(top), bool)),
        ndcg=user_macro(ndcg, export.meta["users"], np.ones(len(top), bool)),
        same_backbone_source=True, selection_or_tuning_performed=False,
        export_manifest_sha256=export.fingerprint, predictions_sha256=digest(output / "predictions.npz"),
        paper_metric_reproduction=False)
    write_json(output / "evaluation.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
