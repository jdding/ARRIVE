#!/usr/bin/env python3
"""Replay fixed-budget coverage controls using supplied complete source lists."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import numpy as np
from export_io import digest, load_export, load_sources, require_fresh_output, write_json
from source_replay import replay_row
from metrics import user_macro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True, dest="export_root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = require_fresh_output(args.output)
    export = load_export(args.export_root)
    sources = load_sources(export)
    records = []
    for row, (lo, hi) in enumerate(zip(export.meta["offsets"][:-1], export.meta["offsets"][1:])):
        truth = export.meta["targets"][row, :int(export.meta["target_counts"][row])]
        records.append(replay_row(export.ids[lo:hi], export.features[lo:hi],
            export.meta["rrf_indices"][row], truth, sources[row]))
    arrays = {key: np.asarray([row[key] for row in records]) for key in records[0]}
    users = export.meta["users"]
    mask = np.ones(len(users), bool)
    report = {"schema": "arrive.reviewer_coverage.v1", "status": "pass",
        "dataset": export.manifest["dataset"], "seed": export.manifest["seed"],
        "surface": export.manifest["surface"], "synthetic": export.manifest["synthetic"],
        "rows": len(users), "paper_metric_reproduction": False, "new_training": False,
        "effect_scope": "candidate_coverage_not_final_top10_quality", "budget": 200,
        "protected_backbone_top10": True, "full_RRF_replay_exact": True,
        "complete_sources": "caller_declared_and_checked_against_observed_ranks_and_RRF200",
        "raw_source_reconstruction_performed": False,
        "full_coverage": user_macro(arrays["full_coverage"], users, mask),
        "removed_coverage": user_macro(arrays["removed_coverage"], users, mask),
        "full_hit_rate": user_macro((arrays["full_coverage"] > 0).astype(float), users, mask),
        "removed_hit_rate": user_macro((arrays["removed_coverage"] > 0).astype(float), users, mask),
        "changed_requests": int(arrays["changed"].sum()),
        "unique_relevant_total": int(arrays["unique_relevant"].sum()),
        "lost_relevant_total": int(arrays["lost_relevant"].sum()),
        "gained_relevant_total": int(arrays["gained_relevant"].sum()),
        "export_manifest_sha256": export.fingerprint}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "coverage.npz", row_ids=export.meta["row_ids"], users=users,
                        bucket_ids=export.meta["bucket_ids"], **arrays)
    report["coverage_sha256"] = digest(output / "coverage.npz")
    write_json(output / "coverage.json", report)
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
