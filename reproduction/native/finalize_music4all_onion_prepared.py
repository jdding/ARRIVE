#!/usr/bin/env python3
"""Finalize the compact Music4All Train/Validation arrays and evidence manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels-manifest", type=Path, required=True)
    args = parser.parse_args()

    stats = json.loads((args.data / "PREP_STATS.json").read_text())
    if stats["source_rows"] != 252_984_396 or stats["users"] != 119_140:
        raise ValueError("source_identity_mismatch")
    if stats["raw_tracks"] != 56_512 or stats["invalid_timestamp_rows"] != 1:
        raise ValueError("track_or_timestamp_identity_mismatch")
    if stats["test_materialized"] is not False:
        raise ValueError("test_must_remain_unmaterialized")
    if digest(args.source, "md5") != "dfe82201036765f7463e6f3ce3d0f991":
        raise ValueError("source_md5_mismatch")

    rows = np.load(args.data / "metric_label_rows.npy", allow_pickle=False)
    values = np.load(args.data / "metric_label_values.npy", allow_pickle=False)
    offsets = np.load(args.data / "metric_label_offsets.npy", allow_pickle=False)
    if len(rows) + 1 != len(offsets) or offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError("metric_label_alignment_mismatch")
    np.savez_compressed(args.data / "metric_labels.npz", row_ids=rows, values=values, offsets=offsets)

    artifact_names = [
        "events.npy",
        "sessions.npy",
        "train_frequencies.npy",
        "metric_labels.npz",
        "track_identity.tsv",
        "PREP_STATS.json",
    ]
    manifest = {
        "schema": "genrec.music4all_onion.prepared.v1",
        "status": "prepared",
        "source": {
            "bytes": args.source.stat().st_size,
            "md5": digest(args.source, "md5"),
            "sha256": digest(args.source),
        },
        "producer_sha256": digest(args.producer),
        "vocabulary_sha256": digest(args.data / "track_identity.tsv"),
        "protocol": {
            "train": "before 2018-01-01",
            "validation": "calendar year 2018",
            "test": "from 2019-01-01; not materialized",
            "minimum_hiatus_days": 30,
            "session_gap_minutes": 30,
            "history_events": 128,
            "observed_return_prefix_tracks": 1,
            "continuation_horizon_events": 64,
            "ambiguous_timestamp_ties": "excluded",
        },
        "counts": stats,
        "artifacts": {
            name: {
                "bytes": (args.data / name).stat().st_size,
                "sha256": digest(args.data / name),
            }
            for name in artifact_names
        },
        "test_accessed": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    args.labels_manifest.write_text(
        json.dumps(
            {
                "schema": "genrec.music4all_onion.metric_labels.v1",
                "sha256": digest(args.data / "metric_labels.npz"),
                "rows": int(len(rows)),
                "values": int(len(values)),
                "split": "validation",
                "test_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"status": "prepared", "counts": stats}, sort_keys=True))


if __name__ == "__main__":
    main()
