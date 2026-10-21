#!/usr/bin/env python3
"""Admit rebuilt event timestamps only after exact prepared-array parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--rebuilt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "events.npy", "sessions.npy", "train_frequencies.npy", "metric_label_rows.npy",
        "metric_label_values.npy", "metric_label_offsets.npy", "track_identity.tsv",
    ):
        if sha(args.frozen / name) != sha(args.rebuilt / name):
            raise ValueError(f"rebuilt_prepared_parity_mismatch:{name}")
    events = np.load(args.frozen / "events.npy", mmap_mode="r")
    times = np.load(args.rebuilt / "event_times.npy", mmap_mode="r")
    if events.shape != times.shape or times.dtype != np.dtype("uint32"):
        raise ValueError("event_time_shape_or_dtype_mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "event_times.npy"
    shutil.copyfile(args.rebuilt / "event_times.npy", destination)
    manifest = {
        "schema": "genrec.music4all_onion.event_times.v1",
        "status": "pass",
        "rows": int(len(times)),
        "dtype": "uint32_unix_seconds",
        "prepared_events_sha256": sha(args.frozen / "events.npy"),
        "event_times_sha256": sha(destination),
        "parity_artifacts": [
            "events.npy", "sessions.npy", "train_frequencies.npy", "metric_label_rows.npy",
            "metric_label_values.npy", "metric_label_offsets.npy", "track_identity.tsv",
        ],
        "test_accessed": False,
    }
    temporary = args.output / ".event_times_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output / "event_times_manifest.json")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
