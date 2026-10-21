#!/usr/bin/env python3
"""Bind the one-shot Music4All Test materialization to the frozen Train data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


EXPECTED_BUCKETS = [5964, 1184, 623, 930]
EXPECTED_TARGETS = 25_673
EXPECTED_TRAIN_SEEN_TARGETS = 21_357
EXPECTED_FUTURE_UNSEEN_TARGETS = 4_316


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    train = json.loads(args.train_manifest.read_text())
    stats = json.loads((args.test_data / "TEST_STATS.json").read_text())
    if train.get("status") != "prepared" or train.get("test_accessed") is not False:
        raise ValueError("frozen_train_manifest_invalid")
    if digest(args.source) != train["source"]["sha256"]:
        raise ValueError("source_identity_mismatch")
    if stats.get("source_rows") != 252_984_396 or stats.get("users") != 119_140:
        raise ValueError("test_source_counts_mismatch")
    if stats.get("raw_tracks") != 56_512 or stats.get("train_catalog_tracks") != 49_393:
        raise ValueError("test_catalog_identity_mismatch")
    if stats.get("test_nonempty_episodes") != 8_701 or stats.get("test_gap_buckets") != EXPECTED_BUCKETS:
        raise ValueError("test_episode_counts_mismatch")

    rows = np.load(args.test_data / "metric_label_rows.npy", allow_pickle=False)
    values = np.load(args.test_data / "metric_label_values.npy", allow_pickle=False)
    offsets = np.load(args.test_data / "metric_label_offsets.npy", allow_pickle=False)
    sessions = np.load(args.test_data / "sessions.npy", mmap_mode="r")
    events = np.load(args.test_data / "events.npy", mmap_mode="r")
    event_times = np.load(args.test_data / "event_times.npy", mmap_mode="r")
    if len(rows) + 1 != len(offsets) or offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError("test_label_alignment_mismatch")
    if len(rows) != len(sessions) or not np.array_equal(rows, np.arange(len(rows))):
        raise ValueError("test_row_identity_mismatch")
    if not np.all(sessions[:, 5] == 2) or len(events) != len(event_times):
        raise ValueError("test_array_shape_or_split_mismatch")
    if len(values) != EXPECTED_TARGETS:
        raise ValueError("test_target_count_mismatch")
    if (
        int((values >= 3).sum()) != EXPECTED_TRAIN_SEEN_TARGETS
        or int((values < 3).sum()) != EXPECTED_FUTURE_UNSEEN_TARGETS
    ):
        raise ValueError("test_seen_unseen_target_counts_mismatch")

    np.savez_compressed(args.test_data / "metric_labels.npz", row_ids=rows, values=values, offsets=offsets)
    shutil.copyfile(args.train_data / "train_frequencies.npy", args.test_data / "train_frequencies.npy")
    shutil.copyfile(args.train_data / "track_identity.tsv", args.test_data / "track_identity.tsv")
    artifacts = {}
    for name in (
        "events.npy", "event_times.npy", "sessions.npy", "metric_labels.npz",
        "train_frequencies.npy", "track_identity.tsv", "TEST_STATS.json",
    ):
        path = args.test_data / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    manifest = {
        "schema": "genrec.music4all_onion.sealed_test.v1",
        "status": "pass",
        "protocol": {
            "test_start": "2019-01-01",
            "test_end": "2021-01-01",
            "observed_source_end": "2020-03-20",
            "minimum_hiatus_days": 30,
            "session_gap_minutes": 30,
            "history_events": 128,
            "observed_return_prefix_tracks": 1,
            "continuation_horizon_events": 64,
            "ambiguous_timestamp_ties": "excluded",
            "right_censoring": "last observed user session excluded",
        },
        "counts": {
            "rows": len(rows),
            "users": int(len(np.unique(sessions[:, 0]))),
            "targets": len(values),
            "train_seen_targets": int((values >= 3).sum()),
            "future_unseen_targets": int((values < 3).sum()),
            "train_seen_target_coverage": float((values >= 3).mean()),
            "gap_buckets": EXPECTED_BUCKETS,
        },
        "source_sha256": train["source"]["sha256"],
        "train_prepared_manifest_sha256": digest(args.train_manifest),
        "vocabulary_sha256": train["vocabulary_sha256"],
        "producer_sha256": digest(args.producer),
        "artifacts": artifacts,
        "test_accessed": True,
        "selection_or_tuning_permitted": False,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "counts": manifest["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
