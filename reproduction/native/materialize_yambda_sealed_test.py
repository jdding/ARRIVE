#!/usr/bin/env python3
"""Materialize the Yambda-500M sealed test split (episodes only, no training).

Mirrors the episode predicate of tools/prepare_arrive_yambda_500m.py EXACTLY
(same dedup keys, session gap, hiatus, boundary fit at AXIS_MAXIMUM_UNITS,
tie exclusion, horizon truncation, history window, interning) but emits only
the test split (return-session start >= validation_end_units) into a
standalone sealed_test directory:

  events.npy / event_times.npy   int32 / uint32, context+x0+targets layout
  sessions.npy                    int64 (N, 6) with split column == 2
  metric_labels.npz               test targets (distinct, first-appearance,
                                   known id or negative raw id)
  train_frequencies.npy / track_identity.tsv   copied verbatim from the
                                   prepared suite
  SEALED_TEST_MANIFEST.json       digest-bound, test_accessed=true

Fail-closed gates:
- refuse an existing output directory
- prepared suite must be finalized (PREP_MANIFEST.json status=finalized)
- source parquet sha256 must equal the registered digest
- the recomputed test census must equal the registered corrected-protocol
  census EXACTLY, else abort
- emitted row count must equal the census

This is the one-time test opening for the Yambda suite. Run it ONCE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc

PAD, UNK, EOS = 0, 1, 2
FIRST_ITEM = 3
# Stored Yambda timestamps are elapsed seconds rounded down to 5-second
# precision, not integer ticks whose duration is five seconds.
AXIS_UNIT_SECONDS = 1
AXIS_MAXIMUM_UNITS = 26_000_000
DEFAULT_TRAIN_END_UNITS = 19_500_000
DEFAULT_VALIDATION_END_UNITS = 22_100_000
SESSION_GAP_UNITS = 1800 // AXIS_UNIT_SECONDS
MINIMUM_HIATUS_UNITS = 30 * 86_400 // AXIS_UNIT_SECONDS
HISTORY_EVENTS = 128
HORIZON_EVENTS = 64

REGISTERED_SOURCE_SHA256 = (
    "bf30497a2b11a50f568cb0bb3efb9ca23786c76b82de4170025c1a9614e1b9cd"
)
# Registered from the v2 preparation census before corrected Test opening.
REGISTERED_TEST_CENSUS = {
    "episodes": 8_213,
    "buckets": [5_906, 1_599, 708, 0],
}


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def bucket_index(gap_units: np.ndarray) -> np.ndarray:
    days = gap_units * (AXIS_UNIT_SECONDS / 86400.0)
    return np.where(days < 90, 0, np.where(days < 180, 1, np.where(days < 365, 2, 3)))


def stream_organic_listen_plus(path: Path):
    dset = ds.dataset(path, format="parquet")
    scanner = dset.scanner(
        columns=["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct"],
        filter=(pc.field("event_type") == "listen"),
        batch_size=5_000_000,
    )
    uid_parts, ts_parts, item_parts = [], [], []
    listen_rows = 0
    invalid_rows = 0
    for batch in scanner.to_batches():
        listen_rows += batch.num_rows
        ts = batch.column("timestamp").to_numpy(zero_copy_only=False)
        ratio = batch.column("played_ratio_pct").to_numpy(zero_copy_only=False).astype(np.float64)
        bad = (ts > AXIS_MAXIMUM_UNITS) | np.isnan(ratio)
        invalid_rows += int(bad.sum()) + int(batch.column("timestamp").null_count)
        keep = (batch.column("is_organic").to_numpy(zero_copy_only=False) == 1) & (ratio > 50) & ~bad
        if keep.any():
            uid_parts.append(batch.column("uid").to_numpy(zero_copy_only=False)[keep])
            ts_parts.append(ts[keep])
            item_parts.append(batch.column("item_id").to_numpy(zero_copy_only=False)[keep].astype(np.int64))
    arrays = (np.concatenate(uid_parts), np.concatenate(ts_parts).astype(np.int64),
              np.concatenate(item_parts))
    return arrays, {"listen_rows": listen_rows, "invalid_rows": invalid_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end-units", type=int, default=DEFAULT_TRAIN_END_UNITS)
    parser.add_argument("--validation-end-units", type=int, default=DEFAULT_VALIDATION_END_UNITS)
    args = parser.parse_args()

    if args.output.exists():
        sys.exit("refuse_existing_output")
    prep = args.prepared
    prep_manifest_path = prep / "PREP_MANIFEST.json"
    if not prep_manifest_path.is_file():
        sys.exit("prepared_suite_not_finalized")
    prepared_manifest = json.loads(prep_manifest_path.read_text())
    if (prepared_manifest.get("schema") != "genrec.yambda500m.prepared.v2"
            or prepared_manifest.get("status") != "finalized"
            or prepared_manifest.get("protocol", {}).get("axis_unit_seconds") != 1):
        sys.exit("prepared_suite_not_finalized")

    source_sha = sha256_file(args.events)
    if source_sha != REGISTERED_SOURCE_SHA256:
        sys.exit("source_identity_mismatch")

    # Encode items with the prepared suite's interning (raw order == numeric
    # ascending in this suite, so sorted identity == catalog order).
    identity: dict[int, int] = {}
    model_by_sorted = []
    with (prep / "track_identity.tsv").open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:3] != ["raw_index", "model_id", "track_id"] and header[:3] != ["raw_index", "model_id", "item_id"]:
            sys.exit("track_identity_header_invalid")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            identity[int(fields[2])] = int(fields[1])
    for name in ("train_frequencies.npy", "track_identity.tsv"):
        if sha256_file(prep / name) != prepared_manifest["artifacts"][name]["sha256"]:
            sys.exit(f"prepared_artifact_digest_mismatch:{name}")
    vocab = len(np.load(prep / "train_frequencies.npy", mmap_mode="r"))
    if len(identity) != vocab - len((PAD, UNK, EOS)):
        sys.exit("identity_vocab_mismatch")
    if len(set(identity.values())) != len(identity):
        sys.exit("identity_model_ids_not_unique")
    sorted_items = np.array(sorted(identity.keys()), dtype=np.int64)
    sorted_models = np.array([identity[int(k)] for k in sorted_items], dtype=np.int32)

    (uid, ts_units, stream_items), counters = stream_organic_listen_plus(args.events)
    if (counters["listen_rows"] != 466_512_103
            or counters["invalid_rows"] != 0
            or len(ts_units) != 139_275_951):
        sys.exit("stream_identity_mismatch")
    if len(np.unique(uid)) != 87_518:
        sys.exit("user_count_mismatch")
    if int(stream_items.max()) >= (1 << 24):
        sys.exit("item_id_exceeds_key_space")

    user_starts = np.flatnonzero(np.concatenate(([True], uid[1:] != uid[:-1])))
    user_ends = np.append(user_starts[1:], len(uid))
    if np.any(uid[1:] < uid[:-1]):
        sys.exit("uid_order_violation")
    del uid

    # model_id for every catalog item, indexed by sorted position
    encoded_by_sorted = np.where(sorted_models >= FIRST_ITEM, sorted_models, UNK)

    events_parts: list[np.ndarray] = []
    times_parts: list[np.ndarray] = []
    events_length = 0
    session_rows: list[list[int]] = []
    label_rows: list[int] = []
    label_values: list[list[int]] = []
    duplicate_rows = 0
    tied_test_sessions = 0
    test_episode_count = 0
    test_bucket_counts = np.zeros(4, dtype=np.int64)

    for user_idx in range(len(user_starts)):
        lo, hi = int(user_starts[user_idx]), int(user_ends[user_idx])
        u_ts = ts_units[lo:hi]
        if np.any(np.diff(u_ts) < 0):
            sys.exit("timestamp_order_violation")
        # Dedup key = (ts, raw item_id). item_id < 2**24 and ts < 2**25, so the
        # packed key preserves the builder's (ts, item) equality and order
        # semantics for all streamed items, catalog or not (builder L224 keys
        # are a strictly monotone transform). Using catalog positions would
        # collapse distinct non-catalog items sharing a timestamp.
        u_item = stream_items[lo:hi]
        keys = (u_ts << 32) | u_item
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        u_item = u_item[order]
        unique_mask = np.empty(len(keys), dtype=bool)
        unique_mask[0] = True
        np.not_equal(keys[1:], keys[:-1], out=unique_mask[1:])
        duplicate_rows += len(keys) - int(unique_mask.sum())
        keys = keys[unique_mask]
        seq_ts = keys >> 32
        seq_item = (keys & 0xFFFFFFFF).astype(np.int64)
        del keys, unique_mask, u_item, order

        starts = np.flatnonzero(np.diff(seq_ts) > SESSION_GAP_UNITS) + 1
        if len(starts) == 0:
            continue
        ends = np.append(starts[1:], len(seq_ts))
        gaps = seq_ts[starts] - seq_ts[starts - 1]
        start_units = seq_ts[starts]
        last_units = seq_ts[ends - 1]
        split = np.where(start_units < args.train_end_units, 0,
                         np.where(start_units < args.validation_end_units, 1, 2))
        boundary = np.where(split == 0, args.train_end_units,
                            np.where(split == 1, args.validation_end_units,
                                     AXIS_MAXIMUM_UNITS))
        tie_global = np.zeros(len(seq_ts), dtype=np.int64)
        tie_global[1:] = np.cumsum((np.diff(seq_ts) == 0).astype(np.int64))
        tied_session = (tie_global[ends - 1] - tie_global[starts]) > 0
        pass_window = ((gaps >= MINIMUM_HIATUS_UNITS)
                       & (last_units + SESSION_GAP_UNITS <= boundary))
        has_target = (ends - starts) >= 2
        qualifying = pass_window & has_target & ~tied_session

        test_sel = (split == 2) & qualifying
        tied_test_sessions += int(np.sum((split == 2) & pass_window & has_target & tied_session))
        test_episode_count += int(test_sel.sum())
        if test_sel.any():
            test_bucket_counts += np.bincount(bucket_index(gaps[test_sel]), minlength=4)

        for pos_index in np.flatnonzero(test_sel):
            start = int(starts[pos_index])
            end = int(ends[pos_index])
            gap_units = int(gaps[pos_index])
            suffix_stop = min(end, start + 1 + HORIZON_EVENTS)
            context_begin = start - HISTORY_EVENTS if start > HISTORY_EVENTS else 0

            def encode(item_ids: np.ndarray) -> np.ndarray:
                hit = np.clip(np.searchsorted(sorted_items, item_ids), 0, len(sorted_items) - 1)
                catalog = sorted_items[hit] == item_ids
                out = np.where(catalog, encoded_by_sorted[np.where(catalog, hit, 0)], UNK)
                return out.astype(np.int32, copy=False)

            context_items = seq_item[context_begin:start + 1]
            target_items = seq_item[start + 1:suffix_stop]
            context_slice = encode(context_items)
            target_slice = encode(target_items)

            offset = events_length
            events_parts.append(context_slice)
            events_parts.append(target_slice)
            times_parts.append((seq_ts[context_begin:start + 1] * AXIS_UNIT_SECONDS).astype(np.uint32))
            times_parts.append((seq_ts[start + 1:suffix_stop] * AXIS_UNIT_SECONDS).astype(np.uint32))
            events_length += len(context_slice) + len(target_slice)
            target_start = offset + len(context_slice) - 1
            target_end = offset + len(context_slice) + len(target_slice)
            session_rows.append([user_idx, offset, target_start, target_end,
                                 gap_units * AXIS_UNIT_SECONDS, 2])
            label_rows.append(len(session_rows) - 1)
            distinct: list[int] = []
            seen: set[int] = set()
            for raw in target_items:
                if int(raw) in seen:
                    continue
                seen.add(int(raw))
                hit = int(np.searchsorted(sorted_items, raw))
                if hit < len(sorted_items) and sorted_items[hit] == raw:
                    model_id = int(encoded_by_sorted[hit])
                    distinct.append(model_id if model_id >= FIRST_ITEM
                                    else -int(raw) - 1)
                else:
                    distinct.append(-int(raw) - 1)  # non-catalog: builder semantics
            label_values.append(distinct)

    # ---- fail-closed census reconciliation ----
    if (test_episode_count != REGISTERED_TEST_CENSUS["episodes"]
            or test_bucket_counts.tolist() != REGISTERED_TEST_CENSUS["buckets"]):
        sys.exit(
            "test_census_mismatch:"
            f"episodes={test_episode_count},buckets={test_bucket_counts.tolist()}"
        )
    if len(session_rows) != test_episode_count:
        sys.exit("materialized_rows_mismatch")

    args.output.mkdir(parents=True)
    np.save(args.output / "events.npy",
            np.concatenate(events_parts).astype(np.int32, copy=False), allow_pickle=False)
    np.save(args.output / "event_times.npy",
            np.concatenate(times_parts), allow_pickle=False)
    np.save(args.output / "sessions.npy",
            np.asarray(session_rows, dtype=np.int64).reshape(-1, 6), allow_pickle=False)
    if label_values:
        values = np.concatenate([np.asarray(v, dtype=np.int64) for v in label_values])
        offsets = np.concatenate(([0], np.cumsum([len(v) for v in label_values])))
    else:
        values = np.zeros(0, np.int64)
        offsets = np.zeros(1, np.int64)
    np.savez(args.output / "metric_labels.npz",
             row_ids=np.asarray(label_rows, dtype=np.int64),
             values=values, offsets=offsets.astype(np.int64))
    shutil.copy2(prep / "train_frequencies.npy", args.output / "train_frequencies.npy")
    shutil.copy2(prep / "track_identity.tsv", args.output / "track_identity.tsv")

    artifacts = {}
    for name in ("events.npy", "event_times.npy", "sessions.npy", "metric_labels.npz",
                 "train_frequencies.npy", "track_identity.tsv"):
        path = args.output / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    manifest = {
        "schema": "genrec.yambda500m.sealed_test.v2",
        "status": "pass",
        "test_accessed": True,
        "selection_or_tuning_permitted": False,
        "source": {"bytes": args.events.stat().st_size, "sha256": source_sha},
        "prepared_manifest_sha256": sha256_file(prep_manifest_path),
        "protocol": {
            "axis_unit_seconds": AXIS_UNIT_SECONDS,
            "timestamp_semantics": "elapsed seconds rounded down to 5-second precision",
            "axis_maximum_units": AXIS_MAXIMUM_UNITS,
            "train_end_units": args.train_end_units,
            "validation_end_units": args.validation_end_units,
            "test_window": f"[{args.validation_end_units}, {AXIS_MAXIMUM_UNITS})",
            "minimum_hiatus_days": 30,
            "session_gap_seconds": 1800,
            "history_events": HISTORY_EVENTS,
            "continuation_horizon_events": HORIZON_EVENTS,
            "ambiguous_timestamp_ties": "excluded",
        },
        "counts": {
            "rows": test_episode_count,
            "episodes": test_episode_count,
            "registered_census_reconciled": True,
            "gap_buckets": test_bucket_counts.tolist(),
            "duplicate_rows_removed": duplicate_rows,
            "tied_test_sessions_excluded": tied_test_sessions,
            "train_seen_target_coverage": float(np.mean(values >= FIRST_ITEM)),
        },
        "artifacts": artifacts,
    }
    (args.output / "SEALED_TEST_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
