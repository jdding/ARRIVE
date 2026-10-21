#!/usr/bin/env python3
"""Prepare the ARRIVE episode contract from Yambda-500M (Yandex Music).

This is the Yambda counterpart of the Music4All-Onion producer
(`prepare_music4all_onion.cpp` + `finalize_music4all_onion_prepared.py`).
The two datasets form two complete, independent suites: this script never
touches Onion artifacts, and the Onion pipeline is left unchanged.

The Onion episode contract is mirrored item by item:

* session split when the inter-event gap exceeds 1800 s (strictly greater);
* episode = return session whose start is at least 30 days after the
  previous event, with at least one continuation target;
* history = up to 128 events before the observed return track (x0), targets
  = up to 64 events after x0 inside the same return session;
* return sessions containing consecutive equal-timestamp events are
  ambiguous and excluded;
* exact (timestamp, item) duplicate events are removed;
* catalog and train frequencies are counted at ingestion (duplicates
  included) over train-period events only;
* validation episodes carry distinct multi-positive metric labels, with
  out-of-catalog items encoded as negative raw item ids;
* the test period is never materialized.

Three declared platform adaptations (the Yambda release has no calendar
timestamps; its anonymized global axis is expressed in seconds rounded down
to 5-second precision):

1. event stream = organic Listen+ (event_type=listen, is_organic=1,
   played_ratio_pct > 50) instead of raw scrobbles;
2. train/validation boundaries are fixed fractions of the frozen axis
   maximum (26,000,000 units) instead of calendar dates; test is the
   remainder and stays sealed (an aggregate episode census is recorded,
   no items or targets);
3. raw items are interned in ascending numeric item_id order (Onion uses
   first-appearance order); both are deterministic.

The released timestamp values are already elapsed seconds; a value such as 10
means 10 seconds, not ten 5-second ticks. Fail-closed gates:
refuse an existing output directory, verify the input parquet sha256 and
size against the cluster-registered digest, verify the per-user timestamp
ordering invariant, and refuse invalid listen rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc

PAD, UNK, EOS, FIRST_ITEM = 0, 1, 2, 3  # EOS unused; kept for the shared id scheme
SESSION_GAP_SECONDS = 1800
MINIMUM_HIATUS_DAYS = 30
HISTORY_EVENTS = 128
HORIZON_EVENTS = 64
# Yambda paper, Eq. (1): floor((T_event - T_start) / 5) * 5.  The stored
# values therefore remain seconds and are merely multiples of five.
AXIS_UNIT_SECONDS = 1
AXIS_MAXIMUM_UNITS = 26_000_000
# Fixed fractions of the frozen axis maximum: train < 75%, validation in
# [75%, 85%), test sealed beyond 85%. Round fractions fixed before any
# per-window episode accounting.
DEFAULT_TRAIN_END_UNITS = 19_500_000
DEFAULT_VALIDATION_END_UNITS = 22_100_000
REGISTERED_SOURCE_SHA256 = (
    "bf30497a2b11a50f568cb0bb3efb9ca23786c76b82de4170025c1a9614e1b9cd"
)
REGISTERED_SOURCE_BYTES = 3_843_474_075
EXPECTED_LISTEN_ROWS = 466_512_103

# Registered run identity asserted by the independent finalize gate
# (second invocation, separate process) before the artifacts may be cited.
REGISTERED_STATS = {
    "organic_listen_plus_rows": 139_275_951,
    "users_with_organic_listen_plus": 87_518,
    "raw_tracks": 1_597_638,
    "train_catalog_tracks": 1_326_189,
    "train_episodes": 17_370,
    "validation_episodes": 4_323,
}
FINALIZED_ARTIFACTS = (
    "events.npy", "event_times.npy", "sessions.npy", "train_frequencies.npy",
    "metric_labels.npz", "track_identity.tsv", "PREP_STATS.json",
)

SESSION_GAP_UNITS = SESSION_GAP_SECONDS // AXIS_UNIT_SECONDS
MINIMUM_HIATUS_UNITS = MINIMUM_HIATUS_DAYS * 86_400 // AXIS_UNIT_SECONDS


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def bucket_of(gap_units: int) -> int:
    days = gap_units * AXIS_UNIT_SECONDS / 86400.0
    if days < 90:
        return 0
    if days < 180:
        return 1
    if days < 365:
        return 2
    return 3


def bucket_index(gap_units: np.ndarray) -> np.ndarray:
    days = gap_units * (AXIS_UNIT_SECONDS / 86400.0)
    return np.where(days < 90, 0, np.where(days < 180, 1, np.where(days < 365, 2, 3)))


def stream_organic_listen_plus(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                    dict[str, int]]:
    """Single pass over the parquet; returns (uid, ts_units, item_id) of the
    organic Listen+ stream plus counters. Columns stay in compact dtypes."""
    dset = ds.dataset(path, format="parquet")
    scanner = dset.scanner(
        columns=["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct"],
        filter=(pc.field("event_type") == "listen"),
        batch_size=5_000_000,
    )
    uid_parts: list[np.ndarray] = []
    ts_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []
    counters = {"listen_rows": 0, "invalid_rows": 0}
    for batch in scanner.to_batches():
        counters["listen_rows"] += batch.num_rows
        ts = batch.column("timestamp").to_numpy(zero_copy_only=False)
        ratio = batch.column("played_ratio_pct").to_numpy(zero_copy_only=False).astype(np.float64)
        bad = (ts > AXIS_MAXIMUM_UNITS) | np.isnan(ratio)
        counters["invalid_rows"] += int(bad.sum()) + int(batch.column("timestamp").null_count)
        keep = (batch.column("is_organic").to_numpy(zero_copy_only=False) == 1) \
            & (ratio > 50) & ~bad
        if keep.any():
            uid_parts.append(batch.column("uid").to_numpy(zero_copy_only=False)[keep])
            ts_parts.append(ts[keep])
            item_parts.append(batch.column("item_id").to_numpy(zero_copy_only=False)[keep]
                              .astype(np.int64))
    return (np.concatenate(uid_parts), np.concatenate(ts_parts),
            np.concatenate(item_parts), counters)


def finalize(output: Path) -> None:
    """Independent finalize gate: re-assert the registered run identity and
    digest every artifact so silent corruption or regeneration is detectable.
    Mirrors the two-process producer/finalize discipline of the Onion suite."""
    manifest_path = output / "PREP_MANIFEST.json"
    stats = json.loads((output / "PREP_STATS.json").read_text())
    for key, expected in REGISTERED_STATS.items():
        if stats.get(key) != expected:
            sys.exit(f"finalize_registered_count_mismatch:{key}")
    if stats.get("test_materialized") is not False:
        sys.exit("finalize_test_must_remain_unmaterialized")
    manifest = json.loads(manifest_path.read_text())
    digests = {}
    for name in FINALIZED_ARTIFACTS:
        path = output / name
        digests[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest["artifacts"] = digests
    manifest["test_accessed"] = False
    manifest["producer_sha256"] = sha256_file(Path(__file__))
    manifest["vocabulary_sha256"] = sha256_file(output / "track_identity.tsv")
    manifest["metric_labels"] = {
        "artifact": "metric_labels.npz",
        "row_scope": "validation episodes only",
        "test_accessed": False,
    }
    manifest["status"] = "finalized"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("FINALIZE_OK " + output.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=None,
                        help="Yambda-500M multi_event.parquet (required for"
                             " preparation; unused with --finalize)")
    parser.add_argument("--output", type=Path, required=True,
                        help="output directory (must not exist)")
    parser.add_argument("--finalize", action="store_true",
                        help="finalize gate only: assert registered counts and"
                             " digest artifacts, then exit")
    parser.add_argument("--train-end-units", type=int, default=DEFAULT_TRAIN_END_UNITS)
    parser.add_argument("--validation-end-units", type=int,
                        default=DEFAULT_VALIDATION_END_UNITS)
    parser.add_argument("--expect-sha256", default=REGISTERED_SOURCE_SHA256)
    parser.add_argument("--expect-bytes", type=int, default=REGISTERED_SOURCE_BYTES)
    parser.add_argument("--expect-listen-rows", type=int, default=EXPECTED_LISTEN_ROWS)
    args = parser.parse_args()

    if args.finalize:
        finalize(args.output)
        return

    if args.events is None:
        sys.exit("events_parquet_required_without_finalize")
    if args.output.exists():
        sys.exit("refuse_existing_output")
    if not (0 < args.train_end_units < args.validation_end_units <= AXIS_MAXIMUM_UNITS):
        sys.exit("invalid_split_boundaries")

    source_sha = sha256_file(args.events)
    if source_sha != args.expect_sha256 or args.events.stat().st_size != args.expect_bytes:
        sys.exit("source_identity_mismatch")
    args.output.mkdir(parents=True)

    uid, ts_units, item, counters = stream_organic_listen_plus(args.events)
    if counters["listen_rows"] != args.expect_listen_rows:
        sys.exit("listen_row_identity_mismatch")
    if counters["invalid_rows"] != 0:
        sys.exit("invalid_listen_rows_present")
    if len(uid) == 0:
        sys.exit("empty_organic_stream")

    # Train-period frequencies are counted at ingestion, duplicates included,
    # mirroring the Onion producer's counting point.
    train_counts_by_item = np.bincount(item[ts_units < args.train_end_units],
                                       minlength=int(item.max()) + 1)
    raw_items, raw_index = np.unique(item, return_inverse=True)
    del item
    raw_train_counts = train_counts_by_item[raw_items]

    # Catalog: train-period count > 0, ordered by count desc then item_id asc.
    order = np.lexsort((raw_items, -raw_train_counts))
    catalog = order[raw_train_counts[order] > 0]
    encoded_track = np.full(len(raw_items), UNK, dtype=np.int32)
    encoded_track[catalog] = np.arange(len(catalog), dtype=np.int32) + FIRST_ITEM
    frequencies = np.zeros(len(catalog) + FIRST_ITEM, dtype=np.int64)
    frequencies[FIRST_ITEM:] = raw_train_counts[catalog]

    # ------------------------------------------------------------------
    # Per-user sessionization and episode construction. The file is
    # pre-sorted by (uid, timestamp); the ordering gate below is mandatory
    # because deduplication re-sorts by the (timestamp, item) key.
    # ------------------------------------------------------------------
    user_starts = np.flatnonzero(np.concatenate(([True], uid[1:] != uid[:-1])))
    user_ends = np.append(user_starts[1:], len(uid))
    # The (uid, timestamp) pre-sortedness is load-bearing: uid blocks must be
    # contiguous and non-decreasing, otherwise users fragment and the
    # sessionization/hiatus semantics silently change.
    if np.any(uid[1:] < uid[:-1]):
        sys.exit("uid_order_violation")
    organic_rows = len(uid)
    del uid
    events_parts: list[np.ndarray] = []
    times_parts: list[np.ndarray] = []
    events_length = 0
    session_rows: list[list[int]] = []
    label_rows: list[int] = []
    label_values: list[list[int]] = []
    duplicate_rows = 0
    tied_sessions = 0
    episode_counts = [0, 0]
    bucket_counts = [[0, 0, 0, 0], [0, 0, 0, 0]]
    test_episode_count = 0
    test_bucket_counts = np.zeros(4, dtype=np.int64)

    for user_idx in range(len(user_starts)):
        lo, hi = int(user_starts[user_idx]), int(user_ends[user_idx])
        u_ts = ts_units[lo:hi].astype(np.int64)
        if np.any(np.diff(u_ts) < 0):
            sys.exit("timestamp_order_violation")
        keys = (u_ts << 32) | raw_index[lo:hi]
        keys = np.sort(keys)
        unique_mask = np.empty(len(keys), dtype=bool)
        unique_mask[0] = True
        np.not_equal(keys[1:], keys[:-1], out=unique_mask[1:])
        duplicate_rows += len(keys) - int(unique_mask.sum())
        keys = keys[unique_mask]
        seq_ts = keys >> 32
        seq_item = (keys & 0xFFFFFFFF).astype(np.int64)

        starts = np.flatnonzero(np.diff(seq_ts) > SESSION_GAP_UNITS) + 1
        if len(starts) == 0:
            continue
        ends = np.append(starts[1:], len(seq_ts))
        # Session-level episode predicate, fully vectorized. The tie term
        # mirrors the Onion producer: ambiguous (consecutive equal
        # timestamps) return sessions are excluded from train/validation.
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
        trainval = split < 2
        tied_sessions += int(np.sum(trainval & pass_window & tied_session))
        qualifying = pass_window & has_target & ~tied_session

        test_sel = (split == 2) & qualifying
        test_episode_count += int(test_sel.sum())
        if test_sel.any():
            test_bucket_counts += np.bincount(bucket_index(gaps[test_sel]), minlength=4)
        cand = np.flatnonzero(trainval & qualifying)
        for pos in cand:
            start = int(starts[pos])
            end = int(ends[pos])
            gap_units = int(gaps[pos])
            split_id = int(split[pos])
            suffix_stop = min(end, start + 1 + HORIZON_EVENTS)
            context_begin = start - HISTORY_EVENTS if start > HISTORY_EVENTS else 0
            offset = events_length
            context_slice = seq_item[context_begin:start + 1]
            target_slice = seq_item[start + 1:suffix_stop]
            events_parts.append(encoded_track[context_slice])
            events_parts.append(encoded_track[target_slice])
            times_parts.append((seq_ts[context_begin:start + 1]
                                * AXIS_UNIT_SECONDS).astype(np.uint32))
            times_parts.append((seq_ts[start + 1:suffix_stop]
                                * AXIS_UNIT_SECONDS).astype(np.uint32))
            events_length += len(context_slice) + len(target_slice)
            target_start = offset + len(context_slice) - 1
            target_end = offset + len(context_slice) + len(target_slice)
            session_rows.append([user_idx, offset, target_start, target_end,
                                 gap_units * AXIS_UNIT_SECONDS, split_id])
            episode_counts[split_id] += 1
            bucket_counts[split_id][bucket_of(gap_units)] += 1
            if split_id == 1:
                label_rows.append(len(session_rows) - 1)
                distinct: list[int] = []
                seen: set[int] = set()
                for raw in target_slice:
                    if int(raw) not in seen:
                        seen.add(int(raw))
                        distinct.append(int(raw))
                label_values.append(
                    [int(encoded_track[raw]) if encoded_track[raw] >= FIRST_ITEM
                     else -int(raw_items[raw]) - 1 for raw in distinct])

    events = (np.concatenate(events_parts).astype(np.int32, copy=False)
              if events_parts else np.zeros(0, np.int32))
    event_times = (np.concatenate(times_parts) if times_parts
                   else np.zeros(0, np.uint32))
    sessions = np.asarray(session_rows, dtype=np.int64).reshape(-1, 6)
    np.save(args.output / "events.npy", events, allow_pickle=False)
    np.save(args.output / "event_times.npy", event_times, allow_pickle=False)
    np.save(args.output / "sessions.npy", sessions, allow_pickle=False)
    np.save(args.output / "train_frequencies.npy", frequencies, allow_pickle=False)

    if label_values:
        values = np.concatenate([np.asarray(v, dtype=np.int64) for v in label_values])
        offsets = np.concatenate(([0], np.cumsum([len(v) for v in label_values])))
    else:
        values = np.zeros(0, np.int64)
        offsets = np.zeros(1, np.int64)
    np.savez(args.output / "metric_labels.npz",
             row_ids=np.asarray(label_rows, dtype=np.int64),
             values=values.astype(np.int64),
             offsets=offsets.astype(np.int64))

    with (args.output / "track_identity.tsv").open("w") as handle:
        handle.write("raw_index\tmodel_id\titem_id\ttrain_events\n")
        for raw in catalog:
            handle.write(f"{int(raw)}\t{int(encoded_track[raw])}\t{int(raw_items[raw])}"
                         f"\t{int(raw_train_counts[raw])}\n")

    stats = {
        "schema": "genrec.yambda500m.prep_stats.v2",
        "source_listen_rows": counters["listen_rows"],
        "invalid_listen_rows": counters["invalid_rows"],
        "organic_listen_plus_rows": organic_rows,
        "users_with_organic_listen_plus": int(len(user_starts)),
        "raw_tracks": int(len(raw_items)),
        "train_catalog_tracks": int(len(catalog)),
        "exact_duplicate_rows_removed": duplicate_rows,
        "ambiguous_tie_sessions_removed": tied_sessions,
        "train_episodes": episode_counts[0],
        "validation_episodes": episode_counts[1],
        "train_gap_buckets": bucket_counts[0],
        "validation_gap_buckets": bucket_counts[1],
        "test_materialized": False,
        "test_aggregate_census_episodes": test_episode_count,
        "test_aggregate_census_gap_buckets": [int(x) for x in test_bucket_counts],
    }
    (args.output / "PREP_STATS.json").write_text(json.dumps(stats, indent=2) + "\n")

    manifest = {
        "schema": "genrec.yambda500m.prepared.v2",
        "status": "prepared",
        "source": {"bytes": args.events.stat().st_size, "sha256": source_sha},
        "producer": Path(__file__).name,
        "protocol": {
            "event_stream": "organic Listen+ (event_type=listen, is_organic=1,"
                            " played_ratio_pct>50)",
            "axis_unit_seconds": AXIS_UNIT_SECONDS,
            "timestamp_semantics": "elapsed seconds rounded down to 5-second precision",
            "axis_maximum_units": AXIS_MAXIMUM_UNITS,
            "train_end_units": args.train_end_units,
            "validation_end_units": args.validation_end_units,
            "minimum_hiatus_days": MINIMUM_HIATUS_DAYS,
            "session_gap_seconds": SESSION_GAP_SECONDS,
            "history_events": HISTORY_EVENTS,
            "observed_return_prefix_tracks": 1,
            "continuation_horizon_events": HORIZON_EVENTS,
            "ambiguous_timestamp_ties": "excluded",
            "test": "sealed; not materialized (aggregate census only)",
            "test_census_fit_boundary_units": AXIS_MAXIMUM_UNITS,
            "test_census_predicate": "hiatus >= 30d AND last event + 1800s <= "
                                     "axis max AND session has >= 2 events AND "
                                     "no consecutive equal timestamps; counts only",
        },
        "counts": stats,
    }
    (args.output / "PREP_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
