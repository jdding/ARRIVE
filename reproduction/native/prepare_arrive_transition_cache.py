#!/usr/bin/env python3
"""Build one train-only transition cache shared by MUSE, MEGAN and FactUBR."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse


FIRST_ITEM = 3
SESSION_GAP_SECONDS = 1800


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def top_neighbors(matrix: sparse.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    items = np.full((matrix.shape[0], k), -1, dtype=np.int32)
    weights = np.zeros((matrix.shape[0], k), dtype=np.float32)
    for row in range(FIRST_ITEM, matrix.shape[0]):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        if start == stop:
            continue
        values = matrix.data[start:stop]
        indices = matrix.indices[start:stop]
        take = min(k, len(values))
        chosen = np.argpartition(values, -take)[-take:]
        chosen = chosen[np.lexsort((indices[chosen], -values[chosen]))]
        items[row, :take] = indices[chosen]
        weights[row, :take] = np.log1p(values[chosen]).astype(np.float32)
    return items, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--event-times", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refuse_existing_output")
    args.output.mkdir(parents=True)

    events = np.load(args.data_dir / "events.npy", mmap_mode="r")
    times = np.load(args.event_times, mmap_mode="r")
    sessions = np.load(args.data_dir / "sessions.npy", mmap_mode="r")
    frequencies = np.load(args.data_dir / "train_frequencies.npy", mmap_mode="r")
    if events.shape != times.shape:
        raise SystemExit("event_time_shape_mismatch")

    sources, targets = array("I"), array("I")
    train_rows = sessions[sessions[:, 5] == 0]
    user_count = 0
    at = 0
    while at < len(train_rows):
        stop = at + 1
        while stop < len(train_rows) and train_rows[stop, 0] == train_rows[at, 0]:
            stop += 1
        rows = train_rows[at:stop]
        user_count += 1
        timeline: dict[tuple[int, int], None] = {}
        for _, begin, _, end, _, _ in rows:
            for item, timestamp in zip(events[begin:end], times[begin:end]):
                if int(item) >= FIRST_ITEM:
                    timeline[(int(timestamp), int(item))] = None
        ordered = sorted(timeline)
        for left, right in zip(ordered, ordered[1:]):
            if right[0] - left[0] <= SESSION_GAP_SECONDS:
                sources.append(left[1])
                targets.append(right[1])
        at = stop

    source_array = np.frombuffer(sources, dtype=np.uint32).astype(np.int64)
    target_array = np.frombuffer(targets, dtype=np.uint32).astype(np.int64)
    values = np.ones(len(source_array), dtype=np.int32)
    transition = sparse.coo_matrix(
        (values, (source_array, target_array)), shape=(len(frequencies), len(frequencies))
    ).tocsr()
    transition.sum_duplicates()
    outgoing_items, outgoing_weights = top_neighbors(transition, args.topk)
    incoming_items, incoming_weights = top_neighbors(transition.T.tocsr(), args.topk)
    cache_path = args.output / "transition_topk.npz"
    np.savez_compressed(
        cache_path,
        outgoing_items=outgoing_items,
        outgoing_weights=outgoing_weights,
        incoming_items=incoming_items,
        incoming_weights=incoming_weights,
    )
    manifest = {
        "schema": "genrec.arrive_transition_cache.v1",
        "status": "pass",
        "train_users": user_count,
        "transition_occurrences": int(len(source_array)),
        "unique_transitions": int(transition.nnz),
        "topk": args.topk,
        "source_hashes": {
            "events": digest(args.data_dir / "events.npy"),
            "sessions": digest(args.data_dir / "sessions.npy"),
            "event_times": digest(args.event_times),
        },
        "artifact": {"path": cache_path.name, "sha256": digest(cache_path)},
        "validation_accessed": False,
        "test_accessed": False,
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
