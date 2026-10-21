#!/usr/bin/env python3
"""Build shared Train-only assets for CoSeRNN and the blocked MEGAN prototype.

The expensive interaction scan is performed once per dataset.  The resulting
HMG neighborhoods and fixed successor-track space are reused by all seeds.
Validation and Test events are never inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


FIRST_ITEM = 3
SESSION_GAP_SECONDS = 1800


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sample_rows(
    matrix: sparse.csr_matrix, k: int, *, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper-style weighted neighbor sampling with replacement, frozen once."""
    ids = np.full((matrix.shape[0], k), -1, dtype=np.int32)
    weights = np.zeros((matrix.shape[0], k), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        if start == stop:
            continue
        values = matrix.data[start:stop]
        columns = matrix.indices[start:stop]
        selected = rng.choice(len(values), size=k, replace=True, p=values / values.sum())
        ids[row] = columns[selected]
        weights[row] = values[selected]
    return ids, weights


def load_track_metadata(identity_path: Path, information_path: Path, tags_path: Path, vocab_size: int):
    raw_to_model: dict[str, int] = {}
    with identity_path.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        id_key = "track_id" if "track_id" in (rows.fieldnames or []) else "item_id"
        for row in rows:
            raw_to_model[row[id_key]] = int(row["model_id"])
    tokens: list[list[str]] = [[] for _ in range(vocab_size)]
    with information_path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            model = raw_to_model.get(row["id"])
            if model is None:
                continue
            if row.get("artist"):
                tokens[model].append("artist:" + row["artist"].strip().lower())
            if row.get("album_name"):
                tokens[model].append("album:" + row["album_name"].strip().lower())
    with tags_path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            model = raw_to_model.get(row["id"])
            if model is None:
                continue
            tokens[model].extend(
                "tag:" + value.strip().lower()
                for value in row.get("tags", "").split(",") if value.strip()
            )
    frequencies = Counter(token for row in tokens for token in set(row))
    vocabulary = {
        token: at + 1 for at, (token, _) in enumerate(
            sorted(frequencies.items(), key=lambda pair: (-pair[1], pair[0]))
        )
    }
    width = 16
    table = np.zeros((vocab_size, width), dtype=np.int32)
    for item, row in enumerate(tokens):
        ordered = sorted(set(row), key=lambda term: (-frequencies[term], term))[:width]
        table[item, :len(ordered)] = [vocabulary[token] for token in ordered]
    return table, len(vocabulary) + 1


def build(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    events = np.load(args.data_dir / "events.npy", mmap_mode="r")
    sessions = np.load(args.data_dir / "sessions.npy", mmap_mode="r")
    times = np.load(args.event_times, mmap_mode="r")
    if events.shape != times.shape or sessions.ndim != 2 or sessions.shape[1] != 6:
        raise ValueError("prepared_input_shape_mismatch")
    train_rows = np.flatnonzero(sessions[:, 5] == 0)
    if not len(train_rows):
        raise ValueError("empty_train_rows")
    user_count = int(sessions[:, 0].max()) + 1
    vocab_size = len(np.load(args.data_dir / "train_frequencies.npy", mmap_mode="r"))

    # Materialization stores one context block per episode.  Later Train
    # episodes may add legitimate history that was absent from the user's
    # first context, so keeping only that first context loses interactions.
    # Take the union of all Train-authorized (timestamp, item) events instead;
    # this retains newly observed history while removing copied context rows.
    user_events: list[list[np.ndarray]] = [[] for _ in range(user_count)]
    for row in train_rows:
        user, first_context, start, end, _, _ = map(int, sessions[row])
        block = np.column_stack((
            np.asarray(times[first_context:end], dtype=np.int64),
            np.asarray(events[first_context:end], dtype=np.int64),
        ))
        user_events[user].append(block)
    user_chunks: list[list[np.ndarray]] = []
    user_timelines: list[np.ndarray] = []
    for blocks in user_events:
        if not blocks:
            user_chunks.append([])
            user_timelines.append(np.empty((0, 2), dtype=np.int64))
            continue
        unique = np.unique(np.concatenate(blocks, axis=0), axis=0)
        unique = unique[np.lexsort((unique[:, 1], unique[:, 0]))]
        user_chunks.append([unique[:, 1].astype(np.int32, copy=False)])
        user_timelines.append(unique)
    lengths = np.fromiter(
        (sum(len(chunk) for chunk in chunks) for chunks in user_chunks),
        dtype=np.int64, count=user_count,
    )
    interaction_count = int(lengths.sum())
    users = np.repeat(np.arange(user_count, dtype=np.int32), lengths)
    items = np.concatenate([chunk for chunks in user_chunks for chunk in chunks])
    if len(items) != interaction_count or (items < FIRST_ITEM).any():
        raise ValueError("train_span_materialization_failed")
    values = np.ones(interaction_count, dtype=np.float32)
    user_music = sparse.coo_matrix(
        (values, (users, items)), shape=(user_count, vocab_size)
    ).tocsr()
    user_music.sum_duplicates()

    # A session node is one authorized Train return-session episode.  Its
    # observed and suffix tracks are all Train interactions at this cutoff.
    session_items: list[np.ndarray] = []
    session_users = np.empty(len(train_rows), dtype=np.int32)
    for session_id, row in enumerate(train_rows):
        user, _, start, end, _, _ = map(int, sessions[row])
        session_users[session_id] = user
        session_items.append(np.unique(np.asarray(events[start:end], dtype=np.int32)))
    session_lengths = np.fromiter((len(row) for row in session_items), dtype=np.int64)
    music = np.concatenate(session_items) if session_items else np.empty(0, dtype=np.int32)
    session_ids = np.repeat(np.arange(len(train_rows), dtype=np.int32), session_lengths)
    music_session = sparse.coo_matrix(
        (np.ones(len(music), dtype=np.float32), (music, session_ids)),
        shape=(vocab_size, len(train_rows)),
    ).tocsr()
    music_session.sum_duplicates()
    user_session = sparse.coo_matrix(
        (np.ones(len(train_rows), dtype=np.float32),
         (session_users, np.arange(len(train_rows), dtype=np.int32))),
        shape=(user_count, len(train_rows)),
    ).tocsr()

    relation_arrays: dict[str, np.ndarray] = {}
    relation_matrices = {
        "user_music": user_music,
        "user_session": user_session,
        "music_user": user_music.T.tocsr(),
        "music_session": music_session,
    }

    transition_sources, transition_targets = [], []
    for timeline in user_timelines:
        if len(timeline) < 2:
            continue
        adjacent = timeline[1:, 0] - timeline[:-1, 0] <= SESSION_GAP_SECONDS
        transition_sources.append(timeline[:-1, 1][adjacent].astype(np.int32))
        transition_targets.append(timeline[1:, 1][adjacent].astype(np.int32))
    source = np.concatenate(transition_sources)
    target = np.concatenate(transition_targets)
    music_music = sparse.coo_matrix(
        (np.ones(len(source), dtype=np.float32), (source, target)),
        shape=(vocab_size, vocab_size),
    ).tocsr()
    music_music.sum_duplicates()
    relation_matrices["music_music"] = music_music
    for offset, (name, matrix) in enumerate(relation_matrices.items()):
        relation_arrays[name + "_ids"], relation_arrays[name + "_weights"] = sample_rows(
            matrix, args.neighbors, seed=args.graph_seed + offset,
        )

    music_tokens, feature_count = load_track_metadata(
        args.data_dir / "track_identity.tsv", args.information, args.tags, vocab_size
    )
    user_tokens = np.zeros((user_count, music_tokens.shape[1]), dtype=np.int32)
    for user in range(user_count):
        counts: Counter[int] = Counter()
        start, end = user_music.indptr[user:user + 2]
        for item, weight in zip(user_music.indices[start:end], user_music.data[start:end]):
            for token in music_tokens[item]:
                if token:
                    counts[int(token)] += float(weight)
        selected = [token for token, _ in counts.most_common(music_tokens.shape[1])]
        user_tokens[user, :len(selected)] = selected

    megan_path = args.output / "megan_hmg_assets.npz"
    np.savez_compressed(
        megan_path, user_tokens=user_tokens, music_tokens=music_tokens,
        feature_count=np.asarray(feature_count), session_count=np.asarray(len(train_rows)),
        **relation_arrays,
    )

    # A fixed track space is the defining CoSeRNN interface.  Use the complete
    # Train-only within-session successor matrix, rather than an already
    # truncated top-k cache, for a deterministic public-data PPMI substitute
    # for Spotify's unavailable proprietary playlist CBOW corpus.
    transition_matrix = music_music
    total = float(transition_matrix.sum())
    row_mass = np.asarray(transition_matrix.sum(1)).ravel()
    col_mass = np.asarray(transition_matrix.sum(0)).ravel()
    coo = transition_matrix.tocoo()
    pmi = np.log(
        np.maximum(coo.data * total, 1e-12)
        / np.maximum(row_mass[coo.row] * col_mass[coo.col], 1e-12)
    )
    ppmi = sparse.coo_matrix(
        (np.maximum(pmi, 0).astype(np.float32), (coo.row, coo.col)),
        shape=transition_matrix.shape,
    ).tocsr()
    track_space = TruncatedSVD(
        n_components=args.track_dim, n_iter=7, random_state=args.embedding_seed
    ).fit_transform(ppmi).astype(np.float32)
    norms = np.linalg.norm(track_space, axis=1, keepdims=True)
    missing = norms[:, 0] == 0
    missing_real_tracks = int(missing[FIRST_ITEM:].sum())
    rng = np.random.default_rng(args.embedding_seed)
    track_space[missing] = rng.standard_normal((int(missing.sum()), args.track_dim), dtype=np.float32)
    track_space /= np.maximum(np.linalg.norm(track_space, axis=1, keepdims=True), 1e-8)
    track_space[:FIRST_ITEM] = 0
    cosernn_path = args.output / "cosernn_track_space.npy"
    np.save(cosernn_path, track_space)

    source_hashes = {
        "events": digest(args.data_dir / "events.npy"),
        "sessions": digest(args.data_dir / "sessions.npy"),
        "event_times": digest(args.event_times),
        "track_identity": digest(args.data_dir / "track_identity.tsv"),
        "information": digest(args.information),
        "tags": digest(args.tags),
    }
    producer_sha = digest(Path(__file__))
    common = {
        "status": "pass", "source_hashes": source_hashes,
        "producer_sha256": producer_sha, "test_accessed": False,
    }
    atomic_json(args.output / "MEGAN_ASSETS.json", {
        **common, "schema": "genrec.arrive_megan_hmg_assets.v1",
        "artifact_sha256": digest(megan_path), "users": user_count,
        "tracks": vocab_size, "sessions": len(train_rows),
        "train_unique_timestamp_item_events": interaction_count,
        "feature_count": feature_count, "neighbors": args.neighbors,
        "edge_weights": "Train interaction/co-occurrence frequency",
        "neighbor_sampling": "weighted with replacement, frozen Train-only sample",
        "graph_seed": args.graph_seed,
    })
    atomic_json(args.output / "COSERNN_ASSETS.json", {
        **common, "schema": "genrec.arrive_cosernn_track_space.v1",
        "artifact_sha256": digest(cosernn_path), "tracks": vocab_size,
        "dimensions": args.track_dim,
        "track_space": "Train-only successor PPMI TruncatedSVD; fixed during CoSeRNN training",
        "transition_support": "complete Train-only within-30-minute successor counts; no top-k truncation",
        "transition_nonzero_pairs": int(transition_matrix.nnz),
        "random_fallback_tracks": missing_real_tracks,
        "random_fallback": "seeded unit vectors for catalog tracks without Train successor support",
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--event-times", type=Path, default=Path("data/music4all_onion/baseline_features_v1/event_times.npy"))
    parser.add_argument("--information", type=Path, default=Path("data/music4all_onion/raw/id_information.csv"))
    parser.add_argument("--tags", type=Path, default=Path("data/music4all_onion/raw/id_tags.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/music4all_onion/faithful_baseline_assets_v1"))
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--track-dim", type=int, default=40)
    parser.add_argument("--embedding-seed", type=int, default=1701)
    parser.add_argument("--graph-seed", type=int, default=20260912)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
