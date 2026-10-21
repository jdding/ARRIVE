#!/usr/bin/env python3
"""Train-only attainability gate for post-A1 multimodal support transport."""

from __future__ import annotations

import argparse
import bz2
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
from scipy import sparse

from lfm_timeconcat_control import LFMTimeConcatData
from run_music4all_candidate_support_a0 import (
    FIRST_ITEM, SCOPES, digest, merge_candidates, source_rows,
)

SCHEMA = "genrec.music4all_onion.component_b2_attainability.v1"
DECISION_SCHEMA = "genrec.music4all_onion.component_b2_attainability_decision.v1"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    )
    temporary = Path(handle.name)
    try:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.close()
        os.replace(temporary, path)
    finally:
        try:
            handle.close()
        except Exception:
            pass
        temporary.unlink(missing_ok=True)


def load_rank_bundle(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        bundle = json.load(handle)
    if (
        bundle.get("schema") != "genrec.music4all_onion.sasrec_ranked_items.v1"
        or bundle.get("status") != "pass"
        or bundle.get("surface") != "train_cv"
        or bundle.get("test_accessed") is not False
        or bundle.get("rank_depth") != 200
        or len(bundle.get("rows", [])) != 20_000
    ):
        raise ValueError("train_cv_rank_bundle_invalid")
    return bundle


def frozen_a1_support(args: argparse.Namespace):
    bundle = load_rank_bundle(args.cv_base)
    x_fit = sparse.load_npz(args.index_dir / "x_fit.npz").tocsr()
    h_fit = sparse.load_npz(args.index_dir / "h_fit.npz").tocsr()
    frequency = np.load(args.index_dir / "fit_target_frequency.npy", allow_pickle=False)
    train_frequency = np.load(args.data_dir / "train_frequencies.npy", allow_pickle=False)
    popularity = [int(i) for i in np.argsort(-train_frequency, kind="stable") if i >= FIRST_ITEM]
    source = source_rows(
        args, bundle, x_fit, h_fit, frequency,
        beta_x=0.0, beta_h=0.0, anchor_count=5,
    )
    candidates = np.asarray([
        merge_candidates(
            row["ranked_items"], row["x_items"], row["h_items"], popularity,
            relation_slots=40, x_fraction=0.75, budget=200,
        )
        for row in source
    ], dtype=np.int32)
    targets = [set(map(int, row["targets"])) for row in source]
    data = LFMTimeConcatData(args.data_dir, max_context=args.max_context, horizon=args.horizon)
    observed = np.asarray([
        int(data.events[int(data.sessions[int(row["row_id"]), 2])]) for row in source
    ], dtype=np.int32)
    if len({int(row["user"]) for row in source}) != len(source):
        raise ValueError("cv_user_not_unique")
    return {"bundle": bundle, "rows": source, "x_fit": x_fit, "frequency": frequency}, candidates, targets, observed


def track_identity(path: Path) -> tuple[dict[str, int], int]:
    mapping: dict[str, int] = {}
    maximum = 0
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:3] != ["raw_index", "model_id", "track_id"]:
            raise ValueError("track_identity_header_invalid")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            model_id = int(fields[1])
            mapping[fields[2]] = model_id
            maximum = max(maximum, model_id)
    return mapping, maximum + 1


def load_dense_feature(path: Path, mapping: dict[str, int], vocab_size: int):
    covered = np.zeros(vocab_size, dtype=bool)
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        width = len(header) - 1
        matrix = np.zeros((vocab_size, width), dtype=np.float32)
        for line in handle:
            track, values = line.rstrip("\n").split("\t", 1)
            model_id = mapping.get(track)
            if model_id is None:
                continue
            row = np.fromstring(values, sep="\t", dtype=np.float32)
            if len(row) != width:
                raise ValueError("dense_feature_width_mismatch")
            matrix[model_id] = row
            covered[model_id] = True
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return matrix, covered


def load_sparse_feature(path: Path, mapping: dict[str, int], vocab_size: int):
    rows: list[int] = []
    cols: list[int] = []
    values_out: list[float] = []
    covered = np.zeros(vocab_size, dtype=bool)
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        width = len(header) - 1
        for line in handle:
            track, values = line.rstrip("\n").split("\t", 1)
            model_id = mapping.get(track)
            if model_id is None:
                continue
            row = np.fromstring(values, sep="\t", dtype=np.float32)
            if len(row) != width:
                raise ValueError("sparse_feature_width_mismatch")
            nz = np.flatnonzero(row)
            rows.extend([model_id] * len(nz))
            cols.extend(map(int, nz))
            values_out.extend(map(float, row[nz]))
            covered[model_id] = True
    matrix = sparse.csr_matrix(
        (np.asarray(values_out, dtype=np.float32), (rows, cols)),
        shape=(vocab_size, width), dtype=np.float32,
    )
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inverse = np.zeros_like(norms)
    inverse[norms > 0] = 1.0 / norms[norms > 0]
    return sparse.diags(inverse).dot(matrix).tocsr(), covered


def prepare_features(args: argparse.Namespace, vocab_size: int):
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache_dir / "feature_cache_manifest.json"
    audio_path = args.cache_dir / "musicnn_normalized.npy"
    genre_path = args.cache_dir / "genre_normalized.npz"
    coverage_path = args.cache_dir / "coverage.npz"
    source_hashes = {
        "musicnn": digest(args.musicnn), "genre": digest(args.genre),
        "identity": digest(args.data_dir / "track_identity.tsv"),
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source_hashes") != source_hashes or manifest.get("vocab_size") != vocab_size:
            raise ValueError("feature_cache_binding_mismatch")
        for path, key in ((audio_path, "audio"), (genre_path, "genre"), (coverage_path, "coverage")):
            if digest(path) != manifest["artifacts"][key]:
                raise ValueError(f"feature_cache_hash_mismatch:{key}")
        return np.load(audio_path, mmap_mode="r"), sparse.load_npz(genre_path).tocsr(), manifest
    mapping, identity_vocab = track_identity(args.data_dir / "track_identity.tsv")
    if identity_vocab != vocab_size:
        raise ValueError("identity_vocab_mismatch")
    audio, audio_covered = load_dense_feature(args.musicnn, mapping, vocab_size)
    genre, genre_covered = load_sparse_feature(args.genre, mapping, vocab_size)
    np.save(audio_path, audio, allow_pickle=False)
    sparse.save_npz(genre_path, genre, compressed=True)
    np.savez_compressed(coverage_path, audio=audio_covered, genre=genre_covered)
    manifest = {
        "schema": "genrec.music4all_onion.component_b2_feature_cache.v1",
        "status": "pass", "vocab_size": vocab_size, "source_hashes": source_hashes,
        "coverage": {"audio": int(audio_covered[FIRST_ITEM:].sum()), "genre": int(genre_covered[FIRST_ITEM:].sum())},
        "artifacts": {"audio": digest(audio_path), "genre": digest(genre_path), "coverage": digest(coverage_path)},
        "test_accessed": False,
    }
    atomic_json(manifest_path, manifest)
    return np.load(audio_path, mmap_mode="r"), genre, manifest


def exact_top_neighbors(features, sources: np.ndarray, *, k: int, batch_size: int):
    ids = np.full((len(sources), k), -1, dtype=np.int32)
    similarities = np.full((len(sources), k), -np.inf, dtype=np.float32)
    catalog = features[FIRST_ITEM:]
    for start in range(0, len(sources), batch_size):
        stop = min(start + batch_size, len(sources))
        scores = features[sources[start:stop]] @ catalog.T
        if sparse.issparse(scores):
            scores = scores.toarray()
        scores = np.asarray(scores, dtype=np.float32)
        for local, source in enumerate(sources[start:stop]):
            source_at = int(source) - FIRST_ITEM
            if 0 <= source_at < scores.shape[1]:
                scores[local, source_at] = -np.inf
            available = np.flatnonzero(scores[local] > 0)
            if not len(available):
                continue
            take = min(k, len(available))
            partial = available[np.argpartition(-scores[local, available], take - 1)[:take]]
            values = scores[local, partial]
            order = np.lexsort((partial, -values))
            partial = partial[order]
            ids[start + local, :take] = partial + FIRST_ITEM
            similarities[start + local, :take] = scores[local, partial]
        print(json.dumps({"event": "content_neighbors", "rows": stop, "total": len(sources)}), flush=True)
    return ids, similarities


def prepare_neighbors(args, audio, genre, sources, feature_manifest):
    path = args.cache_dir / "source_neighbors.npz"
    manifest_path = args.cache_dir / "source_neighbors_manifest.json"
    binding = {
        "feature_manifest_sha256": digest(args.cache_dir / "feature_cache_manifest.json"),
        "source_sha256": hashlib.sha256(sources.tobytes()).hexdigest(),
        "sources": len(sources), "k": args.neighbor_depth,
    }
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("binding") != binding or manifest.get("artifact_sha256") != digest(path):
            raise ValueError("neighbor_cache_binding_mismatch")
        loaded = np.load(path, allow_pickle=False)
        return {key: loaded[key] for key in loaded.files}
    audio_ids, audio_sim = exact_top_neighbors(audio, sources, k=args.neighbor_depth, batch_size=args.dense_neighbor_batch)
    genre_ids, genre_sim = exact_top_neighbors(genre, sources, k=args.neighbor_depth, batch_size=args.sparse_neighbor_batch)
    np.savez_compressed(path, sources=sources, audio_ids=audio_ids, audio_sim=audio_sim, genre_ids=genre_ids, genre_sim=genre_sim)
    atomic_json(manifest_path, {
        "schema": "genrec.music4all_onion.component_b2_neighbors.v1", "status": "pass",
        "binding": binding, "artifact_sha256": digest(path),
        "feature_coverage": feature_manifest["coverage"], "test_accessed": False,
    })
    return {"sources": sources, "audio_ids": audio_ids, "audio_sim": audio_sim, "genre_ids": genre_ids, "genre_sim": genre_sim}


def row_softmax(values: np.ndarray, tau: float) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    for row in range(len(values)):
        valid = np.isfinite(values[row]) & (values[row] > 0)
        if not valid.any():
            continue
        shifted = values[row, valid] / tau
        shifted -= shifted.max()
        weights = np.exp(shifted)
        output[row, valid] = weights / weights.sum()
    return output


def kernel_matrix(ids, similarities, vocab_size, *, k, tau, permutation=None):
    selected_ids = ids[:, :k].copy()
    selected_sim = similarities[:, :k]
    weights = row_softmax(selected_sim, tau)
    valid = (selected_ids >= FIRST_ITEM) & (weights > 0)
    if permutation is not None:
        selected_ids[valid] = permutation[selected_ids[valid]]
    row_ids = np.repeat(np.arange(len(ids), dtype=np.int32), k)[valid.ravel()]
    return sparse.csr_matrix(
        (weights.ravel()[valid.ravel()], (row_ids, selected_ids.ravel()[valid.ravel()])),
        shape=(len(ids), vocab_size), dtype=np.float32,
    )


def row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    sums = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    inverse = np.zeros_like(sums)
    inverse[sums > 0] = 1.0 / sums[sums > 0]
    return sparse.diags(inverse).dot(matrix).tocsr()


def top_sparse_rows(matrix: sparse.csr_matrix, limit: int = 200) -> np.ndarray:
    output = np.full((matrix.shape[0], limit), -1, dtype=np.int32)
    for row_id in range(matrix.shape[0]):
        row = matrix.getrow(row_id)
        keep = row.indices >= FIRST_ITEM
        items, values = row.indices[keep], row.data[keep]
        if not len(items):
            continue
        take = min(limit, len(items))
        partial = np.argpartition(-values, take - 1)[:take]
        order = np.lexsort((items[partial], -values[partial]))
        output[row_id, :take] = items[partial][order]
    return output


def proposal_merge(base: np.ndarray, proposed: np.ndarray, quota: int) -> np.ndarray:
    output = list(map(int, base[:10]))
    seen, original = set(output), set(map(int, base))
    if quota:
        for item in proposed:
            item = int(item)
            if item >= FIRST_ITEM and item not in original and item not in seen:
                output.append(item)
                seen.add(item)
                if len(output) == 10 + quota:
                    break
    for item in base[10:]:
        item = int(item)
        if item not in seen:
            output.append(item)
            seen.add(item)
            if len(output) == len(base):
                break
    if len(output) != len(base) or output[:10] != list(map(int, base[:10])):
        raise ValueError("b2_fixed_budget_merge_failed")
    return np.asarray(output, dtype=np.int32)


def split_masks(rows, seed):
    calibration = np.asarray([
        hashlib.sha256(f"{seed}:{int(row['user'])}".encode()).digest()[0] < 128
        for row in rows
    ], dtype=bool)
    return calibration, ~calibration


def row_recall(candidates, targets):
    return np.asarray([
        len(truth & set(map(int, candidate))) / max(len(truth), 1)
        for candidate, truth in zip(candidates, targets)
    ], dtype=np.float64)


def bootstrap_delta(delta, mask, seed, draws):
    values = delta[mask]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for at in range(draws):
        samples[at] = values[rng.integers(0, len(values), len(values))].mean()
    return {"delta": float(values.mean()), "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975)), "rows": int(len(values))}


def evaluate(base, proposals, source_positions, targets, rows, fit_frequency, mask, quota):
    selected_rows = np.flatnonzero(mask)
    merged = np.asarray([
        proposal_merge(base[row], proposals[source_positions[row]], quota)
        for row in selected_rows
    ], dtype=np.int32)
    selected_targets = [targets[i] for i in selected_rows]
    base_selected = base[selected_rows]
    base_recall = row_recall(base_selected, selected_targets)
    model_recall = row_recall(merged, selected_targets)
    delta_full = np.zeros(len(base), dtype=np.float64)
    delta_full[selected_rows] = model_recall - base_recall
    output: dict[str, Any] = {}
    for scope in SCOPES:
        positions = np.asarray([
            at for at, row in enumerate(selected_rows)
            if scope == "overall" or rows[row]["gap_bucket"] == scope
        ], dtype=np.int64)
        output[scope] = {
            "rows": int(len(positions)),
            "a1_recall_at_200": float(base_recall[positions].mean()),
            "arm_recall_at_200": float(model_recall[positions].mean()),
            "delta": float((model_recall[positions] - base_recall[positions]).mean()),
        }
    residual = recovered = rescued = displaced = 0
    for at, row in enumerate(selected_rows):
        old, new, truth = set(map(int, base[row])), set(map(int, merged[at])), targets[row]
        rescued += len((truth & new) - old)
        displaced += len((truth & old) - new)
        for target in truth:
            if target < len(fit_frequency) and fit_frequency[target] > 0 and target not in old:
                residual += 1
                recovered += int(target in new)
    output["targeted_residual"] = {"train_seen_not_in_a1": residual, "recovered": recovered, "recovery_rate": recovered / residual if residual else 0.0}
    output["coverage"] = {"rescued_targets": rescued, "displaced_targets": displaced, "net_rescued": rescued - displaced}
    return output, delta_full


def choose_quota(base, proposals, source_positions, targets, rows, frequency, calibration, quotas):
    choices = []
    for quota in quotas:
        metrics, _ = evaluate(base, proposals, source_positions, targets, rows, frequency, calibration, quota)
        score = metrics["overall"]["delta"] + metrics[">365d"]["delta"]
        choices.append({"quota": int(quota), "score": score, "metrics": metrics})
    selected = max(choices, key=lambda item: (item["score"], -item["quota"]))
    return int(selected["quota"]), choices


def direct_matrix(neighbors, view, vocab_size):
    ids, sims = neighbors[f"{view}_ids"], neighbors[f"{view}_sim"]
    valid = (ids >= FIRST_ITEM) & np.isfinite(sims) & (sims > 0)
    row_ids = np.repeat(np.arange(len(ids), dtype=np.int32), ids.shape[1])[valid.ravel()]
    return sparse.csr_matrix(
        (sims.ravel()[valid.ravel()], (row_ids, ids.ravel()[valid.ravel()])),
        shape=(len(ids), vocab_size), dtype=np.float32,
    )


def calibrate_transport_view(view, neighbors, transition, base, source_positions, targets, rows, frequency, calibration, args, permutation=None):
    best = None
    records = []
    for k in args.k_grid:
        for tau in args.tau_grid:
            kernel = kernel_matrix(neighbors[f"{view}_ids"], neighbors[f"{view}_sim"], transition.shape[0], k=k, tau=tau, permutation=permutation)
            transported = kernel.dot(transition).tocsr()
            proposals = top_sparse_rows(transported)
            quota, choices = choose_quota(base, proposals, source_positions, targets, rows, frequency, calibration, args.quota_grid)
            selected = next(item for item in choices if item["quota"] == quota)
            score = float(selected["score"])
            record = {"k": k, "tau": tau, "quota": quota, "score": score}
            records.append(record)
            key = (score, -k, -tau, -quota)
            if best is None or key > best[0]:
                best = (key, transported, record)
    return best[1], {"selected": best[2], "grid": records}


def combined_arm(left, right, base, source_positions, targets, rows, frequency, calibration, args):
    records = []
    best = None
    for audio_weight in args.weight_grid:
        proposals = top_sparse_rows((audio_weight * left + (1.0 - audio_weight) * right).tocsr())
        quota, choices = choose_quota(base, proposals, source_positions, targets, rows, frequency, calibration, args.quota_grid)
        selected = next(item for item in choices if item["quota"] == quota)
        record = {"audio_weight": audio_weight, "quota": quota, "score": float(selected["score"])}
        records.append(record)
        key = (float(selected["score"]), -abs(audio_weight - 0.5), -quota)
        if best is None or key > best[0]:
            best = (key, proposals, record)
    return best[1], {"selected": best[2], "grid": records}


def shuffled_permutation(vocab_size, seed):
    output = np.arange(vocab_size, dtype=np.int32)
    catalog = output[FIRST_ITEM:].copy()
    np.random.default_rng(seed).shuffle(catalog)
    output[FIRST_ITEM:] = catalog
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/music4all_onion/a0_candidate_index_v1"))
    parser.add_argument("--cv-base", type=Path, default=Path("refine-logs/music4all_onion/candidate_support_a0/sasrec_train_cv_top200.json.gz"))
    parser.add_argument("--musicnn", type=Path, default=Path("data/music4all_onion/raw/id_musicnn.tsv.bz2"))
    parser.add_argument("--genre", type=Path, default=Path("data/music4all_onion/raw/id_genres_tf-idf.tsv.bz2"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("refine-logs/music4all_onion/component_b2/attainability"))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=129)
    parser.add_argument("--neighbor-depth", type=int, default=200)
    parser.add_argument("--dense-neighbor-batch", type=int, default=128)
    parser.add_argument("--sparse-neighbor-batch", type=int, default=128)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--k-grid", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--tau-grid", type=float, nargs="+", default=[0.1, 0.5])
    parser.add_argument("--weight-grid", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--quota-grid", type=int, nargs="+", default=[0, 5, 10, 20, 40])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("component_b2_attainability_output_exists")

    state, base, targets, observed = frozen_a1_support(args)
    rows, transition, frequency = state["rows"], row_normalize(state["x_fit"]), state["frequency"]
    vocab_size = transition.shape[0]
    calibration, gate = split_masks(rows, args.seed)
    if calibration.sum() < 9000 or gate.sum() < 9000:
        raise ValueError("cv_partition_too_small")
    sources = np.unique(observed)
    source_to_position = {int(source): at for at, source in enumerate(sources)}
    source_positions = np.asarray([source_to_position[int(source)] for source in observed], dtype=np.int32)
    audio, genre, feature_manifest = prepare_features(args, vocab_size)
    neighbors = prepare_neighbors(args, audio, genre, sources, feature_manifest)
    if not np.array_equal(neighbors["sources"], sources):
        raise ValueError("neighbor_source_order_mismatch")

    direct_audio, direct_genre = direct_matrix(neighbors, "audio", vocab_size), direct_matrix(neighbors, "genre", vocab_size)
    direct_proposals, direct_selection = combined_arm(direct_audio, direct_genre, base, source_positions, targets, rows, frequency, calibration, args)
    direct_quota = int(direct_selection["selected"]["quota"])
    direct_metrics, direct_delta = evaluate(base, direct_proposals, source_positions, targets, rows, frequency, gate, direct_quota)

    source_transition = transition[sources]
    diffusion_candidates = []
    for k in args.k_grid:
        truncated = source_transition.copy().tocsr()
        for row_id in range(truncated.shape[0]):
            begin, end = truncated.indptr[row_id:row_id + 2]
            if end - begin > k:
                values = truncated.data[begin:end]
                keep = np.argpartition(-values, k - 1)[:k]
                drop = np.ones(end - begin, dtype=bool)
                drop[keep] = False
                truncated.data[begin:end][drop] = 0
        truncated.eliminate_zeros()
        proposals = top_sparse_rows(row_normalize(truncated).dot(transition).tocsr())
        quota, choices = choose_quota(base, proposals, source_positions, targets, rows, frequency, calibration, args.quota_grid)
        selected = next(item for item in choices if item["quota"] == quota)
        diffusion_candidates.append((float(selected["score"]), -k, proposals, {"k": k, "quota": quota, "grid": choices}))
    _, _, diffusion_proposals, diffusion_selection = max(diffusion_candidates, key=lambda item: item[:2])
    diffusion_metrics, diffusion_delta = evaluate(base, diffusion_proposals, source_positions, targets, rows, frequency, gate, int(diffusion_selection["quota"]))

    audio_transport, audio_selection = calibrate_transport_view("audio", neighbors, transition, base, source_positions, targets, rows, frequency, calibration, args)
    genre_transport, genre_selection = calibrate_transport_view("genre", neighbors, transition, base, source_positions, targets, rows, frequency, calibration, args)
    full_proposals, full_selection = combined_arm(audio_transport, genre_transport, base, source_positions, targets, rows, frequency, calibration, args)
    full_quota = int(full_selection["selected"]["quota"])
    full_metrics, full_delta = evaluate(base, full_proposals, source_positions, targets, rows, frequency, gate, full_quota)

    shuffled_results, shuffled_deltas = [], []
    for shuffle_seed in (2024, 2025, 2026):
        permutation = shuffled_permutation(vocab_size, shuffle_seed)
        shuffled_audio, shuffled_audio_selection = calibrate_transport_view("audio", neighbors, transition, base, source_positions, targets, rows, frequency, calibration, args, permutation=permutation)
        shuffled_genre, shuffled_genre_selection = calibrate_transport_view("genre", neighbors, transition, base, source_positions, targets, rows, frequency, calibration, args, permutation=permutation)
        proposals, selection = combined_arm(shuffled_audio, shuffled_genre, base, source_positions, targets, rows, frequency, calibration, args)
        quota = int(selection["selected"]["quota"])
        metrics, delta = evaluate(base, proposals, source_positions, targets, rows, frequency, gate, quota)
        shuffled_results.append({"seed": shuffle_seed, "audio_selection": shuffled_audio_selection, "genre_selection": shuffled_genre_selection, "mixture_selection": selection, "gate_metrics": metrics})
        shuffled_deltas.append(delta)

    scope_masks = {"overall": gate, ">365d": gate & np.asarray([row["gap_bucket"] == ">365d" for row in rows])}
    paired = {
        "full_vs_a1": {scope: bootstrap_delta(full_delta, mask, args.seed + at, args.bootstrap_draws) for at, (scope, mask) in enumerate(scope_masks.items())},
        "full_vs_direct": {scope: bootstrap_delta(full_delta - direct_delta, mask, args.seed + 10 + at, args.bootstrap_draws) for at, (scope, mask) in enumerate(scope_masks.items())},
        "full_vs_id_diffusion": {scope: bootstrap_delta(full_delta - diffusion_delta, mask, args.seed + 20 + at, args.bootstrap_draws) for at, (scope, mask) in enumerate(scope_masks.items())},
    }
    shuffled_mean = np.mean(np.stack(shuffled_deltas), axis=0)
    paired["full_vs_shuffled_mean"] = {scope: bootstrap_delta(full_delta - shuffled_mean, mask, args.seed + 30 + at, args.bootstrap_draws) for at, (scope, mask) in enumerate(scope_masks.items())}
    shuffled_summary = {scope: {
        "mean_delta_vs_a1": float(np.mean([result["gate_metrics"][scope]["delta"] for result in shuffled_results])),
        "min_delta_vs_a1": float(np.min([result["gate_metrics"][scope]["delta"] for result in shuffled_results])),
        "max_delta_vs_a1": float(np.max([result["gate_metrics"][scope]["delta"] for result in shuffled_results])),
    } for scope in ("overall", ">365d")}
    criteria = {
        "overall_paired_ci95_low_positive": paired["full_vs_a1"]["overall"]["ci95_low"] > 0,
        "long_delta_positive": full_metrics[">365d"]["delta"] > 0,
        "targeted_residual_recovery_at_least_5pct": full_metrics["targeted_residual"]["recovery_rate"] >= 0.05,
        "net_rescued_positive": full_metrics["coverage"]["net_rescued"] > 0,
        "beats_direct_overall_and_long": all(paired["full_vs_direct"][scope]["delta"] > 0 for scope in ("overall", ">365d")),
        "beats_shuffled_mean_overall_and_long": all(paired["full_vs_shuffled_mean"][scope]["delta"] > 0 for scope in ("overall", ">365d")),
    }
    decision = "PASS_LEARNED_A1_PLUS_B_PILOT_UNLOCKED" if all(criteria.values()) else "STOP_THIS_TRANSPORT_OPERATOR"
    result = {
        "schema": SCHEMA, "status": "pass", "decision": decision,
        "surface": "train_only_temporal_cv_gate",
        "partitions": {"calibration_rows": int(calibration.sum()), "gate_rows": int(gate.sum())},
        "predecessor": {"name": "frozen_A1_candidate_support", "construction": "SASRec Top10 + 30 observed-first-track continuation + 10 history continuation + SASRec fill", "budget": 200, "protected_prefix": 10},
        "arms": {
            "direct_content_target_control": {"selection": direct_selection, "gate_metrics": direct_metrics},
            "id_only_transition_diffusion_control": {"selection": diffusion_selection, "gate_metrics": diffusion_metrics},
            "multimodal_transition_transport_B": {"audio_selection": audio_selection, "genre_selection": genre_selection, "mixture_selection": full_selection, "gate_metrics": full_metrics},
            "shuffled_content_transport_control": {"replicates": shuffled_results, "summary": shuffled_summary},
        },
        "paired_user_bootstrap": paired, "criteria": criteria,
        "bindings": {
            "cv_base_sha256": digest(args.cv_base), "index_manifest_sha256": digest(args.index_dir / "index_manifest.json"),
            "feature_cache_manifest_sha256": digest(args.cache_dir / "feature_cache_manifest.json"),
            "neighbor_cache_manifest_sha256": digest(args.cache_dir / "source_neighbors_manifest.json"),
            "musicnn_sha256": digest(args.musicnn), "genre_sha256": digest(args.genre), "producer_sha256": digest(Path(__file__)),
        },
        "interpretation_boundary": "PASS establishes Train-only support attainability only; it is not Top10 gain or a novelty claim.",
        "negative_scope": "A failure closes only the tested content-neighbour transition transport under the frozen A1 support and bounded grids.",
        "validation_accessed": False, "test_accessed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path, decision_path = args.output_dir / "RESULT.json", args.output_dir / "DECISION.json"
    atomic_json(result_path, result)
    atomic_json(decision_path, {"schema": DECISION_SCHEMA, "status": "pass", "decision": decision, "criteria": criteria, "result_sha256": digest(result_path), "validation_accessed": False, "test_accessed": False})
    print(json.dumps({"status": "pass", "decision": decision, "criteria": criteria}, sort_keys=True))


if __name__ == "__main__":
    main()
