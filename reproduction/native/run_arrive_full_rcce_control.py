#!/usr/bin/env python3
"""Full-source controls and shared retained-framework feature export.

One runner exports matched candidate features, fits one common ranker per
seed, and analyzes paired predictions. The legacy control rejects Test;
framework_export supports both datasets and requires a freeze record for Test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import torch
from scipy import sparse

from analyze_arrive_joint_test import paired, user_values
from run_arrive_attribution_controls import FEATURE_NAMES, generic_fit, rrf_pool
from run_arrive_joint_objective import JointARRIVE, joint_universe, select_pool
from run_music4all_candidate_support_a0 import merge_candidates, ranked_history, ranked_sparse_row, recent_anchors
from run_music4all_artist_component_b import artist_members, load_artist_features, load_metadata
from run_music4all_candidate_support_a1 import (atomic_json, atomic_npz, base_topk, digest,
                                                encode_base_states, load_standard_base, stable_topk,
                                                target_lookup, user_partition)
from run_music4all_component_c2_conversion_oof import row_metrics, user_macro


SEEDS = (2024, 2025, 2026)
GRID = ((31, 220), (31, 440), (63, 220), (63, 440))


def parent_paths(project: Path, seed: int, surface: str) -> dict[str, Path]:
    base = project / "runs/paired_20260923/onion"
    checkpoint = base / ("crossfit" if surface == "train_cv" else "sasrec") / f"s{seed}"
    return {
        "feature": base / "features" / f"s{seed}" / "artist" / f"{surface}_features.npz",
        "feature_manifest": base / "features" / f"s{seed}" / "artist" / f"{surface}_features_manifest.json",
        "base_checkpoint": checkpoint / ("final_model.pt" if surface == "train_cv" else "best_model.pt"),
        "base_manifest": checkpoint / "training_manifest.json",
        "ordinary_prediction": base / "heads/ordinary" / f"s{seed}" / "validation_predictions.npz",
        "ordinary_manifest": base / "heads/ordinary" / f"s{seed}" / "training_manifest.json",
    }


def truth_for(data, lookup, row_id: int, surface: str) -> list[int]:
    if surface == "train_cv":
        return list(dict.fromkeys(data.known_suffix(row_id)))
    return list(map(int, lookup[row_id]))


def reconciled_base_order(computed: np.ndarray, raw: dict[str, np.ndarray], row: int,
                          state: np.ndarray, items: np.ndarray) -> list[int]:
    """Restore source ranks recorded by the original CUDA feature export.

    The original PyTorch exact retrieval on CPU can reverse nearly tied ranks.
    Known exact ranks are part of the frozen A0 feature surface; only ranks
    absent from that surface are filled by a fresh catalog retrieval.
    """
    order = [None] * 200
    known = {}
    for candidate_items, ranks in ((raw["candidates"][row], raw["base_ranks"][row]),
                                   (raw["proposal_candidates"][row, 0], raw["proposal_base_ranks"][row, 0])):
        for item, rank in zip(candidate_items, ranks):
            if int(item) >= 3 and 1 <= int(rank) <= 200:
                if int(rank) in known and known[int(rank)] != int(item):
                    raise ValueError("conflicting_original_base_rank")
                known[int(rank)] = int(item)
                order[int(rank) - 1] = int(item)
    computed_set = set(map(int, computed))
    # Original CUDA/CPU score association can differ at the Top-200 edge.
    # Check any missing original candidate against the computed boundary;
    # larger deviations are a checkpoint or retrieval-semantics mismatch.
    # Parent ranks are authoritative where recorded. Batch-shape/platform
    # numerical differences may move an unrecorded tail item across rank 200.
    remainder = iter(int(item) for item in computed if int(item) not in set(known.values()))
    for at, item in enumerate(order):
        if item is None:
            order[at] = next(remainder)
    if len(set(order)) != 200 or not np.array_equal(order[:10], raw["candidates"][row, :10]):
        raise ValueError("reconciled_base_order_invalid")
    return order


def computed_c0(parent_args, data, raw, output: Path):
    row_ids = raw["row_ids"].astype(np.int64)
    binding = {"checkpoint_sha256": digest(parent_args.base_checkpoint),
               "manifest_sha256": digest(parent_args.base_training_manifest),
               "row_ids_sha256": hashlib.sha256(row_ids.tobytes()).hexdigest(),
               "catalog_retrieval": f"original_exact_catalog_topk_{parent_args.device}_parent_rank_stitch_v1"}
    name = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    path = output / "c0_cache" / f"{name}.npz"
    manifest_path = path.with_suffix(".json")
    if path.is_file() and manifest_path.is_file():
        meta = json.loads(manifest_path.read_text())
        if meta.get("binding") != binding or meta.get("sha256") != digest(path):
            raise ValueError("catalog_cache_binding_invalid")
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z["row_ids"], row_ids):
                raise ValueError("catalog_cache_row_mismatch")
            return z["candidates"].copy(), z["states"].copy(), z["items"].copy(), meta
    model, _ = load_standard_base(parent_args, data, checkpoint_path=parent_args.base_checkpoint,
                                  manifest_path=parent_args.base_training_manifest)
    items = model.item_embedding.weight.detach().cpu().numpy().astype(np.float32)
    states = np.empty((len(row_ids), items.shape[1]), dtype=np.float32)
    candidates = np.empty((len(row_ids), 200), dtype=np.int32)
    for start in range(0, len(row_ids), parent_args.eval_batch_size):
        stop = min(start + parent_args.eval_batch_size, len(row_ids))
        with torch.no_grad():
            batch_states, serving = encode_base_states(parent_args, data, model, row_ids[start:stop])
            candidates[start:stop] = base_topk(model, batch_states, serving, k=200).cpu().numpy()
            states[start:stop] = batch_states.cpu().numpy()
        if start == 0 or stop % 2048 == 0 or stop == len(row_ids):
            print(json.dumps({"event": "catalog_retrieval", "rows": stop, "total": len(row_ids)}), flush=True)
    atomic_npz(path, row_ids=row_ids, candidates=candidates, states=states, items=items)
    meta = {"binding": binding, "sha256": digest(path)}
    atomic_json(manifest_path, meta)
    return candidates, states, items, meta


def frozen_rcce_selection(universe: dict, checkpoint_path: Path, manifest_path: Path,
                          seed: int, batch_size: int) -> np.ndarray:
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "pass" or manifest.get("seed") != seed
            or manifest.get("test_accessed") is not False
            or manifest.get("checkpoint_sha256") != digest(checkpoint_path)):
        raise ValueError("frozen_selector_binding_invalid")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("seed") != seed or checkpoint.get("component_variant") != "full"
            or checkpoint.get("configuration", {}).get("training_rule") != "ordinary_multipositive"):
        raise ValueError("frozen_selector_variant_invalid")
    model = JointARRIVE(checkpoint["proposal_width"], checkpoint["behavior_width"],
                        checkpoint["hidden_dim"], "full", checkpoint.get("interaction_mode", "setwise"))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    output = np.empty((len(universe["candidates"]), 200), dtype=np.int32)
    with torch.no_grad():
        for start in range(0, len(output), batch_size):
            stop = min(start + batch_size, len(output))
            raw = torch.as_tensor(universe["raw_scores"][start:stop])
            features = torch.as_tensor(universe["proposal_features"][start:stop])
            valid = torch.as_tensor(universe["valid"][start:stop])
            candidates = torch.as_tensor(universe["candidates"][start:stop])
            scores = model.proposal_scores(raw, features).masked_fill(~valid, float("-inf"))
            chosen = select_pool(scores, valid, candidates)
            output[start:stop] = torch.gather(candidates, 1, chosen).numpy().astype(np.int32)
    return output


def static_feature_rows(frequency: np.ndarray, artists: np.ndarray, audio: np.ndarray,
                        genre: sparse.csr_matrix, tags: sparse.csr_matrix) -> np.ndarray:
    """Catalog-versioned columns shared by every request; no user or cue input."""
    rows = np.zeros((len(artists), len(FEATURE_NAMES)), dtype=np.float32)
    rows[:, 6] = np.log1p(frequency)
    rows[:, 18] = artists >= 0
    rows[:, 20] = np.linalg.norm(audio, axis=1) > 0
    rows[:, 23] = genre.getnnz(axis=1) > 0
    rows[:, 26] = tags.getnnz(axis=1) > 0
    return rows


def row_feature(ids: np.ndarray, state: np.ndarray, items: np.ndarray, source_maps: list[dict[int, int]],
                gap: float, bucket: int, frequency: np.ndarray, observed: int, history: list[int],
                artists: np.ndarray, tags: sparse.csr_matrix, audio: np.ndarray,
                genre: sparse.csr_matrix, normalized_base_score: np.ndarray | None = None,
                precomputed_history: tuple[np.ndarray, sparse.csr_matrix] | None = None,
                catalog_static_rows: np.ndarray | None = None,
                profile_seconds: dict[str, float] | None = None,
                source_ranks: np.ndarray | None = None,
                dense_history_genre: bool = False) -> np.ndarray:
    # Optional measurement only; production and training callers do not enter
    # the timing path.  The same arrays/operations are used on both paths.
    from time import perf_counter
    tick = perf_counter() if profile_seconds is not None else 0.0
    def mark(name: str) -> None:
        nonlocal tick
        if profile_seconds is not None:
            now = perf_counter()
            profile_seconds[name] = profile_seconds.get(name, 0.0) + now - tick
            tick = now
    ids = np.asarray(ids, dtype=np.int32)
    if normalized_base_score is None:
        raw = items[ids] @ state
        z = (raw - raw.mean()) / max(float(raw.std()), 1e-6)
    else:
        z = np.asarray(normalized_base_score)
        if z.shape != (len(ids),) or not np.isfinite(z).all():
            raise ValueError("normalized_base_score_invalid")
    if source_ranks is None:
        ranks = np.asarray([[m.get(int(item), 0) for m in source_maps] for item in ids], dtype=np.float32)
    else:
        ranks = np.asarray(source_ranks, dtype=np.float32)
        if ranks.shape != (len(ids), len(source_maps)) or not np.isfinite(ranks).all():
            raise ValueError("source_ranks_invalid")
    rrf = np.where(ranks > 0, 1.0 / (60.0 + np.maximum(ranks, 1.0)), 0.0).sum(axis=1)
    rr = np.where(ranks > 0, 1.0 / np.log2(np.maximum(ranks, 1.0) + 1.0), 0.0)
    mark("base_and_source_ranks")
    a_obs = np.asarray(audio[observed])
    a_hist = (precomputed_history[0] if precomputed_history is not None else
              np.asarray(audio[history]).mean(axis=0) if history else np.zeros_like(a_obs))
    g_obs = genre.getrow(observed)
    if precomputed_history is not None:
        g_hist = precomputed_history[1]
    elif dense_history_genre:
        # SciPy's sparse mean already produces a dense row. Keep its exact
        # values and avoid converting it back to CSR before the candidate dot.
        g_hist = (np.asarray(genre[history].mean(axis=0)).ravel() if history
                  else np.zeros(genre.shape[1], dtype=genre.dtype))
    else:
        g_hist = (sparse.csr_matrix(genre[history].mean(axis=0)) if history
                  else sparse.csr_matrix((1, genre.shape[1])))
    hist_genre_nnz = (int(np.count_nonzero(g_hist)) if isinstance(g_hist, np.ndarray)
                      else g_hist.nnz)
    mark("return_and_history_context")
    a_cand = np.asarray(audio[ids])
    g_cand = genre[ids]
    t_cand = tags[ids]
    mark("candidate_content_gather")
    same_artist = ((artists[ids] == artists[observed]) & (artists[observed] >= 0)).astype(np.float32)
    tag_cos = t_cand.dot(tags.getrow(observed).T).toarray().ravel()
    mark("artist_and_tag_similarity")
    audio_obs = a_cand @ a_obs
    audio_hist = a_cand @ a_hist
    mark("audio_similarity")
    genre_obs = g_cand.dot(g_obs.T).toarray().ravel()
    genre_hist = (g_cand.dot(g_hist) if isinstance(g_hist, np.ndarray)
                  else g_cand.dot(g_hist.T).toarray().ravel())
    mark("genre_similarity")
    if catalog_static_rows is not None:
        if catalog_static_rows.shape != (len(artists), len(FEATURE_NAMES)):
            raise ValueError("catalog_static_feature_shape_invalid")
        matrix = catalog_static_rows[ids].copy()
        matrix[:, 0] = z
        matrix[:, 1] = rrf
        matrix[:, 2:6] = rr
        matrix[:, 7] = np.log1p(gap) / 8.0
        matrix[:, 8 + bucket] = 1.0
        matrix[:, 12] = same_artist
        matrix[:, 13] = tag_cos
        matrix[:, 14] = audio_obs
        matrix[:, 15] = audio_hist
        matrix[:, 16] = genre_obs
        matrix[:, 17] = genre_hist
        matrix[:, 19] = artists[observed] >= 0
        matrix[:, 21] = np.linalg.norm(a_obs) > 0
        matrix[:, 22] = np.linalg.norm(a_hist) > 0
        matrix[:, 24] = g_obs.nnz > 0
        matrix[:, 25] = hist_genre_nnz > 0
        matrix[:, 27] = tags.getrow(observed).nnz > 0
    else:
        matrix = np.column_stack((
        z, rrf, rr, np.log1p(frequency[ids]), np.full(len(ids), np.log1p(gap) / 8.0),
        np.tile(np.eye(4, dtype=np.float32)[bucket], (len(ids), 1)), same_artist, tag_cos,
        audio_obs, audio_hist,
        genre_obs, genre_hist,
        artists[ids] >= 0, np.full(len(ids), artists[observed] >= 0),
        np.linalg.norm(a_cand, axis=1) > 0, np.full(len(ids), np.linalg.norm(a_obs) > 0),
        np.full(len(ids), np.linalg.norm(a_hist) > 0), g_cand.getnnz(axis=1) > 0,
        np.full(len(ids), g_obs.nnz > 0), np.full(len(ids), hist_genre_nnz > 0),
        t_cand.getnnz(axis=1) > 0, np.full(len(ids), tags.getrow(observed).nnz > 0),
        )).astype(np.float32)
    mark("feature_matrix_assembly")
    if matrix.shape != (len(ids), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError("common_candidate_feature_invalid")
    mark("feature_validation")
    return matrix


def _paired_content_dot(indptr, indices, values, ids, cue_indptr, cue_indices,
                        cue_values, history, cue_out, history_out):
    """CSR row dots, preserving stored candidate-entry accumulation order."""
    for row in range(len(ids)):
        cue_start, cue_stop = cue_indptr[row], cue_indptr[row + 1]
        for col in range(ids.shape[1]):
            item = ids[row, col]
            for edge in range(indptr[item], indptr[item + 1]):
                key = indices[edge]
                left, right = cue_start, cue_stop
                while left < right:
                    mid = (left + right) // 2
                    if cue_indices[mid] < key:
                        left = mid + 1
                    else:
                        right = mid
                if left < cue_stop and cue_indices[left] == key:
                    cue_out[row, col] += values[edge] * cue_values[left]
                if history.shape[1]:
                    history_out[row, col] += values[edge] * history[row, key]


_CONTENT_DOT_KERNEL = None


def batch_row_features(ids, states, items, source_ranks, gaps, buckets, frequency,
                       observed, histories, artists, tags, audio, genre,
                       normalized_base_scores, catalog_static_rows):
    """Opt-in exact feature batching for the existing 28-column contract.

    Catalog/state construction stays outside this helper. Request-dependent
    sparse dots and matrix assembly are batched. Audio BLAS calls retain
    their row shape, since changing their reduction can change tree routes.
    """
    global _CONTENT_DOT_KERNEL
    ids, observed = np.asarray(ids), np.asarray(observed)
    ranks = np.asarray(source_ranks, dtype=np.float32)
    b, width = ids.shape
    if (len(observed) != b or len(histories) != b or ranks.shape != (b, width, 4)
            or np.shape(normalized_base_scores) != (b, width)
            or np.shape(catalog_static_rows) != (len(artists), len(FEATURE_NAMES))
            or not np.isfinite(ranks).all() or (ranks < 0).any()
            or np.shape(gaps) != (b,) or np.shape(buckets) != (b,)
            or np.any(np.asarray(buckets) < 0) or np.any(np.asarray(buckets) > 3)
            or np.any(ids < 0) or np.any(ids >= len(artists))
            or np.any(observed < 0) or np.any(observed >= len(artists))):
        raise ValueError("batch_feature_input_invalid")
    cue_tags, cue_genre = tags[observed].copy(), genre[observed].copy()
    cue_tags.sort_indices()
    cue_genre.sort_indices()
    if not (cue_tags.has_canonical_format and cue_genre.has_canonical_format):
        raise ValueError("batch_features_require_duplicate_free_cue_rows")
    if _CONTENT_DOT_KERNEL is None:
        from numba import njit
        _CONTENT_DOT_KERNEL = njit(_paired_content_dot)  # no fastmath/reassociation
    audio = np.asarray(audio)
    cue_audio = audio[observed]
    history_audio = np.stack([audio[h].mean(axis=0) if len(h) else np.zeros_like(cue_audio[r])
                              for r, h in enumerate(histories)])
    history_genre = np.stack([np.asarray(genre[h].mean(axis=0)).ravel() if len(h)
                              else np.zeros(genre.shape[1], dtype=genre.dtype) for h in histories])
    candidates_audio = audio[ids]
    tag_cue = np.zeros((b, width), dtype=tags.dtype)
    genre_cue = np.zeros((b, width), dtype=genre.dtype)
    genre_history = np.zeros((b, width), dtype=history_genre.dtype)
    _CONTENT_DOT_KERNEL(tags.indptr, tags.indices, tags.data, ids,
                        cue_tags.indptr, cue_tags.indices, cue_tags.data,
                        np.empty((b, 0), dtype=tags.dtype), tag_cue, np.empty_like(tag_cue))
    _CONTENT_DOT_KERNEL(genre.indptr, genre.indices, genre.data, ids,
                        cue_genre.indptr, cue_genre.indices, cue_genre.data,
                        history_genre, genre_cue, genre_history)
    matrix = catalog_static_rows[ids].copy()
    matrix[:, :, 0] = normalized_base_scores
    matrix[:, :, 1] = np.where(ranks > 0, 1.0 / (60.0 + np.maximum(ranks, 1.0)), 0.0).sum(axis=2)
    matrix[:, :, 2:6] = np.where(ranks > 0, 1.0 / np.log2(np.maximum(ranks, 1.0) + 1.0), 0.0)
    matrix[:, :, 7] = (np.log1p(np.asarray(gaps, dtype=np.float64)) / 8.0)[:, None]
    matrix[np.arange(b), :, 8 + np.asarray(buckets, dtype=int)] = 1.0
    matrix[:, :, 12] = (artists[ids] == artists[observed, None]) & (artists[observed, None] >= 0)
    matrix[:, :, 13] = tag_cue
    for row in range(b):
        matrix[row, :, 14] = candidates_audio[row] @ cue_audio[row]
        matrix[row, :, 15] = candidates_audio[row] @ history_audio[row]
        matrix[row, :, 21] = np.linalg.norm(cue_audio[row]) > 0
        matrix[row, :, 22] = np.linalg.norm(history_audio[row]) > 0
    matrix[:, :, 16] = genre_cue
    matrix[:, :, 17] = genre_history
    matrix[:, :, 19] = (artists[observed] >= 0)[:, None]
    matrix[:, :, 24] = (np.diff(genre.indptr)[observed] > 0)[:, None]
    matrix[:, :, 25] = (np.count_nonzero(history_genre, axis=1) > 0)[:, None]
    matrix[:, :, 27] = (np.diff(tags.indptr)[observed] > 0)[:, None]
    if not np.isfinite(matrix).all():
        raise ValueError("common_candidate_feature_invalid")
    return matrix


def export(args: argparse.Namespace) -> None:
    if args.surface not in ("train_cv", "validation"):
        raise ValueError("test_closed")
    destination = args.output / args.surface
    if destination.exists():
        raise FileExistsError(destination)
    paths = parent_paths(args.project, args.seed, args.surface)
    raw, raw_manifest = load_artist_features(paths["feature"], paths["feature_manifest"], args.surface)
    ordinary = json.loads(paths["ordinary_manifest"].read_text())
    if ordinary.get("status") != "pass" or ordinary.get("seed") != args.seed or ordinary.get("test_accessed") is not False:
        raise ValueError("ordinary_parent_invalid")
    if ordinary["binding"]["train_feature_sha256" if args.surface == "train_cv" else "validation_feature_sha256"] != digest(paths["feature"]):
        raise ValueError("ordinary_feature_binding_mismatch")
    if ordinary["binding"]["train_base_checkpoint_sha256" if args.surface == "train_cv" else "serving_base_checkpoint_sha256"] != digest(paths["base_checkpoint"]):
        raise ValueError("ordinary_checkpoint_binding_mismatch")
    if args.limit:
        raw = {key: value[:args.limit] if len(value) == len(raw["row_ids"]) else value for key, value in raw.items()}
    from run_music4all_candidate_support_a1 import load_base_data
    parent_args = SimpleNamespace(
        data_dir=args.project / "data/music4all_onion/prepared_v1",
        event_times=args.project / "data/music4all_onion/baseline_features_v1/event_times.npy",
        ssd4rec_source=args.project / "vendor/ssd4rec_official/ssd4rec.py",
        muse_counts=args.project / "data/music4all_onion/main_table_repair_v1/muse_transition_counts.npz",
        base_checkpoint=paths["base_checkpoint"], base_training_manifest=paths["base_manifest"],
        candidate_cache_dir=args.output / "c0_cache", output=args.output / "unused",
        max_context=129, horizon=64, eval_batch_size=args.batch_size, device=args.device,
    )
    data = load_base_data(parent_args, parent_args.base_training_manifest)
    c0, states, items, c0_identity = computed_c0(parent_args, data, raw, args.output)
    original = joint_universe(raw)
    index_dir = args.project / "data/music4all_onion/a0_candidate_index_v1"
    index_manifest = json.loads((index_dir / "index_manifest.json").read_text())
    suffix = "fit" if args.surface == "train_cv" else "full"
    for name in (f"x_{suffix}.npz", f"h_{suffix}.npz", f"{suffix}_target_frequency.npy"):
        if digest(index_dir / name) != index_manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"train_index_hash_mismatch:{name}")
    x = sparse.load_npz(index_dir / f"x_{suffix}.npz").tocsr()
    h = sparse.load_npz(index_dir / f"h_{suffix}.npz").tocsr()
    frequency = np.load(index_dir / f"{suffix}_target_frequency.npy", allow_pickle=False)
    train_frequency = np.load(parent_args.data_dir / "train_frequencies.npy", mmap_mode="r")
    popularity = [int(i) for i in np.argsort(-train_frequency, kind="stable") if i >= 3]
    content = args.project / "data/music4all_onion/component_b2_attainability_v1"
    audio = np.load(content / "musicnn_normalized.npy", mmap_mode="r")
    genre = sparse.load_npz(content / "genre_normalized.npz").tocsr()
    meta_args = SimpleNamespace(data_dir=parent_args.data_dir,
                                tags=args.project / "data/music4all_onion/raw/id_tags.csv",
                                information=args.project / "data/music4all_onion/raw/id_information.csv")
    tags, artists, _ = load_metadata(meta_args, len(items))
    members = artist_members(artists, frequency)
    lookup = None if args.surface == "train_cv" else target_lookup(parent_args.data_dir)
    if args.surface == "train_cv":
        selected = frozen_rcce_selection(original,
                                         args.project / "runs/paired_20260923/onion/heads/ordinary" / f"s{args.seed}" / "joint_arrive_checkpoint.pt",
                                         paths["ordinary_manifest"], args.seed, args.batch_size)
    else:
        if ordinary.get("predictions_sha256") != digest(paths["ordinary_prediction"]):
            raise ValueError("ordinary_prediction_hash_mismatch")
        with np.load(paths["ordinary_prediction"], allow_pickle=False) as z:
            if not np.array_equal(raw["row_ids"], z["row_ids"][:len(raw["row_ids"])]):
                raise ValueError("ordinary_prediction_row_mismatch")
            # The persisted Validation output is the exact served C200.
            # Replaying the head on CPU can reassociate float32 near-ties.
            selected = z["candidate_top200"][:len(raw["row_ids"])].astype(np.int32)
    feature_rows, ids_rows, label_rows = [], [], []
    selected_indices, rrf_indices = [], []
    targets = np.full((len(raw["row_ids"]), int(raw["target_counts"].max())), -1, dtype=np.int32)
    offsets = [0]
    x_cache: dict[int, list[int]] = {}
    source_exclusive_hits = np.zeros((len(raw["row_ids"]), 4), dtype=np.int16)
    source_additional_vs_base = np.zeros((len(raw["row_ids"]), 4), dtype=np.int16)
    pool_overlap = np.empty(len(raw["row_ids"]), dtype=np.int16)
    retrieval_missing_parent_base = np.zeros(len(raw["row_ids"]), dtype=np.int16)
    retrieval_max_boundary_gap = np.zeros(len(raw["row_ids"]), dtype=np.float32)
    retrieval_parent_rank_swaps = np.zeros(len(raw["row_ids"]), dtype=np.int16)
    for at, rid in enumerate(raw["row_ids"]):
        session = data.sessions[int(rid)]
        observed = int(data.events[int(session[2])])
        history = recent_anchors(data.events, session, 5)
        if observed not in x_cache:
            x_cache[observed] = ranked_sparse_row(x, observed, frequency, 0.0, 200)
        known_by_rank = {}
        for candidate_items, ranks in ((raw["candidates"][at], raw["base_ranks"][at]),
                                       (raw["proposal_candidates"][at, 0], raw["proposal_base_ranks"][at, 0])):
            known_by_rank.update((int(rank), int(item)) for item, rank in zip(candidate_items, ranks)
                                 if int(item) >= 3 and 1 <= int(rank) <= 200)
        missing = set(known_by_rank.values()) - set(map(int, c0[at]))
        retrieval_missing_parent_base[at] = len(missing)
        retrieval_parent_rank_swaps[at] = sum(int(c0[at, rank - 1]) != item
                                              for rank, item in known_by_rank.items())
        if missing:
            threshold = float(items[int(c0[at, -1])] @ states[at])
            retrieval_max_boundary_gap[at] = max(abs(float(items[item] @ states[at]) - threshold) for item in missing)
        base_order = reconciled_base_order(c0[at], raw, at, states[at], items)
        full_artist_order = [int(i) for i in members.get(int(artists[observed]), []) if int(i) != observed]
        sources = [base_order, x_cache[observed],
                   ranked_history(h, history, frequency, 0.0, 200), full_artist_order[:200]]
        merged = merge_candidates(sources[0], sources[1], sources[2], popularity,
                                  relation_slots=40, x_fraction=.75, budget=200)
        if not np.array_equal(merged, raw["candidates"][at]):
            raise ValueError(f"upstream_a0_candidate_parity_failed:{args.surface}:{at}")
        disjoint_artist = [i for i in full_artist_order if i not in set(merged)][:20]
        recorded = raw["proposal_candidates"][at, 0]
        if not np.array_equal(np.asarray(disjoint_artist, dtype=np.int32), recorded[recorded >= 3]):
            raise ValueError(f"upstream_artist_candidate_parity_failed:{args.surface}:{at}")
        rrf_ids, _, _ = rrf_pool(sources, budget=200)
        served = selected[at]
        if len(set(map(int, served))) != len(served) or not np.array_equal(served[:10], sources[0][:10]):
            raise ValueError("rcce_candidate_validity_invalid")
        if not set(map(int, served)).issubset(set(map(int, original["candidates"][at, original["valid"][at]]))):
            raise ValueError("rcce_selected_outside_parent_universe")
        pool_overlap[at] = len(set(map(int, served)) & set(map(int, rrf_ids)))
        # Keep the same U220-plus-RRF feature context at fit and Validation;
        # the served RCCE subset is only a mask at inference.
        universe = list(map(int, original["candidates"][at, original["valid"][at]]))
        seen = set(universe)
        for item in rrf_ids:
            if int(item) not in seen:
                universe.append(int(item)); seen.add(int(item))
        ids = np.asarray(universe, dtype=np.int32)
        position = {int(item): j for j, item in enumerate(ids)}
        selected_indices.append(np.asarray([position[int(item)] for item in served], dtype=np.int16))
        rrf_indices.append(np.asarray([position[int(item)] for item in rrf_ids], dtype=np.int16))
        truth = truth_for(data, lookup, int(rid), args.surface)
        if len(truth) != int(raw["target_counts"][at]):
            raise ValueError("full_target_denominator_mismatch")
        targets[at, :len(truth)] = truth
        truth_set = set(truth)
        label_rows.append(np.asarray([int(int(item) in truth_set) for item in ids], dtype=np.uint8))
        maps = [{int(item): rank + 1 for rank, item in enumerate(source)} for source in sources]
        source_sets = [set(source) for source in sources]
        source_exclusive_hits[at] = [len((truth_set & source_sets[j]) - set().union(*(source_sets[k] for k in range(4) if k != j)))
                                     for j in range(4)]
        source_additional_vs_base[at] = [len((truth_set & source_sets[j]) - source_sets[0]) for j in range(4)]
        feature_rows.append(row_feature(ids, states[at], items, maps, float(raw["gaps"][at]),
                                        int(raw["bucket_ids"][at]), frequency, observed, history,
                                        artists, tags, audio, genre))
        ids_rows.append(ids)
        offsets.append(offsets[-1] + len(ids))
        if at == 0 or (at + 1) % 2000 == 0 or at + 1 == len(raw["row_ids"]):
            print(json.dumps({"event": "source_export", "seed": args.seed, "surface": args.surface,
                              "rows": at + 1, "total": len(raw["row_ids"])}), flush=True)
    destination.mkdir(parents=True)
    np.save(destination / "features.npy", np.concatenate(feature_rows), allow_pickle=False)
    np.save(destination / "candidate_ids.npy", np.concatenate(ids_rows), allow_pickle=False)
    np.save(destination / "labels.npy", np.concatenate(label_rows), allow_pickle=False)
    atomic_npz(destination / "metadata.npz", row_ids=raw["row_ids"], users=raw["users"],
               bucket_ids=raw["bucket_ids"], target_counts=raw["target_counts"], targets=targets,
               offsets=np.asarray(offsets, dtype=np.int64), selected_indices=np.stack(selected_indices),
               rrf_indices=np.stack(rrf_indices), source_exclusive_hits=source_exclusive_hits,
               source_additional_vs_base=source_additional_vs_base, pool_overlap=pool_overlap,
               retrieval_missing_parent_base=retrieval_missing_parent_base,
               retrieval_max_boundary_gap=retrieval_max_boundary_gap,
               retrieval_parent_rank_swaps=retrieval_parent_rank_swaps)
    atomic_json(destination / "EXPORT_MANIFEST.json", {
        "schema": "arrive.full_rcce_control.export.v1", "status": "pass", "surface": args.surface,
        "seed": args.seed, "rows": len(raw["row_ids"]), "features": FEATURE_NAMES,
        "training_policy": "joint_U220_plus_full_source_RRF_C200" if args.surface == "train_cv" else "frozen_C200_vs_RRF_C200",
        "test_accessed": False, "limit": args.limit, "c0_identity": c0_identity,
        "retrieval_parent_disagreement": {
            "platform": args.device,
            "affected_rows": int((retrieval_parent_rank_swaps > 0).sum()),
            "rank_swaps": int(retrieval_parent_rank_swaps.sum()),
            "missing_recorded_items": int(retrieval_missing_parent_base.sum()),
            "max_boundary_score_gap": float(retrieval_max_boundary_gap.max()),
        },
        "bindings": {name: digest(path) for name, path in paths.items() if path.is_file()},
        "index_manifest_sha256": digest(index_dir / "index_manifest.json"),
        "content_sha256": {name: digest(content / name) for name in ("musicnn_normalized.npy", "genre_normalized.npz")},
        "output_sha256": {name: digest(destination / name) for name in
                          ("features.npy", "candidate_ids.npy", "labels.npy", "metadata.npz")},
        "producer_sha256": digest(Path(__file__)),
    })


def load_export(path: Path, seed: int, surface: str):
    manifest = json.loads((path / "EXPORT_MANIFEST.json").read_text())
    if manifest.get("status") != "pass" or manifest.get("seed") != seed or manifest.get("surface") != surface or manifest.get("test_accessed") is not False:
        raise ValueError("export_manifest_invalid")
    for name, expected in manifest["output_sha256"].items():
        if digest(path / name) != expected:
            raise ValueError(f"export_hash_mismatch:{name}")
    with np.load(path / "metadata.npz", allow_pickle=False) as z:
        meta = {key: z[key] for key in z.files}
    features = np.load(path / "features.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(path / "candidate_ids.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(path / "labels.npy", mmap_mode="r", allow_pickle=False)
    if (len(features) != len(ids) or len(ids) != len(labels) or len(ids) != int(meta["offsets"][-1])
            or features.shape[1] != len(FEATURE_NAMES)):
        raise ValueError("export_shape_invalid")
    return features, ids, labels, meta, manifest


def predict(model, features, ids, meta, rows: np.ndarray, arm: str):
    output = np.empty((len(rows), 10), dtype=np.int32)
    for j, row in enumerate(rows):
        start, stop = map(int, meta["offsets"][row:row + 2])
        values = model.predict(features[start:stop], num_threads=8)
        positions = meta[f"{arm}_indices"][row]
        if arm == "selected" and len(positions) > 200:
            # Train's full U220 is a fitting union, not a served RCCE set.
            raise ValueError("train_rcce_serving_not_defined")
        output[j] = stable_topk(values[positions][None, :], ids[start:stop][positions][None, :], 10)[0]
    return output


def fit(args: argparse.Namespace) -> None:
    if args.limit:
        raise ValueError("limited_smoke_cannot_fit_formal_model")
    if (args.output / "training_manifest.json").exists():
        raise FileExistsError("completed_fit_exists")
    train_f, train_ids, train_y, train, train_m = load_export(args.output / "train_cv", args.seed, "train_cv")
    val_f, val_ids, _, val, val_m = load_export(args.output / "validation", args.seed, "validation")
    fit_mask = user_partition(train["users"])
    has_positive = np.asarray([np.any(train_y[int(a):int(b)]) for a, b in zip(train["offsets"][:-1], train["offsets"][1:])])
    fit_mask &= has_positive
    calibration = np.flatnonzero(~user_partition(train["users"]))
    if set(map(int, train["users"][fit_mask])) & set(map(int, train["users"][calibration])):
        raise ValueError("fit_calibration_user_overlap")
    matrices = [train_f[int(a):int(b)] for a, b in zip(train["offsets"][:-1], train["offsets"][1:])]
    labels = [train_y[int(a):int(b)] for a, b in zip(train["offsets"][:-1], train["offsets"][1:])]
    records, best_key, best_model = [], None, None
    for leaves, trees in GRID:
        model = generic_fit(matrices, labels, fit_mask, args.seed + 1000, trees, leaves)
        # Select capacity on the label-independent RRF arm only. The frozen
        # RCCE selector was already fit on these Train-CV rows, so using its
        # calibration score would advantage RCCE through in-sample selection.
        arm_scores = {}
        for arm in ("selected", "rrf"):
            ranked = predict(model.booster_, train_f, train_ids, train, calibration, arm)
            _, ndcg = row_metrics(ranked, train["targets"][calibration], train["target_counts"][calibration])
            arm_scores[arm] = user_macro(ndcg, train["users"][calibration], np.ones(len(calibration), bool))
        score = arm_scores["rrf"]
        records.append({"leaves": leaves, "trees": trees, "train_calibration_ndcg": score,
                        "train_calibration_arm_ndcg": arm_scores})
        key = (score, -trees, -leaves)
        if best_key is None or key > best_key:
            best_key, best_model = key, model
        print(json.dumps({"event": "train_calibration", "seed": args.seed, **records[-1]}), flush=True)
    model_path = args.output / "lambdamart_model.txt"
    best_model.booster_.save_model(str(model_path))
    model = lgb.Booster(model_file=str(model_path))
    rows = np.arange(len(val["row_ids"]))
    tops = {arm: predict(model, val_f, val_ids, val, rows, arm) for arm in ("selected", "rrf")}
    row_values = {arm: row_metrics(top, val["targets"], val["target_counts"]) for arm, top in tops.items()}
    pool_values = {}
    for arm in ("selected", "rrf"):
        values = np.empty(len(rows), dtype=np.float64)
        for at in rows:
            a, b = map(int, val["offsets"][at:at + 2])
            pool = val_ids[a:b][val[f"{arm}_indices"][at]]
            values[at] = sum(int(item in set(val["targets"][at, :int(val["target_counts"][at])])) for item in pool) / max(int(val["target_counts"][at]), 1)
        pool_values[arm] = values
    prediction_path = args.output / "validation_predictions.npz"
    atomic_npz(prediction_path, row_ids=val["row_ids"], users=val["users"], bucket_ids=val["bucket_ids"],
               target_counts=val["target_counts"], targets=val["targets"],
               selected_top10=tops["selected"], rrf_top10=tops["rrf"],
               selected_recall=row_values["selected"][0], rrf_recall=row_values["rrf"][0],
               selected_ndcg=row_values["selected"][1], rrf_ndcg=row_values["rrf"][1],
               selected_pool_recall=pool_values["selected"], rrf_pool_recall=pool_values["rrf"],
               source_exclusive_hits=val["source_exclusive_hits"],
               source_additional_vs_base=val["source_additional_vs_base"],
               pool_overlap=val["pool_overlap"])
    metric = {arm: {"candidate_recall_at_200": user_macro(pool_values[arm], val["users"], np.ones(len(rows), bool)),
                    "recall_at_10": user_macro(row_values[arm][0], val["users"], np.ones(len(rows), bool)),
                    "ndcg_at_10": user_macro(row_values[arm][1], val["users"], np.ones(len(rows), bool))}
              for arm in ("selected", "rrf")}
    atomic_json(args.output / "training_manifest.json", {
        "schema": "arrive.full_rcce_control.v1", "status": "pass", "seed": args.seed,
        "test_accessed": False, "train_rows": len(train["row_ids"]), "validation_rows": len(rows),
        "fit_rows": int(fit_mask.sum()), "calibration_rows": int(len(calibration)),
        "grid": records, "selected_config": max(records, key=lambda x: (x["train_calibration_ndcg"], -x["trees"], -x["leaves"])),
        "train_export_sha256": digest(args.output / "train_cv/EXPORT_MANIFEST.json"),
        "validation_export_sha256": digest(args.output / "validation/EXPORT_MANIFEST.json"),
        "model_sha256": digest(model_path), "prediction_sha256": digest(prediction_path),
        "producer_sha256": digest(Path(__file__)), "metrics": metric,
    })
    print(json.dumps({"event": "fit_complete", "seed": args.seed, "metrics": metric}), flush=True)


def analyze(args: argparse.Namespace) -> None:
    records, differences = [], {key: [] for key in ("pool_recall", "recall", "ndcg")}
    users = buckets = row_ids = targets = counts = None
    for seed in SEEDS:
        path = args.output.parent / f"s{seed}"
        manifest = json.loads((path / "training_manifest.json").read_text())
        if manifest.get("status") != "pass" or manifest.get("seed") != seed or manifest.get("test_accessed") is not False:
            raise ValueError("seed_manifest_invalid")
        for key, filename in (("model_sha256", "lambdamart_model.txt"), ("prediction_sha256", "validation_predictions.npz")):
            if manifest[key] != digest(path / filename):
                raise ValueError(f"seed_artifact_hash_mismatch:{seed}:{key}")
        with np.load(path / "validation_predictions.npz", allow_pickle=False) as z:
            if users is None:
                users, buckets = z["users"].copy(), z["bucket_ids"].copy()
                row_ids, targets, counts = z["row_ids"].copy(), z["targets"].copy(), z["target_counts"].copy()
            elif (not np.array_equal(users, z["users"]) or not np.array_equal(buckets, z["bucket_ids"])
                  or not np.array_equal(row_ids, z["row_ids"]) or not np.array_equal(targets, z["targets"])
                  or not np.array_equal(counts, z["target_counts"])):
                raise ValueError("cross_seed_row_identity_mismatch")
            for arm in ("selected", "rrf"):
                r, n = row_metrics(z[f"{arm}_top10"], z["targets"], z["target_counts"])
                if not np.array_equal(r, z[f"{arm}_recall"]) or not np.array_equal(n, z[f"{arm}_ndcg"]):
                    raise ValueError("row_metric_recomputation_mismatch")
            for key in differences:
                differences[key].append(z[f"selected_{key}"] - z[f"rrf_{key}"])
        records.append({"seed": seed, "metrics": manifest["metrics"],
                        "manifest_sha256": digest(path / "training_manifest.json")})
    comparison = {}
    for j, (key, values) in enumerate(differences.items()):
        delta = np.stack(values)
        per_user = user_values(delta.mean(axis=0), users, np.ones(len(users), bool))
        comparison[key] = {
            "overall": paired(delta, users, np.ones(len(users), bool), 97900 + j),
            "over_365d": paired(delta, users, buckets == 3, 98000 + j),
            "seed_deltas": [record["metrics"]["selected"]["candidate_recall_at_200" if key == "pool_recall" else f"{key}_at_10"] -
                            record["metrics"]["rrf"]["candidate_recall_at_200" if key == "pool_recall" else f"{key}_at_10"] for record in records],
            "improved_users": int((per_user > 1e-12).sum()), "harmed_users": int((per_user < -1e-12).sum()),
        }
    source_summary = []
    for seed in SEEDS:
        with np.load(args.output.parent / f"s{seed}" / "validation_predictions.npz", allow_pickle=False) as z:
            source_summary.append({
                "seed": seed,
                "exclusive_target_hits_total": z["source_exclusive_hits"].sum(axis=0).astype(int).tolist(),
                "additional_target_hits_outside_base_total": z["source_additional_vs_base"].sum(axis=0).astype(int).tolist(),
                "mean_selected_rrf_overlap_at_200": float(z["pool_overlap"].mean()),
                "changed_pool_requests": int((z["pool_overlap"] < 200).sum()),
            })
    atomic_json(args.output / "RESULTS.json", {"schema": "arrive.full_rcce_control.analysis.v1", "status": "pass",
                                            "test_accessed": False, "seeds": records, "comparison": comparison,
                                            "source_summary": source_summary})
    print(json.dumps({"event": "analysis_complete", "comparison": comparison}), flush=True)



def finish_npy(array, rows):
    """Finalize the existing payload in place; no second feature-file write."""
    import io
    if not 0 < rows <= len(array):
        raise ValueError("invalid_export_payload_length")
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {
        "descr":np.lib.format.dtype_to_descr(array.dtype), "fortran_order":False,
        "shape":(rows,*array.shape[1:])})
    value = header.getvalue()
    if len(value) != array.offset:
        raise ValueError("export_header_offset_changed")
    size = array.offset + rows * int(np.prod(array.shape[1:])) * array.dtype.itemsize
    path = array.filename
    array.flush()
    array._mmap.close()
    with open(path,"r+b") as handle:
        handle.write(value)
        handle.truncate(size)


def framework_retrieval(args, data, model, rows, legacy20k=False):
    """Replay historical64-score/128-A0/512-artist batches without saved ranks."""
    block = 512 if legacy20k else args.batch_size
    for start in range(0, len(rows), block):
        batch = rows[start:start+block]
        def encode(size):
            states, candidates = [], []
            for at in range(0, len(batch), size):
                h, serving = encode_base_states(args, data, model, batch[at:at+size])
                candidates.append(base_topk(model, h, serving, k=200).cpu().numpy())
                states.append(h.cpu().numpy())
            return np.concatenate(states), np.concatenate(candidates)
        states, bases = encode(64 if legacy20k else block)
        a0 = encode(128)[1] if legacy20k else None
        artist = encode(512)[1] if legacy20k else None
        yield start, states, bases, a0, artist


def framework_export(args):
    """Export the retained U220-union-RRF feature space, without fitting old heads."""
    import tempfile
    from arrive_framework_datasets import (bundle, metadata_args, feature_input_paths,
        hashes, feature_schema, code_bindings, EXPORT_SCHEMA, EXPORT_FILES)
    from run_music4all_candidate_support_a1 import load_base_data
    from run_arrive_recent_ranker import request_relations
    from lfm_timeconcat_control import LFMTimeConcatData, gap_bucket
    from evaluate_arrive_frozen_test import validate_test
    ds = bundle(args.project, args.dataset)
    legacy20k = getattr(args, "legacy20k_retrieval", False)
    if legacy20k and (args.dataset != "music4all_onion" or args.surface not in ("validation", "test")):
        raise ValueError("legacy20k_retrieval_scope")
    destination = args.output
    if destination.exists():
        raise FileExistsError(destination)
    if args.surface == "test":
        if getattr(args, "reviewer_export_interface", False):
            from reviewer_freeze import validate_before_export
            freeze = validate_before_export(args)
        else:
            from evaluate_arrive_yeti31_fusion_test import validate_before_test
            _, freeze = validate_before_test(args)
        if legacy20k != (freeze.get("retrieval_contract") == "legacy20k_64_128_512"):
            raise ValueError("frozen_retrieval_contract")
        validate_test(args)
        data_dir = args.test_data
    else:
        freeze = None
        data_dir = Path(ds["data_dir"])
    data = LFMTimeConcatData(data_dir, max_context=129, horizon=64)
    parent_args = SimpleNamespace(data_dir=Path(ds["data_dir"]), max_context=129, horizon=64,
        event_times=Path(ds["event_times"]), muse_counts=Path(ds["muse_counts"]),
        ssd4rec_source=args.project/"vendor/ssd4rec_official/ssd4rec.py", device=args.device,
        surface=args.surface)
    train_data = load_base_data(parent_args, args.base_manifest)
    bm = json.loads(args.base_manifest.read_text())
    if bm.get("binding", {}).get("configuration", {}).get("seed") != args.seed:
        raise ValueError("framework_base_seed")
    from arrive_framework_datasets import hashes as input_hashes
    prepared_files = ("events.npy", "sessions.npy", "train_frequencies.npy", "track_identity.tsv", "metric_labels.npz", "PREP_STATS.json")
    prepared = input_hashes({n: Path(ds["data_dir"])/n for n in prepared_files})
    base_prepared = bm["binding"].get("prepared_artifacts", bm["binding"].get("inputs", {}))
    if any(base_prepared.get(n) != h for n,h in prepared.items()):
        raise ValueError("framework_base_prepared_binding")
    index = args.index_dir or Path(ds["index"])
    suffix = "fit" if args.surface == "train_cv" else "full"
    im = json.loads((index/"index_manifest.json").read_text())
    if im.get("status")!="pass" or im.get("test_accessed") is not False or any(im.get("prepared_artifacts",{}).get(n)!=h for n,h in prepared.items()):
        raise ValueError("framework_index_prepared_binding")
    for name, entry in im["artifacts"].items():
        if digest(index/name) != entry["sha256"]:
            raise ValueError("framework_index_hash")
    if args.surface == "train_cv":
        rows = np.load(index/"cv_rows.npy", allow_pickle=False)
        if (bm.get("status") != "pass" or bm.get("test_accessed") is not False or
                bm["binding"].get("cv_rows_sha256") != digest(index/"cv_rows.npy") or
                bm["binding"].get("index_manifest_sha256") != digest(index/"index_manifest.json") or
                bm.get("excluded_cv_rows") != len(rows)):
            raise ValueError("framework_fold_carrier_binding")
        from arrive_framework_datasets import validate_train_cv_exclusion
        validate_train_cv_exclusion(args.project, args.dataset, index, im, train_data, rows)
    else:
        rows = data.row_ids(1 if args.surface == "validation" else 2, returns_only=True)
    model, _ = load_standard_base(parent_args, train_data,
        checkpoint_path=args.base_checkpoint, manifest_path=args.base_manifest)
    items = model.item_embedding.weight.detach().cpu().numpy().astype(np.float32)
    if len(items) != data.vocab_size:
        raise ValueError("framework_test_vocabulary")
    x,h = [sparse.load_npz(index/f"{k}_{suffix}.npz").tocsr() for k in ("x","h")]
    frequency = np.load(index/f"{suffix}_target_frequency.npy")
    audio = np.load(ds["content_audio"], mmap_mode="r")
    genre = sparse.load_npz(ds["content_genre"]).tocsr()
    tags,artists,_ = load_metadata(metadata_args(ds, Path(ds["data_dir"])), len(items))
    tags = sparse.csr_matrix(tags)  # Yambda uses the established audio-as-tag channel.
    members = artist_members(artists, frequency)
    popularity = [int(i) for i in np.argsort(-train_data.frequencies,kind="stable") if i>=3]
    lookup = None if args.surface=="train_cv" else target_lookup(data_dir)
    truth = [truth_for(data, lookup, int(r), args.surface) for r in rows]
    if not len(rows) or any(not t or len(t)!=len(set(t)) for t in truth):
        raise ValueError("framework_empty_or_duplicate_truth")
    targets = np.full((len(rows),max(map(len,truth))),-1,np.int64)
    for at,t in enumerate(truth):
        targets[at,:len(t)] = t
    scopes = ("30-90d","90-180d","180-365d",">365d")
    # Train includes short-gap rows; legacy bucket mapping uses bucket0 for those.
    buckets = np.asarray([scopes.index(gap_bucket(data.gap_days(int(r)))) if data.gap_days(int(r))>=30 else 0 for r in rows])
    relation = request_relations(data,rows,items,audio,genre,tags,artists,x,h,frequency)
    offsets=[0]; indices=[]; xcache={}
    complete_sources = np.zeros((len(rows), 4, 200), dtype=np.int32)
    source_lengths = np.zeros((len(rows), 4), dtype=np.int32)
    actual_cues = np.zeros(len(rows), dtype=np.int64)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".framework-export-",dir=destination.parent) as tmp:
        stage=Path(tmp)/"export"
        stage.mkdir()
        # At most 200 old A0 + 20 artist proposals + 200 RRF candidates.
        capacity=len(rows)*420
        fx=np.lib.format.open_memmap(stage/"features.npy",mode="w+",dtype=np.float32,shape=(capacity,28))
        fi=np.lib.format.open_memmap(stage/"candidate_ids.npy",mode="w+",dtype=np.int32,shape=(capacity,))
        fy=np.lib.format.open_memmap(stage/"labels.npy",mode="w+",dtype=np.uint8,shape=(capacity,))
        with torch.no_grad():
            parent_args.batch_size = args.batch_size
            for start,states,bases,a0_bases,artist_bases in framework_retrieval(parent_args,data,model,rows,legacy20k):
                stop=start+len(states)
                for j,rid in enumerate(rows[start:stop]):
                    at=start+j; session=data.sessions[rid]; cue=int(data.events[int(session[2])])
                    history=recent_anchors(data.events,session,5)
                    if cue not in xcache:
                        xcache[cue]=ranked_sparse_row(x,cue,frequency,0.,200)
                    artist_source=[int(i) for i in members.get(int(artists[cue]),[]) if int(i)!=cue]
                    sources=[bases[j].tolist(),xcache[cue],ranked_history(h,history,frequency,0.,200),artist_source[:200]]
                    if legacy20k:
                        parent_pool=merge_candidates(a0_bases[j].tolist(),sources[1],sources[2],popularity,relation_slots=40,x_fraction=.75,budget=200)
                        parent_set=set(map(int,parent_pool))
                        proposals=[i for i in artist_source if i not in parent_set][:20]
                        a0_map={int(i):k+1 for k,i in enumerate(a0_bases[j])}
                        artist_map={int(i):k+1 for k,i in enumerate(artist_bases[j])}
                        raw={"candidates":np.asarray([parent_pool]),
                             "base_ranks":np.asarray([[a0_map.get(int(i),201) for i in parent_pool]]),
                             "proposal_candidates":np.asarray([[proposals]]),
                             "proposal_base_ranks":np.asarray([[[artist_map.get(int(i),201) for i in proposals]]])}
                        sources[0]=reconciled_base_order(bases[j],raw,0,states[j],items)
                    actual_cues[at] = cue
                    for source_index, source in enumerate(sources):
                        source_lengths[at, source_index] = len(source)
                        complete_sources[at, source_index, :len(source)] = source
                    old=merge_candidates(sources[0],sources[1],sources[2],popularity,relation_slots=40,x_fraction=.75,budget=200)
                    old_set=set(map(int,old))
                    extras=[i for i in artist_source if i not in old_set][:20]
                    rrf,_,_=rrf_pool(sources,budget=200)
                    ids=np.asarray(list(dict.fromkeys([*map(int,old),*extras,*map(int,rrf)])),np.int32)
                    position={int(i):k for k,i in enumerate(ids)}
                    indices.append([position[int(i)] for i in rrf])
                    maps=[{int(i):k+1 for k,i in enumerate(src)} for src in sources]
                    lo=offsets[-1]; hi=lo+len(ids)
                    # Match the legacy export's float32 gap storage before log1p.
                    gap = float(np.float32(data.gap_days(int(rid))))
                    fx[lo:hi]=row_feature(ids,states[j],items,maps,gap,int(buckets[at]),frequency,cue,history,artists,tags,audio,genre)
                    fi[lo:hi]=ids; fy[lo:hi]=np.isin(ids,truth[at]); offsets.append(hi)
        for src in (fx,fi,fy):
            finish_npy(src,offsets[-1])
        del fx,fi,fy
        atomic_npz(stage/"metadata.npz", row_ids=rows,users=data.sessions[rows,0],
            bucket_ids=buckets,targets=targets,target_counts=np.asarray(list(map(len,truth))),
            offsets=np.asarray(offsets),rrf_indices=np.asarray(indices))
        atomic_npz(stage/"REQUEST_CONTEXT.npz",row_ids=rows,users=data.sessions[rows,0],relation=relation)
        atomic_npz(stage/"sources.npz", row_ids=rows, users=data.sessions[rows,0],
            source_ids=complete_sources, source_lengths=source_lengths, cues=actual_cues)
        inputs=hashes(feature_input_paths(ds,index,suffix))
        inputs.update(base_checkpoint=digest(args.base_checkpoint),base_manifest=digest(args.base_manifest))
        m={"schema":EXPORT_SCHEMA,"status":"pass","dataset":args.dataset,"seed":args.seed,
            "surface":args.surface,"rows":len(rows),"limit":0,"test_accessed":args.surface=="test",
            "synthetic":bm.get("synthetic", False),
            "feature_schema":feature_schema(),"code_sha256":code_bindings(),"serving_inputs":inputs,
            "prepared_sha256":prepared,"index_manifest_sha256":digest(index/"index_manifest.json"),
            "base_manifest":str(args.base_manifest.resolve()),"index_dir":str(index.resolve()),
            "output_sha256":{n:digest(stage/n) for n in EXPORT_FILES},
            "training_policy":"joint_U220_plus_full_source_RRF_C200"}
        m["retrieval_contract"]="legacy20k_64_128_512" if legacy20k else "direct"
        m["reproduction_sources_sha256"] = digest(stage / "sources.npz")
        if freeze is not None:
            if inputs!=freeze["serving_inputs"]:
                raise ValueError("framework_test_serving_mismatch")
            m.update(freeze_sha256=digest(args.freeze_record),test_manifest_sha256=digest(args.test_manifest))
        atomic_json(stage/"EXPORT_MANIFEST.json",m)
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)
    print(json.dumps({"event":"framework_export_complete","surface":args.surface,"rows":len(rows)}),flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "fit", "analyze", "framework_export"))
    parser.add_argument("--dataset", choices=("music4all_onion", "yambda500m"))
    for name in ("base-checkpoint", "base-manifest", "index-dir", "model-dir", "freeze-record", "test-data", "test-manifest", "validation-audit"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--surface", choices=("train_cv", "validation", "test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--legacy20k-retrieval", action="store_true")
    parser.add_argument("--reviewer-export-interface", action="store_true",
        help="Use a separately bound reviewer_freeze.v1 record for portable Test reproduction.")
    args = parser.parse_args()
    if args.mode != "analyze" and args.seed is None:
        parser.error("seed_required")
    if args.mode == "export" and args.surface is None:
        parser.error("surface_required")
    torch.set_num_threads(args.threads)
    start = time.monotonic()
    if args.mode == "framework_export" and not all((args.dataset, args.surface, args.base_checkpoint, args.base_manifest)):
        parser.error("framework_export requires dataset/surface/base-checkpoint/base-manifest")
    if args.mode == "framework_export" and args.surface == "test" and not all((args.freeze_record,args.model_dir,args.test_data,args.test_manifest)):
        parser.error("Test export requires frozen model and Test manifest")
    {"export": export, "fit": fit, "analyze": analyze, "framework_export": framework_export}[args.mode](args)
    print(json.dumps({"event": "complete", "mode": args.mode, "seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    main()
