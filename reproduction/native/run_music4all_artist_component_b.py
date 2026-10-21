#!/usr/bin/env python3
"""Learn observed-artist candidate support on top of frozen Music4All A1.

The feature stage appends fixed-budget same-artist proposals to the frozen A1
feature bundle.  The head stage keeps every A1 parameter frozen and learns a
residual only for those proposals.  Alpha zero is exact cached A1 serving.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from scipy import sparse
from torch import nn

from lfm_timeconcat_control import LFMTimeConcatData, exact_catalog_topk
from run_music4all_candidate_support_a0 import digest, ranked_history, ranked_sparse_row, recent_anchors
from run_music4all_candidate_support_a1 import (
    SCOPES,
    SetwiseResidual,
    atomic_json,
    atomic_npz,
    atomic_torch,
    bootstrap_delta,
    base_topk,
    encode_base_states,
    load_base_data,
    load_features,
    numeric_features,
    predict_residual,
    set_seed,
    stable_topk,
    target_lookup,
    user_macro,
    user_partition,
)
from run_music4all_component_b import a1_scores, load_a1_model


FEATURE_SCHEMA = "genrec.music4all_onion.artist_b_feature_pool.v1"
HEAD_SCHEMA = "genrec.music4all_onion.artist_b_head.v1"
GROUPS = ("real", "shuffled_2024", "shuffled_2025", "shuffled_2026")


def _load_metadata_yambda(args: argparse.Namespace, vocab_size: int):
    """Yambda counterpart: artist from artist_item_mapping.parquet, content
    rows from the derived audio array. Declared adaptations: items with
    several artists take the smallest artist id as primary (deterministic);
    the tag-TFIDF channel becomes the audio-embedding cosine (rows are
    L2-normalized, so inner products are cosines); items without content
    hold zero rows and contribute zero terms."""
    identity: dict[int, int] = {}
    with (args.data_dir / "track_identity.tsv").open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:3] != ["raw_index", "model_id", "item_id"]:
            raise ValueError("track_identity_header_invalid")
        for line in handle:
            _, model_id, item_id, _ = line.rstrip("\n").split("\t")
            identity[int(item_id)] = int(model_id)
    if len(identity) != vocab_size - 3:
        raise ValueError("identity_vocab_mismatch")

    # The mapping ships as an npz pair of int64 arrays (converted from the
    # release parquet; the cluster venv has no pyarrow and no external net).
    npz = np.load(args.artist_mapping, allow_pickle=False)
    if not {"item_id", "artist_id"} <= set(npz.files):
        raise ValueError("artist_mapping_schema_invalid")
    ids = npz["item_id"].astype(np.int64)
    artists = npz["artist_id"].astype(np.int64)
    keep = np.isin(ids, np.fromiter(identity.keys(), dtype=np.int64))
    ids, artists = ids[keep], artists[keep]
    order = np.argsort(ids)
    ids, artists = ids[order], artists[order]
    used_artists = np.unique(artists)
    artist_dense = np.searchsorted(used_artists, artists)

    artist_by_item = np.full(vocab_size, -1, dtype=np.int32)
    items_arr = np.fromiter(identity.keys(), dtype=np.int64)
    models_arr = np.fromiter(identity.values(), dtype=np.int64)
    items_order = np.argsort(items_arr)
    sorted_items = items_arr[items_order]
    sorted_models = models_arr[items_order]
    model_ids = sorted_models[np.searchsorted(sorted_items, ids)]
    # multi-artist items: smallest dense artist id wins (first write after
    # sorting by (item, dense artist) keeps that deterministic)
    slot_order = np.lexsort((artist_dense, model_ids))
    model_ids = model_ids[slot_order]
    artist_dense = artist_dense[slot_order]
    first = np.ones(len(model_ids), dtype=bool)
    first[1:] = model_ids[1:] != model_ids[:-1]
    artist_by_item[model_ids[first]] = artist_dense[first]
    if int(artist_by_item[:3].max()) != -1:
        raise ValueError("reserved_rows_overwritten")

    audio = np.asarray(np.load(args.audio_array, mmap_mode="r"), dtype=np.float32)
    if audio.shape[0] != vocab_size:
        raise ValueError("audio_array_vocab_mismatch")
    coverage = {
        "provenance": "yambda",
        "tag_tracks": int((np.linalg.norm(audio, axis=1) > 0).sum()),
        "tag_vocabulary": int(audio.shape[1]),
        "artist_tracks": int((artist_by_item >= 0).sum()),
        "artist_vocabulary": int(len(used_artists)),
    }
    if coverage["artist_tracks"] == 0:
        raise ValueError("artist_mapping_zero_coverage")
    return audio, artist_by_item, coverage


def load_metadata(args: argparse.Namespace, vocab_size: int):
    mapping_set = getattr(args, "artist_mapping", None) is not None
    audio_set = getattr(args, "audio_array", None) is not None
    if mapping_set != audio_set:
        raise ValueError("yambda_metadata_args_incomplete")
    if mapping_set:
        return _load_metadata_yambda(args, vocab_size)
    mapping: dict[str, int] = {}
    with (args.data_dir / "track_identity.tsv").open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:3] != ["raw_index", "model_id", "track_id"]:
            raise ValueError("track_identity_header_invalid")
        maximum = 0
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            model_id = int(fields[1])
            mapping[fields[2]] = model_id
            maximum = max(maximum, model_id)
    if maximum + 1 != vocab_size:
        raise ValueError("identity_vocab_mismatch")

    tags_by_item: dict[int, list[str]] = {}
    document_frequency: Counter[str] = Counter()
    with args.tags.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["id", "tags"]:
            raise ValueError("tag_header_invalid")
        for row in reader:
            item = mapping.get(row["id"])
            if item is None:
                continue
            tags = sorted({tag.strip().casefold() for tag in row["tags"].split(",") if tag.strip()})
            tags_by_item[item] = tags
            document_frequency.update(tags)
    tag_vocab = {tag: at for at, tag in enumerate(sorted(document_frequency))}
    rows, cols, values = [], [], []
    for item, tags in tags_by_item.items():
        for tag in tags:
            rows.append(item)
            cols.append(tag_vocab[tag])
            values.append(np.log((1.0 + len(tags_by_item)) / (1.0 + document_frequency[tag])) + 1.0)
    tag_matrix = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)),
        shape=(vocab_size, len(tag_vocab)), dtype=np.float32,
    )
    norms = np.sqrt(np.asarray(tag_matrix.multiply(tag_matrix).sum(axis=1)).ravel())
    inverse = np.zeros_like(norms)
    inverse[norms > 0] = 1.0 / norms[norms > 0]
    tag_matrix = sparse.diags(inverse).dot(tag_matrix).tocsr()

    artist_by_item = np.full(vocab_size, -1, dtype=np.int32)
    artist_vocab: dict[str, int] = {}
    with args.information.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["id", "artist", "song", "album_name"]:
            raise ValueError("information_header_invalid")
        for row in reader:
            item = mapping.get(row["id"])
            artist = row["artist"].strip().casefold()
            if item is not None and artist:
                artist_by_item[item] = artist_vocab.setdefault(artist, len(artist_vocab))
    return tag_matrix, artist_by_item, {
        "tag_tracks": len(tags_by_item), "tag_vocabulary": len(tag_vocab),
        "artist_tracks": int((artist_by_item >= 0).sum()), "artist_vocabulary": len(artist_vocab),
    }


def artist_members(artist_by_item: np.ndarray, frequency: np.ndarray) -> dict[int, list[int]]:
    members: dict[int, list[int]] = {}
    for item in range(3, len(artist_by_item)):
        artist = int(artist_by_item[item])
        if artist >= 0:
            members.setdefault(artist, []).append(item)
    for values in members.values():
        values.sort(key=lambda item: (-float(frequency[item]), item))
    return members


def fixed_proposals(
    sources: np.ndarray,
    artist_by_item: np.ndarray,
    members: dict[int, list[int]],
    original: np.ndarray,
    quota: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.full((len(sources), quota), -1, dtype=np.int32)
    ranks = np.zeros((len(sources), quota), dtype=np.float32)
    cache: dict[int, list[int]] = {}
    for row, source in enumerate(map(int, sources)):
        if source not in cache:
            cache[source] = [item for item in members.get(int(artist_by_item[source]), []) if item != source]
        forbidden = set(map(int, original[row]))
        selected = [item for item in cache[source] if item not in forbidden][:quota]
        candidates[row, : len(selected)] = selected
        ranks[row, : len(selected)] = np.arange(1, len(selected) + 1, dtype=np.float32)
    return candidates, ranks


def deterministic_source_groups(observed: np.ndarray) -> np.ndarray:
    groups = [observed.astype(np.int32, copy=True)]
    for seed in (2024, 2025, 2026):
        permutation = np.random.default_rng(seed).permutation(len(observed))
        groups.append(observed[permutation].astype(np.int32, copy=False))
    return np.stack(groups, axis=1)


@torch.no_grad()
def build_features(args: argparse.Namespace) -> None:
    if args.output.exists() or args.feature_manifest.exists():
        raise FileExistsError("artist_b_feature_output_exists")
    original, original_manifest = load_features(args.a1_features, args.a1_feature_manifest, args.surface)
    data = load_base_data(args, args.base_training_manifest)
    metadata_args = argparse.Namespace(
        data_dir=args.data_dir, tags=args.tags, information=args.information,
        artist_mapping=getattr(args, "artist_mapping", None),
        audio_array=getattr(args, "audio_array", None),
    )
    tag_matrix, artist_by_item, coverage = load_metadata(metadata_args, len(data.frequencies))
    frequency = np.load(
        args.index_dir / ("fit_target_frequency.npy" if args.surface == "train_cv" else "full_target_frequency.npy"),
        allow_pickle=False,
    )
    members = artist_members(artist_by_item, frequency)
    rows = original["row_ids"].astype(np.int64)
    observed = np.asarray([int(data.events[int(data.sessions[int(row)][2])]) for row in rows], dtype=np.int32)
    source_groups = deterministic_source_groups(observed)
    n, group_count, quota = len(rows), len(GROUPS), args.metadata_slots
    proposal_candidates = np.full((n, group_count, quota), -1, dtype=np.int32)
    proposal_ranks = np.zeros((n, group_count, quota), dtype=np.float32)
    for group in range(group_count):
        proposal_candidates[:, group], proposal_ranks[:, group] = fixed_proposals(
            source_groups[:, group], artist_by_item, members, original["candidates"], quota
        )

    # Load the exact standard SASRec state used by the original A1 feature surface.
    from run_music4all_candidate_support_a1 import load_standard_base
    model, base_manifest = load_standard_base(
        args, data, checkpoint_path=args.base_checkpoint, manifest_path=args.base_training_manifest
    )
    index = json.loads((args.index_dir / "index_manifest.json").read_text())
    from scipy import sparse
    x_matrix = sparse.load_npz(args.index_dir / ("x_fit.npz" if args.surface == "train_cv" else "x_full.npz")).tocsr()
    h_matrix = sparse.load_npz(args.index_dir / ("h_fit.npz" if args.surface == "train_cv" else "h_full.npz")).tocsr()
    proposal_base_scores = np.full_like(proposal_ranks, -1e9, dtype=np.float32)
    proposal_base_ranks = np.full_like(proposal_ranks, 201, dtype=np.uint16)
    proposal_x_ranks = np.full_like(proposal_ranks, 201, dtype=np.uint16)
    proposal_h_ranks = np.full_like(proposal_ranks, 201, dtype=np.uint16)
    proposal_log_popularity = np.zeros_like(proposal_ranks, dtype=np.float32)
    proposal_tag_cosine = np.zeros_like(proposal_ranks, dtype=np.float32)
    proposal_target_mask = np.zeros_like(proposal_ranks, dtype=np.uint8)
    labels = None if args.surface == "train_cv" else target_lookup(args.data_dir)
    x_cache: dict[int, list[int]] = {}
    tag_cache: dict[tuple[int, int], float] = {}
    dense_content = isinstance(tag_matrix, np.ndarray)

    for start in range(0, n, args.eval_batch_size):
        stop = min(start + args.eval_batch_size, n)
        batch_rows = rows[start:stop]
        states, serving = encode_base_states(args, data, model, batch_rows)
        base_top = base_topk(model, states, serving, k=200).cpu().numpy()
        flat = proposal_candidates[start:stop].reshape(stop - start, -1)
        safe = np.maximum(flat, 0)
        candidate_tensor = torch.as_tensor(safe, dtype=torch.long, device=args.device)
        scores = torch.einsum("bd,bkd->bk", states, model.item_embedding(candidate_tensor)).cpu().numpy()
        scores[flat < 0] = -1e9
        proposal_base_scores[start:stop] = scores.reshape(stop - start, group_count, quota)
        for offset, row_id in enumerate(map(int, batch_rows)):
            at = start + offset
            session = data.sessions[row_id]
            source = int(observed[at])
            if source not in x_cache:
                x_cache[source] = ranked_sparse_row(x_matrix, source, frequency, 0.0, 200)
            anchors = recent_anchors(data.events, session, 5)
            h_items = ranked_history(h_matrix, anchors, frequency, 0.0, 200)
            base_map = {int(item): rank + 1 for rank, item in enumerate(base_top[offset])}
            x_map = {item: rank + 1 for rank, item in enumerate(x_cache[source])}
            h_map = {item: rank + 1 for rank, item in enumerate(h_items)}
            truth = set(map(int, data.known_suffix(row_id) if labels is None else labels[row_id]))
            for group in range(group_count):
                assigned_source = int(source_groups[at, group])
                source_vector = (
                    tag_matrix[assigned_source] if dense_content
                    else tag_matrix.getrow(assigned_source)
                )
                for slot, item in enumerate(map(int, proposal_candidates[at, group])):
                    if item < 3:
                        continue
                    proposal_base_ranks[at, group, slot] = base_map.get(item, 201)
                    proposal_x_ranks[at, group, slot] = x_map.get(item, 201)
                    proposal_h_ranks[at, group, slot] = h_map.get(item, 201)
                    proposal_log_popularity[at, group, slot] = math.log1p(float(data.frequencies[item]))
                    key = (assigned_source, item)
                    if key not in tag_cache:
                        if dense_content:
                            tag_cache[key] = float(source_vector @ tag_matrix[item])
                        else:
                            tag_cache[key] = float(source_vector.multiply(tag_matrix.getrow(item)).sum())
                    proposal_tag_cosine[at, group, slot] = tag_cache[key]
                    proposal_target_mask[at, group, slot] = int(item in truth)
        print(json.dumps({"event": "artist_b_features", "surface": args.surface, "rows": stop, "total": n}), flush=True)

    atomic_npz(
        args.output,
        **original,
        observed=observed,
        source_groups=source_groups,
        proposal_candidates=proposal_candidates,
        proposal_ranks=proposal_ranks,
        proposal_base_scores=proposal_base_scores,
        proposal_base_ranks=proposal_base_ranks,
        proposal_x_ranks=proposal_x_ranks,
        proposal_h_ranks=proposal_h_ranks,
        proposal_log_popularity=proposal_log_popularity,
        proposal_tag_cosine=proposal_tag_cosine,
        proposal_target_mask=proposal_target_mask,
    )
    atomic_json(args.feature_manifest, {
        "schema": FEATURE_SCHEMA, "status": "pass", "surface": args.surface,
        "rows": n, "groups": list(GROUPS), "metadata_slots": quota,
        "feature_sha256": digest(args.output), "producer_sha256": digest(Path(__file__)),
        "a1_feature_sha256": original_manifest["feature_sha256"],
        "base_method": original_manifest.get("base_method"),
        "base_checkpoint_sha256": digest(args.base_checkpoint),
        "base_manifest_sha256": digest(args.base_training_manifest),
        "index_manifest_sha256": digest(args.index_dir / "index_manifest.json"),
        **(
            {"artist_mapping_sha256": digest(args.artist_mapping),
             "audio_array_sha256": digest(args.audio_array)}
            if getattr(args, "artist_mapping", None) is not None
            else {"tags_sha256": digest(args.tags), "information_sha256": digest(args.information)}
        ),
        "metadata_coverage": coverage, "test_accessed": args.surface == "test",
        "selection_or_tuning_permitted": False if args.surface == "test" else None,
    })


def load_artist_features(path: Path, manifest_path: Path, surface: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != FEATURE_SCHEMA or manifest.get("status") != "pass" or manifest.get("surface") != surface or manifest.get("feature_sha256") != digest(path) or manifest.get("test_accessed") is not (surface == "test") or (surface == "test" and manifest.get("selection_or_tuning_permitted") is not False):
        raise ValueError(f"artist_b_feature_manifest_invalid:{surface}")
    loaded = np.load(path, allow_pickle=False)
    arrays = {key: loaded[key] for key in loaded.files}
    if len(arrays["row_ids"]) != manifest["rows"] or arrays["proposal_candidates"].shape[1:] != (4, 20):
        raise ValueError("artist_b_feature_shape_invalid")
    return arrays, manifest


def proposal_numeric(arrays: dict[str, np.ndarray], group: int) -> tuple[np.ndarray, np.ndarray]:
    proxy = {
        "base_scores": arrays["proposal_base_scores"][:, group],
        "base_ranks": arrays["proposal_base_ranks"][:, group],
        "x_ranks": arrays["proposal_x_ranks"][:, group],
        "h_ranks": arrays["proposal_h_ranks"][:, group],
        "log_popularity": arrays["proposal_log_popularity"][:, group],
        "gaps": arrays["gaps"],
    }
    base, _ = numeric_features(proxy, "full")
    valid = (arrays["proposal_candidates"][:, group] >= 3).astype(np.float32)
    safe_rank = np.maximum(arrays["proposal_ranks"][:, group], 1.0)
    rank = np.where(valid > 0, 1.0 / np.log2(safe_rank + 1.0), 0.0)
    extra = np.stack((valid, arrays["proposal_tag_cosine"][:, group], rank, valid), axis=-1)
    return np.concatenate((base, extra.astype(np.float32)), axis=-1), valid


def candidate_view(
    arrays: dict[str, np.ndarray], a1_original: np.ndarray, a1_proposals: np.ndarray,
    proposal_features: np.ndarray, group: int, cached_top10: np.ndarray,
) -> dict[str, np.ndarray]:
    n = len(arrays["row_ids"])
    width = proposal_features.shape[-1]
    candidates = np.empty((n, 200), dtype=np.int32)
    scores = np.empty((n, 200), dtype=np.float32)
    features = np.zeros((n, 200, width), dtype=np.float32)
    mask = np.zeros((n, 200), dtype=np.float32)
    positives = np.zeros((n, 200), dtype=np.uint8)
    proposal_hits = np.zeros(n, dtype=np.uint8)
    for row in range(n):
        original_ids = arrays["candidates"][row]
        order = np.lexsort((original_ids, -a1_original[row]))
        top = list(map(int, order[:10]))
        cached_top = [int(v) for v in cached_top10[row]]
        if top != cached_top:
            # Two CUDA evaluations reassociate float32 scores at the ~1e-7
            # level, so near-equal items around the rank-10 boundary may
            # swap or exchange membership between runs. Accept only
            # disagreements whose differing members sit within 1e-5 of the
            # union's 10th-highest score (boundary ties by definition);
            # anything else is a real inconsistency.
            score_of = dict(zip(map(int, original_ids), map(float, a1_original[row])))
            top_items = [int(original_ids[i]) for i in top]
            rec_set, cached_set = set(top_items), set(cached_top)
            if rec_set == cached_set:
                consistent = True  # order-only: cascaded near-equal swaps
            else:
                union = rec_set | cached_set
                union_scores = sorted((score_of[i] for i in union), reverse=True)
                boundary = union_scores[9]
                consistent = all(
                    abs(score_of[i] - boundary) <= 1e-5
                    for i in rec_set.symmetric_difference(cached_set)
                )
            if not consistent:
                raise ValueError("a1_top10_reconstruction_failed")
        proposal_slots = [slot for slot, item in enumerate(arrays["proposal_candidates"][row, group]) if int(item) >= 3]
        remaining = [int(index) for index in order if int(index) not in set(top)]
        original_take = 200 - len(proposal_slots)
        write = 0
        for index in top + remaining[: original_take - 10]:
            candidates[row, write] = original_ids[index]
            scores[row, write] = a1_original[row, index]
            positives[row, write] = arrays["target_mask"][row, index]
            write += 1
        for slot in proposal_slots:
            candidates[row, write] = arrays["proposal_candidates"][row, group, slot]
            scores[row, write] = a1_proposals[row, slot]
            features[row, write] = proposal_features[row, slot]
            mask[row, write] = 1.0
            positives[row, write] = arrays["proposal_target_mask"][row, group, slot]
            proposal_hits[row] |= positives[row, write]
            write += 1
        if write != 200 or len(set(map(int, candidates[row]))) != 200:
            raise ValueError("artist_b_candidate_identity_invalid")
    return {"candidates": candidates, "scores": scores, "features": features, "mask": mask, "positives": positives, "proposal_positive_rows": proposal_hits}


class ProposalResidual(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1) * mask


def proposal_loss(scores: torch.Tensor, positives: torch.Tensor, proposal_mask: torch.Tensor, margin: float, boundary_weight: float) -> torch.Tensor:
    valid = positives.any(dim=1)
    scores, positives, proposal_mask = scores[valid], positives[valid], proposal_mask[valid]
    positive_scores = scores.masked_fill(~positives, float("-inf"))
    set_loss = torch.logsumexp(scores, dim=1) - torch.logsumexp(positive_scores, dim=1)
    proposal_positive = positives & proposal_mask.bool()
    active = proposal_positive.any(dim=1)
    if active.any():
        best = scores.masked_fill(~proposal_positive, float("-inf")).max(dim=1).values
        cutoff = min(10, scores.shape[1])
        tenth_negative = scores.masked_fill(positives, float("-inf")).topk(cutoff, dim=1).values[:, -1]
        boundary = torch.nn.functional.softplus(margin + tenth_negative - best)
        set_loss = set_loss + boundary_weight * torch.where(active, boundary, torch.zeros_like(boundary))
    return set_loss.mean()


@torch.no_grad()
def residual_scores(model: nn.Module, features: np.ndarray, mask: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    output = np.empty(mask.shape, dtype=np.float32)
    model.eval()
    for start in range(0, len(mask), batch_size):
        stop = min(start + batch_size, len(mask))
        output[start:stop] = model(torch.as_tensor(features[start:stop], device=device), torch.as_tensor(mask[start:stop], device=device)).cpu().numpy()
    return output


def metric_table(predictions: np.ndarray, view: dict[str, np.ndarray], arrays: dict[str, np.ndarray]):
    recall = np.zeros(len(predictions), dtype=np.float64)
    ndcg = np.zeros(len(predictions), dtype=np.float64)
    for row, prediction in enumerate(predictions):
        relevant = set(map(int, view["candidates"][row][view["positives"][row].astype(bool)]))
        total = max(int(arrays["target_counts"][row]), 1)
        hits = [int(int(item) in relevant) for item in prediction]
        recall[row] = sum(hits) / total
        dcg = sum(hit / math.log2(rank + 2.0) for rank, hit in enumerate(hits))
        ideal = sum(1.0 / math.log2(rank + 2.0) for rank in range(min(total, 10)))
        ndcg[row] = dcg / ideal if ideal else 0.0
    table = {}
    for scope_index, scope in enumerate(SCOPES):
        selected = np.ones(len(recall), dtype=bool) if scope == "overall" else arrays["bucket_ids"] == scope_index - 1
        table[scope] = {
            "rows": int(selected.sum()),
            "users": int(len(np.unique(arrays["users"][selected]))),
            "recall_at_10": user_macro(recall, arrays["users"], selected),
            "ndcg_at_10": user_macro(ndcg, arrays["users"], selected),
        }
    return table, recall, ndcg


def train_head(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError("artist_b_output_exists")
    train, train_manifest = load_artist_features(args.train_features, args.train_feature_manifest, "train_cv")
    validation_manifest_meta = json.loads(args.validation_feature_manifest.read_text()) if args.validation_feature_manifest else None
    a1_model, a1_checkpoint, _ = load_a1_model(
        args.a1_checkpoint, args.a1_result, args.a1_predictions,
        train_manifest["a1_feature_sha256"], validation_manifest_meta["a1_feature_sha256"] if validation_manifest_meta else train_manifest["a1_feature_sha256"], args.device,
    )
    original_proxy = {key: train[key] for key in ("base_scores", "base_ranks", "x_ranks", "h_ranks", "log_popularity", "gaps")}
    train_a1_original = a1_scores(a1_model, a1_checkpoint, original_proxy, args.device, args.eval_batch_size)
    pfeatures, pmask = proposal_numeric(train, 0)
    proposal_proxy = {
        "base_scores": train["proposal_base_scores"][:, 0], "base_ranks": train["proposal_base_ranks"][:, 0],
        "x_ranks": train["proposal_x_ranks"][:, 0], "h_ranks": train["proposal_h_ranks"][:, 0],
        "log_popularity": train["proposal_log_popularity"][:, 0], "gaps": train["gaps"],
    }
    train_a1_proposals = a1_scores(a1_model, a1_checkpoint, proposal_proxy, args.device, args.eval_batch_size)
    train_a1_top10 = stable_topk(train_a1_original, train["candidates"])
    train_view = candidate_view(train, train_a1_original, train_a1_proposals, pfeatures, 0, train_a1_top10)
    fit_mask = user_partition(train["users"]); calibration_mask = ~fit_mask
    fit_rows = np.flatnonzero(fit_mask & train_view["positives"].any(axis=1))
    set_seed(args.seed)
    model = ProposalResidual(pfeatures.shape[-1], args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    base_cal = train_a1_top10[calibration_mask].copy()
    cal_arrays = {key: value[calibration_mask] for key, value in train.items()}
    cal_view = {key: value[calibration_mask] for key, value in train_view.items()}
    base_metrics, _, _ = metric_table(base_cal, cal_view, cal_arrays)
    long_scope = next(
        scope for scope in reversed(SCOPES[1:]) if base_metrics[scope]["rows"] > 0
    )
    best = None; best_state = None; records = []; batches = 0; started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(args.seed + epoch).permutation(fit_rows); losses = []
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            residual = model(torch.as_tensor(train_view["features"][rows], device=args.device), torch.as_tensor(train_view["mask"][rows], device=args.device))
            scores = torch.as_tensor(train_view["scores"][rows], device=args.device) + residual
            loss = proposal_loss(scores, torch.as_tensor(train_view["positives"][rows].astype(bool), device=args.device), torch.as_tensor(train_view["mask"][rows], device=args.device), args.margin, args.boundary_weight)
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite_artist_b_loss")
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True); optimizer.step()
            losses.append(float(loss.detach())); batches += 1
            if args.stop_after and batches >= args.stop_after:
                args.output.mkdir(parents=True); atomic_json(args.output / "SMOKE.json", {"status": "engineering_pass", "scientific_result": False, "batches": batches, "test_accessed": False}); return
        residual = residual_scores(model, cal_view["features"], cal_view["mask"], args.device, args.eval_batch_size)
        choices = []
        for alpha in args.alpha_grid:
            pred = base_cal if alpha == 0 else stable_topk(cal_view["scores"] + alpha * residual, cal_view["candidates"])
            metrics, _, _ = metric_table(pred, cal_view, cal_arrays)
            score = sum(metrics[s][m] for s in ("overall", long_scope) for m in ("recall_at_10", "ndcg_at_10"))
            short_decline = max(base_metrics["30-90d"][m] - metrics["30-90d"][m] for m in ("recall_at_10", "ndcg_at_10"))
            if short_decline > 0.005: score = -1e9
            choices.append({"alpha": alpha, "score": score, "metrics": metrics})
            key = (score, -abs(alpha), -alpha)
            if best is None or key > best:
                best = key; best_state = {"epoch": epoch, "alpha": float(alpha), "score": float(score), "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        records.append({"epoch": epoch, "mean_loss": float(np.mean(losses)), "calibration": choices})
        print(json.dumps({"event": "artist_b_epoch", "seed": args.seed, "epoch": epoch, "loss": records[-1]["mean_loss"], "alpha": best_state["alpha"]}), flush=True)
    if best_state is None: raise RuntimeError("artist_b_selection_failed")
    model.load_state_dict(best_state["model"], strict=True)
    validation, validation_manifest = load_artist_features(args.validation_features, args.validation_feature_manifest, "validation")
    original_proxy = {key: validation[key] for key in ("base_scores", "base_ranks", "x_ranks", "h_ranks", "log_popularity", "gaps")}
    val_a1_original = a1_scores(a1_model, a1_checkpoint, original_proxy, args.device, args.eval_batch_size)
    cached = np.load(args.a1_predictions, allow_pickle=False)
    if not np.array_equal(cached["row_ids"], validation["row_ids"]):
        raise ValueError("a1_prediction_row_mismatch")
    results = {}
    views = {}
    for name, group in (("real", 0), ("shuffled", GROUPS.index(f"shuffled_{args.seed}"))):
        pf, _ = proposal_numeric(validation, group)
        proxy = {"base_scores": validation["proposal_base_scores"][:, group], "base_ranks": validation["proposal_base_ranks"][:, group], "x_ranks": validation["proposal_x_ranks"][:, group], "h_ranks": validation["proposal_h_ranks"][:, group], "log_popularity": validation["proposal_log_popularity"][:, group], "gaps": validation["gaps"]}
        pa1 = a1_scores(a1_model, a1_checkpoint, proxy, args.device, args.eval_batch_size)
        view = candidate_view(validation, val_a1_original, pa1, pf, group, cached["a1_top10"])
        residual = residual_scores(model, view["features"], view["mask"], args.device, args.eval_batch_size)
        predictions = cached["a1_top10"] if best_state["alpha"] == 0 else stable_topk(view["scores"] + best_state["alpha"] * residual, view["candidates"])
        metrics, recall, ndcg = metric_table(predictions, view, validation)
        results[name] = {"predictions": predictions, "metrics": metrics, "recall": recall, "ndcg": ndcg}
        views[name] = view
    base_view = views["real"]
    base_metrics, base_recall, base_ndcg = metric_table(cached["a1_top10"], base_view, validation)
    paired = {}
    for scope_index, scope in enumerate(SCOPES):
        selected = np.ones(len(base_recall), dtype=bool) if scope == "overall" else validation["bucket_ids"] == scope_index - 1
        paired[scope] = {"recall_delta": bootstrap_delta(results["real"]["recall"], base_recall, validation["users"], selected, seed=args.seed), "ndcg_delta": bootstrap_delta(results["real"]["ndcg"], base_ndcg, validation["users"], selected, seed=args.seed + 1)}
    args.output.mkdir(parents=True)
    binding = {
        "schema": HEAD_SCHEMA, "seed": args.seed, "epochs": args.epochs,
        "hidden_dim": args.hidden_dim, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "margin": args.margin,
        "boundary_weight": args.boundary_weight, "alpha_grid": list(args.alpha_grid),
        "producer_sha256": digest(Path(__file__)),
        "train_feature_sha256": train_manifest["feature_sha256"],
        "validation_feature_sha256": validation_manifest["feature_sha256"],
        "a1_checkpoint_sha256": digest(args.a1_checkpoint),
        "a1_result_sha256": digest(args.a1_result), "test_accessed": False,
    }
    checkpoint = args.output / "artist_b_checkpoint.pt"
    atomic_torch(checkpoint, {"schema": HEAD_SCHEMA, "binding": binding, "model": best_state["model"], "selection": {k:v for k,v in best_state.items() if k != "model"}})
    ProposalResidual(pfeatures.shape[-1], args.hidden_dim).load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    predictions_path = args.output / "validation_predictions.npz"
    atomic_npz(predictions_path, row_ids=validation["row_ids"], users=validation["users"], bucket_ids=validation["bucket_ids"], a1_top10=cached["a1_top10"], artist_b_top10=results["real"]["predictions"], shuffled_top10=results["shuffled"]["predictions"], a1_recall=base_recall, artist_b_recall=results["real"]["recall"], shuffled_recall=results["shuffled"]["recall"], a1_ndcg=base_ndcg, artist_b_ndcg=results["real"]["ndcg"], shuffled_ndcg=results["shuffled"]["ndcg"])
    result = {"schema": HEAD_SCHEMA, "status": "pass", "seed": args.seed, "selection": {k:v for k,v in best_state.items() if k != "model"}, "metrics": {"a1": base_metrics, "artist_b": results["real"]["metrics"], "shuffled": results["shuffled"]["metrics"]}, "paired_bootstrap": paired, "rescued_rows": int(((results["real"]["recall"] > 0) & (base_recall == 0)).sum()), "displaced_rows": int(((base_recall > 0) & (results["real"]["recall"] == 0)).sum()), "proposal_positive_rows": int(base_view["proposal_positive_rows"].sum()), "binding": binding, "checkpoint_sha256": digest(checkpoint), "predictions_sha256": digest(predictions_path), "checkpoint_strict_reload": True, "seconds": time.monotonic() - started, "validation_accessed": True, "test_accessed": False}
    atomic_json(args.output / "ARTIST_B_RESULT.json", result)
    atomic_json(args.output / "training_manifest.json", {"schema": HEAD_SCHEMA, "status": "pass", "binding": binding, "selection": result["selection"], "epoch_records": records, "checkpoint_sha256": result["checkpoint_sha256"], "predictions_sha256": result["predictions_sha256"], "validation_accessed_once_after_freeze": True, "test_accessed": False})
    print(json.dumps({"event": "artist_b_complete", "seed": args.seed, "selection": result["selection"], "overall": result["metrics"]["artist_b"]["overall"]}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build-features", "train-head"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/music4all_onion/a0_candidate_index_v1"))
    parser.add_argument("--tags", type=Path, default=Path("data/music4all_onion/raw/id_tags.csv"))
    parser.add_argument("--artist-mapping", type=Path, default=None,
                        help="Yambda: artist_item_mapping.parquet; selects the Yambda metadata path")
    parser.add_argument("--audio-array", type=Path, default=None,
                        help="Yambda: derived audio_normalized.npy (content rows)")
    parser.add_argument("--information", type=Path, default=Path("data/music4all_onion/raw/id_information.csv"))
    parser.add_argument("--event-times", type=Path, default=Path("data/music4all_onion/baseline_features_v1/event_times.npy"))
    parser.add_argument("--ssd4rec-source", type=Path, default=Path("vendor/ssd4rec_official/ssd4rec.py"))
    parser.add_argument(
        "--muse-counts",
        type=Path,
        default=Path("data/music4all_onion/main_table_repair_v1/muse_transition_counts.npz"),
    )
    parser.add_argument("--a1-features", type=Path)
    parser.add_argument("--a1-feature-manifest", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--base-training-manifest", type=Path)
    parser.add_argument("--surface", choices=("train_cv", "validation", "test"))
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--train-features", type=Path)
    parser.add_argument("--train-feature-manifest", type=Path)
    parser.add_argument("--validation-features", type=Path)
    parser.add_argument("--validation-feature-manifest", type=Path)
    parser.add_argument("--a1-checkpoint", type=Path)
    parser.add_argument("--a1-result", type=Path)
    parser.add_argument("--a1-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--metadata-slots", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0, 2.0))
    parser.add_argument("--max-context", type=int, default=129)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(); torch.set_num_threads(args.threads)
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "build-features": build_features(args)
    else: train_head(args)


if __name__ == "__main__":
    main()
