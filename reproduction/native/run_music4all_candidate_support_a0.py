#!/usr/bin/env python3
"""Build and evaluate the Music4All candidate-support A0 gate.

The gate is intentionally split into three modes:

* ``build-index`` is CPU-only and uses Train rows only;
* ``export-base`` performs exact SASRec Top-200 inference for the Train-internal
  temporal holdout or the untouched Validation rows;
* ``evaluate`` freezes source/merge choices on the Train holdout and opens
  Validation once.

No Test rows are materialized or accessed.
"""

from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
from scipy import sparse
import torch

from lfm_timeconcat_control import LFMTimeConcatData, exact_catalog_topk, gap_bucket
from music_sequence_core import set_metrics
from run_music4all_baselines import Music4AllSASRec


FIRST_ITEM = 3
INDEX_SCHEMA = "genrec.music4all_onion.candidate_support_index.v1"
RANK_SCHEMA = "genrec.music4all_onion.sasrec_ranked_items.v1"
RESULT_SCHEMA = "genrec.music4all_onion.candidate_support_a0_result.v1"
DECISION_SCHEMA = "genrec.music4all_onion.candidate_support_a0_decision.v1"
SCOPES = ("overall", "30-90d", "90-180d", "180-365d", ">365d")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
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


def atomic_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as compressed:
            json.dump(payload, compressed, separators=(",", ":"), allow_nan=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_sparse(path: Path, matrix: sparse.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    sparse.save_npz(temporary, matrix, compressed=True)
    os.replace(temporary, path)


def load_prepared(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text())
    # Two complete dataset suites share this loader (Onion original +
    # Yambda-500M counterpart).
    if manifest.get("schema") not in (
        "genrec.music4all_onion.prepared.v1", "genrec.yambda500m.prepared.v2"
    ):
        raise ValueError("prepared_schema_mismatch")
    if manifest.get("test_accessed") is not False:
        raise ValueError("test_accessed_before_method_freeze")
    for name, entry in manifest["artifacts"].items():
        path = args.data_dir / name
        if digest(path) != entry["sha256"]:
            raise ValueError(f"input_hash_mismatch:{name}")
    return manifest


def known_targets(events: np.ndarray, row: np.ndarray, horizon: int) -> list[int]:
    _, _, start, end, _, _ = map(int, row)
    values = events[start + 1 : min(end, start + 1 + horizon)]
    return list(dict.fromkeys(int(value) for value in values if int(value) >= FIRST_ITEM))


def recent_anchors(events: np.ndarray, row: np.ndarray, limit: int) -> list[int]:
    _, first_user, start, _, _, _ = map(int, row)
    observed = int(events[start])
    anchors: list[int] = []
    seen = {observed}
    begin = max(first_user, start - 128)
    for value in reversed(events[begin:start]):
        item = int(value)
        if item >= FIRST_ITEM and item not in seen:
            anchors.append(item)
            seen.add(item)
            if len(anchors) == limit:
                break
    return anchors


def temporal_holdout_rows(
    sessions: np.ndarray,
    events: np.ndarray,
    *,
    horizon: int,
    limit: int,
    seed: int,
) -> np.ndarray:
    by_user: dict[int, list[int]] = defaultdict(list)
    for row_id, row in enumerate(sessions):
        user, _, _, _, gap, split = map(int, row)
        if split == 0 and gap >= 30 * 86400 and known_targets(events, row, horizon):
            by_user[user].append(row_id)
    candidates = [(user, rows[-1]) for user, rows in by_user.items() if len(rows) >= 2]
    candidates.sort(
        key=lambda value: hashlib.sha256(f"{seed}:{value[0]}:{value[1]}".encode()).digest()
    )
    selected = candidates[:limit] if limit else candidates
    return np.asarray(sorted(row_id for _, row_id in selected), dtype=np.int64)


def pair_matrix(
    rows: array,
    cols: array,
    vocab_size: int,
) -> sparse.csr_matrix:
    row_values = np.frombuffer(rows, dtype=np.uint32).astype(np.int64, copy=False)
    col_values = np.frombuffer(cols, dtype=np.uint32).astype(np.int64, copy=False)
    values = np.ones(len(row_values), dtype=np.int32)
    matrix = sparse.coo_matrix(
        (values, (row_values, col_values)), shape=(vocab_size, vocab_size), dtype=np.int32
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def build_index(args: argparse.Namespace) -> None:
    prepared = load_prepared(args)
    manifest_path = args.index_dir / "index_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("candidate_support_index_exists")
    events = np.load(args.data_dir / "events.npy", mmap_mode="r")
    sessions = np.load(args.data_dir / "sessions.npy", mmap_mode="r")
    vocab_size = len(np.load(args.data_dir / "train_frequencies.npy", mmap_mode="r"))
    if getattr(args, "cv_rows", None):
        cv_rows = np.load(args.cv_rows, allow_pickle=False)
        eligible = np.asarray([r for r, row in enumerate(sessions)
                               if int(row[5]) == 0 and known_targets(events, row, args.horizon)])
        if (cv_rows.ndim != 1 or not np.issubdtype(cv_rows.dtype, np.integer)
                or not len(cv_rows) or len(np.unique(cv_rows)) != len(cv_rows)
                or not np.isin(cv_rows, eligible).all()):
            raise ValueError("explicit_cv_rows_invalid")
        remaining = np.setdiff1d(eligible, cv_rows)
        if np.intersect1d(sessions[cv_rows, 0], sessions[remaining, 0]).size:
            raise ValueError("explicit_cv_users_overlap")
    else:
        cv_rows = temporal_holdout_rows(
            sessions, events, horizon=args.horizon, limit=args.cv_limit, seed=args.seed
        )
    cv_set = set(map(int, cv_rows))

    x_rows, x_cols = array("I"), array("I")
    h_rows, h_cols = array("I"), array("I")
    cv_x_rows, cv_x_cols = array("I"), array("I")
    cv_h_rows, cv_h_cols = array("I"), array("I")
    full_frequency = np.zeros(vocab_size, dtype=np.int64)
    cv_frequency = np.zeros(vocab_size, dtype=np.int64)
    used_rows = 0
    for row_id, row in enumerate(sessions):
        if int(row[5]) != 0:
            continue
        targets = known_targets(events, row, args.horizon)
        if not targets:
            continue
        used_rows += 1
        _, _, start, _, _, _ = map(int, row)
        observed = int(events[start])
        anchors = recent_anchors(events, row, args.max_anchors)
        is_cv = row_id in cv_set
        for target in targets:
            full_frequency[target] += 1
            if is_cv:
                cv_frequency[target] += 1
            if observed >= FIRST_ITEM and observed != target:
                x_rows.append(observed)
                x_cols.append(target)
                if is_cv:
                    cv_x_rows.append(observed)
                    cv_x_cols.append(target)
            for anchor in anchors:
                if anchor != target:
                    h_rows.append(anchor)
                    h_cols.append(target)
                    if is_cv:
                        cv_h_rows.append(anchor)
                        cv_h_cols.append(target)
        if used_rows % 50_000 == 0:
            print(json.dumps({"event": "index_scan", "train_rows": used_rows}), flush=True)

    x_full = pair_matrix(x_rows, x_cols, vocab_size)
    h_full = pair_matrix(h_rows, h_cols, vocab_size)
    x_cv = pair_matrix(cv_x_rows, cv_x_cols, vocab_size)
    h_cv = pair_matrix(cv_h_rows, cv_h_cols, vocab_size)
    x_fit = (x_full - x_cv).tocsr()
    h_fit = (h_full - h_cv).tocsr()
    x_fit.eliminate_zeros()
    h_fit.eliminate_zeros()
    if (x_fit.data < 0).any() or (h_fit.data < 0).any():
        raise ValueError("negative_crossfit_relation_count")
    fit_frequency = full_frequency - cv_frequency
    if (fit_frequency < 0).any():
        raise ValueError("negative_crossfit_frequency")

    args.index_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.index_dir / "cv_rows.npy", cv_rows, allow_pickle=False)
    np.save(args.index_dir / "fit_target_frequency.npy", fit_frequency, allow_pickle=False)
    np.save(args.index_dir / "full_target_frequency.npy", full_frequency, allow_pickle=False)
    atomic_sparse(args.index_dir / "x_fit.npz", x_fit)
    atomic_sparse(args.index_dir / "h_fit.npz", h_fit)
    atomic_sparse(args.index_dir / "x_full.npz", x_full)
    atomic_sparse(args.index_dir / "h_full.npz", h_full)
    artifacts = {
        name: {"sha256": digest(args.index_dir / name), "bytes": (args.index_dir / name).stat().st_size}
        for name in (
            "cv_rows.npy",
            "fit_target_frequency.npy",
            "full_target_frequency.npy",
            "x_fit.npz",
            "h_fit.npz",
            "x_full.npz",
            "h_full.npz",
        )
    }
    payload = {
        "schema": INDEX_SCHEMA,
        "status": "pass",
        "producer_sha256": digest(Path(__file__)),
        "prepared_sha256": digest(args.manifest),
        "prepared_artifacts": {name: value["sha256"] for name, value in prepared["artifacts"].items()},
        "configuration": {
            "seed": args.seed,
            "horizon": args.horizon,
            "max_anchors": args.max_anchors,
            "cv_limit": args.cv_limit,
            "cv_rule": ("explicit user-disjoint fold" if getattr(args, "cv_rows", None) else
                        "last chronological >=30d Train return per user with a prior Train return; deterministic hash sample"),
            "relation_rows": "all Train sessions except temporal holdout for fit; all Train sessions for final index",
            "targets": "distinct known continuation tracks within horizon",
        },
        "counts": {
            "train_rows": used_rows,
            "cv_rows": int(len(cv_rows)),
            "x_fit_edges": int(x_fit.nnz),
            "h_fit_edges": int(h_fit.nnz),
            "x_full_edges": int(x_full.nnz),
            "h_full_edges": int(h_full.nnz),
        },
        "artifacts": artifacts,
        "test_accessed": False,
    }
    atomic_json(manifest_path, payload)
    print(json.dumps({"status": "pass", "counts": payload["counts"]}, sort_keys=True))


def validate_index(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads((args.index_dir / "index_manifest.json").read_text())
    if payload.get("schema") != INDEX_SCHEMA or payload.get("status") != "pass":
        raise ValueError("index_manifest_invalid")
    if payload.get("prepared_sha256") != digest(args.manifest) or payload.get("test_accessed") is not False:
        raise ValueError("index_prepared_binding_mismatch")
    for name, entry in payload["artifacts"].items():
        if digest(args.index_dir / name) != entry["sha256"]:
            raise ValueError(f"index_artifact_hash_mismatch:{name}")
    return payload


def export_base(args: argparse.Namespace) -> None:
    prepared = load_prepared(args)
    index = validate_index(args)
    if args.output.exists():
        raise FileExistsError("base_rank_export_exists")
    training_manifest = json.loads(args.training_manifest.read_text())
    if (
        training_manifest.get("status") != "pass"
        or training_manifest.get("baseline") != "sasrec_standard"
        or training_manifest.get("test_accessed") is not False
    ):
        raise ValueError("sasrec_training_manifest_invalid")
    if digest(args.checkpoint) != training_manifest.get("best_model_sha256"):
        raise ValueError("sasrec_checkpoint_hash_mismatch")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if checkpoint.get("binding") != training_manifest.get("binding"):
        raise ValueError("sasrec_checkpoint_binding_mismatch")

    data = LFMTimeConcatData(args.data_dir, max_context=args.max_context, horizon=args.horizon)
    # The noise buffer is unused at inference; construct the exact training model
    # through the reviewed factory to avoid any probability-shape ambiguity.
    from run_music4all_baselines import create_model

    model_args = argparse.Namespace(**training_manifest["binding"]["configuration"])
    model = create_model(model_args, data).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    if args.surface == "train_cv":
        row_ids = np.load(args.index_dir / "cv_rows.npy", allow_pickle=False)
        labels = None
    else:
        identities = np.load(args.data_dir / "metric_labels.npz", allow_pickle=False)
        row_ids = identities["row_ids"]
        labels = {
            int(row): identities["values"][identities["offsets"][at] : identities["offsets"][at + 1]]
            for at, row in enumerate(identities["row_ids"])
        }
    rows: list[dict[str, Any]] = []
    for at in range(0, len(row_ids), args.eval_batch_size):
        ids = row_ids[at : at + args.eval_batch_size]
        contexts = [data.context(int(row_id)) for row_id in ids]
        gaps = [data.gap_days(int(row_id)) for row_id in ids]
        serving = data.serving_batch(contexts, gaps, torch.device(args.device))
        hidden = model.encode_all(serving["sequence"], serving["gap_days"])
        states = model.gather_states(hidden, serving["state_positions"])
        ranked = exact_catalog_topk(model, states, k=args.rank_depth).cpu().numpy()
        for offset, row_id in enumerate(ids):
            row = data.sessions[int(row_id)]
            truth = (
                known_targets(data.events, row, args.horizon)
                if labels is None
                else list(dict.fromkeys(map(int, labels[int(row_id)])))
            )
            user, _, _, _, gap, split = map(int, row)
            expected_split = 0 if args.surface == "train_cv" else 1
            if split != expected_split:
                raise ValueError("base_rank_surface_split_mismatch")
            rows.append(
                {
                    "row_id": int(row_id),
                    "user": user,
                    "gap_days": gap / 86400.0,
                    "gap_bucket": gap_bucket(gap / 86400.0),
                    "targets": truth,
                    "ranked_items": list(map(int, ranked[offset])),
                }
            )
        print(
            json.dumps(
                {"event": "base_rank_export", "surface": args.surface, "rows": min(at + len(ids), len(row_ids)), "total": len(row_ids)}
            ),
            flush=True,
        )
    payload = {
        "schema": RANK_SCHEMA,
        "status": "pass",
        "surface": args.surface,
        "rank_depth": args.rank_depth,
        "checkpoint_sha256": training_manifest["best_model_sha256"],
        "training_manifest_sha256": digest(args.training_manifest),
        "prepared_sha256": digest(args.manifest),
        "index_manifest_sha256": digest(args.index_dir / "index_manifest.json"),
        "prepared_artifacts": {name: value["sha256"] for name, value in prepared["artifacts"].items()},
        "index_configuration": index["configuration"],
        "producer_sha256": digest(Path(__file__)),
        "rows": rows,
        "test_accessed": False,
    }
    atomic_gzip_json(args.output, payload)
    print(json.dumps({"status": "pass", "surface": args.surface, "rows": len(rows)}))


def load_rank_bundle(args: argparse.Namespace, path: Path, surface: str) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("schema") != RANK_SCHEMA
        or payload.get("status") != "pass"
        or payload.get("surface") != surface
        or payload.get("rank_depth") != 200
        or payload.get("prepared_sha256") != digest(args.manifest)
        or payload.get("index_manifest_sha256") != digest(args.index_dir / "index_manifest.json")
        or payload.get("producer_sha256") != digest(Path(__file__))
        or payload.get("test_accessed") is not False
    ):
        raise ValueError(f"rank_bundle_invalid:{surface}")
    row_ids = [int(row["row_id"]) for row in payload.get("rows", [])]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"rank_bundle_duplicate_rows:{surface}")
    expected = (
        np.load(args.index_dir / "cv_rows.npy", allow_pickle=False)
        if surface == "train_cv"
        else np.load(args.data_dir / "metric_labels.npz", allow_pickle=False)["row_ids"]
    )
    if row_ids != list(map(int, expected)):
        raise ValueError(f"rank_bundle_row_identity_mismatch:{surface}")
    for row in payload["rows"]:
        ranked = list(map(int, row.get("ranked_items", [])))
        if len(ranked) != 200 or len(set(ranked)) != 200 or min(ranked) < FIRST_ITEM:
            raise ValueError(f"rank_bundle_candidate_contract:{surface}")
    return payload


def ranked_sparse_row(
    matrix: sparse.csr_matrix,
    anchor: int,
    frequency: np.ndarray,
    beta: float,
    limit: int,
) -> list[int]:
    if anchor < FIRST_ITEM or anchor >= matrix.shape[0]:
        return []
    row = matrix.getrow(anchor)
    if row.nnz == 0:
        return []
    items = row.indices
    values = row.data.astype(np.float64)
    scores = np.log1p(values) - beta * np.log1p(frequency[items])
    order = np.lexsort((items, -scores))[:limit]
    return list(map(int, items[order]))


def ranked_history(
    matrix: sparse.csr_matrix,
    anchors: list[int],
    frequency: np.ndarray,
    beta: float,
    limit: int,
) -> list[int]:
    anchors = anchors[:]
    if not anchors:
        return []
    scores_by_item: dict[int, float] = {}
    for position, anchor in enumerate(anchors):
        if anchor < FIRST_ITEM or anchor >= matrix.shape[0]:
            continue
        row = matrix.getrow(anchor)
        recency = 1.0 / math.log2(position + 2.0)
        for item, count in zip(row.indices, row.data):
            value = recency * float(count)
            scores_by_item[int(item)] = scores_by_item.get(int(item), 0.0) + value
    if not scores_by_item:
        return []
    items = np.fromiter(scores_by_item.keys(), dtype=np.int64)
    counts = np.fromiter((scores_by_item[int(item)] for item in items), dtype=np.float64)
    scores = np.log1p(counts) - beta * np.log1p(frequency[items])
    order = np.lexsort((items, -scores))[:limit]
    return list(map(int, items[order]))


def merge_candidates(
    base: list[int],
    x_items: list[int],
    h_items: list[int],
    popularity: list[int],
    *,
    relation_slots: int,
    x_fraction: float,
    budget: int = 200,
) -> list[int]:
    if not (0 <= relation_slots <= budget - 10):
        raise ValueError("invalid_relation_slots")
    output = list(dict.fromkeys(map(int, base[:10])))
    seen = set(output)

    base_tail_slots = budget - 10 - relation_slots
    if base_tail_slots:
        for item in base[10:]:
            if item not in seen:
                output.append(int(item))
                seen.add(int(item))
                if len(output) == 10 + base_tail_slots:
                    break

    x_quota = int(round(relation_slots * x_fraction))
    h_quota = relation_slots - x_quota
    for source, quota in ((x_items, x_quota), (h_items, h_quota)):
        if quota == 0 or len(output) >= budget:
            continue
        added = 0
        for item in source:
            if item >= FIRST_ITEM and item not in seen:
                output.append(int(item))
                seen.add(int(item))
                added += 1
                if added == quota or len(output) == budget:
                    break

    for source in (x_items, h_items, base[10:], popularity):
        if len(output) == budget:
            break
        for item in source:
            if item >= FIRST_ITEM and item not in seen:
                output.append(int(item))
                seen.add(int(item))
                if len(output) == budget:
                    break
        if len(output) == budget:
            break
    if len(output) != budget or len(set(output)) != budget or output[:10] != base[:10]:
        raise ValueError("candidate_merge_contract_failed")
    return output


def user_macro(rows: list[dict[str, Any]], field: str, scope: str) -> float:
    by_user: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if (scope == "overall" or row["gap_bucket"] == scope) and row[field] is not None:
            by_user[int(row["user"])].append(float(row[field]))
    return float(np.mean([np.mean(values) for values in by_user.values()])) if by_user else 0.0


def source_rows(
    args: argparse.Namespace,
    bundle: dict[str, Any],
    x_matrix: sparse.csr_matrix,
    h_matrix: sparse.csr_matrix,
    frequency: np.ndarray,
    *,
    beta_x: float,
    beta_h: float,
    anchor_count: int,
) -> list[dict[str, Any]]:
    data = LFMTimeConcatData(args.data_dir, max_context=args.max_context, horizon=args.horizon)
    cache_x: dict[int, list[int]] = {}
    rows = []
    for entry in bundle["rows"]:
        row_id = int(entry["row_id"])
        row = data.sessions[row_id]
        _, _, start, _, _, _ = map(int, row)
        observed = int(data.events[start])
        if observed not in cache_x:
            cache_x[observed] = ranked_sparse_row(x_matrix, observed, frequency, beta_x, 200)
        anchors = recent_anchors(data.events, row, anchor_count)
        h_items = ranked_history(h_matrix, anchors, frequency, beta_h, 200)
        rows.append(
            {
                **entry,
                "x_items": cache_x[observed],
                "h_items": h_items,
            }
        )
    return rows


def evaluate_candidate_list(
    prediction: list[int], targets: list[int]
) -> tuple[float | None, float | None]:
    relevant = set(int(item) for item in targets if int(item) >= FIRST_ITEM)
    if not relevant:
        return None, None
    found = len(relevant & set(map(int, prediction)))
    return found / len(relevant), float(found > 0)


def score_source_rows(rows: list[dict[str, Any]], field: str, scope: str = "overall") -> float:
    evaluated = []
    for row in rows:
        recall, hit = evaluate_candidate_list(row[field], row["targets"])
        evaluated.append({**row, "source_recall": recall, "source_hit": hit})
    return user_macro(evaluated, "source_recall", scope)


def bootstrap_delta(
    rows: list[dict[str, Any]],
    field_a: str,
    field_b: str,
    scope: str,
    *,
    seed: int,
    draws: int,
) -> dict[str, float | int | None]:
    by_user: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if scope == "overall" or row["gap_bucket"] == scope:
            if row[field_a] is None or row[field_b] is None:
                continue
            by_user[int(row["user"])].append(float(row[field_a]) - float(row[field_b]))
    values = np.asarray([np.mean(items) for items in by_user.values()], dtype=np.float64)
    if not len(values):
        return {
            "delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "users": 0,
        }
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for at in range(draws):
        samples[at] = values[rng.integers(0, len(values), size=len(values))].mean()
    return {
        "delta": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "users": int(len(values)),
    }


def longest_nonempty_gap_scope(rows: list[dict[str, Any]]) -> str:
    """Return the longest hiatus bucket actually represented on this surface."""
    for scope in reversed(SCOPES[1:]):
        if any(row["gap_bucket"] == scope for row in rows):
            return scope
    raise ValueError("no_nonempty_gap_scope")


def evaluate(args: argparse.Namespace) -> None:
    if args.result.exists() or args.decision.exists():
        raise FileExistsError("a0_validation_decision_output_exists")
    load_prepared(args)
    index_manifest = validate_index(args)
    cv_bundle = load_rank_bundle(args, args.cv_base, "train_cv")
    fit_frequency = np.load(args.index_dir / "fit_target_frequency.npy", allow_pickle=False)
    full_frequency = np.load(args.index_dir / "full_target_frequency.npy", allow_pickle=False)
    train_event_frequency = np.load(args.data_dir / "train_frequencies.npy", allow_pickle=False)
    x_fit = sparse.load_npz(args.index_dir / "x_fit.npz").tocsr()
    h_fit = sparse.load_npz(args.index_dir / "h_fit.npz").tocsr()
    x_full = sparse.load_npz(args.index_dir / "x_full.npz").tocsr()
    h_full = sparse.load_npz(args.index_dir / "h_full.npz").tocsr()
    popularity = list(map(int, np.argsort(-train_event_frequency, kind="stable")))
    popularity = [item for item in popularity if item >= FIRST_ITEM]

    beta_grid = (0.0, 0.5, 1.0)
    x_candidates = []
    for beta in beta_grid:
        rows = source_rows(
            args, cv_bundle, x_fit, h_fit, fit_frequency,
            beta_x=beta, beta_h=0.0, anchor_count=1,
        )
        x_candidates.append((score_source_rows(rows, "x_items"), -beta, beta))
    beta_x = max(x_candidates)[2]

    h_candidates = []
    for anchors in (1, 3, 5):
        for beta in beta_grid:
            rows = source_rows(
                args, cv_bundle, x_fit, h_fit, fit_frequency,
                beta_x=beta_x, beta_h=beta, anchor_count=anchors,
            )
            h_candidates.append((score_source_rows(rows, "h_items"), -anchors, -beta, anchors, beta))
    _, _, _, anchor_count, beta_h = max(h_candidates)
    cv_rows = source_rows(
        args, cv_bundle, x_fit, h_fit, fit_frequency,
        beta_x=beta_x, beta_h=beta_h, anchor_count=anchor_count,
    )
    selection_long_scope = longest_nonempty_gap_scope(cv_rows)

    merge_grid = []
    for relation_slots in (40, 80, 120, 160, 190):
        for x_fraction in (0.25, 0.5, 0.75):
            evaluated = []
            for row in cv_rows:
                merged = merge_candidates(
                    row["ranked_items"], row["x_items"], row["h_items"], popularity,
                    relation_slots=relation_slots, x_fraction=x_fraction,
                )
                recall, hit = evaluate_candidate_list(merged, row["targets"])
                base_recall, base_hit = evaluate_candidate_list(row["ranked_items"], row["targets"])
                evaluated.append(
                    {**row, "u_recall": recall, "u_hit": hit, "base_recall": base_recall, "base_hit": base_hit}
                )
            overall = user_macro(evaluated, "u_recall", "overall")
            long = user_macro(evaluated, "u_recall", selection_long_scope)
            base_overall = user_macro(evaluated, "base_recall", "overall")
            base_long = user_macro(evaluated, "base_recall", selection_long_scope)
            objective = (overall - base_overall) + (long - base_long)
            merge_grid.append(
                (objective, -relation_slots, -abs(x_fraction - 0.5), relation_slots, x_fraction, overall, long)
            )
    _, _, _, relation_slots, x_fraction, _, _ = max(merge_grid)

    # Freeze every selectable configuration on Train-CV before opening the
    # one-shot Validation bundle.
    validation_bundle = load_rank_bundle(args, args.validation_base, "validation")
    if (
        cv_bundle.get("checkpoint_sha256") != validation_bundle.get("checkpoint_sha256")
        or cv_bundle.get("training_manifest_sha256")
        != validation_bundle.get("training_manifest_sha256")
    ):
        raise ValueError("base_rank_checkpoint_identity_mismatch")
    validation_rows = source_rows(
        args, validation_bundle, x_full, h_full, full_frequency,
        beta_x=beta_x, beta_h=beta_h, anchor_count=anchor_count,
    )
    decision_long_scope = longest_nonempty_gap_scope(validation_rows)
    positive_frequency = np.log1p(full_frequency[FIRST_ITEM:][full_frequency[FIRST_ITEM:] > 0])
    decile_edges = np.quantile(positive_frequency, np.linspace(0.0, 1.0, 11))
    decile_stats = {
        str(decile): {
            name: {"targets": 0, "hits": 0}
            for name in ("base", "x", "h", "union", "popularity")
        }
        for decile in range(10)
    }
    rows = []
    for row in validation_rows:
        base = list(map(int, row["ranked_items"]))
        x_arm = merge_candidates(base, row["x_items"], [], popularity, relation_slots=190, x_fraction=1.0)
        h_arm = merge_candidates(base, [], row["h_items"], popularity, relation_slots=190, x_fraction=0.0)
        union = merge_candidates(
            base, row["x_items"], row["h_items"], popularity,
            relation_slots=relation_slots, x_fraction=x_fraction,
        )
        pop = merge_candidates(
            base,
            popularity,
            [],
            popularity,
            relation_slots=relation_slots,
            x_fraction=1.0,
        )
        target_set = set(int(item) for item in row["targets"] if int(item) >= FIRST_ITEM)
        output = {
            "row_id": int(row["row_id"]),
            "user": int(row["user"]),
            "gap_bucket": row["gap_bucket"],
            "targets": len(target_set),
            "base_top10": base[:10],
        }
        arms = {"base": base, "x": x_arm, "h": h_arm, "union": union, "popularity": pop}
        base_hits = target_set & set(base)
        for name, prediction in arms.items():
            recall, hit = evaluate_candidate_list(prediction, list(target_set))
            output[f"{name}_recall200"] = recall
            output[f"{name}_hit200"] = hit
            output[f"{name}_target_hits"] = len(target_set & set(prediction))
        for target in target_set:
            target_decile = int(
                np.searchsorted(decile_edges[1:-1], math.log1p(full_frequency[target]), side="right")
            )
            for name, prediction in arms.items():
                decile_stats[str(target_decile)][name]["targets"] += 1
                decile_stats[str(target_decile)][name]["hits"] += int(target in set(prediction))
        union_hits = target_set & set(union)
        output["union_rescued_targets"] = len(union_hits - base_hits)
        output["union_displaced_targets"] = len(base_hits - union_hits)
        output["union_benefited"] = float(len(union_hits) > len(base_hits))
        output["union_harmed"] = float(len(union_hits) < len(base_hits))
        output["top10_preserved"] = union[:10] == base[:10]
        if not output["top10_preserved"]:
            raise ValueError("base_top10_not_preserved")
        rows.append(output)

    metrics: dict[str, Any] = {}
    for scope in SCOPES:
        metrics[scope] = {
            name: {
                "user_macro_recall_at_200": user_macro(rows, f"{name}_recall200", scope),
                "user_macro_any_hit_at_200": user_macro(rows, f"{name}_hit200", scope),
            }
            for name in ("base", "x", "h", "union", "popularity")
        }
        metrics[scope]["union_vs_base"] = bootstrap_delta(
            rows, "union_recall200", "base_recall200", scope, seed=args.seed, draws=args.bootstrap_draws
        )
        metrics[scope]["union_vs_popularity"] = bootstrap_delta(
            rows, "union_recall200", "popularity_recall200", scope,
            seed=args.seed + 1, draws=args.bootstrap_draws,
        )
        selected = rows if scope == "overall" else [row for row in rows if row["gap_bucket"] == scope]
        metrics[scope]["coverage"] = {
            "rows": len(selected),
            "users": len({row["user"] for row in selected}),
            "rescued_targets": int(sum(row["union_rescued_targets"] for row in selected)),
            "displaced_targets": int(sum(row["union_displaced_targets"] for row in selected)),
            "benefited_users": len({row["user"] for row in selected if row["union_benefited"]}),
            "harmed_users": len({row["user"] for row in selected if row["union_harmed"]}),
        }

    critical = [
        metrics[scope][comparison]
        for scope in ("overall", decision_long_scope)
        for comparison in ("union_vs_base", "union_vs_popularity")
    ]
    point_positive = all(item["delta"] is not None and item["delta"] > 0 for item in critical)
    ci_positive = all(item["ci95_low"] is not None and item["ci95_low"] > 0 for item in critical)
    decision = "PASS" if ci_positive else "PARTIAL" if point_positive else "NEGATIVE"
    result = {
        "schema": RESULT_SCHEMA,
        "status": "pass",
        "surface": "validation",
        "base": "sasrec_standard",
        "frozen_configuration": {
            "beta_x": beta_x,
            "beta_h": beta_h,
            "anchor_count": anchor_count,
            "relation_slots": relation_slots,
            "x_fraction": x_fraction,
            "budget": 200,
            "protected_base_prefix": 10,
            "selection_objective": (
                "Train-internal temporal-holdout overall plus "
                f"{selection_long_scope} user-macro Recall@200 delta"
            ),
            "popularity_control": "all-Train event frequency with the same B* tail replacement count as union",
        },
        "train_internal_selection": {
            "x_grid": [
                {"beta": beta, "user_macro_recall_at_200": score}
                for score, _, beta in sorted(x_candidates, reverse=True)
            ],
            "h_grid": [
                {"anchor_count": anchors, "beta": beta, "user_macro_recall_at_200": score}
                for score, _, _, anchors, beta in sorted(h_candidates, reverse=True)
            ],
            "merge_grid": [
                {
                    "objective_delta_sum": item[0],
                    "relation_slots": item[3],
                    "x_fraction": item[4],
                    "overall_recall_at_200": item[5],
                    ">365d_recall_at_200": item[6],
                }
                for item in sorted(merge_grid, reverse=True)
            ],
        },
        "metrics": metrics,
        "source_overlap": {
            "mean_x_h_jaccard": float(
                np.mean(
                    [
                        len(set(row["x_items"]) & set(row["h_items"]))
                        / max(1, len(set(row["x_items"]) | set(row["h_items"])))
                        for row in validation_rows
                    ]
                )
            )
        },
        "target_popularity_deciles": {
            decile: {
                name: {
                    **counts,
                    "target_recall_at_200": counts["hits"] / counts["targets"]
                    if counts["targets"]
                    else 0.0,
                }
                for name, counts in arms.items()
            }
            for decile, arms in decile_stats.items()
        },
        "bindings": {
            "prepared_sha256": digest(args.manifest),
            "index_manifest_sha256": digest(args.index_dir / "index_manifest.json"),
            "cv_base_sha256": digest(args.cv_base),
            "validation_base_sha256": digest(args.validation_base),
            "producer_sha256": digest(Path(__file__)),
        },
        "index_counts": index_manifest["counts"],
        "decision_scopes": ["overall", decision_long_scope],
        "decision": decision,
        "negative_scope": "only the tested Train-only continuation/history relation sources and fixed-budget merge",
        "test_accessed": False,
    }
    decision_payload = {
        "schema": DECISION_SCHEMA,
        "status": "pass",
        "decision": decision,
        "base": "sasrec_standard",
        "validation_rows": len(rows),
        "criteria": {
            "all_four_point_deltas_positive": point_positive,
            "all_four_paired_ci95_lower_bounds_positive": ci_positive,
            "base_top10_preserved_all_rows": all(row["top10_preserved"] for row in rows),
        },
        "decision_scopes": ["overall", decision_long_scope],
        "result_sha256_pending": True,
        "test_accessed": False,
    }
    atomic_json(args.result, result)
    decision_payload["result_sha256"] = digest(args.result)
    decision_payload.pop("result_sha256_pending")
    atomic_json(args.decision, decision_payload)
    print(json.dumps({"status": "pass", "decision": decision, "configuration": result["frozen_configuration"]}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build-index", "export-base", "evaluate"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("refine-logs/music4all_onion/PREPARED_DATASET.json"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/music4all_onion/a0_candidate_index_v1"))
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--surface", choices=("train_cv", "validation"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cv-base", type=Path)
    parser.add_argument("--validation-base", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=129)
    parser.add_argument("--max-anchors", type=int, default=5)
    parser.add_argument("--cv-limit", type=int, default=20_000)
    parser.add_argument("--cv-rows", type=Path, help="Explicit user-disjoint eligible Train rows for full-Train OOF")
    parser.add_argument("--rank-depth", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.mode == "build-index":
        build_index(args)
    elif args.mode == "export-base":
        required = (args.training_manifest, args.checkpoint, args.surface, args.output)
        if any(value is None for value in required):
            raise ValueError("export_base_arguments_missing")
        export_base(args)
    else:
        required = (args.cv_base, args.validation_base, args.result, args.decision)
        if any(value is None for value in required):
            raise ValueError("evaluate_arguments_missing")
        evaluate(args)


if __name__ == "__main__":
    main()
