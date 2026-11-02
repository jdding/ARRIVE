"""Selected native ARRIVE primitives; see coverage.md for scope."""
from __future__ import annotations

import json
import numpy as np


SEEDS = (2024, 2025, 2026)


ITERATIONS = (220, 440)


LOSS = "YetiRank:mode=NDCG;top=10;dcg_type=Base;dcg_denominator=LogPosition"


CONTEXT_MODES = ("base", "statistics", "relations")


CUE_MATCH_COLUMNS = (12, 13, 14, 16)


CUE_MATCH_MODES = ("true", "none", "wrong", "identity")


class IdentityTreatment:
    """Replace matching with observable song identity on a fresh feature block."""
    def __init__(self, cues):
        self.cues = np.asarray(cues)

    def apply(self, block, ids, query_ids):
        if block.ndim != 2 or block.shape[1] != 31 or block.dtype != np.float32 or not block.flags.writeable:
            raise ValueError("identity_requires_writable_statistics31")
        ids, query_ids = np.asarray(ids), np.asarray(query_ids)
        if (ids.shape != (len(block),) or query_ids.shape != ids.shape
                or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 3)
                or not np.issubdtype(query_ids.dtype, np.integer)
                or np.any(query_ids < 0) or np.any(query_ids >= len(self.cues))):
            raise ValueError("identity_candidate_query_alignment")
        block[:, CUE_MATCH_COLUMNS] = 0
        block[:, 12] = ids == self.cues[query_ids]
        return block


def apply_cue_match(block: np.ndarray, mode: str, treatment=None, ids=None, query_ids=None) -> np.ndarray:
    """Treat only a freshly allocated training/prediction block, never its source."""
    if mode not in CUE_MATCH_MODES:
        raise ValueError("unknown_cue_match_mode")
    if mode == "none":
        if block.ndim != 2 or block.shape[1] != 31 or block.dtype != np.float32 or not block.flags.writeable:
            raise ValueError("cue_match_ablation_requires_writable_statistics31")
        block[:, CUE_MATCH_COLUMNS] = 0
    elif mode in ("wrong", "identity"):
        if treatment is None or ids is None or query_ids is None:
            raise ValueError(f"{mode}_cue_treatment_required")
        treatment.apply(block,ids,query_ids)
    return block


def request_relations(data, row_ids, items, audio, genre, tags, artists, x, h, frequency):
    """Request-only fields. Deliberately never read continuation targets."""
    from scipy import sparse
    from candidate_sources import recent_anchors, ranked_sparse_row, ranked_history
    norm_items = items / np.maximum(np.linalg.norm(items, axis=1, keepdims=True), 1e-8)
    result = np.empty((len(row_ids), 18), np.float32)
    x_cache = {}
    for at, rid in enumerate(row_ids):
        session = data.sessions[int(rid)]
        start = int(session[2])
        cue = int(data.events[start])
        past = list(dict.fromkeys(map(int, data.events[max(int(session[1]), start - 128):start][::-1])))[:20]
        q = []
        for matrix in (audio, genre, tags, norm_items):
            if sparse.issparse(matrix):
                v = (matrix.getrow(cue) @ matrix[past].T).toarray().ravel() if past else np.zeros(1)
            else:
                v = matrix[past] @ matrix[cue] if past else np.zeros(1)
            q.extend((float(v.mean()), float(v.max()), float(v.std())))
        if cue not in x_cache:
            x_cache[cue] = ranked_sparse_row(x, cue, frequency, 0., 200)
        history = recent_anchors(data.events, session, 5)
        h_source = ranked_history(h, history, frequency, 0., 200)
        a = artists[past]
        trans = x.getrow(cue).data
        q.extend((float(np.mean((a == artists[cue]) & (a >= 0))) if past else 0.,
                  float(cue in past), len(past) / 20., float(np.log1p(trans.sum())),
                  float(trans.max() / max(float(trans.sum()), 1e-8)) if len(trans) else 0.,
                  len(set(x_cache[cue]) & set(h_source)) / 200.))
        result[at] = q
        if (at + 1) % 2000 == 0:
            print(json.dumps({"event": "context_rows", "rows": at+1, "total": len(row_ids)}), flush=True)
    if not np.isfinite(result).all():
        raise ValueError("request_relation_nonfinite")
    return result


def with_request_context(features: np.ndarray, query_ids: np.ndarray,
                         context: np.ndarray | None, mode: str) -> np.ndarray:
    """Append observable request fields; no base scores or second learner."""
    if mode not in CONTEXT_MODES:
        raise ValueError("unknown_request_context_mode")
    block = np.asarray(features, dtype=np.float32)
    if block.ndim != 2 or block.shape[1] != 28 or len(query_ids) != len(block):
        raise ValueError("invalid_candidate_feature_alignment")
    if mode == "base":
        return block
    if context is None or context.ndim != 2 or context.shape[1] != 18:
        raise ValueError("missing_18_request_fields")
    if not np.isfinite(context).all():
        raise ValueError("nonfinite_request_context")
    if len(query_ids) and (query_ids.min() < 0 or query_ids.max() >= len(context)):
        raise ValueError("invalid_request_context_index")
    columns = slice(14, 17) if mode == "statistics" else slice(None)
    return np.concatenate((block, context[query_ids, columns].astype(np.float32)), axis=1)


def selected_rows(meta: dict, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = meta["offsets"]
    positions = np.concatenate([np.arange(int(offsets[r]), int(offsets[r + 1]), dtype=np.int64) for r in rows])
    group_id = np.repeat(np.arange(len(rows), dtype=np.int32), np.diff(offsets)[rows])
    return positions, group_id


def fit_pool(features, labels, meta, rows, context, mode, threads, chunk=65536,
             cue_match_mode="true", cue_treatment=None, candidate_ids=None):
    """One column-major input; bounded gathers, identical query/candidate order."""
    from catboost import Pool
    widths = np.diff(meta["offsets"])[rows]
    offsets = np.r_[0, np.cumsum(widths)]
    count = int(offsets[-1])
    columns = {"base": 28, "statistics": 31, "relations": 46}[mode]
    if chunk <= 0 or not count:
        raise ValueError("empty_fit_pool_or_invalid_chunk")
    if mode != "base" and (context is None or context.shape != (len(meta["offsets"])-1,18)
                            or not np.isfinite(context).all()):
        raise ValueError("invalid_fit_context")
    values = np.empty((count, columns), dtype=np.float32, order="F")
    targets = np.empty(count, dtype=np.int32)
    groups = np.empty(count, dtype=np.int32)
    for lo in range(0, count, chunk):
        hi = min(lo + chunk, count)
        local = np.arange(lo, hi, dtype=np.int64)
        query = np.searchsorted(offsets, local, side="right") - 1
        positions = meta["offsets"][rows[query]] + local - offsets[query]
        values[lo:hi, :28] = features[positions]
        if mode != "base":
            cols = slice(14,17) if mode == "statistics" else slice(None)
            values[lo:hi, 28:] = context[rows[query], cols]
        apply_cue_match(values[lo:hi], cue_match_mode, cue_treatment,
                        candidate_ids[positions] if candidate_ids is not None else None, rows[query])
        targets[lo:hi] = labels[positions]
        groups[lo:hi] = query
    print(json.dumps({"event":"fit_input_ready", "candidates":count, "features":columns,
                      "feature_bytes":values.nbytes, "layout":"F", "gather_chunk":chunk}), flush=True)
    return Pool(values, targets, group_id=groups, thread_count=threads)


def top10(model: CatBoostRanker, features: np.ndarray, ids: np.ndarray,
          meta: dict, rows: np.ndarray, trees: int,
          context: np.ndarray | None = None, context_mode: str = "base", threads: int = -1,
          batch_rows: int = 256, cue_match_mode: str = "true", cue_treatment=None) -> np.ndarray:
    from data_splits import stable_topk

    offsets = meta["offsets"]
    source = meta["rrf_indices"]
    if batch_rows <= 0:
        raise ValueError("invalid_prediction_batch")
    result = np.empty((len(rows),10),dtype=np.int32)
    for lo in range(0,len(rows),batch_rows):
        current = rows[lo:lo+batch_rows]
        flat = (offsets[current,None] + source[current]).reshape(-1)
        qids = np.repeat(current,source.shape[1])
        block = with_request_context(np.asarray(features[flat]),qids,context,context_mode)
        apply_cue_match(block, cue_match_mode, cue_treatment, ids[flat], qids)
        scores = np.asarray(model.predict(block,ntree_end=trees,thread_count=threads),dtype=np.float64)
        chosen_ids = np.asarray(ids[flat],dtype=np.int32).reshape(len(current),-1)
        result[lo:lo+len(current)] = stable_topk(scores.reshape(len(current),-1),chosen_ids,10)
    return result


def check_train_cv_rows(meta: dict, expected: int | None) -> None:
    """Verify the prepared cohort; never silently resample a larger export."""
    if expected is None:
        return
    rows = meta["row_ids"]
    if not 0 < expected <= 20000 or len(rows) != expected or len(np.unique(rows)) != expected:
        raise ValueError("train_cv_row_contract_mismatch")
