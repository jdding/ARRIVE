"""Selected native ARRIVE primitives; see coverage.md for scope."""
from __future__ import annotations

import math
import numpy as np
from scipy import sparse


FIRST_ITEM = 3


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
