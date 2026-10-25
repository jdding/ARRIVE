"""Selected native ARRIVE primitives; see coverage.md for scope."""
from __future__ import annotations

import math
from collections import defaultdict
import numpy as np


def row_metrics(predictions: np.ndarray, targets: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    recall = np.zeros(len(predictions), dtype=np.float64)
    ndcg = np.zeros(len(predictions), dtype=np.float64)
    for row, prediction in enumerate(predictions):
        truth = set(map(int, targets[row, : int(counts[row])]))
        hits = [int(int(item) in truth) for item in prediction]
        recall[row] = sum(hits) / max(int(counts[row]), 1)
        ideal_count = min(int(counts[row]), 10)
        ideal = sum(1.0 / math.log2(rank + 2.0) for rank in range(ideal_count))
        ndcg[row] = sum(hit / math.log2(rank + 2.0) for rank, hit in enumerate(hits)) / ideal if ideal else 0.0
    return recall, ndcg


def user_macro(values: np.ndarray, users: np.ndarray, selected: np.ndarray) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for value, user in zip(values[selected], users[selected]):
        grouped[int(user)].append(float(value))
    return float(np.mean([np.mean(group) for group in grouped.values()])) if grouped else 0.0
