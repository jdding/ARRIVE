"""Selected native ARRIVE primitives; see coverage.md for scope."""
from __future__ import annotations

import hashlib
import numpy as np


def user_partition(users: np.ndarray, fraction: float = 0.8) -> np.ndarray:
    """Return a stable fit mask; rows from one user never cross the boundary."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("invalid_user_partition_fraction")
    unique = np.unique(users.astype(np.int64, copy=False))
    fit_users = {
        int(user)
        for user in unique
        if int.from_bytes(hashlib.sha256(f"a1-fit:{int(user)}".encode()).digest()[:8], "big")
        < int(fraction * (1 << 64))
    }
    mask = np.fromiter((int(user) in fit_users for user in users), dtype=bool, count=len(users))
    if not mask.any() or mask.all():
        raise ValueError("degenerate_user_partition")
    return mask


def stable_topk(scores: np.ndarray, candidates: np.ndarray, k: int = 10) -> np.ndarray:
    output = np.empty((len(scores), k), dtype=np.int32)
    for row in range(len(scores)):
        order = np.lexsort((candidates[row], -scores[row]))[:k]
        output[row] = candidates[row, order]
    return output
