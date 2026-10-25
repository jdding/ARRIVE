"""Selected native ARRIVE primitives; see coverage.md for scope."""
from __future__ import annotations

import numpy as np


def artist_members(artist_by_item: np.ndarray, frequency: np.ndarray) -> dict[int, list[int]]:
    members: dict[int, list[int]] = {}
    for item in range(3, len(artist_by_item)):
        artist = int(artist_by_item[item])
        if artist >= 0:
            members.setdefault(artist, []).append(item)
    for values in members.values():
        values.sort(key=lambda item: (-float(frequency[item]), item))
    return members
