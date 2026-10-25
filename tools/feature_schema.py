"""Selected native ARRIVE primitives; see coverage.md for scope."""
from __future__ import annotations



def feature_schema():
    from candidate_fusion import FEATURE_NAMES
    return list(FEATURE_NAMES) + ["recent_unique_fraction", "log_first_track_transition_count", "first_track_transition_concentration"]
