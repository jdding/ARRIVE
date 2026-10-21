"""Selected native ARRIVE primitives; see coverage.md for scope."""
from __future__ import annotations



def feature_schema():
    from run_arrive_attribution_controls import FEATURE_NAMES
    return list(FEATURE_NAMES) + ["recent_unique_fraction", "log_first_track_transition_count", "first_track_transition_concentration"]
