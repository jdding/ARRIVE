"""Selected native ARRIVE primitives; see coverage.md for scope."""
from __future__ import annotations

import numpy as np


FEATURE_NAMES = ['encoder_score_z','rrf_score','encoder_rank','first_track_transition_rank',
                 'history_transition_rank','artist_rank','log_popularity','log_gap',
                 'gap_30_90','gap_90_180','gap_180_365','gap_gt365',
                 'same_artist','tag_cosine_first','audio_cosine_first','audio_cosine_history',
                 'genre_cosine_first','genre_cosine_history','artist_known','first_artist_known',
                 'audio_available','first_audio_available','history_audio_available',
                 'genre_available','first_genre_available','history_genre_available',
                 'tag_available','first_tag_available']


def rrf_pool(sources, budget=200, k=60):
    """Pure inference inputs, deterministic ties, protects exact encoder Top-10."""
    maps = [{int(item): at + 1 for at, item in enumerate(source) if int(item) >= 3} for source in sources]
    universe = sorted(set().union(*maps))
    score = {i: sum(1. / (k + m[i]) for m in maps if i in m) for i in universe}
    protected = list(map(int, sources[0][:10]))
    ranked = sorted((i for i in universe if i not in protected), key=lambda i: (-score[i], i))
    pool = protected + ranked[:budget-len(protected)]
    if len(pool) != budget or len(set(pool)) != budget:
        raise ValueError('rrf_budget_or_identity_invalid')
    ranks = np.array([[m.get(i, 0) for m in maps] for i in pool], np.float32)
    return np.array(pool, np.int32), np.array([score[i] for i in pool], np.float32), ranks
