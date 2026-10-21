"""Selected native ARRIVE primitives; see coverage.md for scope."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from run_arrive_attribution_controls import FEATURE_NAMES


def static_feature_rows(frequency: np.ndarray, artists: np.ndarray, audio: np.ndarray,
                        genre: sparse.csr_matrix, tags: sparse.csr_matrix) -> np.ndarray:
    """Catalog-versioned columns shared by every request; no user or cue input."""
    rows = np.zeros((len(artists), len(FEATURE_NAMES)), dtype=np.float32)
    rows[:, 6] = np.log1p(frequency)
    rows[:, 18] = artists >= 0
    rows[:, 20] = np.linalg.norm(audio, axis=1) > 0
    rows[:, 23] = genre.getnnz(axis=1) > 0
    rows[:, 26] = tags.getnnz(axis=1) > 0
    return rows


def row_feature(ids: np.ndarray, state: np.ndarray, items: np.ndarray, source_maps: list[dict[int, int]],
                gap: float, bucket: int, frequency: np.ndarray, observed: int, history: list[int],
                artists: np.ndarray, tags: sparse.csr_matrix, audio: np.ndarray,
                genre: sparse.csr_matrix, normalized_base_score: np.ndarray | None = None,
                precomputed_history: tuple[np.ndarray, sparse.csr_matrix] | None = None,
                catalog_static_rows: np.ndarray | None = None,
                profile_seconds: dict[str, float] | None = None,
                source_ranks: np.ndarray | None = None,
                dense_history_genre: bool = False) -> np.ndarray:
    # Optional measurement only; production and training callers do not enter
    # the timing path.  The same arrays/operations are used on both paths.
    from time import perf_counter
    tick = perf_counter() if profile_seconds is not None else 0.0
    def mark(name: str) -> None:
        nonlocal tick
        if profile_seconds is not None:
            now = perf_counter()
            profile_seconds[name] = profile_seconds.get(name, 0.0) + now - tick
            tick = now
    ids = np.asarray(ids, dtype=np.int32)
    if normalized_base_score is None:
        raw = items[ids] @ state
        z = (raw - raw.mean()) / max(float(raw.std()), 1e-6)
    else:
        z = np.asarray(normalized_base_score)
        if z.shape != (len(ids),) or not np.isfinite(z).all():
            raise ValueError("normalized_base_score_invalid")
    if source_ranks is None:
        ranks = np.asarray([[m.get(int(item), 0) for m in source_maps] for item in ids], dtype=np.float32)
    else:
        ranks = np.asarray(source_ranks, dtype=np.float32)
        if ranks.shape != (len(ids), len(source_maps)) or not np.isfinite(ranks).all():
            raise ValueError("source_ranks_invalid")
    rrf = np.where(ranks > 0, 1.0 / (60.0 + np.maximum(ranks, 1.0)), 0.0).sum(axis=1)
    rr = np.where(ranks > 0, 1.0 / np.log2(np.maximum(ranks, 1.0) + 1.0), 0.0)
    mark("base_and_source_ranks")
    a_obs = np.asarray(audio[observed])
    a_hist = (precomputed_history[0] if precomputed_history is not None else
              np.asarray(audio[history]).mean(axis=0) if history else np.zeros_like(a_obs))
    g_obs = genre.getrow(observed)
    if precomputed_history is not None:
        g_hist = precomputed_history[1]
    elif dense_history_genre:
        # SciPy's sparse mean already produces a dense row. Keep its exact
        # values and avoid converting it back to CSR before the candidate dot.
        g_hist = (np.asarray(genre[history].mean(axis=0)).ravel() if history
                  else np.zeros(genre.shape[1], dtype=genre.dtype))
    else:
        g_hist = (sparse.csr_matrix(genre[history].mean(axis=0)) if history
                  else sparse.csr_matrix((1, genre.shape[1])))
    hist_genre_nnz = (int(np.count_nonzero(g_hist)) if isinstance(g_hist, np.ndarray)
                      else g_hist.nnz)
    mark("return_and_history_context")
    a_cand = np.asarray(audio[ids])
    g_cand = genre[ids]
    t_cand = tags[ids]
    mark("candidate_content_gather")
    same_artist = ((artists[ids] == artists[observed]) & (artists[observed] >= 0)).astype(np.float32)
    tag_cos = t_cand.dot(tags.getrow(observed).T).toarray().ravel()
    mark("artist_and_tag_similarity")
    audio_obs = a_cand @ a_obs
    audio_hist = a_cand @ a_hist
    mark("audio_similarity")
    genre_obs = g_cand.dot(g_obs.T).toarray().ravel()
    genre_hist = (g_cand.dot(g_hist) if isinstance(g_hist, np.ndarray)
                  else g_cand.dot(g_hist.T).toarray().ravel())
    mark("genre_similarity")
    if catalog_static_rows is not None:
        if catalog_static_rows.shape != (len(artists), len(FEATURE_NAMES)):
            raise ValueError("catalog_static_feature_shape_invalid")
        matrix = catalog_static_rows[ids].copy()
        matrix[:, 0] = z
        matrix[:, 1] = rrf
        matrix[:, 2:6] = rr
        matrix[:, 7] = np.log1p(gap) / 8.0
        matrix[:, 8 + bucket] = 1.0
        matrix[:, 12] = same_artist
        matrix[:, 13] = tag_cos
        matrix[:, 14] = audio_obs
        matrix[:, 15] = audio_hist
        matrix[:, 16] = genre_obs
        matrix[:, 17] = genre_hist
        matrix[:, 19] = artists[observed] >= 0
        matrix[:, 21] = np.linalg.norm(a_obs) > 0
        matrix[:, 22] = np.linalg.norm(a_hist) > 0
        matrix[:, 24] = g_obs.nnz > 0
        matrix[:, 25] = hist_genre_nnz > 0
        matrix[:, 27] = tags.getrow(observed).nnz > 0
    else:
        matrix = np.column_stack((
        z, rrf, rr, np.log1p(frequency[ids]), np.full(len(ids), np.log1p(gap) / 8.0),
        np.tile(np.eye(4, dtype=np.float32)[bucket], (len(ids), 1)), same_artist, tag_cos,
        audio_obs, audio_hist,
        genre_obs, genre_hist,
        artists[ids] >= 0, np.full(len(ids), artists[observed] >= 0),
        np.linalg.norm(a_cand, axis=1) > 0, np.full(len(ids), np.linalg.norm(a_obs) > 0),
        np.full(len(ids), np.linalg.norm(a_hist) > 0), g_cand.getnnz(axis=1) > 0,
        np.full(len(ids), g_obs.nnz > 0), np.full(len(ids), hist_genre_nnz > 0),
        t_cand.getnnz(axis=1) > 0, np.full(len(ids), tags.getrow(observed).nnz > 0),
        )).astype(np.float32)
    mark("feature_matrix_assembly")
    if matrix.shape != (len(ids), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError("common_candidate_feature_invalid")
    mark("feature_validation")
    return matrix
