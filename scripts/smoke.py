#!/usr/bin/env python3
"""Deterministic synthetic native-path checks; never reproduce paper metrics."""
import argparse
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))

import numpy as np
from scipy import sparse
from feature_schema import feature_schema
from cue_donors import (availability_codes, controlled_mapping, CueTreatment,
                               verify_unchanged_rows)
from candidate_fusion import rrf_pool
from candidate_features import row_feature, static_feature_rows
from ranking import (LOSS, apply_cue_match, fit_pool, request_relations,
                                     selected_rows, top10, with_request_context)
from catalog_metadata import artist_members
from candidate_sources import (merge_candidates, ranked_history,
                                               ranked_sparse_row, recent_anchors)
from data_splits import stable_topk, user_partition
from metrics import row_metrics, user_macro
import test_cue_donors as native_tests


def sample():
    for name in sorted(n for n in vars(native_tests) if n.startswith('test_')):
        getattr(native_tests, name)()
    rng = np.random.default_rng(2024)
    n, catalog = 6, 300
    items = rng.normal(size=(catalog, 8)).astype(np.float32)
    audio = rng.normal(size=(catalog, 6)).astype(np.float32)
    audio /= np.maximum(np.linalg.norm(audio, axis=1, keepdims=True), 1e-8)
    artists = np.arange(catalog, dtype=np.int32) % 5
    tag_rows = np.eye(7, dtype=np.float32)[np.arange(catalog) % 7]
    genre_rows = np.eye(5, dtype=np.float32)[np.arange(catalog) % 5]
    audio[:3] = 0
    artists[:3] = -1
    tag_rows[:3] = 0
    genre_rows[:3] = 0
    tags = sparse.csr_matrix(tag_rows)
    genre = sparse.csr_matrix(genre_rows)
    frequency = np.arange(catalog, dtype=np.float32) % 16
    frequency[:3] = 0
    edge_rows, edge_cols, values = [], [], []
    for anchor in range(3, 60):
        for shift in range(1, 21):
            item = 223 + ((anchor + shift) % 70)
            edge_rows.append(anchor)
            edge_cols.append(item)
            values.append(21 - shift)
    x = sparse.csr_matrix((np.asarray(values, np.float32), (edge_rows, edge_cols)), shape=(catalog, catalog))
    h = x.copy()
    events, sessions, cues = [], [], []
    for row in range(n):
        begin = len(events)
        events.extend([20 + row, 30 + row, 20 + row, 40 + row, 50 + row])
        start = len(events)
        cue = row + 3 if row != n - 1 else 1
        cues.append(cue)
        events.extend([cue, 4, 5, 4])
        sessions.append([100 + row, begin, start, len(events), 0, 0])
    data = SimpleNamespace(events=np.asarray(events, np.int32), sessions=np.asarray(sessions, np.int64))
    context = request_relations(data, np.arange(n), items, audio, genre, tags, artists, x, h, frequency)
    static = static_feature_rows(frequency, artists, audio, genre, tags)
    members = artist_members(artists, frequency)
    popularity = sorted(range(3, catalog), key=lambda i: (-frequency[i], i))
    matrices, ids, labels, offsets, indices, pools, targets = [], [], [], [0], [], [], []
    for row, cue in enumerate(cues):
        base = np.roll(np.arange(3, 223, dtype=np.int32), row * 11).tolist()
        history = recent_anchors(data.events, data.sessions[row], 5)
        assert len(history) <= 5 and cue not in history
        artist = [i for i in members.get(int(artists[cue]), []) if i != cue][:200]
        sources = [base, ranked_sparse_row(x, cue, frequency, 0., 200),
                   ranked_history(h, history, frequency, 0., 200), artist]
        pool, fused, ranks = rrf_pool(sources)
        assert pool.shape == (200,) and np.array_equal(pool[:10], base[:10])
        assert len(np.unique(pool)) == 200 and np.all(pool >= 3)
        dropped, _, _ = rrf_pool([sources[0], [], sources[2], sources[3]])
        assert len(dropped) == 200 and np.array_equal(dropped[:10], pool[:10])
        old = merge_candidates(*sources[:3], popularity, relation_slots=40, x_fraction=.75, budget=200)
        extras = [i for i in artist if i not in set(old)][:20]
        union = np.asarray(list(dict.fromkeys([*old, *extras, *map(int, pool)])), np.int32)
        position = {int(i): at for at, i in enumerate(union)}
        indices.append([position[int(i)] for i in pool])
        maps = [{int(i): at + 1 for at, i in enumerate(src)} for src in sources]
        state = rng.normal(size=8).astype(np.float32)
        block = row_feature(union, state, items, maps, 45., 0, frequency, cue, history,
                            artists, tags, audio, genre)
        fast = row_feature(union, state, items, maps, 45., 0, frequency, cue, history,
                           artists, tags, audio, genre, catalog_static_rows=static)
        assert np.array_equal(block, fast)
        rrf_at = np.asarray(indices[-1])
        np.testing.assert_allclose(block[rrf_at, 1], fused, rtol=1e-6, atol=1e-8)
        assert np.array_equal(block[rrf_at, 2:6], np.where(ranks > 0, 1 / np.log2(np.maximum(ranks, 1) + 1), 0))
        truth = np.asarray([pool[0], pool[25], 9999], np.int32)
        targets.append(truth)
        matrices.append(block)
        ids.append(union)
        labels.append(np.isin(union, truth).astype(np.int32))
        pools.append(pool)
        offsets.append(offsets[-1] + len(union))
    features, ids, labels = np.concatenate(matrices), np.concatenate(ids), np.concatenate(labels)
    assert len(ids) <= 2000
    meta = dict(offsets=np.asarray(offsets, np.int64), rrf_indices=np.asarray(indices, np.int32),
                row_ids=np.arange(n), users=data.sessions[:, 0], targets=np.asarray(targets),
                target_counts=np.full(n, 3, np.int32), bucket_ids=np.zeros(n, np.int8))
    queries = np.repeat(np.arange(n), np.diff(meta['offsets']))
    values = with_request_context(features, queries, context, 'statistics')
    assert values.shape == (len(ids), 31) and values.dtype == np.float32
    assert len(feature_schema()) == 31 and np.isfinite(values).all()
    removed = apply_cue_match(values.copy(), 'none')
    keep = np.setdiff1d(np.arange(31), [12, 13, 14, 16])
    assert len(keep) == 27 and np.array_equal(removed[:, keep], values[:, keep])
    assert not removed[:, [12, 13, 14, 16]].any()
    av = availability_codes(artists, tags, audio, genre)
    bins = np.floor(np.log2(1 + frequency)).astype(np.int64)
    cues = np.asarray(cues)
    part = (~user_partition(meta['users'])).astype(np.uint8)
    catalog_cues = np.flatnonzero((frequency > 0) & (np.arange(catalog) >= 3))
    donor, source, tier, eligible = controlled_mapping(cues, av[cues], bins[cues], part, 2024,
        catalog_cues, av[catalog_cues], bins[catalog_cues])
    treatment = CueTreatment(donor, artists, tags, audio, genre, eligible)
    wrong = apply_cue_match(values.copy(), 'wrong', treatment, ids, queries)
    assert np.array_equal(wrong[:, keep], values[:, keep])
    assert np.all(donor[eligible] != cues[eligible]) and np.array_equal(av[donor], av[cues])
    verify_unchanged_rows(features, meta['offsets'], eligible)
    assert not eligible[-1] and np.array_equal(wrong[queries == n - 1], values[queries == n - 1])
    pos, groups = selected_rows(meta, np.array([2, 0]))
    assert np.array_equal(ids[pos], np.concatenate([ids[offsets[2]:offsets[3]], ids[:offsets[1]]]))
    assert np.array_equal(groups, np.repeat(np.arange(2), np.diff(meta['offsets'])[[2, 0]]))
    ties = stable_topk(np.ones((1, 12)), np.arange(14, 2, -1)[None], 10)
    assert np.array_equal(ties, np.arange(3, 13)[None])
    ranked = np.asarray(pools)[:, :10]
    recall, ndcg = row_metrics(ranked, meta['targets'], meta['target_counts'])
    assert np.all(recall == 1 / 3)
    ideal = sum(1 / math.log2(i + 2) for i in range(3))
    assert np.allclose(ndcg, 1 / ideal)
    assert user_macro(np.array([0., 1., 1.]), np.array([1, 1, 2]), np.ones(3, bool)) == .75
    return features, ids, labels, meta, context, (donor, artists, tags, audio, genre, eligible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-only', action='store_true', help='Explicitly skip CatBoost model checks.')
    args = parser.parse_args()
    features, ids, labels, meta, context, cue_args = sample()
    report = dict(status='pass', scope='synthetic_native_primitives', requests=len(meta['users']),
        candidate_rows=len(ids), data='deterministic synthetic only', paper_metric_reproduction=False,
        checks=['source ranking', 'protected RRF200', 'source removal', 'union feature normalization',
                '31 FP32 fields', 'static feature parity', 'four-column removal', 'label-free donor matching',
                'unknown cue retained', 'stable ties', 'full-target denominator', 'user macro aggregation',
                'seven unchanged native donor regression checks'])
    if args.core_only:
        report['catboost'] = dict(status='NOT_RUN', reason='explicit --core-only; no substitute predictor')
    else:
        if importlib.util.find_spec('catboost') is None:
            print(json.dumps(dict(report, status='incomplete', catboost=dict(status='NOT_RUN', reason='CatBoost is not installed')), indent=2))
            raise SystemExit(2)
        import catboost
        from catboost import CatBoostRanker
        rows = np.arange(len(meta['users']))
        before = features.copy()
        with tempfile.TemporaryDirectory(prefix='synthetic-smoke-', dir=Path(__file__).resolve().parents[1]) as temp:
            for mode in ('true', 'none', 'wrong'):
                treatment = CueTreatment(*cue_args) if mode == 'wrong' else None
                pool = fit_pool(features, labels, meta, rows, context, 'statistics', 1, chunk=37,
                                cue_match_mode=mode, cue_treatment=treatment, candidate_ids=ids)
                model = CatBoostRanker(iterations=5, depth=6, learning_rate=.035, loss_function=LOSS,
                    random_seed=3024, thread_count=1, task_type='CPU', verbose=False, allow_writing_files=False)
                model.fit(pool)
                ranked = top10(model, features, ids, meta, rows, 5, context, 'statistics', threads=1,
                               batch_rows=2, cue_match_mode=mode, cue_treatment=treatment)
                assert ranked.shape == (len(rows), 10) and len(model.feature_names_) == 31
                assert all(set(top) <= set(ids[meta['offsets'][r] + meta['rrf_indices'][r]]) for r, top in enumerate(ranked))
                path = Path(temp) / (mode + '.cbm')
                model.save_model(str(path))
                restored = CatBoostRanker()
                restored.load_model(str(path))
                assert np.array_equal(ranked, top10(restored, features, ids, meta, rows, 5,
                    context, 'statistics', threads=1, cue_match_mode=mode, cue_treatment=treatment))
                recall, ndcg = row_metrics(ranked, meta['targets'], meta['target_counts'])
                assert np.isfinite(recall).all() and np.isfinite(ndcg).all()
        assert np.array_equal(features, before)
        report['catboost'] = dict(status='pass', version=catboost.__version__, trees=5,
                                  threads=1, arms=['true', 'none', 'wrong'], fit_reload_predict=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
