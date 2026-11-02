"""Check metric accounting against directly recomputed full-list NDCG."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from metrics import row_metrics
from export_io import digest
from result_summary import other_song_rank_terms, song_type_contributions, summarize


def record(top, targets, counts, cues):
    top, targets, counts, cues = map(np.asarray, (top, targets, counts, cues))
    n = len(top)
    recall, ndcg = row_metrics(top, targets, counts)
    return dict(top10=top, targets=targets, target_counts=counts, cues=cues,
                users=np.arange(n) // 2, row_ids=np.arange(n), bucket_ids=np.zeros(n, int),
                recall=recall, ndcg=ndcg)


def test_original_ranks_and_full_target_denominator_are_preserved():
    rng = np.random.default_rng(42)
    n = 40
    top_a = np.asarray([rng.choice(np.arange(3, 33), 10, replace=False) for _ in range(n)])
    top_b = np.asarray([rng.choice(np.arange(3, 33), 10, replace=False) for _ in range(n)])
    # ID 999 is outside both candidate lists but remains in the target denominator.
    targets = np.asarray([rng.choice(np.r_[np.arange(3, 33), 999], 14, replace=False) for _ in range(n)])
    counts = rng.integers(1, 15, size=n)
    cues = rng.choice(np.r_[1, np.arange(3, 33)], size=n)
    a = record(top_a, targets, counts, cues)
    b = record(top_b, targets, counts, cues)
    left, right = song_type_contributions(a), song_type_contributions(b)
    for metric in ("recall", "ndcg"):
        np.testing.assert_allclose(left["replay"][metric] + left["other"][metric], a[metric], atol=1e-15)
    terms = other_song_rank_terms(a, b)
    np.testing.assert_allclose(sum(terms.values()), left["other"]["ndcg"] - right["other"]["ndcg"], atol=1e-15)
    drift = {**b, "cues": b["cues"] + 1}
    with pytest.raises(ValueError, match="paired_cohort_difference:cues"):
        other_song_rank_terms(a, drift)


def test_identity_joint_summaries_include_paired_song_types_and_no_replay():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        entries = []
        for seed in (2024, 2025, 2026):
            for method in ("arrive", "identity", "joint-removal"):
                top = np.tile(np.arange(3, 13), (3, 1))
                if method != "arrive":
                    top[:, :2] = [4, 3]
                values = record(top, [[3, 4, 999], [4, 999, 0], [4, 999, 0]], [3, 2, 2], [3, 3, 1])
                path = root / f"{method}_{seed}.npz"
                np.savez(path, **values)
                entries.append(dict(dataset="synthetic", backbone="ssd4rec", method=method,
                                    seed=seed, path=path.name))
        out = summarize(dict(rankings=entries, bootstrap_draws={"ssd4rec": 20}), root)
        assert len(out["paired_comparisons"]) == 2
        for comparison in out["paired_comparisons"]:
            assert comparison["contrast"] in ("arrive_minus_identity", "arrive_minus_joint-removal")
            assert comparison["ndcg"]["interval_role"] == "98.75_percent_control_interval"
            for metric in ("recall", "ndcg"):
                parts = comparison["song_type"]
                assert parts["replay"][metric]["delta"] + parts["other"][metric]["delta"] == pytest.approx(comparison[metric]["delta"])
            terms = comparison["other_song_rank_terms"]
            assert sum(x["delta"] for x in terms.values()) == pytest.approx(comparison["song_type"]["other"]["ndcg"]["delta"])
            assert comparison["no_replay_requests"]["ndcg"]["requests"] == 1
            assert comparison["no_replay_scope"] == "known_catalog_cue_absent_from_full_target_set"
        # External eligibility-only bundles remain supported, but are now bound
        # in the result receipt and checked when the caller supplies an expectation.
        bundle = root / "cues.npz"
        np.savez(bundle, row_ids=values["row_ids"], users=values["users"], eligible=np.ones(3, bool))
        entries[0].update(cue_bundle=bundle.name, cue_bundle_sha256="0" * 64)
        with pytest.raises(ValueError, match="cue_bundle_hash_mismatch"):
            summarize(dict(rankings=entries, bootstrap_draws={"ssd4rec": 20}), root)
        entries[0]["cue_bundle_sha256"] = digest(bundle)
        out = summarize(dict(rankings=entries, bootstrap_draws={"ssd4rec": 20}), root)
        receipt = next(entry for entry in out["inputs"] if entry["method"] == "arrive" and entry["seed"] == 2024)
        assert receipt["cue_bundle_sha256"] == digest(bundle)
