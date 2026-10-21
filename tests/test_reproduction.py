"""Synthetic bridge and reporting checks, not a paper experiment rerun."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_export_io import add_wrong_cue_assets, make_export

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from convert_native_export import convert
from export_io import digest, load_cue_treatment, load_export, read_json
from result_summary import contrast, load_predictions, summarize, user_values


class ReproductionBridgeTests(unittest.TestCase):
    def test_native_conversion_preserves_arrays_and_rebuilds_wrong_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_export(root / "native")
            catalog = root / "catalog"
            add_wrong_cue_assets(source, catalog)
            manifest = read_json(source / "EXPORT_MANIFEST.json")
            with np.load(source / "sources.npz", allow_pickle=False) as z:
                sources = {k: z[k] for k in z.files}
            with np.load(source / "CUE_DONORS.npz", allow_pickle=False) as z:
                sources["cues"] = z["cues"]
            np.savez(source / "sources.npz", **sources)
            manifest.update(schema="arrive.framework.export.v1", status="pass", test_accessed=False,
                output_sha256={name: manifest["sha256"][name] for name in (
                    "features.npy", "candidate_ids.npy", "labels.npy", "metadata.npz", "REQUEST_CONTEXT.npz")},
                code_sha256={"arrive_framework_datasets.py": digest(REPO / "reproduction/native/arrive_framework_datasets.py")},
                reproduction_sources_sha256=digest(source / "sources.npz"))
            (source / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest))
            output = root / "converted"
            report = convert(source, output, catalog)
            self.assertIs(report["wrong_cue_bundle"], True)
            for name in manifest["output_sha256"]:
                self.assertEqual(digest(source / name), digest(output / name))
            load_cue_treatment(load_export(output), catalog)
            with self.assertRaises(FileExistsError):
                convert(source, output, catalog)

    def test_wrong_source_version_is_rejected_before_copying(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_export(root / "native")
            manifest = read_json(source / "EXPORT_MANIFEST.json")
            manifest.update(schema="arrive.framework.export.v1", status="pass", test_accessed=False,
                output_sha256={name: manifest["sha256"][name] for name in (
                    "features.npy", "candidate_ids.npy", "labels.npy", "metadata.npz", "REQUEST_CONTEXT.npz")},
                code_sha256={"arrive_framework_datasets.py": "0" * 64})
            (source / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "source_version"):
                convert(source, root / "output")
            self.assertFalse((root / "output").exists())


class ResultSummaryTests(unittest.TestCase):
    def fixture(self, root):
        entries = []
        for seed in (2024, 2025, 2026):
            for method in ("backbone", "arrive", "removed", "mismatched"):
                top = np.tile(np.arange(3, 13), (4, 1))
                if method != "arrive":
                    top[0, 0] = 13
                path = root / f"{method}_{seed}.npz"
                np.savez(path, row_ids=np.arange(4), users=np.asarray([1, 1, 1, 2]),
                    targets=np.tile([3], (4, 1)), target_counts=np.ones(4, dtype=np.int64),
                    bucket_ids=np.arange(4), top10=top)
                entries.append(dict(dataset="synthetic", backbone="sasrec", method=method, seed=seed, path=path.name))
        return dict(rankings=entries, bootstrap_draws={"sasrec": 20}, bootstrap_seed=20261007)

    def test_user_macro_not_request_macro_and_three_seed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            result = summarize(config, root)
            self.assertEqual(len(result["ranking_rows"]), 4)
            comparison = next(row for row in result["paired_comparisons"] if row["contrast"] == "arrive_minus_backbone")
            self.assertAlmostEqual(comparison["recall"]["delta"], 1 / 6)
            self.assertEqual(comparison["users"]["improved_fraction"], 0.5)
            self.assertIs(result["new_training"], False)
            self.assertIs(result["inference_performed"], False)
            self.assertAlmostEqual(result["ranking_rows"][0]["ndcg"]["sample_sd"], 0.0)

    def test_missing_seed_or_target_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "three_matched_seeds"):
                summarize({**config, "rankings": config["rankings"][:-1]}, root)
            path = root / "arrive_2026.npz"
            with np.load(path, allow_pickle=False) as z:
                values = {k: z[k] for k in z.files}
            values["targets"][0, 0] = 14
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "paired_cohort_difference"):
                summarize(config, root)

    def test_hash_and_duplicate_prediction_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.fixture(root)["rankings"][0]
            with self.assertRaisesRegex(ValueError, "hash_mismatch"):
                load_predictions({**entry, "sha256": "0" * 64}, root)
            path = root / entry["path"]
            with np.load(path, allow_pickle=False) as z:
                values = {k: z[k] for k in z.files}
            values["top10"][0, 1] = values["top10"][0, 0]
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "ten_distinct"):
                load_predictions(entry, root)

    def test_paired_interval_averages_seeds_before_resampling(self):
        users = np.asarray([1, 1, 2])
        a = np.asarray([[1., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
        b = np.zeros_like(a)
        result = contrast(a, b, users, np.ones(3, bool), 30, 2024)
        self.assertAlmostEqual(result["delta"], 2 / 3)
        self.assertEqual(result["resampling_unit"], "user_after_seed_averaging")
        np.testing.assert_allclose(result["ci95_pp"], [100 * 2 / 3, 100 * 2 / 3], rtol=0, atol=1e-12)

    def test_coverage_report_contains_recall_and_hit_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coverage = []
            for seed in (2024, 2025, 2026):
                path = root / f"coverage_{seed}.npz"
                np.savez(path, row_ids=np.arange(3), users=np.asarray([1, 1, 2]),
                    full_coverage=np.asarray([.5, 0, 1]), removed_coverage=np.asarray([0., 0, 1]))
                coverage.append(dict(dataset="synthetic", seed=seed, path=path.name))
            result = summarize(dict(coverage=coverage, bootstrap_draws={"sasrec": 20}), root)["coverage"][0]
            self.assertAlmostEqual(result["recall"]["full"]["mean"], .625)
            self.assertAlmostEqual(result["hit_rate"]["full"]["mean"], .75)

    def test_coverage_rejects_nonfinite_or_duplicate_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "bad.npz"
            config = dict(coverage=[dict(dataset="synthetic", seed=2024, path=path.name)])
            for rows, values in (([1, 1], [0., 1.]), ([1, 2], [0., np.nan]), ([1, 2], [0., 2.])):
                np.savez(path, row_ids=np.asarray(rows), users=np.asarray([1, 2]),
                    full_coverage=np.asarray(values), removed_coverage=np.zeros(2))
                with self.assertRaisesRegex(ValueError, "coverage_shape_identity_or_range"):
                    summarize(config, root)


if __name__ == "__main__":
    unittest.main()
