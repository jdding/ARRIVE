"""CPU-only CLI integration on synthetic exports; never paper-result evidence."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_export_io import add_wrong_cue_assets, make_export


REPO = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("catboost"), "CatBoost is not installed")
class CliSmokeTests(unittest.TestCase):
    def command(self, script, *arguments, expected=0):
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / script), *map(str, arguments)],
            cwd=REPO, capture_output=True, text=True, timeout=90,
        )
        if expected == 0:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_train_reload_three_arms_and_source_replay(self):
        with tempfile.TemporaryDirectory(prefix="arrive-cli-smoke-") as directory:
            root = Path(directory)
            train = make_export(root / "train", "train_cv", 0)
            validation = make_export(root / "validation", "validation", 1000)
            test = make_export(root / "test", "test", 2000)
            catalog = root / "catalog"
            for export in (train, validation, test):
                add_wrong_cue_assets(export, catalog)
            for arm in ("true", "none", "wrong"):
                model = root / ("model_" + arm)
                extra = ["--catalog", catalog] if arm == "wrong" else []
                self.command("train_ranker.py", "--train-export", train,
                    "--validation-export", validation, "--output", model,
                    "--cue-match-mode", arm, "--threads", 1, "--smoke", *extra)
                manifest = json.loads((model / "training_manifest.json").read_text())
                self.assertIn(manifest["selected_trees"], (5, 10))
                self.assertIs(manifest["test_accessed"], False)
                self.assertIs(manifest["synthetic"], True)
                self.assertIs(manifest["paper_metric_reproduction"], False)
                for surface, export in (("validation", validation), ("test", test)):
                    output = root / (arm + "_" + surface)
                    self.command("evaluate_ranker.py", "--export", export,
                        "--model-dir", model, "--output", output, "--threads", 1, *extra)
                    evaluation = json.loads((output / "evaluation.json").read_text())
                    self.assertEqual(evaluation["selected_trees"], manifest["selected_trees"])
                    self.assertIs(evaluation["selection_or_tuning_performed"], False)
                    if surface == "validation":
                        with np.load(model / "validation_predictions.npz", allow_pickle=False) as before:
                            with np.load(output / "predictions.npz", allow_pickle=False) as after:
                                for key in ("top10", "recall", "ndcg"):
                                    np.testing.assert_array_equal(before[key], after[key])
                refused = self.command("evaluate_ranker.py", "--export", train,
                    "--model-dir", model, "--output", root / (arm + "_invalid"),
                    "--threads", 1, *extra, expected=1)
                self.assertIn("evaluation_requires_validation_or_test_export", refused.stderr)
                refused = self.command("train_ranker.py", "--train-export", train,
                    "--validation-export", validation, "--output", model,
                    "--cue-match-mode", arm, "--threads", 1, "--smoke", *extra, expected=1)
                self.assertIn("output_already_exists", refused.stderr)
            coverage = root / "coverage"
            self.command("replay_coverage.py", "--export", test, "--output", coverage)
            report = json.loads((coverage / "coverage.json").read_text())
            self.assertEqual(report["full_coverage"], 0.5)
            self.assertEqual(report["removed_coverage"], 0.5)
            self.assertEqual(report["full_hit_rate"], 1.0)
            self.assertEqual(report["removed_hit_rate"], 1.0)
            self.assertIs(report["paper_metric_reproduction"], False)


if __name__ == "__main__":
    unittest.main()
