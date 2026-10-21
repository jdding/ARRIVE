# ARRIVE

Code for **ARRIVE: First-Song Conditioning for Post-Hiatus Music Recommendation**.
ARRIVE combines sequence-model candidates, first-song transitions, historical
transitions and same-artist songs in a protected RRF200 pool, then ranks that
pool with 31-feature CatBoost YetiRank.

This release contains the main-result source chain: data preparation, SASRec
and SSD4Rec backbone training, Train-only indexes, candidate/feature export,
matching controls, evaluation and result analysis. The
[reproduction guide](reproduction/README.md) gives the stage order, commands
and recorded settings. Datasets and trained models are not distributed.
The chain has not been rerun on the real datasets for this release;
[coverage and limits](docs/coverage.md) separates source coverage from verification.

## Start here

Use Python 3.12 and a CPU. No music data, GPU or network access is needed after
installing dependencies:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/smoke.py
```

The smoke builds six synthetic requests and checks retrieval, RRF200, features,
the three matching treatments, metrics and five-tree CatBoost fit/save/reload.
It should finish in under a minute and end with `"status": "pass"` and
`"catboost": {"status": "pass", ...}`. Temporary models are removed. The printed
values are test outputs, not paper results.

For a NumPy/SciPy-only check, use `python scripts/smoke.py --core-only`.
This explicitly reports CatBoost as `NOT_RUN`; it is not a model-training check.
See the [quickstart](ARTIFACT_QUICKSTART.md) for the test suite and export commands.

## Read the method and protocol

| File | Contents |
|---|---|
| [docs/protocol.md](docs/protocol.md) | Task, fixed candidate pools, controls, fitting/calibration, metrics and uncertainty |
| [docs/features.md](docs/features.md) | All 31 columns, transformations, missing content and Yambda substitutions |
| [docs/exports.md](docs/exports.md) | Prepared-input schema and executable export-level commands |
| [docs/baselines.md](docs/baselines.md) | Baseline adaptations and recorded configuration scope |
| [docs/data_acquisition.md](docs/data_acquisition.md) | Dataset-provider links and preparation boundaries |
| [reproduction/README.md](reproduction/README.md) | Main-result pipeline, both backbones, input preparation and analysis |
| [docs/coverage.md](docs/coverage.md) | What can be checked here and what is not distributed |
| [configs/protocol.json](configs/protocol.json) | Machine-readable method settings |
| [configs/feature_schema.json](configs/feature_schema.json) | Exact feature order |

The matching controls change only columns `12, 13, 14, 16`; the other 27 inputs
and the candidate pool remain fixed. Correct, removed and mismatched arms use
the same fitting and calibration procedure. The wrong-song arm needs a
label-free donor bundle and catalog metadata; it cannot be reconstructed from
the four matching values alone.

## Code layout

- `tools/` contains the native retrieval, feature, donor, ranking and metric
  functions, plus validated export I/O. Historical module names are retained
  to keep the native functions and imports intact.
- `scripts/` contains the synthetic smoke and prepared-export entry points.
- `reproduction/native/` contains the main-result dependency closure and data
  producers; `reproduction/configs/` records the 12 backbone runs and 30
  comparison-baseline configurations.
- `tests/` checks export contracts and error handling.
- `docs/native_training_selection.txt` preserves the original fitting and
  calibration excerpt for comparison with the runnable wrapper.

The ranker is trained on the exported candidate union. Calibration and inference
rank only the RRF200 subset. Saved model manifests record `selected_trees`; do
not assume that every tree in a saved model was selected for evaluation.

## Reproduction boundary

Synthetic checks test implementation behavior. Running these entry points on
new exports tests those exports; it does not recreate the frozen paper cohort.
The paper's complete-system comparisons use SASRec and SSD4Rec, while its
fixed-pool matching and transition-source controls use SASRec. No new result,
backbone-general matching claim or online-business result is supplied here.
The original baseline search history is incomplete. SSD4Rec needs the pinned
upstream blocks and a compatible Linux CUDA/Mamba environment. Source inclusion
does not establish fresh-run metric equality or equal baseline tuning budgets.

Source is provided under the existing [MIT license](LICENSE). Dataset and
dependency rights remain with their respective providers; see [NOTICE.md](NOTICE.md).
