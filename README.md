# ARRIVE

Code for **ARRIVE: Return-Song-Guided Music Recommendation after Listening Hiatuses**.

ARRIVE combines candidates from a frozen sequence model, first-song transitions,
history transitions and the first song's artist. It protects the backbone's
top ten within a 200-song pool, then ranks candidates with 31-feature CatBoost
YetiRank. Both SASRec and SSD4Rec source pipelines are included.

## Quick check

Use Python 3.12. After installing dependencies, this check needs no music data,
network connection or GPU:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/smoke.py
```

It builds six synthetic requests, checks retrieval, protected RRF200, features,
the three matching treatments and metrics, and fits/saves/reloads a five-tree
CatBoost model. It normally finishes within a minute and prints `"status": "pass"`
with a passing `catboost` check. Temporary models are removed. The printed
values are test outputs, not paper results.

To run the tests:

```sh
.venv/bin/python -m pip install -r requirements/test.txt
.venv/bin/python -m pytest -q tests
```

`scripts/smoke.py --core-only` runs the NumPy/SciPy checks and explicitly skips
CatBoost. Native training/export checks additionally need
`requirements/native.txt`; the SSD4Rec CUDA/Mamba environment is described in
the reproduction guide.

## Reproduce the paper

Follow [reproduction/README.md](reproduction/README.md) for data acquisition,
preparation, backbone training, Train-only indexes, candidate exports, matching
controls and saved-result analysis. It contains the stage order, commands and
recorded settings. Obtain the public datasets under their providers' terms;
datasets, paper checkpoints and original per-user predictions are not bundled.

| Paper result | Code and protocol |
|---|---|
| Table 1: complete-system results | Both backbone pipelines and comparison adapters; each ARRIVE/backbone pair shares its actual frozen checkpoint within each seed |
| Table 2: matching controls | Correct/removed/mismatched fields with candidates and the other 27 inputs fixed; both backbones reuse Table 1's ARRIVE models as arm A and share its frozen checkpoints within each seed |
| Table 2: identity and joint controls | SSD4Rec identity-only ranking on the full pool; joint removal rebuilds pools without explicit return-song sources and fields, using the same fitting and selection rules |
| Table 3: candidate coverage | SASRec source replay with a fixed 200-song budget and protected backbone top ten |
| Figure 2 and user analysis | Seed-averaged user gains, harm/zero/improvement, strata and ECDF coordinates |

Analysis uses 10,000 user bootstrap draws. The four original SASRec matching
contrasts and four SSD4Rec follow-up contrasts are adjusted as separate families.
See [docs/protocol.md](docs/protocol.md) for aggregation and interval definitions,
and [metric decomposition](docs/metric_decomposition.md) for replay/other-song
contributions and gained/lost/shared-hit accounting.

The source chain is supplied for inspection and reproduction. It has not been
rerun on the real datasets for this release. See [docs/coverage.md](docs/coverage.md)
for execution checks and the remaining historical asset/configuration limits.

## Repository layout

| Directory | Purpose |
|---|---|
| `scripts/` | Commands for smoke checks, prepared exports, fitting, evaluation and analysis |
| `tools/` | Lightweight retrieval, feature, donor, ranking, metric and export-I/O functions |
| `reproduction/` | Raw-data preparation and full backbone/export implementation, plus recorded configurations |
| `configs/` | Task protocol and the ordered 31-feature schema |
| `requirements/` | Optional test and native-pipeline dependencies |
| `tests/` | Synthetic algorithm, input-contract and CLI checks |
| `docs/` | Protocol, features, data sources, adapters and reproduction scope |

Useful references: [features](docs/features.md), [export schema](docs/exports.md),
[baseline adaptations](docs/baselines.md), [data acquisition](docs/data_acquisition.md).
The ranker fits the exported candidate union; calibration and inference rank
the RRF200 subset. Evaluation uses the recorded `selected_trees` value.

Code uses the [MIT license](LICENSE). Dataset and dependency rights remain with
their providers; see [NOTICE.md](NOTICE.md).
