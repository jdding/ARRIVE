# Quickstart

Download the repository ZIP, extract it, and open a terminal in its root.
Use Python 3.12; no private paths or credentials are required.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python scripts/smoke.py
.venv/bin/python -m pytest -q tests tools/test_arrive_cue_donors.py
```

The smoke uses synthetic data only. It should finish within a minute and print
`status: pass` with a passing CatBoost fit/reload/predict check. The tests should
report no failures. Neither command downloads music data or reports scientific
performance. `scripts/smoke.py --core-only` skips the CatBoost check explicitly.

Prepared-data entry points:

```sh
.venv/bin/python scripts/train_ranker.py --help
.venv/bin/python scripts/evaluate_ranker.py --help
.venv/bin/python scripts/replay_coverage.py --help
```

Read [the export schema](docs/exports.md) before supplying data. Training and
evaluation require aligned candidate exports, not raw listening logs. Source
coverage replay additionally requires complete ranked source lists. Wrong-song
matching requires the donor/content contract described there. Outputs use a
new directory rather than overwrite an existing result.

Read [protocol](docs/protocol.md), [features](docs/features.md), and
[coverage](docs/coverage.md) to map these checks to the paper. The original
datasets, models and frozen exports are not included.

For the raw-data-to-results source chain and both backbone configurations,
continue with [reproduction/README.md](reproduction/README.md). Its optional
`python scripts/smoke_reproduction.py` check additionally needs the reproduction
dependencies. It exercises native SASRec forward/export and all three ranker
arms on a tiny synthetic fixture, not a real-data or GPU reproduction.
