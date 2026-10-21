# Prepared-export interface

These entry points start after backbone scoring, candidate retrieval and feature
construction. They fit and evaluate the ARRIVE ranker on supplied arrays; they
do not prepare Music4All-Onion/Yambda, train SASRec/SSD4Rec, or reproduce the
paper's frozen results. The raw-event export producer is not included.

Install the dependencies in the [quickstart](../ARTIFACT_QUICKSTART.md) first.
All commands below run from the repository root. Output directories must not
already exist. Training uses CPU CatBoost; no GPU is required by these wrappers.

## Required export files

Each export represents one dataset, seed and surface (`train_cv`, `validation`
or `test`). Candidate rows are concatenated across requests. Let `N` be the
number of requests and `M` the total number of candidate rows.

| File | Contract |
|---|---|
| `features.npy` | Finite FP32, shape `(M, 28)`; the base columns in [features.md](features.md) |
| `candidate_ids.npy` | Integer `(M,)`, IDs from 3 through the int32 maximum, unique within each request |
| `labels.npy` | Binary integer `(M,)`, exactly the membership of each candidate in the request's complete target set |
| `metadata.npz` | The arrays below; no Python objects |
| `REQUEST_CONTEXT.npz` | Aligned `row_ids`, `users`, and finite FP32 `relation` of shape `(N, 18)` |
| `EXPORT_MANIFEST.json` | Identity, feature order and SHA-256 bindings, as below |

`metadata.npz` contains:

- `offsets`: integer `(N+1,)`, starting at zero and ending at `M`. Each request
  has at least 200 candidate rows. The union, not just RRF200, is used for fitting.
- `rrf_indices`: integer `(N,200)`, unique **request-local** indices into that
  request's union. They are not global row offsets. Their order is the original
  protected RRF200 order.
- `row_ids`, `users`: nonnegative integer `(N,)`; row IDs are unique and must
  remain disjoint between the supplied Train and Validation exports. User IDs
  must retain their identity across requests; temporal surfaces may share users.
- `targets`: integer `(N,L)`; `target_counts` gives each row's valid prefix.
  Targets are distinct, with at least one per request. IDs `-1,0,1,2` are
  reserved and cannot appear in a valid prefix. Distinct unseen targets may use
  IDs below `-1` and remain in metric denominators. Do not replace them by one
  shared unknown ID or discard them.
- `target_counts`: integer `(N,)`, between 1 and `L`.
- `bucket_ids`: integer `(N,)`, 0 through 3 for the four native gap bins.
  The upstream producer must preserve the original endpoint rules.

`REQUEST_CONTEXT.npz` must have exactly the same request order. Its columns
14–16 (zero-based) supply the three extra request statistics; the full 18-column
native relation matrix is retained as the input contract. See
[features.md](features.md) for their definitions. All NumPy loads disable pickle.

The manifest has this structure; replace the placeholders and populate the
complete feature-name list from `configs/feature_schema.json` (each `fields`
entry's `name`, in index order):

```json
{
  "schema": "arrive.reviewer_export.v1",
  "dataset": "DATASET_NAME",
  "surface": "train_cv",
  "seed": 2024,
  "synthetic": false,
  "feature_schema": ["ALL_31_NAMES_IN_ORDER"],
  "uses_future_labels_for_features": false,
  "sha256": {
    "features.npy": "SHA256_HEX",
    "candidate_ids.npy": "SHA256_HEX",
    "labels.npy": "SHA256_HEX",
    "metadata.npz": "SHA256_HEX",
    "REQUEST_CONTEXT.npz": "SHA256_HEX"
  }
}
```

Supported seeds are 2024, 2025 and 2026. Use the same dataset, seed, synthetic
flag and feature order across paired exports. A file hash checks byte identity;
it does **not** establish that features, counts or sources were created without
future data. The producer must enforce the time boundaries in
[protocol.md](protocol.md), including the Train-frozen catalog and indexes.
The evaluator cannot establish Train/Test request disjointness or chronology
from hashes and dataset names alone; those are also the producer's responsibility.

## Train, calibrate and evaluate

```sh
python scripts/train_ranker.py \
  --train-export data/train_cv --validation-export data/validation \
  --cue-match-mode true --threads 4 --output outputs/correct

python scripts/evaluate_ranker.py \
  --export data/test --model-dir outputs/correct \
  --threads 4 --output outputs/correct_test
```

The training wrapper divides `train_cv` by the native stable user hash. It fits
on positive-containing requests from the fitting users and calibrates on all
requests from the other users. It selects 220 or 440 trees by Train-internal
user-macro NDCG@10, preferring fewer trees on ties. Validation is evaluated only
after selection; no Test export is accepted by the training command.

Training writes `yeti_ndcg.cbm`, `training_manifest.json` and
`validation_predictions.npz`. The model can contain 440 trees even when 220
were selected. The evaluator uses `selected_trees`, checks the model hash and
export identity, and writes `evaluation.json` plus `predictions.npz`. Scores
are fractions, not percentages. No selection or tuning occurs in evaluation.

For matching removal, rerun the same command with `--cue-match-mode none` and a
different output directory. This zeros only columns 12, 13, 14 and 16 in both
fitting and prediction; it refits the ranker. Do not zero a fitted correct-arm
model's inputs and call that the removal control.

The CLI accepts `--smoke` only for exports declared `synthetic: true`; it uses
5/10 trees rather than 220/440. Synthetic fixture builders are in
`tests/test_export_io.py`. Neither smoke scores nor results on replacement
exports should be presented as reproduction of the paper's metrics.

## Wrong-song matching

Use the same exports and fitting procedure, adding a label-free donor bundle
and a shared Train-only catalog:

```sh
python scripts/train_ranker.py \
  --train-export data/train_cv --validation-export data/validation \
  --cue-match-mode wrong --catalog data/catalog \
  --threads 4 --output outputs/mismatched

python scripts/evaluate_ranker.py \
  --export data/test --model-dir outputs/mismatched \
  --catalog data/catalog --threads 4 --output outputs/mismatched_test
```

Each export additionally contains `CUE_DONORS.npz`, with aligned `row_ids` and
`users`, integer `(N,)` arrays `cues`, `donors`, `source_rows`, `tiers`,
`partition`, and a boolean `(N,)` `eligible` array. `source_rows` uses local
request indices, with `-1` for catalog fallback/unchanged cases. Unknown cues
use ID 1. The native selector defines the tier codes. For `train_cv`, partition
0 means fitting users and 1 means calibration users; Validation and Test each
use their own all-zero partition. Do not pool donors across these surfaces.

The export manifest must bind `CUE_DONORS.npz` in `sha256` and include
`catalog_id`, `catalog_manifest_sha256`, and
`donor_selector_reads_targets: false`.

The catalog directory contains:

| File | Contract |
|---|---|
| `artists.npy` | Integer `(V,)`, scalar primary artist or `-1` for missing |
| `audio.npy` | Finite FP32 `(V,D)` using the native representation |
| `tags.npz`, `genre.npz` | Finite FP32 sparse CSR content matrices with `V` rows |
| `train_frequencies.npy` | Finite, nonnegative integer-valued `(V,)` Train counts |
| `CATALOG_MANIFEST.json` | Schema `arrive.reviewer_catalog.v1`, matching `catalog_id`, `counts_scope: "train_only"`, `uses_future_labels: false`, and `sha256` bindings for the five assets |

Rows 0–2 are reserved: artist `-1`, zero content and zero counts. All candidate
IDs must address catalog rows. The catalog and the cue matching fields must use
the representations and normalizations in [features.md](features.md).

The loader reruns the native availability/popularity-bin donor selector and
checks its mapping, partition and eligibility against the supplied bundle. It
also checks the actual cue's content against the original matching and
availability fields. It does not reconstruct cues or Train counts from raw
events. The other 27 features, candidate pools and actual-cue availability flags
are kept unchanged. A usable cue without a valid donor is an error; zero-content
and unknown cues retain their zero matching fields.

## Equal-budget source replay

```sh
python scripts/replay_coverage.py \
  --export data/test --output outputs/source_coverage
```

The export must additionally contain `sources.npz` with aligned `row_ids`,
`users`, integer `source_ids` of shape `(N,4,200)` padded with zeros, and integer
`source_lengths` of shape `(N,4)`. The native v2 replay interface requires 200
backbone candidates and at most 200 items in each other complete source list.
This is the supported replay input contract, not a claim about all possible
retrieval implementations. Partial lists must not be padded and declared
complete.

Bind `sources.npz` in the export manifest's `sha256` and add:

```json
{
  "complete_source_lists": true,
  "source_order": ["backbone", "first_track_transition", "history_transition", "artist"]
}
```

The replay checks source-rank encodings, reconstructs the full RRF200 exactly,
then removes the first-song transition source and refills to the same 200-item
budget while protecting the backbone Top-10. It reports user-macro Recall@200
and Hit Rate@200 for both pools, plus request changes and target exchanges.
`lost_relevant_total` counts targets present in the full pool but not the removed
pool; `gained_relevant_total` counts the reverse. These are candidate-coverage
results, not final ranking quality. The wrapper does not compute the paper's
confidence intervals or aggregate seeds.
