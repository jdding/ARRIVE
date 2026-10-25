# Reproducing the main results

The main-result chain is supplied as source: raw-data preparation, full and
cross-fit backbones, Train-only indexes, candidate/feature export, the three
matching arms, Test evaluation, source replay and three-seed analysis. This
release does not distribute datasets or checkpoints. Obtain the public data
from the providers in [data_acquisition.md](../docs/data_acquisition.md).

The added entry points were checked with synthetic inputs and CLI/import tests.
The real-data training and GPU chain was **not rerun for this release**. Supplying
source code is not evidence that a fresh run has matched the paper's scores.

## Environments

For CPU checks and SASRec source execution, use Python 3.12 and:

```sh
python -m pip install -r requirements/native.txt -r requirements/test.txt
python scripts/smoke.py
python scripts/smoke_reproduction.py
python -m pytest -q
```

`requirements/native.txt` records the CPU import/test environment, not
the historical GPU environment. The recorded SSD4Rec environment was Python
3.10.15, PyTorch 2.1.1+cu118, Mamba-SSM 2.2.2, causal-conv1d 1.4.0, RecBole
1.2.0 and transformers 4.46.3. Use a separate compatible Linux CUDA environment
for SSD4Rec; its native BiSSD blocks are not replaced by a CPU approximation.
The adapter loads the official blocks rather than the RecBole model shell.
CatBoost is version 1.2.10 in both the paper and this release.

Acquire the pinned upstream SSD4Rec file when using that backbone:

```sh
python scripts/fetch_ssd4rec_source.py
```

The downloader verifies both the upstream SHA-256 and the recorded terminal-
newline normalization. It does not install CUDA dependencies or train a model.
The upstream source retains its own rights and terms; it is not relicensed by
this repository. No third-party SSD4Rec source is bundled here.

## Stage map

All commands run from the repository root. `scripts/native.py MODULE ...`
executes the included file in `reproduction/native/` with its local imports.
The historical filenames are retained. Some dependency modules contain earlier
research branches; they are not the short-paper method and are not required
by the commands below. Do not substitute joint/neural/ILMART branches for the
`statistics` CatBoost ranker.

| Stage | Entry point | Output / invariant |
|---|---|---|
| Onion preparation | C++ producers and Python finalizers below | Train-frozen vocabulary, aligned events/sessions, complete targets |
| Yambda preparation | `prepare_arrive_yambda_500m`, `materialize_yambda_sealed_test` | Correct elapsed-second v2 protocol, not the superseded five-second-tick interpretation |
| Content | `prepare_onion_content.py`, `derive_arrive_yambda_content`, `prepare_artist_mapping.py` | Catalog-aligned normalized content and deterministic primary artist |
| Full backbone | `train_backbone.py` | Recorded SASRec or SSD4Rec settings, Validation checkpoint selection |
| Train index | `run_music4all_candidate_support_a0 build-index` | `x/h_fit`, `x/h_full`, frequencies and exact held-out Train rows |
| Auxiliary backbone | `run_music4all_candidate_support_a1 train-crossfit-base` | Excludes those held-out rows; epoch matches full-backbone selection |
| Candidate export | `run_arrive_full_rcce_control framework_export` | Native 28+3 features, fixed RRF200, full four-source lists and actual cues |
| Portable binding | `convert_native_export.py` | Array bytes unchanged; schema/source hashes checked; optional wrong-cue mapping |
| Matching controls | `train_ranker.py`, `evaluate_ranker.py` | Three independent fits, Train-internal tree selection, no Test tuning |
| Coverage | `replay_coverage.py` | Full vs removed first-song source, same budget 200 |
| Tables/user analysis | `summarize_results.py` | User-macro means, sample SD, paired intervals, harm/zero/improvement and unsmoothed ECDF data |

The bounded paper fitting cohort is 20,000 Onion Train-held-out requests and
17,370 Yambda Train requests. The retained full-Train fold utility is not an
instruction to expand Onion fitting to 293,201 requests.

## 1. Prepare data and content

Keep the registry layout in `native/arrive_framework_datasets.py`, relative to
your `--project` directory. In particular, Onion uses
`data/music4all_onion/prepared_v1`; Yambda uses
`prepared_data/yambda500m_prepared_v2_seconds`. Do not mix Train/Validation
prepared arrays with the separately materialized Test arrays.

For Onion, compile the two existing C++17 producers:

```sh
c++ -O3 -std=c++17 reproduction/native/prepare_music4all_onion.cpp -o /tmp/arrive_prepare_onion
c++ -O3 -std=c++17 reproduction/native/materialize_music4all_onion_test.cpp -o /tmp/arrive_prepare_onion_test
```

Feed the decompressed provider event TSV on stdin, with header
`user_id<TAB>track_id<TAB>timestamp`. The first program takes a new output
directory. The Test producer takes a different new directory and the Train
`track_identity.tsv`; it does not rebuild the vocabulary using Test.
The producer excludes ambiguous timestamps and retains unseen target identities
in the metric denominator. Its date cutoffs are before 2018 for Train, calendar
2018 for Validation and 2019–2020 for Test.

Run `finalize_music4all_onion_prepared --data ... --source COMPRESSED_SOURCE
--producer /tmp/arrive_prepare_onion --manifest ... --labels-manifest ...` and
`finalize_music4all_onion_test --help` through `scripts/native.py`. These gates
deliberately require the original release counts and source digest. A renamed
or different provider release is not the frozen paper input.

The current C++ Train/Validation producer also writes `event_times.npy`. Run
`finalize_music4all_event_times --frozen PREPARED --rebuilt PREPARED --output
data/music4all_onion/baseline_features_v1` to make its timestamp binding; that
finalizer checks array parity before admitting the timestamps.

Normalize Onion musicnn and genre features without running the earlier neighbor
study:

```sh
python scripts/prepare_onion_content.py \
  --data-dir data/music4all_onion/prepared_v1 \
  --musicnn data/music4all_onion/raw/id_musicnn.tsv.bz2 \
  --genre data/music4all_onion/raw/id_genres_tf-idf.tsv.bz2 \
  --output data/music4all_onion/component_b2_attainability_v1
```

Use the provider `id_tags.csv` and `id_information.csv` files at the registry
locations; despite their extensions, the native loader expects the provider's
tab-separated headers. It implements the exact TF-IDF/normalization and scalar
artist mapping, rather than a replacement similarity function.

For Yambda:

```sh
python scripts/native.py prepare_arrive_yambda_500m \
  --events datasets/Yambda/multi_event.parquet \
  --output prepared_data/yambda500m_prepared_v2_seconds
python scripts/native.py prepare_arrive_yambda_500m --finalize \
  --output prepared_data/yambda500m_prepared_v2_seconds
python scripts/native.py derive_arrive_yambda_content \
  --prepared prepared_data/yambda500m_prepared_v2_seconds \
  --embeddings datasets/Yambda/embeddings.parquet \
  --artist-mapping datasets/Yambda/artist_item_mapping.parquet \
  --output data/arrive_yambda_500m/content_v1
python scripts/prepare_artist_mapping.py \
  --input datasets/Yambda/artist_item_mapping.parquet \
  --output datasets/Yamba_50M/artist_mapping.npz
```

`Yamba_50M` is the retained registry folder name, not a choice of the 50M event
subset. The paper uses the registered **500M event file**. Timestamps are already
seconds rounded to multiples of five: do not multiply them by five. Train ends
at 19,500,000 seconds and Validation at 22,100,000. Read
`materialize_yambda_sealed_test --help` to materialize Test against the finalized
Train vocabulary. The producer/finalizers contain the release hashes and count
checks; all data still comes from the provider, not this repository.

## 2. Train each backbone and auxiliary model

The exact 12 recorded dataset/backbone/seed configurations, selected epochs and
available stopping/optimizer fields are in [backbone_training.json](configs/backbone_training.json).
The SASRec objective is `continuation_only`; the SSD4Rec task adapter uses
`oneshot_only`. Do not overwrite these with a shared guessed objective.

`train_backbone.py` prints its argv by default. Add `--execute` only when you
intend to run training on your own machine:

```sh
python scripts/train_backbone.py --dataset music4all_onion --backbone sasrec \
  --seed 2024 --data-dir data/music4all_onion/prepared_v1 \
  --prepared-manifest refine-logs/music4all_onion/PREPARED_DATASET.json \
  --labels-manifest refine-logs/music4all_onion/METRIC_LABELS.json \
  --output outputs/music4all_onion/sasrec/s2024/backbone_full --device cuda
```

For SSD4Rec, additionally provide `--event-times`, `--timestamp-manifest` and
the pinned `--ssd4rec-source`. Use the corresponding finalized Yambda manifest
paths for that dataset. Checkpoint selection uses Validation; it is not the
Train-internal calibration used later for the ranker.

Build the registered full/fit index with `run_music4all_candidate_support_a0
build-index --data-dir ... --manifest ... --index-dir ... --cv-limit 20000
--seed 2024`. For Onion this is the original 20k temporal-row exclusion:
the last qualifying Train return per sampled user is excluded, while earlier
Train returns may remain. This is **not** backbone-level user-disjoint OOF;
the later ranker fitting/calibration split is user-disjoint. The exporter
checks the exact deterministic 20k selection at the registered Onion index.
Reuse that cohort for all arms and model seeds.

Then call `run_music4all_candidate_support_a1 train-crossfit-base`, passing
`--data-dir`, `--manifest`, `--index-dir`, `--full-base-manifest`, `--seed`,
`--epochs`, `--output`, `--event-times`, `--device`, and the recorded batch/LR/
negative/architecture settings. `--epochs` must equal **that full model's newly
selected epoch**, not a Test-selected value or the training cap. For SSD4Rec,
pass `--ssd4rec-source`. Use `--a0-result reproduction/configs/a0_recipe.json`:
this is a fixed method recipe, not a fabricated historical gate result.
The native trainer validates the input/model bindings and excluded Train rows.
This single auxiliary-model route is the Onion route, not Yambda's full Train
coverage route.

For **Yambda**, retain the recorded **K=5**, assignment seed **20260927**:
the old A0 temporal subset alone is insufficient. For each model seed, run
`arrive_train_fold_coverage run-fold` for folds 0 through 4 with `--project`,
`--dataset yambda500m`, `--seed`, `--data-dir`, `--prepared-manifest`,
`--training-manifest` (the selected full backbone), `--folds 5`, `--fold K`,
`--assignment-seed 20260927`, `--output FOLD_ROOT`, `--device` and `--threads`.
It reuses the existing index builder, selected-epoch auxiliary trainer and
exporter, with each fold's users excluded. It supports both SASRec and SSD4Rec.

Merge the five exports in global request order:

```sh
python scripts/native.py arrive_train_fold_coverage merge \
  --project . --dataset yambda500m --seed 2024 \
  --data-dir prepared_data/yambda500m_prepared_v2_seconds \
  --folds 5 --assignment-seed 20260927 --fold-root FOLD_ROOT \
  --storage physical --output NATIVE_TRAIN_CV
```

The merger verifies fold/user coverage and input/model bindings. Use `physical`
for the portable converter below: it creates contiguous candidate arrays and
needs extra disk space. The historical sharded reader remains included but is
not the portable converter's input format. The result must cover all **17,370**
eligible Train requests. Use the registered index's `full` matrices for
Validation/Test; each fold's `fit` matrices are only for its held-out rows.

## 3. Export, fit and freeze

For Onion, generate native `train_cv` with the auxiliary checkpoint. For both
datasets generate `validation` with the selected full-backbone checkpoint.
Yambda's `train_cv` is the five-fold merge above, not this single export:

```sh
python scripts/native.py run_arrive_full_rcce_control framework_export \
  --project . --dataset music4all_onion --seed 2024 --surface train_cv \
  --index-dir data/music4all_onion/a0_candidate_index_v1 \
  --base-checkpoint outputs/music4all_onion/sasrec/s2024/backbone_aux/final_model.pt \
  --base-manifest outputs/music4all_onion/sasrec/s2024/backbone_aux/training_manifest.json \
  --output outputs/music4all_onion/sasrec/s2024/native/train_cv --device cuda
```

For Validation/Test, preserve the paper's retrieval contract: Onion's legacy
64/128/512 rank-stitch path uses `--legacy20k-retrieval` where recorded; Yambda
uses direct retrieval. Do not infer this from the final 200-item pool size.
Use the same retrieval contract for Validation and its frozen Test continuation.

The exporter now also saves `sources.npz`: the **complete** four lists, their
lengths, aligned request IDs and actual cues. No source list is reconstructed
from only those items that survived the exported union.

Build a catalog for the wrong-song control with `prepare_catalog.py --data-dir
PREPARED --audio NORMALIZED_AUDIO --genre NORMALIZED_CATEGORY --output CATALOG`.
Onion also requires `--tags` and `--information`; Yambda instead requires
`--artist-mapping` pointing to the NPZ above. Yambda reuses audio as the tag
channel and artist multi-hot as the category channel.

Convert each native export, preserving array bytes:

```sh
python scripts/convert_native_export.py \
  --input outputs/music4all_onion/sasrec/s2024/native/train_cv \
  --output outputs/music4all_onion/sasrec/s2024/exports/train_cv \
  --catalog data/music4all_onion/reviewer_catalog
```

For Yambda's merged Train export, also pass `--prepared-data
prepared_data/yambda500m_prepared_v2_seconds`: each fold retains its complete
source bundle, but the merge uses the aligned prepared row IDs to recover cues.
Validation/Test exports carry their own cues and complete source lists.

Repeat for Validation. See [exports.md](../docs/exports.md) for the three
`train_ranker.py` commands: `true` → `models/arrive`, `none` → `models/removed`,
`wrong` → `models/mismatched`. The wrong arm needs `--catalog`. The three arms
share exports, other fields and fitting/calibration policy, but are independently
refitted. No labels are used by the donor selector.

Before generating Test features, call `reviewer_freeze` through `native.py`,
passing the correct arm's `--model-dir`, `--validation-export`, full-backbone
`--base-checkpoint/--base-manifest`, `--project`, `--dataset`, `--seed`,
`--index-dir`, `--test-data`, `--test-manifest`, `--output` and the recorded
`--legacy20k-retrieval` flag when applicable. It binds the selected ranker,
Validation serving inputs, native source hashes, Test manifest and retrieval
contract without opening Test arrays. The `reviewer_freeze.v1` schema is a new
input interface, not a replacement for or rewrite of a historical freeze.

Generate the native Test export with `framework_export --surface test`, the
same full backbone/index, `--model-dir`, `--freeze-record`, `--test-data`,
`--test-manifest`, `--reviewer-export-interface`, and the same retrieval flag.
The native sealed-Test artifact and denominator checks remain active. Convert
this export, with the same catalog, then run the saved A/B/C models using
`evaluate_ranker.py`. Test data does not choose trees or refit models.

## 4. Produce the paper-facing analyses

For each Test export:

```sh
python scripts/backbone_predictions.py --export TEST_EXPORT --output BACKBONE_EVALUATION
python scripts/replay_coverage.py --export TEST_EXPORT --output COVERAGE_EVALUATION
```

The first uses the **same frozen source backbone**, not a separately retrained
baseline. The second removes only the first-song transition source, refills to
200 and keeps backbone Top-10 protection. Source coverage is not Top-10 ranking.
Run fixed-pool matching controls for both SASRec and SSD4Rec. Within each
backbone's A/B/C comparison, retain the same candidate pools, fitting requests,
other 27 inputs and Train-internal selection rule. The SSD4Rec Table 2 follow-up
uses separate backbone checkpoints and refitted rankers from Table 1; its ARRIVE
arm must not be replaced with Table 1's outputs. Run Table 3 source-coverage
replay for SASRec.

Prepare an analysis JSON with `rankings` entries containing `dataset`,
`backbone`, `method`, `seed` and `path` to each saved prediction NPZ. Methods are
`backbone`, `arrive`, `removed`, `mismatched`, or a named comparison baseline.
Every row requires all three seeds. If a native baseline prediction omits
complete targets, add `metadata` pointing to the matching Test-export
`metadata.npz`; request/user identities are checked, not assumed.
An optional `cue_bundle` points to `CUE_DONORS.npz` for metadata-availability harm
analysis. `coverage` entries contain `dataset`, `seed` and their `coverage.npz`
path. Optional SHA-256 fields bind individual inputs. Paths are relative to the
analysis JSON. A layout example is [results.example.json](configs/results.example.json).

```sh
python scripts/summarize_results.py --config MY_RESULTS.json --output outputs/summary
```

`summary.json` supplies Table 1/2 ranking means and sample SD. The four
prespecified SASRec NDCG matching contrasts and the four SSD4Rec follow-up
contrasts are adjusted as separate families, each using 98.75% intervals.
Other comparisons and Table 3 coverage use descriptive 95% intervals.
Matched-seed per-user differences are averaged before 10,000 user bootstrap
draws for both backbones. The default settings and example config follow this
current protocol; historical native analysis scripts retain their original
settings. It also
provides harmed/unchanged/improved shares, the four gap and four target-count
strata, Recall among NDCG-harmed users, optional cue-availability groups, and
unsmoothed user-gain ECDF coordinates for Figure 2. All denominators include
out-of-catalog targets. See [protocol.md](../docs/protocol.md).

## Comparison baselines and remaining limits

The five comparison adapters, their native trainer/evaluator and available asset
producers are included. Read [baselines.md](../docs/baselines.md) and
[comparison_training.json](configs/comparison_training.json); use the recorded
`adapter_revision`, optimizer and method-specific settings, not CLI defaults.
CoSeRNN/MEGAN need their Train-only track-space/graph assets, with MEGAN LINE
initialization; MUSE needs its Train transition counts. Null settings remain
unrecorded. The complete historical tuning search is not recoverable from these
records, and equal tuning/compute is not asserted.

Source inclusion and synthetic checks do not certify fresh 500M preprocessing,
Linux CUDA/Mamba execution, real-data metric equality, or exact recovery of every
historical baseline asset. There are no paper weights, original prepared arrays,
user logs or original per-user predictions in the release. A full run must
obtain the registered public sources, regenerate inputs and train. Do not
hand-edit hashes, silently discard a missing stage, change the fitting cohort,
or report synthetic scores as paper replication.
