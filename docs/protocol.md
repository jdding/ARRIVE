# ARRIVE protocol

This document records the paper protocol and the behavior of the included native
helpers. The package supports inspection and execution on supplied inputs;
[coverage.md](coverage.md) lists the data preparation, model, and analysis paths
that are not included. Synthetic checks do not reproduce the paper's results.

## Task and data

A request contains up to 128 pre-hiatus events, a listening gap of at least
30 days, and the observed first return song. Sessions end after 30 minutes of
inactivity. The target set contains distinct songs in the next at most 64 events
of the same session, including any later replay of the first song. The output is
a ten-song ranking.

The paper uses global temporal splits and Train-frozen catalogs; users may recur
across splits. Yambda uses elapsed seconds and retains organic listens exceeding
half the song's duration. Ranker development uses 20,000 Train-held-out Onion
requests and all 17,370 eligible Yambda Train requests. Auxiliary backbones and
indexes exclude held-out targets from training features; evaluation uses
full-Train assets. Both Test cohorts had been evaluated in earlier versions.

## Candidate sources and RRF200

The four ranked sources are:

1. A frozen SASRec or SSD4Rec backbone, given pre-hiatus history followed by the
   first song.
2. Train-frozen transitions from the first song.
3. Transitions from up to five recent distinct historical anchors.
4. Songs by the first song's artist.

Historical anchors are scanned from newest to oldest within the last 128 events;
reserved items, duplicates, and the first song itself are excluded as anchors.
The native transition helper sorts by descending `log1p(count)` when its
popularity penalty is zero, then by ascending item ID. Historical counts are
weighted by `1/log2(anchor_position + 2)` before ranking. Artist candidates are
ordered by descending Train frequency, then ascending item ID.

Reciprocal rank fusion sums `1/(60 + r)` over sources containing an item, with
one-based source rank `r`. The pool first retains the exact backbone Top-10,
then fills to 200 unique items by descending fused score and ascending item ID
on ties. The helper fails if it cannot produce exactly 200 unique items; it does
not pad. Protection guarantees candidate inclusion: the fitted ranker can
reorder every item in the pool.

Source lengths follow the dataset's native export contract. The 200-song pool
budget is not a universal per-source length limit. Backbone scoring, production
retrieval batches, and their rank-stitch contracts are outside this package.

## Ranker fitting and selection

The ranker consumes the 31 FP32 fields described in [features.md](features.md).
The backbone and transition statistics remain frozen.

Fitting and calibration users are separated by a fixed user hash. A user is in
the fitting partition when the unsigned big-endian integer represented by the
first eight bytes of `SHA256("a1-fit:<user_id>")` is below
`int(0.8 * 2**64)`. All requests from one user stay on one side; this split is
independent of the experiment seed.

Fitting omits requests with no positive candidate in their exported fitting
union. Calibration retains its complete partition, including requests with no
positive candidate. Empty fitting/calibration sets or overlapping users fail
validation. Fitting uses the exported candidate union, which can exceed 200
items; calibration and inference rank only each request's RRF200 subset.

CatBoost uses CPU training, depth 6, learning rate 0.035, and this loss:

```text
YetiRank:mode=NDCG;top=10;dcg_type=Base;dcg_denominator=LogPosition
```

The model fits up to 440 trees. Train-internal calibration compares 220 and
440 trees by user-macro NDCG@10; an exact score tie selects fewer trees. The
experiment seeds are 2024, 2025, and 2026, and CatBoost uses the experiment seed
plus 1000 as `random_seed`. The recorded paper configuration uses four CPU
threads. At inference,
scores are sorted in descending order, with ascending item ID breaking ties.

## Matching controls

The three arms use identical fitting requests and candidate pools:

| Arm | Four matching fields | Other 27 fields |
|---|---|---|
| ARRIVE / A | Actual first song | Unchanged |
| Removed / B | Zero | Unchanged |
| Mismatched / C | Another song's correspondence | Unchanged |

Only zero-based columns `12, 13, 14, 16` are treated. Each mismatched request uses
one donor for all its candidates, including across processing chunks. The arms
use the same calibration rule; their selected tree counts may differ.

Donors use cue identities, content availability, Train-popularity bins, and
partition membership without continuation labels. Availability is a four-bit
code: artist `1`, audio `2`, category `4`, and tag `8`. Popularity bins are
`floor(log2(1 + Train_count))`; prepared Train-event frequencies define these
bins on fitting, calibration, and evaluation surfaces.

For a usable cue, the selector applies these rules in order:

1. Search request donors in the same partition and availability pattern, first
   in the same popularity bin and then by increasing absolute bin distance.
   A distance tie prefers the lower bin.
2. Within the chosen bin, exclude all occurrences of the actual cue identity
   and sample uniformly from eligible request occurrences. This is not uniform
   sampling over distinct songs. Groups are stably sorted by cue ID, and the
   selector uses NumPy `default_rng(experiment_seed)`.
3. If no request bin supplies an alternative, search Train-known catalog songs
   with the same availability pattern, using the same nearest-bin/lower-bin
   ordering. A catalog song must have positive Train frequency and item ID at
   least 3.

Fitting and calibration request donors cannot cross their user-partition
boundary. Validation and Test each construct their own isolated request mapping.
Catalog fallback uses Train-known songs rather than requests from another
partition. Donor reuse is allowed. If no alternative exists for a usable cue,
the control fails; it does not silently retain that cue.

Unknown cues and known cues with no usable content remain in the cohort. Their
original zero matching and cue-availability fields are preserved. Shared-ranker
refitting can still change their predictions. The low-level selector exposes
unresolved donors as `-1`; the control wrapper rejects unresolved usable cues.
The included package provides this selector and treatment logic; construction
of real-data mappings from native dataset manifests is an omitted loader path.

## Metrics and uncertainty

Recall@10 divides hits by all distinct targets, including out-of-catalog songs.
NDCG@10 uses binary relevance, discount `1/log2(r + 1)`, and an ideal list of
`min(10, target_count)` songs. Metric inputs must preserve the full target
denominator; the known-item retrieval helper is not a replacement for those
labels.

Metrics first average requests within each user, then average users. Paper
results average seeds 2024--2026 and report sample SD (`ddof=1`). Paired analysis
averages matched-seed differences per user before resampling users with
replacement. SASRec-backed comparisons use 5,000 draws; SSD4Rec comparisons use
10,000. These intervals condition on the fitted models and do not estimate
variation from retraining on new population samples.

The four main NDCG matching contrasts (A--B and A--C on two datasets) use
Bonferroni-adjusted 98.75% intervals. Mismatch--removal, Recall, subgroups, and
candidate-coverage intervals are descriptive 95% intervals. Users may enter
multiple strata; stratified analyses average requests within each user and
stratum. Bootstrap analysis code and the paper's saved outputs are not included.

## Paper-to-code mapping

| Paper result | Protocol | Included helpers / limits |
|---|---|---|
| Table 1 | Full ARRIVE compared with its actual matched backbone checkpoint; three seeds | Retrieval, features, ranker, and metrics are inspectable. Backbones, baseline runners, paper models, and real outputs are omitted. |
| Table 2 | SASRec-backed A/B/C comparison with fixed pools and identical fitting requests | Four-column removal and wrong-song treatment are included. A reuses Table 1's SASRec-backed models and outputs; paper controls and intervals are not reproduced. |
| Table 3 | SASRec-backed source replay with and without first-song transitions | The other three sources, 200-song budget, and protected backbone Top-10 stay fixed. The paper's replay needed no padding. Real source exports and coverage results are omitted. |
| User analyses | Seed-averaged paired per-user gains and within-user/stratum averages | The basic metric aggregation is included; saved predictions, interval analysis, and figure generation are omitted. |
