# ARRIVE protocol

This document records the paper protocol and the behavior of the included native
helpers. The package supports inspection and execution on supplied inputs;
[coverage.md](coverage.md) separates included source from unverified execution.
Synthetic checks do not reproduce the paper's results.

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
Onion uses the recorded 20k temporal-row exclusion, not user-disjoint backbone
folds: earlier Train episodes of the selected users can remain. Yambda covers
all eligible Train rows through five user-disjoint folds with assignment seed
20260927. This is distinct from the user-disjoint ranker fit/calibration split.
See [the native stage guide](../reproduction/README.md) for both routes.

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
retrieval batches and rank-stitch contracts are included in the native exporter;
the portable interface consumes their hash-bound exports.

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

The fixed-pool matching arms use identical fitting requests and candidate pools:

| Arm | Four matching fields | Other 27 fields |
|---|---|---|
| ARRIVE / A | Actual first song | Unchanged |
| Removed / B | Zero | Unchanged |
| Mismatched / C | Another song's correspondence | Unchanged |
| Identity / I | Column 12 is `candidate_id == actual_return_song_id`; the other three are zero | Unchanged |

Only zero-based columns `12, 13, 14, 16` are treated. Each mismatched request uses
one donor for all its candidates, including across processing chunks. The arms
use the same calibration rule; their selected tree counts may differ.
Identity uses the observed cue from a hash-bound, request-aligned bundle without
consulting continuation targets. Known identity remains available even when
content is missing; the unknown ID 1 cannot match catalog candidates. A versus I
compares alternative representations, not matching conditional on identity.

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
`convert_native_export.py` constructs the mapping from exported cue IDs (or
aligned prepared rows for a folded Train merge) and the Train-only catalog.
It reruns this selector and validates the resulting treatment bundle.

## Joint explicit return-song control

The `joint-removal` source policy removes the return-song transition and artist
sources, their source ranks, matching fields, cue availability and transition
support/concentration. H-only anchors retain the recent distinct pre-hiatus
items without comparing them to the return song. Frozen backbone weights still
receive the same H-minus/return-song prefix. Retained U200 union RRF200 is
regenerated, including RRF and score-feature normalization, rather than filtered
from an incomplete old export. Inference keeps 200 candidates and the exact
protected backbone Top-10.

Each arm retains the same Train rows/targets, user partition and positive-group
fitting policy. The effective fitted groups may differ after candidate removal;
calibration and full Test cohorts stay fixed. The comparison estimates the
joint pipeline value, including training-support and component interactions.
It is not an individual-source or serving-only effect. Do not add it to the
fixed-pool matching increment or interpret it as a percentage of Table 1 gains.
New exports and models declare `source_policy` explicitly; historical missing
values mean `full`, and paired training/evaluation must keep the policy matched.

Identity and joint controls were completed for Onion/Yambda with the original
Table 2 SSD4Rec checkpoints and seeds 2024–2026. Missing original SASRec carrier
assets do not establish results under that carrier. The source interface can
be executed with new, fully bound assets, but this is not recovery of the
historical paper checkpoint lineage.

## Metrics and uncertainty

Recall@10 divides hits by all distinct targets, including out-of-catalog songs.
NDCG@10 uses binary relevance, discount `1/log2(r + 1)`, and an ideal list of
`min(10, target_count)` songs. Metric inputs must preserve the full target
denominator; the known-item retrieval helper is not a replacement for those
labels.

Metrics first average requests within each user, then average users. Paper
results average seeds 2024--2026 and report sample SD (`ddof=1`). Paired analysis
averages matched-seed differences per user before resampling users with
replacement. Both SASRec and SSD4Rec comparisons use 10,000 draws. These
intervals condition on the fitted models and do not estimate
variation from retraining on new population samples.

The four prespecified SASRec NDCG matching contrasts (A--B and A--C on two
datasets) use Bonferroni-adjusted 98.75% intervals. The four SSD4Rec follow-up
contrasts form a separate family with the same adjustment. This does not assert
one joint 5% family-wise error bound over all eight contrasts.
The identity and joint-removal comparisons also retain 98.75% intervals from
their intended four dataset/carrier contrasts, while the completed reported
scope is two SSD4Rec settings for each control.
Mismatch--removal, Recall, subgroups, and
candidate-coverage intervals are descriptive 95% intervals. Users may enter
multiple strata; stratified analyses average requests within each user and
stratum. `summarize_results.py` implements this aggregation and interval
analysis on saved predictions; the paper's original predictions are not supplied.

## Paper-to-code mapping

| Paper result | Protocol | Included helpers / limits |
|---|---|---|
| Table 1 | Full ARRIVE compared with its actual matched backbone checkpoint; three seeds | Both backbone source chains, comparison adapters, export/ranking/evaluation and summary code are included. Data, paper models and original outputs are not distributed. |
| Table 2 | SASRec and SSD4Rec A/B/C comparisons with fixed pools and identical fitting requests within each comparison | Four-column removal and wrong-song treatment are included. SASRec A reuses Table 1's models. SSD4Rec uses separate backbone checkpoints and refitted rankers from Table 1; pair A/B/C within that follow-up. Original paper predictions and intervals are not distributed. |
| Table 2 I/J | SSD4Rec identity replacement and joint source/field removal on both datasets | Portable identity mode and native joint-removal export policy are included. Keep the same Table 2 carrier lineage; J refits after regeneration and can change positive-fit counts. Original paper weights/exports remain omitted. |
| Table 3 | SASRec-backed source replay with and without first-song transitions | The other three sources, 200-song budget, and protected backbone Top-10 stay fixed. The paper's replay needed no padding. Real source exports and coverage results are omitted. |
| User analyses | Seed-averaged paired per-user gains and within-user/stratum averages | Summary code includes paired intervals, harm/zero/improvement, strata and ECDF coordinates. Original predictions and figure styling are not distributed. |

## Additional result details

Differences use unrounded three-seed user-macro means. The paired bootstrap resamples users after matched-seed averaging, with fitted models held fixed. Wrong-song donors are selected separately in fitting, calibration and evaluation partitions, preserving metadata availability and using the training-popularity bin where possible; no continuation labels enter donor selection.

The table below gives ARRIVE minus w/o matching in percentage points. Replay and other-song contributions retain the original ranks and the full-target metric denominator.

| Dataset | Backbone | Overall NDCG increment | Other-song Recall increment | Other-song NDCG increment |
|---|---|---:|---:|---:|
| Onion | SASRec | 0.563346 | 0.145205 | -0.018713 |
| Onion | SSD4Rec | 0.231916 | 0.132451 | 0.084442 |
| Yambda | SASRec | 1.210682 | 0.094328 | -0.288241 |
| Yambda | SSD4Rec | 1.145671 | 0.277168 | 0.003601 |

For Yambda/SASRec, the other-song NDCG change has a 95% interval of [-0.40, -0.18] points. Its gained/lost-hit net is +0.04 points and shared-hit rank term is -0.33 points, while the overall matching gain remains +1.21 points.

Correct matching exceeds the wrong-song control by 0.56 and 1.19 NDCG points on Onion and Yambda with SASRec, and by 0.25 and 1.18 with SSD4Rec. The four mismatch-minus-removal 95% intervals include zero; this is not an equivalence test.

For source coverage, Onion's Recall@200 increment is 0.09 points with 95% interval [-0.03, 0.22]. At the fixed budget, adding the return-song transition source adds 111 and displaces 100 covered target occurrences per seed on average. For known-return-song requests without later replay, Onion's Hit Rate increment is 0.26 points ([0.04, 0.48]); its Recall interval includes zero. Yambda's full-cohort and no-replay gains are reported in the manuscript.

The user-level analysis uses ARRIVE versus its actual Table 1 backbone, not the separate Table 2 SSD4Rec checkpoints. On Onion, 79.5% of users losing NDCG have unchanged or higher Recall with either backbone. Their loss shares are 7.1%/7.6% among users with any unusable return-song request and 14.4%/16.2% among users with all requests usable (SASRec/SSD4Rec).

### Metadata groups and mean gains

A request has usable metadata when its actual return-song ID is in the catalog and at least one matching-metadata channel is available. A user is in the all-usable group if every observed request is usable; this does not require all metadata modalities. The other group has at least one unusable request. Means below are full ARRIVE minus the actual paired Table 1 backbone, in NDCG percentage points, using the paper's user-macro and three-seed aggregation.

| Dataset | Backbone | All-usable users | Mean gain | Users with any unusable request | Mean gain |
|---|---|---:|---:|---:|---:|
| Onion | SASRec | 4,495 | 7.5377 | 1,236 | 2.5007 |
| Onion | SSD4Rec | 4,495 | 6.6827 | 1,236 | 2.0207 |
| Yambda | SASRec | 6,508 | 13.6526 | 1,606 | 2.3102 |
| Yambda | SSD4Rec | 6,508 | 9.1329 | 1,606 | 2.1172 |

These are observed user-group means; the analysis does not estimate a causal effect of metadata availability or test the between-group difference. The original aggregate report has SHA-256 `df23bd53c4932324c5e2b9b2d607a284d401358baf8461ca77d8e442f51f047a`. Its 24 source prediction hashes match the Table 1 full-system/backbone inputs.
