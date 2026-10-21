# Actual artifact coverage

| Path | Included / verification |
|---|---|
| Candidate sources | Native first-transition/history ranking, recent distinct anchors and same-artist ordering; synthetic checks pass |
| RRF200 | Native 1/(60+r), protected backbone Top-10, deterministic item ties; synthetic checks pass |
| Features | Native 28-column row builder plus three request statistics; FP32, union score normalization and static-row parity checked |
| Removed / wrong-cue controls | Native four columns 12/13/14/16, preserved 27 columns, availability/bin/partition donor rules and unknown-cue retention checked |
| Ranker | Native CatBoost Pool and RRF-only top10 functions and fixed YetiRank loss included; synthetic fit/reload/predict passed for all three arms |
| Train selection | Runnable export wrapper uses native user partition, positive fitting filter, Train-internal 220/440 calibration and fewer-tree tie rule; original excerpt retained for comparison |
| Evaluation | Native Recall/NDCG full-target denominator and request-then-user averaging checked on synthetic inputs |
| Export interface | Hash, identity, shape, target/label, RRF-index and request-alignment checks; model persistence and selected-tree evaluation |
| Source replay | Native v2 full-source replay with exact RRF200 parity and equal-budget removal; Recall/Hit Rate@200, not final Top-10 quality |

The packaged code is a semantic-preserving projection of selected native bodies,
not a full copy of the research repository. Module names are retained so native
imports and their original provenance remain legible. No alternate neural,
joint-objective or LightGBM ranking branch is provided.

Unverified or omitted paths:

- Trained SASRec/SSD4Rec scoring and the production retrieval batches/rank-stitch
  contract. The synthetic smoke supplies deterministic backbone ranks/states.
- Raw data preprocessing, normalized audio/tag/category construction, Train
  transition-index construction and user/time split preparation.
- Production batched feature acceleration, source replay against full frozen
  exports and the exact frozen Test cohort. Supplied-file hash checks cannot
  prove the declared Train-only/time boundaries of an omitted producer.
- Original raw-event export, request-context and donor-bundle producers. The
  new wrapper validates supplied bundles and reruns the native donor selector;
  it does not recover their raw-event provenance.
- Paper Table 1 baseline implementations, Table 2 scientific results,
  Table 3 real candidate coverage, confidence intervals and user-harm analysis.
  Native semantics are inspectable; those scientific numbers are not reproduced.
  Recorded baseline adaptations and budgets are documented in `baselines.md`;
  the complete tuning history remains unavailable.

There are no raw data, content matrices, private identifiers, trained paper
models, original predictions or original manifests. The package is sufficient
for the explicitly scoped core, full synthetic smoke and prepared-export
interfaces described in `exports.md`. The full model check
passed with CatBoost 1.2.10, five trees per arm and one CPU thread. Real-data
result reproduction and exact frozen-model execution have not been verified.
