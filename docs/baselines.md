# Table 1 baselines

This document describes the five comparison baselines in Table 1. Their
recorded training configurations correspond to the saved Test results,
selected checkpoints, and selected epochs underlying those rows. Their source
implementations, training runners, data assets, and paper checkpoints are not
included in this core release. The SASRec/SSD4Rec pairs with ARRIVE are described
in [protocol.md](protocol.md).

## Task adaptations

All baselines observe the first return song. Their common task is to recommend
distinct later songs in the same return session.

| Method | Recorded adaptation |
|---|---|
| CoSeRNN | Played-session pathway with available play/time/count/gap context; fixed Train successor-track space and session recurrence. Missing skip/device/source fields are not fabricated. Yambda constructs its track space from PPMI+SVD. |
| CAR | Observed-prefix context with the user branch omitted. The manifests describe the creator branch as disabled and use uniform negatives excluding observed and suffix-positive items. |
| MEGAN | Train-only heterogeneous music graph, with pre-hiatus history plus the observed return anchor as prefix input and an episode set-target sampled objective. Yambda builds its graph from organic-listen transitions. |
| TiSASRec | Relative-time key/value self-attention with the same return-session set target and evaluator. Yambda uses elapsed-second time rounded to five-second precision. |
| MUSE | Transition augmentation within observed sessions, preserving the return anchor; up to 129 real visible events plus synthetic bridges up to 257. Onion records that shuffle labels are unavailable. |

## Recorded configurations

The three seeds are 2024, 2025, and 2026. Configuration fields below are identical
across those seeds except for the seed itself. All records specify
`max_context=129`, `negative_samples=1024`, and `objective=oneshot_only`.
The 129-event limit includes the observed first song. All records select the
checkpoint by Validation user-macro NDCG@10; this is separate from ARRIVE's
Train-internal ranker calibration.

`E/L/H` lists the recorded `embed_dim/num_layers/num_heads` fields. These are
shared-runner configuration fields, not a claim that every field controls every
architecture. Optimizer entries show optimizer name, recorded beta2, and weight
decay. Unrecorded fields are not inferred from another method or dataset.

| Dataset | Method | Batch | Learning rate | E/L/H | Dropout | Optimizer; beta2; weight decay | Epoch cap |
|---|---|---:|---:|---|---:|---|---:|
| Onion | CoSeRNN | 256 | 0.0005 | 40/1/2 | 0 | Adam; 0.999; 0 | 30 |
| Onion | CAR | 64 | 0.001 | 128/2/2 | 0.2 | Adam; 0.98; 0 | 30 |
| Onion | MEGAN | 256 | 0.001 | 256/1/2 | 0 | Adam; 0.999; 0 | 30 |
| Onion | TiSASRec | 64 | 0.001 | 128/2/2 | 0.2 | Not recorded in the manifest | 15 |
| Onion | MUSE | 64 | 0.001 | 128/1/2 | 0 | Adam; 0.999; 0 | 30 |
| Yambda | CoSeRNN | 256 | 0.001 | 40/1/2 | 0 | Adam; 0.999; 0 | 15 |
| Yambda | CAR | 64 | 0.001 | 128/2/2 | 0.2 | Adam; 0.98; 0 | 15 |
| Yambda | MEGAN | 256 | 0.001 | 256/2/2 | 0.2 | Adam; 0.999; 0 | 15 |
| Yambda | TiSASRec | 64 | 0.001 | 128/2/2 | 0.2 | AdamW; 0.999; 0.0001 | 15 |
| Yambda | MUSE | 64 | 0.001 | 128/2/2 | 0 | AdamW; 0.999; 0.0001 | 15 |

CoSeRNN additionally records hidden dimension 400 and fully connected dimension
200 on both datasets. The records contain 293,201 training sessions per Onion
run and 17,370 per Yambda run. These baseline counts are not ARRIVE's 20,000-row
Onion ranker-development limit.

## Completed budgets and checkpoint selection

All listed runs completed 15 epochs. Onion CoSeRNN, CAR, MEGAN, and MUSE record
a 30-epoch cap, minimum 15 epochs, patience 5, minimum improvement 0.0001, and
`validation_patience` as the stopping reason. Onion TiSASRec records a 15-epoch
cap but no convergence-policy or stopping-reason field. Every Yambda run records
a 15-epoch cap, patience 0, and `max_epochs` as its stopping reason.

| Dataset | Method | Completed updates per seed | Selected epoch, seeds 2024/2025/2026 |
|---|---|---:|---|
| Onion | CoSeRNN | 17,190 | 10 / 5 / 8 |
| Onion | CAR | 68,730 | 2 / 3 / 2 |
| Onion | MEGAN | 17,190 | 8 / 8 / 7 |
| Onion | TiSASRec | 68,730 | 3 / 2 / 2 |
| Onion | MUSE | 68,730 | 4 / 5 / 6 |
| Yambda | CoSeRNN | 1,020 | 11 / 10 / 9 |
| Yambda | CAR | 4,080 | 12 / 11 / 11 |
| Yambda | MEGAN | 1,020 | 13 / 9 / 12 |
| Yambda | TiSASRec | 4,080 | 6 / 5 / 7 |
| Yambda | MUSE | 4,080 | 15 / 15 / 15 |

## Interpretation limits

These are settings and completed budgets of the recorded runs, not an exhaustive
hyperparameter-search history. The records do not establish the complete search
spaces, number of discarded trials, equal optimization effort, or equal compute
cost across methods. Some optimizer details, including Onion TiSASRec's
optimizer, are unrecorded in these manifests. Method-specific epoch/update
budgets are not proof of fair tuning. Correspondence between saved metadata and
Table 1 does not replace a baseline source audit or an independent rerun.
