# Artifact coverage and verification

The release contains source for the main-result chain, not the original data
or model artifacts. Follow [the reproduction guide](../reproduction/README.md)
in stage order; installing the minimal smoke dependencies alone is insufficient
for the native pipeline.

| Paper component | Included source | Verification in this release |
|---|---|---|
| Data and content | Onion C++ producers/finalizers; Yambda 500M seconds-protocol preparation and sealed Test; native content/artist construction | Import/CLI checks; full datasets not rebuilt |
| Backbones | SASRec trainer; SSD4Rec task adapter, pinned upstream-source downloader; 12 recorded configurations | Command/argument checks; no real-data training or CUDA execution |
| Train-only index and auxiliary model | Transition-index producer, fixed candidate recipe, held-out-row auxiliary trainer | Included with native binding checks; no full index/model rebuilt |
| Native candidate export | Full four-source lists, protected RRF200, native 28+3 features and request context | Synthetic checks for the core functions and export bridge |
| Matching controls | Correct/removed/mismatched arms, exact four columns 12/13/14/16, partitioned label-free donors | Synthetic tests for unchanged 27 fields, donor rules and three independent fits |
| Calibration and evaluation | User partition, positive-only fitting filter, Train-internal 220/440 tree selection, complete-target metrics | Synthetic fit/save/reload and selected-tree evaluation |
| Test binding | Portable freeze record plus original sealed-Test schema/hash/denominator gates | Source/contract checks; no Test export rebuilt |
| Table 1 | Backbone/ARRIVE predictions; five comparison adapters and available asset producers; 30 recorded configurations | Historical configuration bindings inspected; no baseline rerun |
| Table 2 | Fixed-pool three-arm fitting/evaluation and paired analysis | Synthetic tests; paper's real-data scores not recomputed |
| Table 3 | Native equal-budget full-source replay and three-seed Recall/Hit Rate analysis | Synthetic RRF parity and reporting tests; real-data replay not repeated |
| Figure 2 / user analysis | Paired user gains, harm/zero/improvement, strata and unsmoothed ECDF coordinates | Synthetic user-macro/seed-averaging tests; no original predictions distributed |

`tools/` retains the small tested native projection for the portable interface.
`reproduction/native/` contains the wider dependency closure used by the native
producers. Older research branches remain where imports require them; they are
not extra short-paper methods. The new portable freeze schema is distinct from
the historical freeze records and does not rewrite them.

Remaining limits:

- Obtain the registered public source versions and regenerate prepared arrays,
  content, indexes, auxiliary/full backbones and exports. Release hashes and
  count gates deliberately reject a different input release.
- No raw events, private identifiers, original arrays, checkpoints or per-user
  predictions are distributed. Exact frozen-model execution therefore requires
  assets beyond this repository.
- Linux CUDA/Mamba compatibility, large-file preparation, real-data execution
  and equality with the paper's numbers have not been verified in this release.
  The synthetic checks are implementation tests, not paper replication.
- The five comparison methods have method-specific adapters and budgets. Their
  recorded settings and available asset builders are supplied, but not every
  historical asset or the complete search history. Equal tuning is not claimed.
- Core dependencies, CPU reproduction dependencies and the historical SSD4Rec
  GPU environment are separate; see the reproduction guide before installing.

The repository can be used to inspect and attempt the main-result reproduction
from public data. It does not certify a completed independent reproduction.
