# Data and required contracts

Core and full synthetic smokes require no music data and make no downloads.
For independent real-data work, obtain Music4All-Onion from
[the provider's Zenodo record](https://zenodo.org/records/15394646) and Yambda from
[its official dataset repository](https://huggingface.co/datasets/yandex/yambda).
Music4All-Onion is described by Moscati et al., *Music4All-Onion: A Large-Scale
Multi-Faceted Content-Centric Music Recommendation Dataset* (CIKM 2022);
use the provider's current access and licensing terms. This artifact does not
redistribute either dataset or offer an unverified automated download command.
These provider pages were checked when preparing this artifact; use the frozen
input version/hashes for scientific reproduction, rather than assuming the
current provider version matches the original experiment bytes.

The scientific chain also needs prepared events/sessions, Train frequencies,
fixed vocabulary, normalized content rows and primary artists, Train-only
transition indexes, the matching frozen backbone, full exported candidate
unions, RRF200 indices, labels and per-request context. Original export/model
manifests must validate these assets before real-data fitting or evaluation.
Generic replacement arrays cannot reproduce the reported metrics.

The task uses 30-minute sessions, a gap of at least 30 days, up to 128
pre-hiatus events and the observed first song. Targets are distinct songs in
the next at most 64 events. Split by the frozen time boundaries; this is not a
user cold-start split. Never build indexes/content from Validation/Test targets.
The [reproduction guide](../reproduction/README.md) identifies the frozen
cutoffs, source-length/batch contracts, held-out Train preparation and native
producers. Recorded configurations include selected epochs and original
manifest fingerprints, not trained weights. A fresh run must regenerate its
own model bindings; it must not copy an old manifest onto a new checkpoint.
