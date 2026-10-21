# Candidate features

ARRIVE uses 31 FP32 inputs per candidate. The first 28 come from the native
candidate row builder; three request statistics are appended in `statistics`
mode. The schema is [configs/feature_schema.json](../configs/feature_schema.json).
Indices below are zero-based.

| Index | Schema name | Value |
|---|---|---|
| 0 | `encoder_score_z` | Backbone dot-product score, standardized across the supplied candidate feature universe; standard-deviation floor `1e-6` |
| 1 | `rrf_score` | Sum of `1/(60 + r)` over sources containing the candidate |
| 2 | `encoder_rank` | `1/log2(r + 1)` for the backbone source, or zero if absent |
| 3 | `first_track_transition_rank` | Same rank transform for first-song transitions |
| 4 | `history_transition_rank` | Same rank transform for historical transitions |
| 5 | `artist_rank` | Same rank transform for the artist source |
| 6 | `log_popularity` | `log1p(Train frequency)` |
| 7 | `log_gap` | `log1p(gap) / 8` |
| 8 | `gap_30_90` | Gap-bucket indicator |
| 9 | `gap_90_180` | Gap-bucket indicator |
| 10 | `gap_180_365` | Gap-bucket indicator |
| 11 | `gap_gt365` | Gap-bucket indicator |
| 12 | `same_artist` | Candidate and first song have equal known scalar artist identity |
| 13 | `tag_cosine_first` | Candidate tag-channel vector dot first-song tag-channel vector |
| 14 | `audio_cosine_first` | Candidate audio vector dot first-song audio vector |
| 15 | `audio_cosine_history` | Candidate audio vector dot mean historical audio vector |
| 16 | `genre_cosine_first` | Candidate category-channel vector dot first-song category-channel vector |
| 17 | `genre_cosine_history` | Candidate category-channel vector dot mean historical category vector |
| 18 | `artist_known` | Candidate scalar artist identity is known |
| 19 | `first_artist_known` | First-song scalar artist identity is known |
| 20 | `audio_available` | Candidate audio vector has nonzero norm |
| 21 | `first_audio_available` | First-song audio vector has nonzero norm |
| 22 | `history_audio_available` | Mean historical audio vector has nonzero norm |
| 23 | `genre_available` | Candidate category-channel vector has a nonzero entry |
| 24 | `first_genre_available` | First-song category-channel vector has a nonzero entry |
| 25 | `history_genre_available` | Mean historical category vector has a nonzero entry |
| 26 | `tag_available` | Candidate tag-channel vector has a nonzero entry |
| 27 | `first_tag_available` | First-song tag-channel vector has a nonzero entry |
| 28 | `recent_unique_fraction` | Number of up to 20 recent distinct historical items, divided by 20 |
| 29 | `log_first_track_transition_count` | `log1p(sum(first-song transition counts))` |
| 30 | `first_track_transition_concentration` | Largest first-song transition count divided by their sum; zero for an empty row |

The `*_rank` names are retained from the native schema, but columns 2--5 contain
inverse log-rank values, not raw ranks. The row builder can accept precomputed
normalized backbone scores; their feature-universe contract must match the
export. The gap-bucket ID is an upstream input. This package does not reconstruct
gap-bucket endpoint rules or the data-preparation pipeline.

## Content and history

The paper uses row-normalized tag, audio, and category vectors, so first-song
dot products are cosine similarities. Missing content has zero vectors and
therefore zero similarities, accompanied by availability fields. The included
row builder consumes these representations; it does not construct or normalize
the original content matrices.

Historical audio and category vectors average up to five recent distinct
anchors. Anchors come from the last 128 pre-hiatus events, newest first, excluding
the actual first song, duplicates, and reserved items. Their mean is not
renormalized before its candidate dot product. Empty history gives zero vectors
and zero historical similarities.

The request statistic in column 28 is computed separately: it scans the last
128 historical events in reverse order, takes up to 20 distinct item identities,
and divides their count by 20. It does not use the five-anchor list or explicitly
remove a historical replay of the first song. Columns 29--30 use the frozen
first-song transition row. These three request fields do not read continuation
targets.

## Yambda substitutions

Yambda has no tag or genre metadata in this protocol. The tag channel reuses its
row-normalized audio embeddings, while the category/genre channel uses
row-normalized artist multi-hot vectors. Consequently, columns 13 and 14 use the
same underlying audio representation on Yambda; these channels are not
independent modalities. Artist multi-hot similarity can capture any shared
artist represented in that vector.

The scalar artist identity used by `same_artist` and the artist source is a
separate representation. For a Yambda item with several artists, it selects the
smallest artist ID deterministically. Thus scalar artist equality and multi-hot
artist similarity have different meanings.

## Four-field controls

Matching removal zeros only `same_artist`, `tag_cosine_first`,
`audio_cosine_first`, and `genre_cosine_first` (columns `12, 13, 14, 16`).
Mismatch recomputes those four fields against one donor song per request. It
preserves every other field, including the actual cue's availability flags,
backbone/source context, historical similarities, and request statistics.
Candidate identities and labels are unchanged. Donor availability matching and
unknown-cue retention are specified in [protocol.md](protocol.md).
