# Interpreting replay and other-song contributions

The paper evaluates complete ten-song lists. Later replays of the observed
return song are valid targets, alongside other continuation songs. The
song-type analysis partitions each original metric; it does not filter and
refill a recommendation list or recompute an ideal list for a smaller target
set.

For a request with full target set \(Y\), let
\(d(r)=1/\log_2(r+1)\) and
\(D(Y)=\sum_{r=1}^{\min(10,|Y|)}d(r)\).
The NDCG contribution from return-song replay uses hits equal to the observed
return song; the other-song contribution uses the remaining hits. Both retain
their original ranks and the same \(D(Y)\), including targets outside the
Train-frozen catalog. Thus the two contributions sum exactly to the original
NDCG. Recall is partitioned in the same way, with denominator \(|Y|\).

For lists A and B, let \(H_A\) and \(H_B\) be their other-song relevant-hit
sets. Their other-song NDCG difference has the exact accounting identity

\[
\Delta N_{\mathrm{other}}=\frac{1}{D(Y)}\left[
\sum_{i\in H_A\setminus H_B}d(r_A(i))
-\sum_{i\in H_B\setminus H_A}d(r_B(i))
+\sum_{i\in H_A\cap H_B}\{d(r_A(i))-d(r_B(i))\}
\right].
\]

To obtain it, expand the two NDCG sums and partition their hits into gained,
lost and shared sets. Averaging requests within users and then averaging
matched seeds preserves this identity. Interval endpoints do not add; the
bootstrap must operate on the summed paired user-level quantities.

With a fixed candidate pool, gained and lost top-ten hits describe ranking
selection, not candidate retrieval. When candidate pools differ, the identity
still holds but cannot by itself separate availability from ranking. A shared
hit's position change also does not establish which other song displaced it.

## Running the analysis

Use `scripts/summarize_results.py` with three seeds per method. The existing
`rankings` configuration accepts methods `arrive`, `backbone`, `removed`,
`mismatched`, `identity` and `joint-removal`. In addition to the three-arm
comparisons, it reports ARRIVE minus identity and ARRIVE minus joint removal.
New prediction exports include the actual observed `cues` array. When all
paired files carry it, the summary also reports:

- replay and other-song Recall/NDCG contributions;
- gained, lost and shared-hit rank terms for other-song NDCG;
- metrics restricted to requests with a known catalog return song that is
  absent from the full target set. Unknown cue ID 1 is excluded from this
  subgroup because its physical replay identity is unrecoverable; its metric
  contribution remains in the full-cohort other-song partition.

An external `cue_bundle` remains usable for legacy eligibility metadata. Its
actual SHA-256 is recorded in `inputs`; supply `cue_bundle_sha256` in the
configuration to require an expected hash before analysis.

The summary uses user-macro aggregation and 10,000 matched-user bootstrap draws
by default. It reports both 95% and 98.75% intervals. The latter is the interval
level retained for the paper's identity and joint controls; it is not a new
claim of one multiplicity correction covering all comparisons in the paper.

For the Yambda/SASRec matching-minus-removal comparison, the saved analysis
gives +0.6354 points from gained other-song hits, -0.5915 from lost hits, and
-0.3321 from rank changes of shared hits. Their sum is -0.2882 points. These
are components of the other-song difference, not of the total matching gain.
