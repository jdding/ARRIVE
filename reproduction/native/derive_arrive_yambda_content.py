#!/usr/bin/env python3
"""Derive the Yambda-500M content arrays for the ARRIVE joint (RGSC) stage.

Counterpart of the Onion content arrays (musicnn_normalized.npy /
genre_normalized.npz). The grounding contract verified from the Onion
artifacts is mirrored exactly:

* rows are indexed by model_id and the row count equals the prepared
  vocabulary size (catalog + 3 reserved ids);
* reserved rows 0/1/2 are exactly zero (invalid grounding slots map to
  id 0 in the joint trainer);
* audio rows are L2-normalized unit vectors (cosine inner products);
  catalog items without an embedding get a zero row;
* the categorical array is an L2-normalized multi-hot per item. Yambda has
  no genre metadata, so the declared adaptation uses artists instead
  (artist_item_mapping.parquet), with deterministic dense columns ordered
  by ascending artist id. The array stays sparse: densifying
  vocab x n_artists is infeasible at Yambda scale.

Artist rows use model_id positions recovered through ``order[hit]``,
not positions in the item-id-sorted lookup. CSR normalization uses
``row_of_entry`` from indptr; scalar assignment would insert explicit zeros.

Fail-closed gates: refuse an existing output directory; verify the
prepared vocabulary identity; verify the embeddings file bytes + sha256
against the official LFS digest; bound the zero-row (missing content)
fraction by the release-level coverage (the catalog is popularity-skewed,
so its coverage must be at least the release-wide 82.2%); a provenance
manifest with artifact and input digests closes the audit chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import scipy
from scipy import sparse

PAD, UNK, EOS = 0, 1, 2
RESERVED = (PAD, UNK, EOS)
EXPECTED_EMBED_ROWS = 7_721_749
EXPECTED_EMBED_SHA256 = (
    "c8959a584257473ab9f5dab7e88b05ea3d03b4daf7c16e9a617e8e216d811c83"
)
EXPECTED_EMBED_BYTES = 13_814_230_943
RELEASE_TRACKS = 9_390_623
# Release-level coverage: 7,721,749 / 9,390,623 = 82.24%. The catalog is
# popularity-skewed, so its coverage must be at least release-wide; the
# missing fraction must stay below the release-level missing fraction.
EXPECTED_RELEASE_MISSING_FRAC = 1.0 - EXPECTED_EMBED_ROWS / RELEASE_TRACKS
NORM_TOL = 1e-5


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_catalog(prepared: Path) -> tuple[np.ndarray, int, int, dict]:
    """Return (catalog item_ids by model_id, vocab, first id, digest block)."""
    frequencies = np.load(prepared / "train_frequencies.npy", mmap_mode="r")
    vocab = int(len(frequencies))
    model_ids: list[int] = []
    item_ids: list[int] = []
    with (prepared / "track_identity.tsv").open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header != ["raw_index", "model_id", "item_id", "train_events"]:
            sys.exit("track_identity_header_mismatch")
        for line in handle:
            _, model_id, item_id, _ = line.rstrip("\n").split("\t")
            model_ids.append(int(model_id))
            item_ids.append(int(item_id))
    if len(model_ids) != vocab - len(RESERVED):
        sys.exit("catalog_vocab_mismatch")
    if model_ids != list(range(len(RESERVED), vocab)):
        sys.exit("model_ids_not_contiguous")
    digests = {
        "train_frequencies": sha256_file(prepared / "train_frequencies.npy"),
        "prepared_manifest": sha256_file(prepared / "PREP_MANIFEST.json"),
    }
    return (np.asarray(item_ids, dtype=np.int64), vocab, len(RESERVED), digests)


def build_audio(
    embeddings: Path,
    catalog_items: np.ndarray,
    order: np.ndarray,
    sorted_items: np.ndarray,
    first_item: int,
    vocab: int,
    embeddings_sha: str,
) -> tuple[np.ndarray, int, int]:
    """Stream the parquet once, placing matched rows directly by model_id.

    Per matched batch row the producer's unit norm is verified (float64),
    zero-norm rows count as missing, and the float32 renormalized vector is
    written into a preallocated buffer at ``first_item + order[hit]`` — the
    composition of the item-id-sorted position with the sorting permutation
    recovers the model_id row. Returns (audio, missing, dim).
    """
    if embeddings.stat().st_size != EXPECTED_EMBED_BYTES:
        sys.exit("embeddings_byte_identity_mismatch")
    if embeddings_sha != EXPECTED_EMBED_SHA256:
        sys.exit("embeddings_sha256_identity_mismatch")
    pf = pq.ParquetFile(embeddings)
    if pf.metadata.num_rows != EXPECTED_EMBED_ROWS:
        sys.exit("embeddings_row_identity_mismatch")
    names = {f.name for f in pf.schema_arrow}
    if not {"item_id", "normalized_embed"} <= names:
        sys.exit("embeddings_schema_mismatch")

    n_cat = len(catalog_items)
    dim: int | None = None
    buf: np.ndarray | None = None
    present = np.zeros(n_cat, dtype=bool)
    for batch in pf.iter_batches(batch_size=500_000,
                                 columns=["item_id", "normalized_embed"]):
        ids = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        vec_array = batch.column("normalized_embed")
        if vec_array.null_count:
            sys.exit("null_embedding_values")
        hit = np.searchsorted(sorted_items, ids)
        hit = np.clip(hit, 0, n_cat - 1)
        member = sorted_items[hit] == ids
        sel = np.flatnonzero(member)
        if len(sel) == 0:
            continue
        offsets = np.asarray(vec_array.offsets, dtype=np.int64)
        values = vec_array.values
        lengths = offsets[sel + 1] - offsets[sel]
        if dim is None:
            dim = int(lengths[0])
            if dim < 8:
                sys.exit("embedding_dimension_invalid")
            buf = np.zeros((n_cat, dim), dtype=np.float32)
        if not np.all(lengths == dim):
            sys.exit("embedding_dimension_inconsistent")
        flat = (offsets[sel][:, None] + np.arange(dim, dtype=np.int64)).ravel()
        rows = np.asarray(values.take(flat)).astype(np.float64).reshape(-1, dim)
        norms = np.linalg.norm(rows, axis=1)
        nonzero = norms > 1e-12
        if not np.all(np.abs(norms[nonzero] - 1.0) < 1e-6):
            sys.exit("producer_normalized_embed_not_unit")
        model_pos = order[hit[member]]
        if present[model_pos[nonzero]].any():
            sys.exit("duplicate_embedding_rows")
        if len(np.unique(model_pos[nonzero])) != len(model_pos[nonzero]):
            sys.exit("duplicate_embedding_rows")
        block = rows[nonzero].astype(np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True).astype(np.float32)
        buf[model_pos[nonzero]] = block
        present[model_pos[nonzero]] = True
    if dim is None or buf is None:
        sys.exit("no_catalog_embeddings_found")
    missing = int(n_cat - present.sum())
    if missing > EXPECTED_RELEASE_MISSING_FRAC * n_cat:
        sys.exit("audio_missing_fraction_exceeded")
    audio = np.zeros((vocab, dim), dtype=np.float32)
    audio[first_item:first_item + n_cat] = buf
    del buf
    live = np.linalg.norm(audio[first_item:first_item + n_cat], axis=1)
    if not np.all(np.abs(live[present] - 1.0) < NORM_TOL):
        sys.exit("audio_norm_violation")
    if not np.all(live[~present] == 0.0):
        sys.exit("audio_absent_rows_not_zero")
    return audio, missing, int(dim)


def build_artist(
    artist_mapping: Path,
    catalog_items: np.ndarray,
    order: np.ndarray,
    sorted_items: np.ndarray,
    first_item: int,
    vocab: int,
) -> tuple[sparse.csr_matrix, int, int]:
    """Build the L2-normalized artist multi-hot CSR, rows by model_id."""
    table = pq.read_table(artist_mapping)
    names = {f.name for f in table.schema}
    if not {"item_id", "artist_id"} <= names:
        sys.exit("artist_mapping_schema_mismatch")
    ids = table.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
    artists = table.column("artist_id").to_numpy(zero_copy_only=False).astype(np.int64)
    hit = np.searchsorted(sorted_items, ids)
    hit = np.clip(hit, 0, len(sorted_items) - 1)
    member = sorted_items[hit] == ids
    if not member.any():
        sys.exit("no_artist_relations_for_catalog")
    model_pos = order[hit[member]] + first_item
    unique_artists, artist_col = np.unique(artists[member], return_inverse=True)
    n_cols = len(unique_artists) + 1
    # (item, artist) pairs may repeat across mapping rows; deduplicate pairs.
    # Overflow check: (vocab + n_cat) * n_cols << 2^63 for vocab 1.33M.
    pair_keys = np.unique(model_pos.astype(np.int64) * n_cols + artist_col)
    final_rows = pair_keys // n_cols
    final_cols = pair_keys % n_cols
    data = np.ones(len(pair_keys), dtype=np.float32)
    matrix = sparse.csr_matrix(
        (data, (final_rows, final_cols)), shape=(vocab, len(unique_artists))
    )
    matrix = matrix.tocsr()
    if matrix.indptr[len(RESERVED)] != 0:
        sys.exit("reserved_rows_not_empty")
    # L2-normalize each multi-hot row via order-independent entry ownership.
    row_sq = np.bincount(final_rows, weights=np.ones(len(pair_keys)),
                         minlength=vocab)
    norms = np.sqrt(row_sq)
    row_of_entry = np.repeat(np.arange(vocab), np.diff(matrix.indptr))
    if not np.all(norms[row_of_entry] > 0):
        sys.exit("artist_row_norm_zero")
    matrix.data = (matrix.data.astype(np.float64) / norms[row_of_entry]).astype(
        np.float32
    )
    post = np.bincount(row_of_entry, weights=matrix.data.astype(np.float64) ** 2,
                       minlength=vocab)
    touched = final_rows
    if not np.all(np.abs(np.sqrt(post[touched]) - 1.0) < NORM_TOL):
        sys.exit("artist_norm_violation")
    empty_rows = int((np.asarray((matrix != 0).sum(1)).ravel() == 0)
                     [first_item:first_item + len(catalog_items)].sum())
    if empty_rows > EXPECTED_RELEASE_MISSING_FRAC * len(catalog_items):
        sys.exit("artist_missing_fraction_exceeded")
    return matrix, len(unique_artists), empty_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--artist-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="output content directory (must not exist)")
    parser.add_argument("--expect-vocab", type=int, default=1_326_192)
    args = parser.parse_args()

    if args.output.exists():
        sys.exit("refuse_existing_output")

    catalog_items, vocab, first_item, prepared_digests = load_catalog(args.prepared)
    if vocab != args.expect_vocab:
        sys.exit("vocab_identity_mismatch")
    order = np.argsort(catalog_items)
    sorted_items = catalog_items[order]

    embeddings_sha = sha256_file(args.embeddings)
    audio, audio_missing, dim = build_audio(
        args.embeddings, catalog_items, order, sorted_items,
        first_item, vocab, embeddings_sha,
    )
    artist, n_artists, artist_missing = build_artist(
        args.artist_mapping, catalog_items, order, sorted_items,
        first_item, vocab,
    )

    args.output.mkdir(parents=True)
    np.save(args.output / "audio_normalized.npy", audio, allow_pickle=False)
    sparse.save_npz(args.output / "artist_normalized.npz", artist)

    artifacts = {}
    for name in ("audio_normalized.npy", "artist_normalized.npz"):
        path = args.output / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema": "genrec.yambda500m.content.v1",
        "status": "prepared",
        "test_accessed": False,
        "producer": Path(__file__).name,
        "producer_sha256": sha256_file(Path(__file__)),
        "inputs": {
            "embeddings": {
                "bytes": args.embeddings.stat().st_size,
                "sha256": embeddings_sha,
                "official_lfs_sha256": EXPECTED_EMBED_SHA256,
            },
            "artist_mapping": {"sha256": sha256_file(args.artist_mapping)},
            "prepared_vocabulary": prepared_digests,
        },
        "protocol": {
            "row_index_basis": "model_id; row count = vocab = catalog + 3",
            "reserved_rows_zero": True,
            "audio_source": "release normalized_embed (CLMR-style CNN), "
                            "renormalized in float32",
            "categorical_adaptation": "artist multi-hot replaces genre "
                                      "(no genre metadata in the release); "
                                      "L2-normalized rows, sparse CSR, dense "
                                      "columns ordered by ascending artist id",
            "missing_row_semantics": "zero row; terms degrade to zero for "
                                     "items without content",
        },
        "counts": {
            "vocab": vocab,
            "catalog_items": len(catalog_items),
            "audio_dim": dim,
            "audio_missing_items": audio_missing,
            "audio_missing_frac": round(audio_missing / len(catalog_items), 6),
            "release_missing_frac": round(EXPECTED_RELEASE_MISSING_FRAC, 6),
            "artists": n_artists,
            "artist_missing_items": artist_missing,
            "artist_missing_frac": round(artist_missing / len(catalog_items), 6),
        },
        "library_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "artifacts": artifacts,
    }
    (args.output / "CONTENT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
