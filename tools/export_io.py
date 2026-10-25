"""Reviewer export contract; validates supplied arrays, not their raw-data producer.

See docs/exports.md for the file layout. Hash checks bind supplied bytes;
the caller remains responsible for the declared Train-only and observable inputs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from feature_schema import feature_schema


EXPORT_FILES = ("features.npy", "candidate_ids.npy", "labels.npy", "metadata.npz", "REQUEST_CONTEXT.npz")
SOURCE_ORDER = ["backbone", "first_track_transition", "history_transition", "artist"]
CATALOG_FILES = ("artists.npy", "audio.npy", "tags.npz", "genre.npz", "train_frequencies.npy")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, record):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checked_file(root, manifest, name):
    path = Path(root) / name
    expected = manifest.get("sha256", {}).get(name)
    if not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected:
        raise ValueError(f"file_hash_mismatch:{name}")
    return path


def npz_dict(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def integer_array(array, shape=None, minimum=None, name="array"):
    if not np.issubdtype(array.dtype, np.integer) or (shape is not None and array.shape != shape):
        raise ValueError(f"integer_shape:{name}")
    if minimum is not None and np.any(array < minimum):
        raise ValueError(f"integer_range:{name}")


def finite_float32(array, shape, name):
    if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"finite_float32_shape:{name}")


@dataclass
class Export:
    root: Path
    features: np.ndarray
    ids: np.ndarray
    labels: np.ndarray
    meta: dict
    context: np.ndarray
    manifest: dict

    @property
    def rows(self):
        return np.arange(len(self.meta["row_ids"]))

    @property
    def fingerprint(self):
        return digest(self.root / "EXPORT_MANIFEST.json")


def load_export(root, surface=None):
    root = Path(root)
    manifest = read_json(root / "EXPORT_MANIFEST.json")
    if (manifest.get("schema") != "arrive.reviewer_export.v1"
            or manifest.get("surface") not in ("train_cv", "validation", "test")
            or (surface is not None and manifest.get("surface") != surface)
            or not isinstance(manifest.get("dataset"), str) or not manifest["dataset"]
            or type(manifest.get("seed")) is not int or manifest["seed"] not in (2024, 2025, 2026)
            or type(manifest.get("synthetic")) is not bool
            or manifest.get("feature_schema") != feature_schema()
            or manifest.get("uses_future_labels_for_features") is not False):
        raise ValueError("export_manifest_contract")
    paths = {name: checked_file(root, manifest, name) for name in EXPORT_FILES}
    features = np.load(paths["features.npy"], mmap_mode="r", allow_pickle=False)
    ids = np.load(paths["candidate_ids.npy"], mmap_mode="r", allow_pickle=False)
    labels = np.load(paths["labels.npy"], mmap_mode="r", allow_pickle=False)
    meta = npz_dict(paths["metadata.npz"])
    required = ("offsets", "rrf_indices", "row_ids", "users", "targets", "target_counts", "bucket_ids")
    if not set(required) <= set(meta):
        raise ValueError("metadata_fields_missing")
    rows = meta["row_ids"]
    if rows.ndim != 1 or not len(rows):
        raise ValueError("nonempty_export_required")
    n = len(rows)
    m = len(ids)
    finite_float32(features, (m, 28), "features")
    integer_array(ids, (m,), 3, "candidate_ids")
    if np.any(ids > np.iinfo(np.int32).max):
        raise ValueError("candidate_id_exceeds_native_int32")
    integer_array(labels, (m,), 0, "labels")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("binary_labels_required")
    integer_array(rows, (n,), 0, "row_ids")
    if len(np.unique(rows)) != n:
        raise ValueError("duplicate_row_ids")
    for name in ("users", "target_counts", "bucket_ids"):
        integer_array(meta[name], (n,), 0, name)
    if np.any(rows > np.iinfo(np.int64).max) or np.any(meta["users"] > np.iinfo(np.int64).max):
        raise ValueError("row_or_user_id_exceeds_native_int64")
    if np.any(meta["bucket_ids"] > 3):
        raise ValueError("bucket_ids_outside_four_gap_bins")
    integer_array(meta["offsets"], (n + 1,), 0, "offsets")
    offsets = meta["offsets"]
    if (offsets[0] != 0 or offsets[-1] != m or np.any(offsets > m)
            or np.any(offsets[1:] <= offsets[:-1]) or np.any(np.diff(offsets) < 200)):
        raise ValueError("candidate_offsets_or_C200_width")
    integer_array(meta["rrf_indices"], (n, 200), 0, "rrf_indices")
    targets = meta["targets"]
    integer_array(targets, name="targets")
    if targets.ndim != 2 or targets.shape[0] != n:
        raise ValueError("target_matrix_shape")
    if np.any(meta["target_counts"] < 1) or np.any(meta["target_counts"] > targets.shape[1]):
        raise ValueError("positive_complete_target_counts_required")
    for row, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        candidates = ids[lo:hi]
        indices = meta["rrf_indices"][row]
        truth = targets[row, :int(meta["target_counts"][row])]
        # Unseen negative identities remain in the denominator. -1 and 0/1/2
        # are reserved padding/unknown IDs and are not target identities.
        if len(np.unique(truth)) != len(truth) or np.isin(truth, (-1, 0, 1, 2)).any():
            raise ValueError("distinct_target_identities_required")
        if len(np.unique(candidates)) != len(candidates):
            raise ValueError("duplicate_candidates_in_request")
        if np.any(indices >= hi - lo) or len(np.unique(indices)) != 200:
            raise ValueError("rrf_local_indices_invalid")
        if not np.array_equal(labels[lo:hi], np.isin(candidates, truth).astype(labels.dtype)):
            raise ValueError("labels_disagree_with_complete_targets")
    context = npz_dict(paths["REQUEST_CONTEXT.npz"])
    if not {"row_ids", "users", "relation"} <= set(context):
        raise ValueError("request_context_fields_missing")
    for name in ("row_ids", "users"):
        integer_array(context[name], (n,), 0, "context_" + name)
    if not (np.array_equal(context["row_ids"], rows) and np.array_equal(context["users"], meta["users"])):
        raise ValueError("request_context_order_mismatch")
    finite_float32(context["relation"], (n, 18), "request_context")
    return Export(root, features, ids, labels, meta, context["relation"], manifest)


def require_same_run(train, other):
    for key in ("dataset", "seed", "synthetic"):
        if train.manifest[key] != other.manifest[key]:
            raise ValueError(f"export_run_mismatch:{key}")
    if np.intersect1d(train.meta["row_ids"], other.meta["row_ids"]).size:
        raise ValueError("train_validation_request_overlap")


def load_sources(export):
    manifest = export.manifest
    if manifest.get("complete_source_lists") is not True or manifest.get("source_order") != SOURCE_ORDER:
        raise ValueError("declared_complete_four_sources_required")
    data = npz_dict(checked_file(export.root, manifest, "sources.npz"))
    if not {"row_ids", "users", "source_ids", "source_lengths"} <= set(data):
        raise ValueError("source_fields_missing")
    n = len(export.rows)
    for name in ("row_ids", "users"):
        integer_array(data[name], (n,), 0, "source_" + name)
    if not (np.array_equal(data["row_ids"], export.meta["row_ids"])
            and np.array_equal(data["users"], export.meta["users"])):
        raise ValueError("source_request_order_mismatch")
    integer_array(data["source_ids"], (n, 4, 200), 0, "source_ids")
    integer_array(data["source_lengths"], (n, 4), 0, "source_lengths")
    if np.any(data["source_lengths"] > 200) or np.any(data["source_lengths"][:, 0] != 200):
        raise ValueError("native_v2_source_length_contract")
    if np.any(data["source_ids"] > np.iinfo(np.int32).max):
        raise ValueError("source_id_exceeds_native_int32")
    for row in range(n):
        for source, count in zip(data["source_ids"][row], data["source_lengths"][row]):
            if np.any(source[:count] < 3) or np.any(source[count:] != 0):
                raise ValueError("source_items_or_padding")
            if len(np.unique(source[:count])) != count:
                raise ValueError("duplicate_source_items")
    return [[data["source_ids"][r, s, :int(data["source_lengths"][r, s])]
             for s in range(4)] for r in range(n)]


def load_cue_treatment(export, catalog_root):
    """Verify a supplied mapping by rerunning the unchanged label-free selector.

    This does not reconstruct cues or Train counts from raw events. The mapping
    producer is omitted; supplied cues are checked against all matching fields.
    """
    from scipy import sparse
    from cue_donors import (CueTreatment, availability_codes, controlled_mapping,
                                    cue_match_values, verify_unchanged_rows)
    from data_splits import user_partition

    if catalog_root is None:
        raise ValueError("wrong_mode_requires_catalog_and_precomputed_CUE_DONORS")
    catalog_root = Path(catalog_root)
    cm = read_json(catalog_root / "CATALOG_MANIFEST.json")
    if (cm.get("schema") != "arrive.reviewer_catalog.v1" or cm.get("counts_scope") != "train_only"
            or cm.get("uses_future_labels") is not False
            or not isinstance(cm.get("catalog_id"), str) or not cm["catalog_id"]
            or export.manifest.get("catalog_id") != cm["catalog_id"]
            or export.manifest.get("catalog_manifest_sha256") != digest(catalog_root / "CATALOG_MANIFEST.json")
            or export.manifest.get("donor_selector_reads_targets") is not False):
        raise ValueError("catalog_observability_or_identity_contract")
    paths = {name: checked_file(catalog_root, cm, name) for name in CATALOG_FILES}
    artists = np.load(paths["artists.npy"], allow_pickle=False)
    audio = np.load(paths["audio.npy"], mmap_mode="r", allow_pickle=False)
    freq = np.load(paths["train_frequencies.npy"], allow_pickle=False)
    tags, genre = [sparse.load_npz(paths[name]).tocsr() for name in ("tags.npz", "genre.npz")]
    if artists.ndim != 1 or len(artists) < 4:
        raise ValueError("catalog_artists_shape")
    size = len(artists)
    integer_array(artists, (size,), -1, "artists")
    if audio.ndim != 2 or audio.shape[1] < 1:
        raise ValueError("catalog_audio_shape")
    finite_float32(audio, (size, audio.shape[1]), "audio")
    if (freq.shape != (size,) or not np.issubdtype(freq.dtype, np.number)
            or not np.isfinite(freq).all() or np.any(freq < 0) or np.any(freq != np.floor(freq))):
        raise ValueError("catalog_train_counts")
    for matrix in (tags, genre):
        if matrix.shape[0] != size or matrix.shape[1] < 1 or not np.isfinite(matrix.data).all():
            raise ValueError("catalog_sparse_content")
        if matrix.dtype != np.float32:
            raise ValueError("catalog_sparse_content_float32_required")
    if (np.any(artists[:3] != -1) or np.any(audio[:3] != 0) or tags[:3].nnz
            or genre[:3].nnz or np.any(freq[:3] != 0) or np.any(export.ids >= size)):
        raise ValueError("catalog_special_or_candidate_identity")
    mapping = npz_dict(checked_file(export.root, export.manifest, "CUE_DONORS.npz"))
    required = {"row_ids", "users", "cues", "donors", "source_rows", "tiers", "eligible", "partition"}
    if not required <= set(mapping):
        raise ValueError("donor_fields_missing")
    n = len(export.rows)
    for name in ("row_ids", "users"):
        integer_array(mapping[name], (n,), 0, "donor_" + name)
    if not (np.array_equal(mapping["row_ids"], export.meta["row_ids"])
            and np.array_equal(mapping["users"], export.meta["users"])):
        raise ValueError("donor_request_order_mismatch")
    for name in ("cues", "donors", "tiers", "partition"):
        integer_array(mapping[name], (n,), 0, name)
    integer_array(mapping["source_rows"], (n,), -1, "source_rows")
    if mapping["eligible"].dtype != bool or mapping["eligible"].shape != (n,):
        raise ValueError("donor_eligibility_shape")
    cues = mapping["cues"]
    if np.any(cues >= size) or np.any((cues < 3) & (cues != 1)):
        raise ValueError("actual_cue_identity")
    partition = ((~user_partition(export.meta["users"])).astype(np.uint8)
                 if export.manifest["surface"] == "train_cv" else np.zeros(n, np.uint8))
    if not np.array_equal(mapping["partition"], partition):
        raise ValueError("donor_fit_calibration_partition_policy")
    availability = availability_codes(artists, tags, audio, genre)
    bins = np.floor(np.log2(1 + freq.astype(np.float64))).astype(np.int64)
    catalog = np.flatnonzero((freq > 0) & (np.arange(size) >= 3))
    actual = controlled_mapping(cues, availability[cues], bins[cues], partition, export.manifest["seed"],
                                catalog, availability[catalog], bins[catalog])
    for name, expected in zip(("donors", "source_rows", "tiers", "eligible"), actual):
        if not np.array_equal(mapping[name], expected):
            raise ValueError(f"donor_native_selector_mismatch:{name}")
    verify_unchanged_rows(export.features, export.meta["offsets"], actual[3])
    for row, (lo, hi) in enumerate(zip(export.meta["offsets"][:-1], export.meta["offsets"][1:])):
        block = export.features[lo:hi]
        bits = ((int(availability[cues[row]]) >> np.arange(4)) & 1).astype(np.float32)
        if not np.all(block[:, [19, 21, 24, 27]] == bits):
            raise ValueError("actual_cue_availability_mismatch")
        expected = cue_match_values(export.ids[lo:hi], int(cues[row]), artists, tags, audio, genre)
        if not np.allclose(block[:, [12, 13, 14, 16]], expected, rtol=1e-5, atol=1e-6):
            raise ValueError("actual_cue_matching_content_mismatch")
    return CueTreatment(actual[0], artists, tags, audio, genre, actual[3])


def require_fresh_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"output_already_exists:{path}")
    return path
