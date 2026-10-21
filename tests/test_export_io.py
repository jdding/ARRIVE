"""Small synthetic contract checks; no raw data, trained models or remote work."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import numpy as np
from scipy import sparse
from arrive_cue_donors import availability_codes, controlled_mapping, cue_match_values
from arrive_framework_datasets import feature_schema
from export_io import (CATALOG_FILES, EXPORT_FILES, SOURCE_ORDER, digest, load_cue_treatment,
                       load_export, load_sources, npz_dict, require_fresh_output, require_same_run)
from replay_arrive_cue_source_v2 import replay_row
from run_arrive_attribution_controls import rrf_pool
from run_music4all_candidate_support_a1 import user_partition


def save_manifest(root, record):
    (root / "EXPORT_MANIFEST.json").write_text(json.dumps(record), encoding="utf-8")


def rehash(root, names):
    record = json.loads((root / "EXPORT_MANIFEST.json").read_text())
    record["sha256"].update({name: digest(root / name) for name in names})
    save_manifest(root, record)


def make_export(root, surface="train_cv", row_start=0):
    """Reusable synthetic input writer for local CLI smokes, not paper exports."""
    root.mkdir()
    n = 8
    base = np.arange(3, 203, dtype=np.int32)
    sources = [base, np.asarray([203, 204, 205], np.int32), np.asarray([], np.int32), np.asarray([], np.int32)]
    pool, _, _ = rrf_pool(sources)
    ids = np.arange(3, 206, dtype=np.int32)
    position = {int(item): at for at, item in enumerate(ids)}
    features = np.zeros((len(ids), 28), np.float32)
    for column, source in enumerate(sources, 2):
        rank = {int(item): at + 1 for at, item in enumerate(source)}
        ranks = np.asarray([rank.get(int(item), 0) for item in ids], np.float32)
        features[:, column] = np.where(ranks > 0, 1 / np.log2(np.maximum(ranks, 1) + 1), 0)
    # Additional signal just makes the tiny fit path nonconstant.
    features[:, 0] = np.linspace(-1, 1, len(ids), dtype=np.float32)
    truth = np.asarray([pool[0], -3], np.int64)
    labels = np.isin(ids, truth).astype(np.uint8)
    meta = dict(offsets=np.arange(n + 1, dtype=np.int64) * len(ids),
        rrf_indices=np.tile(np.asarray([position[int(i)] for i in pool], np.int32), (n, 1)),
        row_ids=np.arange(row_start, row_start + n, dtype=np.int64), users=np.arange(100, 100 + n, dtype=np.int64),
        targets=np.tile(truth, (n, 1)), target_counts=np.full(n, 2, np.int32), bucket_ids=np.zeros(n, np.int8))
    np.save(root / "features.npy", np.tile(features, (n, 1)))
    np.save(root / "candidate_ids.npy", np.tile(ids, n))
    np.save(root / "labels.npy", np.tile(labels, n))
    np.savez(root / "metadata.npz", **meta)
    np.savez(root / "REQUEST_CONTEXT.npz", row_ids=meta["row_ids"], users=meta["users"], relation=np.zeros((n, 18), np.float32))
    padded = np.zeros((n, 4, 200), np.int32)
    for source, items in enumerate(sources):
        padded[:, source, :len(items)] = items
    np.savez(root / "sources.npz", row_ids=meta["row_ids"], users=meta["users"], source_ids=padded,
             source_lengths=np.tile(np.asarray([len(items) for items in sources], np.int32), (n, 1)))
    record = {"schema": "arrive.reviewer_export.v1", "dataset": "synthetic_contract_fixture",
        "surface": surface, "seed": 2024, "synthetic": True, "feature_schema": feature_schema(),
        "uses_future_labels_for_features": False, "complete_source_lists": True,
        "source_order": SOURCE_ORDER,
        "sha256": {name: digest(root / name) for name in (*EXPORT_FILES, "sources.npz")}}
    save_manifest(root, record)
    return root


def add_wrong_cue_assets(export_root, catalog_root):
    """Write a supplied mapping using the public native selector on fake content."""
    catalog_root.mkdir(exist_ok=True)
    size = 206
    artists = (np.arange(size) % 2).astype(np.int32)
    audio = np.eye(2, dtype=np.float32)[np.arange(size) % 2]
    tag_rows = np.eye(3, dtype=np.float32)[np.arange(size) % 3]
    genre_rows = np.eye(2, dtype=np.float32)[np.arange(size) % 2]
    counts = np.full(size, 5, np.int64)
    artists[:3] = -1
    audio[:3] = tag_rows[:3] = genre_rows[:3] = 0
    counts[:3] = 0
    tags, genre = sparse.csr_matrix(tag_rows), sparse.csr_matrix(genre_rows)
    np.save(catalog_root / "artists.npy", artists)
    np.save(catalog_root / "audio.npy", audio)
    np.save(catalog_root / "train_frequencies.npy", counts)
    sparse.save_npz(catalog_root / "tags.npz", tags)
    sparse.save_npz(catalog_root / "genre.npz", genre)
    cm = {"schema": "arrive.reviewer_catalog.v1", "catalog_id": "synthetic_catalog",
        "counts_scope": "train_only", "uses_future_labels": False,
        "sha256": {name: digest(catalog_root / name) for name in CATALOG_FILES}}
    (catalog_root / "CATALOG_MANIFEST.json").write_text(json.dumps(cm), encoding="utf-8")
    export = load_export(export_root)
    cues = np.arange(3, 3 + len(export.rows), dtype=np.int64)
    availability = availability_codes(artists, tags, audio, genre)
    bins = np.floor(np.log2(1 + counts)).astype(np.int64)
    partition = (~user_partition(export.meta["users"])).astype(np.uint8) if export.manifest["surface"] == "train_cv" else np.zeros(len(cues), np.uint8)
    catalog = np.flatnonzero((counts > 0) & (np.arange(size) >= 3))
    donors, source, tiers, eligible = controlled_mapping(cues, availability[cues], bins[cues], partition,
        2024, catalog, availability[catalog], bins[catalog])
    features = export.features.copy()
    for row, (lo, hi) in enumerate(zip(export.meta["offsets"][:-1], export.meta["offsets"][1:])):
        features[lo:hi, [12, 13, 14, 16]] = cue_match_values(export.ids[lo:hi], int(cues[row]), artists, tags, audio, genre)
        features[lo:hi, [19, 21, 24, 27]] = ((int(availability[cues[row]]) >> np.arange(4)) & 1)
    np.save(export_root / "features.npy", features)
    np.savez(export_root / "CUE_DONORS.npz", row_ids=export.meta["row_ids"], users=export.meta["users"],
        cues=cues, donors=donors, source_rows=source, tiers=tiers, eligible=eligible, partition=partition)
    record = export.manifest
    record.update(catalog_id=cm["catalog_id"], catalog_manifest_sha256=digest(catalog_root / "CATALOG_MANIFEST.json"),
                  donor_selector_reads_targets=False)
    save_manifest(export_root, record)
    rehash(export_root, ("features.npy", "CUE_DONORS.npz"))


class ExportContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="arrive-export-contract-")
        self.base = Path(self.temp.name)
        self.root = make_export(self.base / "train")

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_export_and_complete_source_replay(self):
        export = load_export(self.root, "train_cv")
        sources = load_sources(export)
        lo, hi = export.meta["offsets"][:2]
        result = replay_row(export.ids[lo:hi], export.features[lo:hi], export.meta["rrf_indices"][0],
                            export.meta["targets"][0], sources[0])
        self.assertEqual(result["full_coverage"], 0.5)
        self.assertEqual(result["removed_coverage"], 0.5)
        self.assertEqual(export.context.shape, (8, 18))

    def test_hash_mismatch_is_rejected(self):
        with (self.root / "features.npy").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "file_hash_mismatch"):
            load_export(self.root)

    def test_rrf_global_or_duplicate_indices_are_rejected(self):
        meta = npz_dict(self.root / "metadata.npz")
        meta["rrf_indices"][1, 0] = meta["offsets"][1] + 1
        np.savez(self.root / "metadata.npz", **meta)
        rehash(self.root, ("metadata.npz",))
        with self.assertRaisesRegex(ValueError, "rrf_local_indices_invalid"):
            load_export(self.root)

    def test_context_order_is_checked(self):
        data = npz_dict(self.root / "REQUEST_CONTEXT.npz")
        data["row_ids"] = data["row_ids"][::-1]
        np.savez(self.root / "REQUEST_CONTEXT.npz", **data)
        rehash(self.root, ("REQUEST_CONTEXT.npz",))
        with self.assertRaisesRegex(ValueError, "request_context_order_mismatch"):
            load_export(self.root)

    def test_targets_cannot_disagree_with_binary_labels(self):
        labels = np.load(self.root / "labels.npy")
        labels[0] = 1 - labels[0]
        np.save(self.root / "labels.npy", labels)
        rehash(self.root, ("labels.npy",))
        with self.assertRaisesRegex(ValueError, "labels_disagree"):
            load_export(self.root)

    def test_nonfinite_features_and_duplicate_candidates_are_rejected(self):
        features = np.load(self.root / "features.npy")
        features[0, 0] = np.nan
        np.save(self.root / "features.npy", features)
        rehash(self.root, ("features.npy",))
        with self.assertRaisesRegex(ValueError, "finite_float32"):
            load_export(self.root)
        features[0, 0] = 0
        np.save(self.root / "features.npy", features)
        ids = np.load(self.root / "candidate_ids.npy")
        ids[1] = ids[0]
        np.save(self.root / "candidate_ids.npy", ids)
        rehash(self.root, ("features.npy", "candidate_ids.npy"))
        with self.assertRaisesRegex(ValueError, "duplicate_candidates"):
            load_export(self.root)

    def test_complete_sources_must_be_declared_and_match_ranks(self):
        export = load_export(self.root)
        export.manifest["complete_source_lists"] = False
        with self.assertRaisesRegex(ValueError, "declared_complete"):
            load_sources(export)
        export.manifest["complete_source_lists"] = True
        sources = load_sources(export)
        sources[0][1] = sources[0][1][:-1]
        lo, hi = export.meta["offsets"][:2]
        with self.assertRaisesRegex(ValueError, "complete_source_observed_ranks_differ"):
            replay_row(export.ids[lo:hi], export.features[lo:hi], export.meta["rrf_indices"][0],
                       export.meta["targets"][0], sources[0])

    def test_train_validation_identity_and_output_guard(self):
        train = load_export(self.root)
        validation = load_export(make_export(self.base / "validation", "validation", 1000))
        require_same_run(train, validation)
        validation.meta["row_ids"][0] = train.meta["row_ids"][0]
        with self.assertRaisesRegex(ValueError, "request_overlap"):
            require_same_run(train, validation)
        with self.assertRaises(FileExistsError):
            require_fresh_output(self.root)

    def test_wrong_mapping_reuses_native_selector_and_rejects_tampering(self):
        catalog = self.base / "catalog"
        add_wrong_cue_assets(self.root, catalog)
        treatment = load_cue_treatment(load_export(self.root), catalog)
        self.assertTrue(treatment.eligible.all())
        mapping = npz_dict(self.root / "CUE_DONORS.npz")
        mapping["donors"][0] = mapping["cues"][0]
        np.savez(self.root / "CUE_DONORS.npz", **mapping)
        rehash(self.root, ("CUE_DONORS.npz",))
        with self.assertRaisesRegex(ValueError, "native_selector_mismatch"):
            load_cue_treatment(load_export(self.root), catalog)

    def test_wrong_mapping_partition_and_original_content_are_checked(self):
        catalog = self.base / "catalog"
        add_wrong_cue_assets(self.root, catalog)
        mapping = npz_dict(self.root / "CUE_DONORS.npz")
        mapping["partition"][0] = 1 - mapping["partition"][0]
        np.savez(self.root / "CUE_DONORS.npz", **mapping)
        rehash(self.root, ("CUE_DONORS.npz",))
        with self.assertRaisesRegex(ValueError, "partition_policy"):
            load_cue_treatment(load_export(self.root), catalog)
        mapping["partition"][0] = 1 - mapping["partition"][0]
        np.savez(self.root / "CUE_DONORS.npz", **mapping)
        features = np.load(self.root / "features.npy")
        features[0, 13] += 0.25
        np.save(self.root / "features.npy", features)
        rehash(self.root, ("CUE_DONORS.npz", "features.npy"))
        with self.assertRaisesRegex(ValueError, "matching_content_mismatch"):
            load_cue_treatment(load_export(self.root), catalog)


if __name__ == "__main__":
    unittest.main()
