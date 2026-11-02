#!/usr/bin/env python3
"""Bind native candidate exports to the portable train/evaluation/control CLI."""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "tools"))

import numpy as np
from scipy import sparse
from cue_donors import availability_codes, controlled_mapping
from feature_schema import feature_schema
from export_io import (CATALOG_FILES, EXPORT_FILES, SOURCE_ORDER, checked_file, digest,
                       load_cue_treatment, load_export, load_sources, npz_dict, read_json, write_json)
from data_splits import user_partition


def convert(source, output, catalog=None, prepared=None):
    if output.exists():
        raise FileExistsError(output)
    record = read_json(source / "EXPORT_MANIFEST.json")
    policy = record.get("source_policy", "full")
    if policy not in ("full", "joint-removal"):
        raise ValueError("unknown_source_policy")
    if (record.get("schema") != "arrive.framework.export.v1" or record.get("status") != "pass"
            or record.get("surface") not in ("train_cv", "validation", "test")
            or record.get("feature_schema") != feature_schema()
            or record.get("test_accessed") != (record["surface"] == "test")
            or set(record.get("output_sha256", {})) != set(EXPORT_FILES)):
        raise ValueError("native_export_contract")
    native_binding = record.get("code_sha256", {})
    if not native_binding or any(digest(REPO / "reproduction/native" / name) != value
                                 for name, value in native_binding.items()
                                 if Path(name).name == name):
        raise ValueError("native_export_source_version")
    if any(Path(name).name != name for name in native_binding):
        raise ValueError("native_source_binding_path")
    verified = {name: checked_file(source, {"sha256": record["output_sha256"]}, name) for name in EXPORT_FILES}
    sources_path = source / "sources.npz"
    sources = None
    if sources_path.exists():
        if record.get("reproduction_sources_sha256") != digest(sources_path):
            raise ValueError("native_complete_sources_hash")
        sources = npz_dict(sources_path)
    output.mkdir(parents=True, exist_ok=False)
    for name, path in verified.items():
        shutil.copyfile(path, output / name)
    manifest = dict(schema="arrive.reviewer_export.v1", dataset=record["dataset"],
        seed=record["seed"], surface=record["surface"], synthetic=record.get("synthetic", False),
        source_policy=policy,
        feature_schema=feature_schema(), uses_future_labels_for_features=False,
        sha256={name: digest(output / name) for name in EXPORT_FILES},
        native_manifest_sha256=digest(source / "EXPORT_MANIFEST.json"),
        native_serving_inputs=record.get("serving_inputs"), native_code_sha256=native_binding,
        native_retrieval_contract=record.get("retrieval_contract"))
    if sources is not None:
        shutil.copyfile(sources_path, output / "sources.npz")
        manifest.update(complete_source_lists=True, source_order=SOURCE_ORDER)
        manifest["sha256"]["sources.npz"] = digest(output / "sources.npz")
    write_json(output / "EXPORT_MANIFEST.json", manifest)
    export = load_export(output)
    if sources is not None:
        load_sources(export)
    if catalog is not None:
        cm = read_json(catalog / "CATALOG_MANIFEST.json")
        paths = {name: checked_file(catalog, cm, name) for name in CATALOG_FILES}
        artists = np.load(paths["artists.npy"], allow_pickle=False)
        audio = np.load(paths["audio.npy"], mmap_mode="r", allow_pickle=False)
        tags, genre = [sparse.load_npz(paths[n]).tocsr() for n in ("tags.npz", "genre.npz")]
        frequency = np.load(paths["train_frequencies.npy"], allow_pickle=False)
        if sources is not None and "cues" in sources:
            cues = sources["cues"]
        elif prepared is not None:
            sessions = np.load(prepared / "sessions.npy", mmap_mode="r", allow_pickle=False)
            events = np.load(prepared / "events.npy", mmap_mode="r", allow_pickle=False)
            selected = sessions[export.meta["row_ids"]]
            if not np.array_equal(selected[:, 0], export.meta["users"]):
                raise ValueError("prepared_request_user_identity")
            cues = np.asarray(events[selected[:, 2]], dtype=np.int64)
        else:
            raise ValueError("wrong_control_requires_exported_cues_or_prepared_data")
        if cues.shape != (len(export.rows),) or not np.issubdtype(cues.dtype, np.integer) or np.any(cues < 0) or np.any(cues >= len(artists)):
            raise ValueError("cue_identity_or_shape")
        availability = availability_codes(artists, tags, audio, genre)
        bins = np.floor(np.log2(1 + frequency.astype(np.float64))).astype(np.int64)
        partition = ((~user_partition(export.meta["users"])).astype(np.uint8)
                     if record["surface"] == "train_cv" else np.zeros(len(cues), np.uint8))
        known = np.flatnonzero((frequency > 0) & (np.arange(len(frequency)) >= 3))
        donors, source_rows, tiers, eligible = controlled_mapping(cues, availability[cues], bins[cues],
            partition, record["seed"], known, availability[known], bins[known])
        np.savez_compressed(output / "CUE_DONORS.npz", row_ids=export.meta["row_ids"], users=export.meta["users"],
            cues=cues, donors=donors, source_rows=source_rows, tiers=tiers, eligible=eligible, partition=partition)
        manifest.update(catalog_id=cm["catalog_id"], catalog_manifest_sha256=digest(catalog / "CATALOG_MANIFEST.json"),
                        donor_selector_reads_targets=False)
        manifest["sha256"]["CUE_DONORS.npz"] = digest(output / "CUE_DONORS.npz")
        # Update only the new manifest; copied candidate/feature/label bytes stay unchanged.
        (output / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        load_cue_treatment(load_export(output), catalog)
    return dict(status="pass", surface=record["surface"], rows=len(export.rows),
                source_policy=policy,
                copied_candidate_arrays_unchanged=True, complete_sources=sources is not None,
                wrong_cue_bundle=catalog is not None, paper_metric_reproduction=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--prepared-data", type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.output, args.catalog, args.prepared_data)))


if __name__ == "__main__":
    main()
