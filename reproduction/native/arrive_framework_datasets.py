#!/usr/bin/env python3
"""Dataset registry for the ARRIVE final-framework Yeti31 chain.

Single source of per-dataset paths for the YetiRank carrier, its feature
exporter and the frozen Yeti31 Test evaluator, so no runner carries
per-dataset path copies again (lesson: the Onion-only paths in
run_arrive_recent_ranker.py blocked the corrected-Yambda matrix).

Path truth comes from the executed v2 scripts and manifests:
- data/prepared/... and index/muse-count bundles mirror
  the recorded per-dataset backbone runs;
- the Yambda content arrays are reused from content_v1 because the v2
  protocol repair changed time semantics, not the catalog;
- `parent_root` is the Yeti31 parent EXPORT root (features/candidate_ids/
  labels/metadata with rrf_indices, produced by the full-RCCE-control
  exporter). It is registered for Onion and deliberately REQUIRED-BUT-
  UNREGISTERED for Yambda until the corrected-Yambda SASRec-chain exports
  are produced and inventoried (EXPERIMENT_PLAN.md, gap 2).

Nothing here executes experiments; verify() is fail-closed and prints the
precise missing path so a single cluster run completes the inventory.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

DATASETS = ("music4all_onion", "yambda500m")
EXPORT_SCHEMA = "arrive.framework.export.v1"
PREPARED_SCHEMAS = {"music4all_onion": "genrec.music4all_onion.prepared.v1", "yambda500m": "genrec.yambda500m.prepared.v2"}
TEST_SCHEMAS = {"music4all_onion": "genrec.music4all_onion.sealed_test.v1", "yambda500m": "genrec.yambda500m.sealed_test.v2"}
EXPORT_FILES = ("features.npy", "candidate_ids.npy", "labels.npy", "metadata.npz", "REQUEST_CONTEXT.npz")
SOURCE_POLICIES = ("full", "joint-removal")
ONION_TEMPORAL_CV_ROWS = 20000
ONION_TEMPORAL_CV_RULE = "last chronological >=30d Train return per user with a prior Train return; deterministic hash sample"


def source_policy(record):
    policy = record.get("source_policy", "full")
    if policy not in SOURCE_POLICIES:
        raise ValueError("unknown_source_policy")
    return policy


def validate_train_cv_exclusion(project, dataset, index, im, data, rows):
    """Keep the recorded Onion temporal-row holdout distinct from K-fold OOF.

    The caller checks carrier/CV/index hashes first. Only the registered Onion
    index can use the recorded temporal rule; custom indexes require user
    isolation. Fitting/calibration user isolation is checked by the ranker.
    """
    import numpy as np
    from run_music4all_candidate_support_a0 import temporal_holdout_rows
    all_rows = data.row_ids(0)
    if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
            or len(np.unique(rows)) != len(rows) or not len(rows)
            or not np.isin(rows, all_rows).all()):
        raise ValueError("framework_fold_rows")
    cfg = im.get("configuration", {})
    temporal = (dataset == "music4all_onion"
        and Path(index).resolve() == Path(bundle(project, dataset)["index"]).resolve()
        and cfg.get("cv_rule") == ONION_TEMPORAL_CV_RULE)
    if temporal:
        if (cfg.get("cv_limit") != ONION_TEMPORAL_CV_ROWS or cfg.get("seed") != 2024
                or cfg.get("horizon") != 64 or len(rows) != ONION_TEMPORAL_CV_ROWS):
            raise ValueError("framework_temporal_cv_configuration")
        expected = temporal_holdout_rows(data.sessions, data.events,
            horizon=64, limit=ONION_TEMPORAL_CV_ROWS, seed=2024)
        if not np.array_equal(rows, expected):
            raise ValueError("framework_temporal_cv_rows")
        return "temporal_row_exclusion"
    if np.intersect1d(data.sessions[rows, 0],
                      data.sessions[np.setdiff1d(all_rows, rows), 0]).size:
        raise ValueError("framework_fold_user_overlap")
    return "user_disjoint_fold"


@lru_cache(maxsize=512)
def _file_hash_version(path, signature):
    from run_music4all_candidate_support_a1 import digest
    result = digest(Path(path))
    s = Path(path).stat()
    if signature != (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns):
        raise ValueError("file_changed_during_hash")
    return result


def file_hash(path):
    """Reuse checks within this process only; any observed file change invalidates."""
    path=Path(path).resolve()
    s=path.stat()
    return _file_hash_version(str(path),(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))


def admit_parallel_fit(measured, binding_hash, memory_max, memory_stat, source_input):
    """Full-input native fit evidence plus fresh non-reclaimable cgroup memory."""
    if (measured.get("status") != "pass" or measured.get("runtime_binding_sha256") != binding_hash
            or measured.get("schema") != "arrive.parallel_capacity.v1"
            or measured.get("feature_count") != 31 or measured.get("threads") != 4
            or measured.get("measured_trees",0) < 100
            or source_input.get("full_train") is not True
            or source_input.get("eligible_train_rows") != 293201
            or measured.get("fit_candidates") != source_input.get("fit_candidates")
            or measured.get("coverage_sha256") != source_input.get("coverage_sha256")):
        raise ValueError("parallel_capacity_evidence_missing_or_stale")
    peak=int(measured["full_fit_native_peak_rss_bytes"])
    if peak <= 0 or memory_max <= 0:
        raise ValueError("parallel_capacity_invalid_memory")
    # Clean file cache is reclaimable. Include dirty/writeback/unevictable pages.
    used=sum(memory_stat[k] for k in ("anon","kernel","file_dirty","file_writeback","unevictable"))
    required=used+peak+peak//5+2*1024**3
    if required > memory_max:
        raise ValueError(f"parallel_capacity_exceeded:{required}>{memory_max}")
    return required


def feature_schema():
    from run_arrive_attribution_controls import FEATURE_NAMES
    return list(FEATURE_NAMES) + ["recent_unique_fraction", "log_first_track_transition_count", "first_track_transition_concentration"]


def code_bindings():
    from run_music4all_candidate_support_a1 import digest
    root = Path(__file__).resolve().parent
    names = ("arrive_framework_datasets.py", "run_arrive_full_rcce_control.py",
             "run_arrive_recent_ranker.py", "evaluate_arrive_yeti31_fusion_test.py",
             "run_arrive_attribution_controls.py", "run_music4all_artist_component_b.py",
             "run_music4all_candidate_support_a0.py", "run_music4all_candidate_support_a1.py",
             "evaluate_music4all_arrive_test.py", "evaluate_arrive_frozen_test.py",
             "lfm_timeconcat_control.py", "run_lfm_timeconcat_control.py", "music_sequence_core.py", "run_music4all_baselines.py",
             "run_music4all_component_c2_conversion_oof.py", "arrive_train_fold_coverage.py")
    return {name: digest(root / name) for name in names}


def hashes(paths):
    return {name: file_hash(path) for name, path in paths.items()}


def check_export(root: Path, dataset: str, seed: int, surface: str, *, expected_code=None):
    """Validate the producer contract before returning memory-mapped arrays."""
    import json
    import numpy as np
    m = json.loads((root / "EXPORT_MANIFEST.json").read_text())
    source_policy(m)
    expected = {"schema": EXPORT_SCHEMA, "status": "pass", "dataset": dataset,
                "seed": seed, "surface": surface, "test_accessed": surface == "test",
                "feature_schema": feature_schema(), "code_sha256": code_bindings() if expected_code is None else expected_code}
    if any(m.get(k) != v for k, v in expected.items()):
        raise ValueError("framework_export_identity_or_code")
    if set(m.get("output_sha256", {})) != set(EXPORT_FILES):
        raise ValueError("framework_export_file_inventory")
    for name in EXPORT_FILES:
        if file_hash(root / name) != m["output_sha256"][name]:
            raise ValueError(f"framework_export_hash:{name}")
    with np.load(root / "metadata.npz", allow_pickle=False) as z:
        meta = {k: z[k] for k in z.files}
    with np.load(root / "REQUEST_CONTEXT.npz", allow_pickle=False) as z:
        if not np.array_equal(z["row_ids"], meta["row_ids"]) or not np.array_equal(z["users"], meta["users"]):
            raise ValueError("framework_context_row_identity")
        context = z["relation"].copy()
    x, ids, y = [np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                 for name in ("features", "candidate_ids", "labels")]
    n = len(meta["row_ids"])
    offsets = meta["offsets"]
    if (n != m["rows"] or len(np.unique(meta["row_ids"])) != n or
            offsets.shape != (n + 1,) or offsets[0] != 0 or
            np.any(np.diff(offsets) < 200) or offsets[-1] != len(ids) or
            ids.ndim != 1 or y.shape != ids.shape or x.shape != (len(ids), 28) or
            context.shape != (n, 18) or meta["rrf_indices"].shape != (n, 200) or
            len(meta["users"]) != n or len(meta["target_counts"]) != n or
            meta["targets"].shape[0] != n or meta["bucket_ids"].shape != (n,) or
            not all(np.issubdtype(a.dtype,np.integer) for a in (offsets,ids,meta["row_ids"],meta["users"],meta["rrf_indices"],meta["target_counts"])) or
            not np.isfinite(context).all()):
        raise ValueError("framework_export_shape")
    for lo in range(0,len(x),65536):
        if not np.isfinite(x[lo:lo+65536]).all():
            raise ValueError("framework_export_shape")
    for at, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        candidates = ids[lo:hi]
        pos = meta["rrf_indices"][at]
        count = int(meta["target_counts"][at])
        truth = meta["targets"][at, :count]
        if (len(np.unique(candidates)) != len(candidates) or np.any(candidates < 3) or
                len(np.unique(pos)) != 200 or np.any(pos < 0) or np.any(pos >= hi-lo) or
                not 0 < count <= meta["targets"].shape[1] or len(np.unique(truth)) != count or
                not np.array_equal(y[lo:hi], np.isin(candidates, truth).astype(y.dtype))):
            raise ValueError("framework_export_candidates_or_targets")
    return x, ids, y, meta, m, context


def _bundle(project: Path, dataset: str) -> dict:
    if dataset == "music4all_onion":
        return {
            "data_dir": "data/music4all_onion/prepared_v1",
            "event_times": "data/music4all_onion/baseline_features_v1/event_times.npy",
            "muse_counts": "data/music4all_onion/main_table_repair_v1/muse_transition_counts.npz",
            "index": "data/music4all_onion/a0_candidate_index_v1",
            "a0_result": "refine-logs/music4all_onion/candidate_support_a0/A0_RESULT.json",
            "content_dir": "data/music4all_onion/component_b2_attainability_v1",
            "content_audio": "data/music4all_onion/component_b2_attainability_v1/musicnn_normalized.npy",
            "content_genre": "data/music4all_onion/component_b2_attainability_v1/genre_normalized.npz",
            "tags": "data/music4all_onion/raw/id_tags.csv",
            "information": "data/music4all_onion/raw/id_information.csv",
            "artist_mapping": None,
            "metadata_kind": "onion",
            "prepared_label": "prepared_v1",
            # Yeti31 parent export root (features/candidate_ids/labels/metadata).
            "parent_root": "runs/arrive_full_rcce_control_20260924/onion",
            "parent_root_registered": True,
            "test_feature_note": "Onion sealed Test was opened 2026-09-19; evaluator reads frozen test exports only.",
        }
    if dataset == "yambda500m":
        return {
            "data_dir": "prepared_data/yambda500m_prepared_v2_seconds",
            "event_times": "prepared_data/yambda500m_prepared_v2_seconds/event_times.npy",
            "muse_counts": "data/arrive_yambda_500m/main_table_repair_v2_seconds/muse_transition_counts.npz",
            "index": "data/arrive_yambda_500m/a0_candidate_index_v2_seconds",
            "a0_result": "refine-logs/arrive_yambda_500m/candidate_support_a0/A0_RESULT.json",
            "content_dir": "data/arrive_yambda_500m/content_v1",
            "content_audio": "data/arrive_yambda_500m/content_v1/audio_normalized.npy",
            "content_genre": "data/arrive_yambda_500m/content_v1/artist_normalized.npz",
            "tags": None,
            "information": None,
            "artist_mapping": "datasets/Yamba_50M/artist_mapping.npz",
            "metadata_kind": "yambda",
            "prepared_label": "prepared_v2_seconds",
            # No corrected-Yambda Yeti31 export root exists yet (EXPERIMENT_PLAN
            # gap 2). Callers MUST pass --parent-root explicitly; verify()
            # rejects the None default instead of silently reading v1 paths.
            "parent_root": None,
            "parent_root_registered": False,
            "test_feature_note": "Test features are consumed from the executed chain's test_features build; hash-bound in TEST bindings.",
        }
    raise ValueError(f"unknown_dataset:{dataset}")


def bundle(project: Path, dataset: str) -> dict:
    """Resolve registry paths to absolute Paths under `project`."""
    spec = _bundle(Path(project), dataset)
    out = {"dataset": dataset, "project": str(project)}
    for key, value in spec.items():
        if key in {"data_dir", "event_times", "muse_counts", "index", "a0_result", "content_dir", "content_audio", "content_genre", "tags", "information", "artist_mapping", "parent_root"} and value:
            out[key] = str(Path(project) / value) if not Path(value).is_absolute() else value
        else:
            out[key] = value
    out["paths_absolute"] = True
    return out


def verify(project: Path, dataset: str, *, require_parent: bool = True) -> list[str]:
    """Fail-closed existence check; returns the list of missing paths.

    parent_root is special: yambda has no registered Yeti31 export root yet,
    so verify reports `parent_root_unregistered` instead of pretending the
    chain is runnable.
    """
    b = bundle(Path(project), dataset)
    missing = []
    required = ["data_dir", "event_times", "muse_counts", "index", "a0_result",
                "content_audio", "content_genre"]
    if dataset == "yambda500m":
        required.append("artist_mapping")
    for key in required:
        value = b[key]
        if value is None or not Path(value).exists():
            missing.append(f"{dataset}:{key}={value}")
    if dataset == "music4all_onion":
        for key in ("tags", "information"):
            if not Path(b[key]).exists():
                missing.append(f"{dataset}:{key}={b[key]}")
    if require_parent and not b["parent_root_registered"]:
        missing.append(f"{dataset}:parent_root_unregistered (pass --parent-root explicitly)")
    elif require_parent and not Path(b["parent_root"]).exists():
        missing.append(f"{dataset}:parent_root={b['parent_root']}")
    return missing


def metadata_args(ds: dict, data_dir: Path):
    from types import SimpleNamespace
    return SimpleNamespace(data_dir=data_dir,
        tags=Path(ds["tags"]) if ds["tags"] else None,
        information=Path(ds["information"]) if ds["information"] else None,
        artist_mapping=Path(ds["artist_mapping"]) if ds["artist_mapping"] else None,
        audio_array=Path(ds["content_audio"]) if ds["artist_mapping"] else None)


def feature_input_paths(ds: dict, index: Path, suffix: str) -> dict:
    paths = {k: Path(ds[k]) for k in ("content_audio", "content_genre", "tags", "information", "artist_mapping") if ds[k]}
    paths.update({name: index / name for name in
                  (f"x_{suffix}.npz", f"h_{suffix}.npz", f"{suffix}_target_frequency.npy")})
    paths["track_identity"] = Path(ds["data_dir"]) / "track_identity.tsv"
    paths["train_frequencies"] = Path(ds["data_dir"]) / "train_frequencies.npy"
    return paths


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else ""
    project = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    if dataset not in DATASETS:
        print(json_dumps({"datasets": list(DATASETS)}))
        raise SystemExit(2)
    b = bundle(project, dataset)
    missing = verify(project, dataset)
    print(json_dumps({"dataset": dataset, "missing_count": len(missing), "missing": missing, "bundle": b}))
    raise SystemExit(0 if not missing else 3)


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=1)


if __name__ == "__main__":
    main()
