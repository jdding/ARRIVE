#!/usr/bin/env python3
"""Evaluate a frozen Yeti31 model using the shared framework export.

Freeze verifies the approved Test manifest digest, serving inputs, feature order
and code before opening Test arrays. The Test exporter uses that same record;
it does not select a model or alter candidate budgets.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from arrive_framework_datasets import (bundle, check_export, feature_schema,
    code_bindings, feature_input_paths, hashes, TEST_SCHEMAS)
from evaluate_music4all_arrive_test import atomic_json, atomic_npz, digest, score_rows
from evaluate_arrive_frozen_test import validate_test
from lfm_timeconcat_control import LFMTimeConcatData


def validate_frozen_ranker(args):
    root = args.model_dir
    paths = {key: root / name for key, name in (
        ("model_sha256", "yeti_ndcg.cbm"),
        ("prediction_sha256", "validation_predictions.npz"),
        ("prediction_receipt_sha256", "validation_predictions.npz.json"))}
    manifest = json.loads((root / "training_manifest.json").read_text())
    legacy = "code_sha256" not in manifest
    required = {"schema": "arrive.recent_ranker.v1", "status": "pass",
        "dataset": args.dataset, "seed": args.seed, "surface": "validation",
        "test_accessed": False, "feature_mode": "statistics", "feature_count": 31,
        "feature_schema": feature_schema(), "code_sha256": code_bindings()}
    if legacy:
        from run_arrive_recent_ranker import LOSS
        if args.dataset != "music4all_onion" or not getattr(args, "validation_audit", None):
            raise ValueError("legacy20k_requires_validation_audit")
        required = {k:v for k,v in required.items() if k not in ("dataset","surface","feature_schema","code_sha256")}
        required.update(loss_function=LOSS, canary_rows=0)
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError("frozen_ranker_manifest_mismatch")
    for key, path in paths.items():
        if legacy and key == "prediction_receipt_sha256":
            continue  # Original manifest binds predictions directly, not a receipt file.
        if not path.is_file() or digest(path) != manifest.get(key):
            raise ValueError(f"frozen_ranker_hash:{key}")
    if not isinstance(manifest.get("selected_trees"), int) or manifest["selected_trees"] <= 0:
        raise ValueError("frozen_ranker_selected_trees")
    from catboost import CatBoostRanker
    model=CatBoostRanker()
    model.load_model(str(paths["model_sha256"]))
    if manifest["selected_trees"] > model.tree_count_ or len(model.feature_names_) != 31:
        raise ValueError("frozen_model_tree_or_feature_count")
    if legacy:
        audit = json.loads(args.validation_audit.read_text())
        expected = dict(status="pass", dataset=args.dataset, seed=args.seed, test_accessed=False,
            train_cv_rows=20000, rows=manifest["validation_rows"],
            training_manifest_sha256=digest(root / "training_manifest.json"),
            model_sha256=manifest["model_sha256"], prediction_sha256=manifest["prediction_sha256"],
            train_export_sha256=manifest["train_export_sha256"],
            validation_export_sha256=manifest["validation_export_sha256"],
            model_replay_exact=True, labels_and_row_identity_exact=True,
            framework_parity_exact=True, code_sha256=code_bindings())
        expected["retrieval_contract"] = "legacy20k_64_128_512"
        if any(audit.get(k) != v for k,v in expected.items()) or not audit.get("serving_inputs"):
            raise ValueError("legacy20k_validation_audit_mismatch")
        # A derived in-memory view only; never rewrite the historical training record.
        return {**manifest, "serving_inputs": audit["serving_inputs"]}, audit
    for key in ("train_export_sha256", "validation_export_sha256", "coverage_sha256", "serving_inputs"):
        if not manifest.get(key):
            raise ValueError(f"frozen_ranker_binding_missing:{key}")
    receipt = json.loads(paths["prediction_receipt_sha256"].read_text())
    if receipt.get("prediction_sha256") != manifest["prediction_sha256"]:
        raise ValueError("frozen_ranker_prediction_receipt_mismatch")
    return manifest, receipt


def serving_inputs(args):
    ds = bundle(args.project, args.dataset)
    paths = feature_input_paths(ds, Path(ds["index"]), "full")
    paths.update(base_checkpoint=args.base_checkpoint, base_manifest=args.base_manifest)
    return hashes(paths)


def validate_freeze_record(path, dataset, seed, model_sha256, training_manifest_sha256):
    record = json.loads(path.read_text())
    expected = {"schema": "arrive.final_framework.freeze.v2", "dataset": dataset,
        "seed": seed, "status": "frozen", "validation_audit_passed": True,
        "model_sha256": model_sha256, "training_manifest_sha256": training_manifest_sha256,
        "feature_schema": feature_schema(), "code_sha256": code_bindings()}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("freeze_record_mismatch")
    return record


def validate_before_test(args):
    manifest, _ = validate_frozen_ranker(args)
    freeze = validate_freeze_record(args.freeze_record, args.dataset, args.seed,
        manifest["model_sha256"], digest(args.model_dir / "training_manifest.json"))
    if "code_sha256" not in manifest and freeze.get("retrieval_contract") != "legacy20k_64_128_512":
        raise ValueError("frozen_retrieval_contract")
    if getattr(args, "validation_audit", None) and digest(args.validation_audit) != freeze["validation_audit_sha256"]:
        raise ValueError("frozen_validation_audit_changed")
    actual = serving_inputs(args)
    if actual != manifest["serving_inputs"] or actual != freeze.get("serving_inputs"):
        raise ValueError("frozen_serving_input_mismatch")
    if digest(args.test_manifest) != freeze.get("test_manifest_sha256"):
        raise ValueError("frozen_test_manifest_mismatch")
    tm = json.loads(args.test_manifest.read_text())
    if tm.get("schema") != TEST_SCHEMAS[args.dataset]:
        raise ValueError("frozen_test_dataset_version")
    required_files={"events.npy","sessions.npy","train_frequencies.npy","metric_labels.npz","track_identity.tsv"}
    if not required_files <= set(tm.get("artifacts",{})):
        raise ValueError("frozen_test_artifact_inventory")
    if Path(args.test_data).resolve() != Path(freeze["test_data"]).resolve():
        raise ValueError("frozen_test_data_path")
    return manifest, freeze


def write_freeze(args):
    """Bind an externally approved Validation audit; do not evaluate Test here."""
    if args.freeze_record.exists(): raise FileExistsError(args.freeze_record)
    manifest,validated_audit=validate_frozen_ranker(args)
    audit=json.loads(args.validation_audit.read_text())
    model_hash=digest(args.model_dir/"training_manifest.json")
    if (audit.get("status")!="pass" or audit.get("dataset")!=args.dataset or audit.get("seed")!=args.seed
            or audit.get("test_accessed") is not False or audit.get("training_manifest_sha256")!=model_hash
            or audit.get("prediction_sha256")!=manifest["prediction_sha256"]):
        raise ValueError("validation_audit_binding")
    actual=serving_inputs(args)
    if actual!=manifest["serving_inputs"]: raise ValueError("freeze_serving_input_mismatch")
    tm=json.loads(args.test_manifest.read_text())
    if tm.get("schema")!=TEST_SCHEMAS[args.dataset]: raise ValueError("freeze_test_schema")
    atomic_json(args.freeze_record,{"schema":"arrive.final_framework.freeze.v2","status":"frozen",
        "dataset":args.dataset,"seed":args.seed,"validation_audit_passed":True,
        "validation_audit_sha256":digest(args.validation_audit),"model_sha256":manifest["model_sha256"],
        "training_manifest_sha256":model_hash,"feature_schema":feature_schema(),"code_sha256":code_bindings(),
        "serving_inputs":actual,"test_manifest_sha256":digest(args.test_manifest),
        "retrieval_contract":validated_audit.get("retrieval_contract", "direct"),
        "test_data":str(args.test_data.resolve()),"test_arrays_accessed":False})


def validate_cohort(data, labels, meta, expected_counts=None, *, split=2):
    from lfm_timeconcat_control import gap_bucket
    rows = data.row_ids(split, returns_only=True)
    if not (np.array_equal(rows, labels["row_ids"]) and np.array_equal(rows, meta["row_ids"])):
        raise ValueError("yeti31_test_labels_row_identity_mismatch")
    if not np.array_equal(meta["users"], data.sessions[rows, 0]):
        raise ValueError("yeti31_test_users_mismatch")
    scopes = ("30-90d", "90-180d", "180-365d", ">365d")
    buckets = np.asarray([gap_bucket(data.gap_days(int(r))) for r in rows])
    counts = [int((buckets == s).sum()) for s in scopes]
    if expected_counts is not None and (len(rows) != expected_counts["rows"] or counts != expected_counts["gap_buckets"]):
        raise ValueError("yeti31_test_cohort_counts")
    offsets = labels["offsets"]
    if offsets.shape != (len(rows)+1,) or offsets[0] != 0 or offsets[-1] != len(labels["values"]) or np.any(np.diff(offsets) <= 0):
        raise ValueError("yeti31_test_label_offsets")
    for at in range(len(rows)):
        truth = labels["values"][offsets[at]:offsets[at+1]]
        n = int(meta["target_counts"][at])
        if not np.array_equal(truth, meta["targets"][at, :n]) or len(truth) != n:
            raise ValueError("yeti31_test_targets_mismatch")
    if not np.array_equal(meta["bucket_ids"], np.asarray([scopes.index(b) for b in buckets])):
        raise ValueError("yeti31_test_buckets_mismatch")


def main():
    from catboost import CatBoostRanker
    from run_arrive_recent_ranker import top10
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, choices=(2024,2025,2026), required=True)
    parser.add_argument("--dataset", choices=tuple(TEST_SCHEMAS), required=True)
    for name in ("test-data", "test-manifest", "model-dir", "features-root",
                 "freeze-record", "base-checkpoint", "base-manifest", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--validation-audit", type=Path)
    args = parser.parse_args()
    if args.freeze_only:
        if args.validation_audit is None: parser.error("--freeze-only requires --validation-audit")
        write_freeze(args)
        return
    if args.output.exists():
        raise FileExistsError("sealed_yeti31_test_output_exists")
    manifest, freeze = validate_before_test(args)
    # Validate export and all inexpensive model bindings before opening Test.
    x, ids, _, meta, em, context = check_export(args.features_root, args.dataset, args.seed, "test")
    if (em.get("freeze_sha256") != digest(args.freeze_record) or
            em.get("retrieval_contract", "direct") != freeze.get("retrieval_contract", "direct") or
            em.get("serving_inputs") != freeze["serving_inputs"] or
            em.get("test_manifest_sha256") != freeze["test_manifest_sha256"]):
        raise ValueError("frozen_test_export_binding")
    model = CatBoostRanker()
    model.load_model(str(args.model_dir / "yeti_ndcg.cbm"))
    if manifest["selected_trees"] > model.tree_count_ or len(model.feature_names_) != 31:
        raise ValueError("frozen_model_tree_or_feature_count")
    tm = validate_test(args)
    data = LFMTimeConcatData(args.test_data, max_context=129, horizon=64)
    with np.load(args.test_data / "metric_labels.npz", allow_pickle=False) as z:
        labels = {k:z[k] for k in z.files}
    validate_cohort(data, labels, meta, tm["counts"])
    predictions = top10(model, x, ids, meta, np.arange(len(meta["row_ids"])),
                        manifest["selected_trees"], context, "statistics", threads=args.threads)
    metrics, arrays = score_rows(predictions, data, labels, meta["row_ids"])
    # Enforce strict JSON before creating the final output directory.
    json.dumps(metrics, allow_nan=False)
    args.output.mkdir(parents=True, exist_ok=False)
    pred = args.output / "test_predictions.npz"
    atomic_npz(pred, row_ids=meta["row_ids"], top10=predictions, **arrays)
    atomic_json(args.output / "TEST_RESULT.json", {
        "schema":"genrec.arrive.yeti31.sealed_test_result.v1", "status":"pass",
        "dataset":args.dataset, "seed":args.seed, "rows":len(predictions),
        "metrics":metrics, "selected_trees":manifest["selected_trees"],
        "model_sha256":manifest["model_sha256"],
        "training_manifest_sha256":digest(args.model_dir / "training_manifest.json"),
        "freeze_record_sha256":digest(args.freeze_record),
        "test_manifest_sha256":digest(args.test_manifest),
        "test_export_manifest_sha256":digest(args.features_root / "EXPORT_MANIFEST.json"),
        "predictions_sha256":digest(pred), "test_accessed":True,
        "selection_or_tuning_performed":False, "new_untouched_test_claim":False})
    print(json.dumps({"event":"sealed_yeti31_test_complete", "rows":len(predictions)}))


if __name__ == "__main__":
    main()
