#!/usr/bin/env python3
"""Verify frozen Yeti Validation predictions using the existing evaluator."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from arrive_framework_datasets import bundle, check_export, code_bindings
from evaluate_arrive_yeti31_fusion_test import validate_cohort, validate_frozen_ranker, TRAIN_CV_ROWS
from lfm_timeconcat_control import LFMTimeConcatData
from run_arrive_recent_ranker import top10, load_request_context, check_train_cv_rows, LOSS
from run_music4all_candidate_support_a1 import atomic_json, digest
from run_music4all_component_c2_conversion_oof import row_metrics, user_macro


def verify_predictions(pred, meta, ranked):
    for name in ("row_ids", "users", "bucket_ids", "target_counts", "targets"):
        if not np.array_equal(pred[name], meta[name]):
            raise ValueError(f"validation_identity:{name}")
    if not np.array_equal(pred["top10"], ranked):
        raise ValueError("validation_model_replay")
    recall, ndcg = row_metrics(ranked, meta["targets"], meta["target_counts"])
    for name, actual in (("recall", recall), ("ndcg", ndcg)):
        if actual.shape != pred[name].shape or not np.allclose(actual, pred[name], rtol=0, atol=3e-16):
            raise ValueError(f"validation_metric:{name}")
    mask = np.ones(len(ranked), bool)
    return {"recall": user_macro(recall, meta["users"], mask),
            "ndcg": user_macro(ndcg, meta["users"], mask)}


def verify_served_features(old, new):
    """Compare exactly the served200 candidates,28 candidate+3 request columns."""
    x, ids, meta, context = old
    nx, nids, nm, nc = new
    for key in ("row_ids", "users", "bucket_ids", "target_counts", "targets"):
        if not np.array_equal(meta[key], nm[key]):
            raise ValueError(f"framework_parity:{key}")
    if not np.array_equal(context[:,14:17], nc[:,14:17]):
        raise ValueError("framework_parity:statistics3")
    for lo in range(0, len(meta["row_ids"]), 256):
        rows = np.arange(lo, min(lo+256, len(meta["row_ids"])))
        a = (meta["offsets"][rows,None] + meta["rrf_indices"][rows]).ravel()
        b = (nm["offsets"][rows,None] + nm["rrf_indices"][rows]).ravel()
        if not np.array_equal(ids[a], nids[b]) or not np.array_equal(x[a], nx[b]):
            raise ValueError(f"framework_parity:candidates_or_features:{lo}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--dataset", choices=("music4all_onion", "yambda500m"), required=True)
    p.add_argument("--seed", type=int, choices=(2024, 2025, 2026), required=True)
    p.add_argument("--parent-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--request-context-root", type=Path,
                   help="Original20k export lineage; preserves the legacy model manifest")
    p.add_argument("--framework-validation-root", type=Path)
    p.add_argument("--source-code-bindings", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    parent = a.parent_root / f"s{a.seed}"
    if a.request_context_root:
        if a.dataset != "music4all_onion":
            raise ValueError("legacy20k_onion_only")
        from run_arrive_full_rcce_control import load_export
        m = json.loads((a.model_dir / "training_manifest.json").read_text())
        expected = dict(schema="arrive.recent_ranker.v1", status="pass", seed=a.seed,
                        feature_mode="statistics", feature_count=31, test_accessed=False,
                        loss_function=LOSS, canary_rows=0, validation_rows=12419)
        if any(m.get(k) != v for k, v in expected.items()):
            raise ValueError("legacy20k_ranker_identity")
        for key, path in (("model_sha256", a.model_dir / "yeti_ndcg.cbm"),
                          ("prediction_sha256", a.model_dir / "validation_predictions.npz"),
                          ("train_export_sha256", parent / "train_cv/EXPORT_MANIFEST.json"),
                          ("validation_export_sha256", parent / "validation/EXPORT_MANIFEST.json"),
                          ("reference_manifest_sha256", parent / "training_manifest.json")):
            if digest(path) != m[key]:
                raise ValueError(f"legacy20k_binding:{key}")
        _, _, _, train_meta, _ = load_export(parent / "train_cv", a.seed, "train_cv")
        check_train_cv_rows(train_meta, 20000)
        _, train_receipt = load_request_context(a.request_context_root, parent, train_meta,
                                               a.seed, "train_cv", "statistics")
        x, ids, _, meta, _ = load_export(parent / "validation", a.seed, "validation")
        context, val_receipt = load_request_context(a.request_context_root, parent, meta,
                                                   a.seed, "validation", "statistics")
        if train_receipt != m["train_request_context"] or val_receipt != m["validation_request_context"]:
            raise ValueError("legacy20k_context_binding")
    else:
        m, _ = validate_frozen_ranker(SimpleNamespace(
            model_dir=a.model_dir, dataset=a.dataset, seed=a.seed))
        from arrive_train_fold_coverage import load_train_sessions
        prepared = Path(bundle(a.project, a.dataset)["data_dir"])
        native_bounded = m.get("training_policy") == "bounded_native_framework_v1"
        expected_train_rows = TRAIN_CV_ROWS[a.dataset] if native_bounded else len(load_train_sessions(prepared)[0])
        expected_validation_rows = len(LFMTimeConcatData(prepared).row_ids(1, returns_only=True))
        if native_bounded:
            _, _, _, train_meta, train_manifest, _ = check_export(
                parent / "train_cv", a.dataset, a.seed, "train_cv")
            check_train_cv_rows(train_meta, expected_train_rows)
            index = Path(train_manifest["index_dir"])
            cv = np.load(index / "cv_rows.npy", allow_pickle=False)
            data = LFMTimeConcatData(prepared)
            if (m["train_cv_contract_sha256"] != digest(index / "index_manifest.json")
                    or train_manifest["index_manifest_sha256"] != m["train_cv_contract_sha256"]
                    or not np.array_equal(train_meta["row_ids"], cv)
                    or not np.array_equal(train_meta["users"], data.sessions[cv, 0])):
                raise ValueError("validation_native_train_contract")
            coverage = {"code_sha256": code_bindings()}
        else:
            coverage = json.loads((parent / "train_cv/COVERAGE.json").read_text())
        if (m["eligible_train_rows"] != expected_train_rows or m["validation_rows"] != expected_validation_rows
            or (not native_bounded and digest(parent / "train_cv/COVERAGE.json") != m["coverage_sha256"])
            or digest(parent / "validation/EXPORT_MANIFEST.json") != m["validation_export_sha256"]
            or digest(parent / "train_cv/EXPORT_MANIFEST.json") != m["train_export_sha256"]):
            raise ValueError("validation_full_train_binding")
        x, ids, _, meta, _, context = check_export(parent / "validation", a.dataset,
            a.seed, "validation", expected_code=coverage.get("source_code_sha256", coverage["code_sha256"]))
    prepared = Path(bundle(a.project, a.dataset)["data_dir"])
    with np.load(prepared / "metric_labels.npz", allow_pickle=False) as z:
        labels = {k: z[k] for k in z.files}
    validate_cohort(LFMTimeConcatData(prepared), labels, meta, split=1)
    from catboost import CatBoostRanker
    model = CatBoostRanker()
    model.load_model(str(a.model_dir / "yeti_ndcg.cbm"))
    if len(model.feature_names_) != 31 or not 0 < m["selected_trees"] <= model.tree_count_:
        raise ValueError("validation_model_shape")
    ranked = top10(model, x, ids, meta, np.arange(len(meta["row_ids"])),
        m["selected_trees"], context, "statistics", threads=4)
    with np.load(a.model_dir / "validation_predictions.npz", allow_pickle=False) as z:
        metrics = verify_predictions(z, meta, ranked)
    if any(not np.isclose(metrics[k], m["overall_" + k], rtol=0, atol=3e-16) for k in metrics):
        raise ValueError("validation_macro_metric")
    extra = {}
    if a.framework_validation_root:
        if not a.source_code_bindings or not a.request_context_root:
            raise ValueError("framework_parity_requires_source_bindings_and_legacy_context")
        binding = json.loads(a.source_code_bindings.read_text())
        nx, nids, _, nm, em, nc = check_export(a.framework_validation_root,
            "music4all_onion", a.seed, "validation", expected_code=binding["code_sha256"])
        verify_served_features((x,ids,meta,context), (nx,nids,nm,nc))
        if not np.array_equal(ranked, top10(model,nx,nids,nm,np.arange(len(nm["row_ids"])),
                                           m["selected_trees"],nc,"statistics",threads=4)):
            raise ValueError("framework_parity:model_replay")
        extra = dict(framework_parity_exact=True, serving_inputs=em["serving_inputs"],
                     retrieval_contract=em.get("retrieval_contract", "direct"),
                     code_sha256=code_bindings(),
                     framework_export_sha256=digest(a.framework_validation_root / "EXPORT_MANIFEST.json"))
    result = dict(status="pass", dataset=a.dataset, seed=a.seed,
        rows=len(ranked), train_cv_rows=20000 if a.request_context_root else m["eligible_train_rows"], metrics=metrics,
        training_manifest_sha256=digest(a.model_dir / "training_manifest.json"),
        prediction_sha256=m["prediction_sha256"], test_accessed=False,
        model_sha256=m["model_sha256"], model_replay_exact=True, labels_and_row_identity_exact=True,
        train_export_sha256=m["train_export_sha256"], validation_export_sha256=m["validation_export_sha256"],
        evaluation_code_sha256=digest(Path(__file__)), **extra)
    atomic_json(a.output, result)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
