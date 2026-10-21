#!/usr/bin/env python3
"""Aggregate a frozen backbone-backed joint ARRIVE Test and ablation ladder."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_music4all_arrive_test import digest
from run_music4all_candidate_support_a1 import atomic_json


SEEDS = (2024, 2025, 2026)
SCOPES = ("overall", "30-90d", "90-180d", "180-365d", ">365d")
ARMS = ("backbone", "rcce", "arrive")
ALLOWED_TEST_PRODUCERS = {
    # Both evaluators share the same inference and metric path. The latter adds
    # only the exact forward-runner compatibility audit and its binding fields.
    "22139ddd60762fbe59078113b995d4b2d092fcb109084b9a2000bb68cc4f9dae",
    "09e7abf7a5aa2ba34dd734b249a7d7287df9b59802e2b7e3cae536bffc91d086",
}

# These exact producer revisions have identical non-main function/class ASTs.
# The second adds training checkpoint/resume plumbing only. Per-seed Test
# evaluation still uses and verifies its exact training-bound forward source.
REPORTING_FORWARD_FAMILY = {
    "cff22a4e499066ef6ab07cc2bcdab97ad08d22fae48e6b2e401dc1fa6e92adfb":
        "onion_portability_support_safe_v4",
    "3c13dd65024f47f799245c5238267aee8fe434dc34e5524cca77f4aa5572841e":
        "onion_portability_support_safe_v4",
}


def mask_for(buckets: np.ndarray, scope: str) -> np.ndarray:
    return np.ones(len(buckets), dtype=bool) if scope == "overall" else buckets == scope


def user_values(values: np.ndarray, users: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grouped: dict[int, list[float]] = defaultdict(list)
    for value, user in zip(values[mask], users[mask]):
        grouped[int(user)].append(float(value))
    return np.asarray([np.mean(grouped[user]) for user in sorted(grouped)], dtype=np.float64)


def paired(delta: np.ndarray, users: np.ndarray, mask: np.ndarray, seed: int) -> dict:
    unit = user_values(delta.mean(axis=0), users, mask)
    if len(unit) == 0:
        return {
            "delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "users": 0,
        }
    rng = np.random.default_rng(seed)
    samples = np.asarray([rng.choice(unit, len(unit), replace=True).mean() for _ in range(5000)])
    return {
        "delta": float(unit.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "users": int(len(unit)),
    }


def independent_components(root: Path, project: Path, output: Path) -> None:
    """Analyze E2's independently trained heads, never cumulative readouts."""
    arms = ("backbone", "rcce_only", "rgsc_only", "full")
    audit = []
    groups = {}
    inventory = json.loads((root / "ASSET_REUSE_INVENTORY.json").read_text())
    full_inventory = {(x["dataset"], x["backbone"], x["seed"]): x
                      for x in inventory["full_reuse_cells"]}

    def require(condition, detail):
        if not condition:
            raise ValueError(detail)

    def loaded(directory, seed, backbone, variant):
        path = directory / "TEST_RESULT.json"
        result = json.loads(path.read_text())
        predictions = directory / "test_predictions.npz"
        require(result["status"] == "pass" and result["seed"] == seed
                and result["backbone"] == backbone
                and result.get("component_variant", "full") == variant
                and result["objective_version"] == "support_safe_v4"
                and result["inference_only"] is True
                and result["selection_or_tuning_performed"] is False
                and result["predictions_sha256"] == digest(predictions),
                f"independent_test_binding:{directory}")
        arrays = dict(np.load(predictions, allow_pickle=False))
        require(len(arrays["row_ids"]) == result["rows"], f"row_count:{directory}")
        audit.append({"result": str(path.relative_to(project)),
                      "result_sha256": digest(path),
                      "predictions_sha256": result["predictions_sha256"],
                      "bindings": result["bindings"], "rows": result["rows"]})
        return result, arrays

    for dataset in ("music4all_onion", "yambda500m"):
        for backbone in ("sasrec", "ssd4rec"):
            values = {arm: {m: [] for m in ("recall", "ndcg", "pool_recall")}
                      for arm in arms}
            reference = None
            for seed in SEEDS:
                if dataset == "music4all_onion":
                    full_dir = project / "refine-logs/arrive_joint_convergence_20260916/test" / backbone / f"s{seed}"
                else:
                    full_dir = project / "refine-logs/arrive_yambda_protocol_repair_20260920/sealed_test" / f"joint_{backbone}" / f"s{seed}"
                full_result, full = loaded(full_dir, seed, backbone, "full")
                require(full_result["bindings"]["joint_checkpoint_sha256"] ==
                        full_inventory[dataset, backbone, seed]["checkpoint_sha256"],
                        f"full_inventory_mismatch:{dataset}:{backbone}:{seed}")
                if reference is None:
                    reference = full
                for key in ("row_ids", "users", "gap_buckets"):
                    require(np.array_equal(reference[key], full[key]), f"seed_identity:{key}")
                train_bindings = None
                label_reference = None
                for variant in ("rcce_only", "rgsc_only"):
                    directory = root / "remote_results/ablations" / dataset / backbone / variant / f"s{seed}"
                    result, arrays = loaded(directory / "test", seed, backbone, variant)
                    training = json.loads((directory / "training_manifest.json").read_text())
                    b = result["bindings"]
                    require(training["status"] == "pass" and training["test_accessed"] is False
                            and training["component_variant"] == variant
                            and training["checkpoint_sha256"] == b["joint_checkpoint_sha256"]
                            and digest(directory / "VALIDATION_RESULT.json") == b["joint_validation_result_sha256"]
                            and training["binding"]["producer_sha256"] == b["forward_runner_sha256"],
                            f"training_binding:{directory}")
                    for key in ("test_manifest_sha256", "base_checkpoint_sha256", "base_training_manifest_sha256",
                                "feature_sha256", "audio_sha256", "genre_sha256", "architecture_contract"):
                        require(b[key] == full_result["bindings"][key], f"matched_binding:{key}:{directory}")
                    selected_binding = {k: training["binding"][k] for k in
                                        ("train_base_checkpoint_sha256", "train_feature_sha256", "validation_feature_sha256")}
                    require(train_bindings is None or train_bindings == selected_binding,
                            f"ablation_train_identity:{directory}")
                    train_bindings = selected_binding
                    cfg = training["configuration"]
                    for key, expected in {"seed": seed, "epochs": 30, "min_epochs": 15,
                                          "patience": 5, "batch_size": 48, "hidden_dim": 64,
                                          "lambda_topk": 1.0, "margin": .25}.items():
                        require(cfg[key] == expected, f"configuration:{key}:{directory}")
                    for key in ("row_ids", "users", "gap_buckets", "backbone_top10", "backbone_recall", "backbone_ndcg"):
                        require(np.array_equal(full[key], arrays[key]), f"matched_rows:{key}:{directory}")
                    labels = {k: arrays[k] for k in ("target_counts", "target_values", "target_offsets")}
                    if label_reference is not None:
                        for key in labels:
                            require(np.array_equal(labels[key], label_reference[key]), f"target_identity:{directory}")
                    label_reference = labels
                    pool = arrays["candidate_top200"]
                    require(hashlib.sha256(np.ascontiguousarray(pool).tobytes()).hexdigest()
                            == result["candidate_top200_sha256"], f"pool_hash:{directory}")
                    if variant == "rgsc_only":
                        require(result["candidate_identity"]["kind"] == "pure_backbone_exact_top200",
                                f"wrong_rgsc_only_pool:{directory}")
                        require(result["candidate_identity"]["candidate_ids_sha256"] ==
                                result["candidate_top200_sha256"], f"c0_identity:{directory}")
                        # The legacy readout re-sorts recomputed scores. C0's
                        # identity is bound above to the exact retrieval cache;
                        # it need not share that readout's internal order.
                        require(all(set(top).issubset(set(candidates)) for top, candidates
                                    in zip(full["backbone_top10"], pool)),
                                f"c0_top10_containment:{directory}")
                        audit[-1]["c0_top10_order_differences"] = int(
                            np.any(pool[:, :10] != full["backbone_top10"], axis=1).sum())
                    recomputed_pool = []
                    for at, candidates in enumerate(pool):
                        lo, hi = labels["target_offsets"][at:at + 2]
                        truth = set(map(int, labels["target_values"][lo:hi]))
                        require(len(truth) == labels["target_counts"][at] and len(truth) > 0, "target_count")
                        require(len(set(map(int, candidates))) == 200, "candidate_duplicates")
                        recomputed_pool.append(len(truth.intersection(candidates)) / len(truth))
                        for source, prefix in ((arrays, "arrive"), (full, "arrive"), (full, "backbone")):
                            hits = np.asarray([int(i) in truth for i in source[f"{prefix}_top10"][at]])
                            r = hits.sum() / len(truth)
                            discounts = 1 / np.log2(np.arange(2, 12))
                            n = np.sum(hits * discounts) / discounts[:min(10, len(truth))].sum()
                            require(np.isclose(r, source[f"{prefix}_recall"][at], atol=1e-12, rtol=0)
                                    and np.isclose(n, source[f"{prefix}_ndcg"][at], atol=1e-12, rtol=0),
                                    f"metric_recompute:{directory}:{at}:{prefix}")
                    require(np.allclose(recomputed_pool, arrays["pool_recall"], atol=1e-12, rtol=0), "pool_metric")
                    for metric in ("recall", "ndcg"):
                        values[variant][metric].append(arrays[f"arrive_{metric}"])
                    values[variant]["pool_recall"].append(arrays["pool_recall"])
                    if variant == "rgsc_only":
                        values["backbone"]["pool_recall"].append(arrays["pool_recall"])
                for arm, prefix in (("backbone", "backbone"), ("full", "arrive")):
                    for metric in ("recall", "ndcg"):
                        values[arm][metric].append(full[f"{prefix}_{metric}"])
                if "pool_recall" in full:
                    values["full"]["pool_recall"].append(full["pool_recall"])
            users, buckets = reference["users"], reference["gap_buckets"]
            summary = {}
            for arm in arms:
                values[arm] = {m: np.stack(v) for m, v in values[arm].items() if len(v) == len(SEEDS)}
                if "pool_recall" in values[arm]:
                    values[arm]["ranking_misses"] = values[arm]["pool_recall"] - values[arm]["recall"]
                summary[arm] = {}
                for scope in SCOPES:
                    mask = mask_for(buckets, scope)
                    summary[arm][scope] = {}
                    for metric, array in values[arm].items():
                        per_seed = [float(user_values(v, users, mask).mean()) for v in array] if mask.any() else []
                        summary[arm][scope][metric] = {"mean": float(np.mean(per_seed)) if per_seed else None,
                            "std": float(np.std(per_seed, ddof=1)) if per_seed else None, "seeds": per_seed}
            comparisons = {}
            for a, b in (("full", "rcce_only"), ("full", "rgsc_only"), ("full", "backbone"),
                         ("rcce_only", "backbone"), ("rgsc_only", "backbone")):
                contrasts = {}
                for si, scope in enumerate(SCOPES):
                    mask = mask_for(buckets, scope)
                    contrasts[scope] = {}
                    for mi, metric in enumerate(("recall", "ndcg")):
                        delta = values[a][metric] - values[b][metric]
                        estimate = paired(delta, users, mask, 97000 + 100 * si + mi)
                        base_mean = summary[b][scope][metric]["mean"]
                        estimate["relative_percent"] = 100 * estimate["delta"] / base_mean if base_mean else None
                        estimate["seed_deltas"] = [float(user_values(v, users, mask).mean()) for v in delta] if mask.any() else []
                        contrasts[scope][metric] = estimate
                for scope in SCOPES:
                    mask = mask_for(buckets, scope)
                    aa = values[a]["recall"].mean(axis=0)[mask]
                    bb = values[b]["recall"].mean(axis=0)[mask]
                    contrasts[scope]["episode_coverage"] = {
                        "improved": int((aa > bb).sum()), "harmed": int((aa < bb).sum()),
                        "unchanged": int((aa == bb).sum()),
                        "rescued_zero_to_positive": int(((aa > 0) & (bb == 0)).sum()),
                        "displaced_positive_to_zero": int(((aa == 0) & (bb > 0)).sum())}
                comparisons[f"{a}_vs_{b}"] = contrasts
            groups[f"{dataset}/{backbone}"] = {"rows": len(users), "users": len(np.unique(users)),
                "longest_nonempty_scope": next(s for s in reversed(SCOPES[1:]) if (buckets == s).any()),
                "summary": summary, "comparisons": comparisons,
                "full_top200_status": "available" if "pool_recall" in values["full"] else "not_exported_in_legacy_full_checkpoint_evaluation"}
    atomic_json(output, {"schema": "arrive.independent_component_analysis.v1", "status": "pass",
        "integrity_status": "accepted", "plan_sha256": digest(root / "PLAN_TO_RUN_GATE.json"),
        "seeds": list(SEEDS), "std_ddof": 1, "bootstrap": "5000 user-resamples after matched seed averaging; descriptive unadjusted 95% CI",
        "independently_trained_ablations": True, "groups": groups, "integrity": audit,
        "limitations": ["Full-model Top-200 recall is missing when not exported by the historical evaluator; never infer it from Top-10.",
                        "Components jointly change candidate sources and scoring, not pure information-matched architecture.",
                        "Test was already observed; this is preregistered supplementary attribution, not untouched confirmation."]})
    print(json.dumps({k: {a: v["summary"][a]["overall"]["ndcg"]["mean"] for a in arms}
                      for k, v in groups.items()}, indent=2))


def candidate_interaction(root: Path, output: Path) -> None:
    """Supplementary Validation attribution on identical frozen selected pools."""
    remote = json.loads((root / "REMOTE_ARTIFACT_AUDIT.json").read_text())
    assert remote["status"] == "pass"
    label_audit = json.loads((root / "FULL_LABEL_AUDIT.json").read_text())
    assert label_audit["status"] == "pass"
    label_receipts = {(x["dataset"], x["seed"], x["mode"]): x for x in label_audit["cells"]}
    receipts = {(x["dataset"], x["seed"], x["mode"]): x for x in remote["mechanism"]}
    groups, audit = {}, []
    discount = 1 / np.log2(np.arange(2, 12))
    for dataset in ("music4all_onion", "yambda500m"):
        values = {mode: {metric: [] for metric in ("recall", "ndcg", "pool_recall")}
                  for mode in ("pointwise", "setwise")}
        reference = None
        for seed in SEEDS:
            arms, results = {}, {}
            for mode in values:
                directory = root / "remote" / dataset / "sasrec" / f"mechanism_{mode}" / f"s{seed}"
                receipt = receipts[dataset, seed, mode]
                assert receipt["selector_exact_parent"] and receipt["exit_code"] == 0
                for filename, checksum in receipt["files"].items():
                    assert digest(directory / filename) == checksum, (directory, filename)
                m = json.loads((directory / "training_manifest.json").read_text())
                r = json.loads((directory / "VALIDATION_RESULT.json").read_text())
                a = dict(np.load(directory / "validation_predictions.npz", allow_pickle=False))
                assert m["status"] == r["status"] == "pass" and not m["test_accessed"]
                assert m["seed"] == r["seed"] == seed
                assert m["checkpoint_sha256"] == r["checkpoint_sha256"] == receipt["checkpoint_sha256"]
                assert m["predictions_sha256"] == r["predictions_sha256"] == receipt["files"]["validation_predictions.npz"]
                assert label_receipts[dataset,seed,mode]["prediction_sha256"] == m["predictions_sha256"]
                assert label_receipts[dataset,seed,mode]["full_truth_metric_recompute"]
                assert r["mechanism_attribution"]["proposal_updated"] is False
                assert r["configuration"]["interaction_mode"] == mode
                assert m["binding"] == r["binding"] and m["configuration"] == r["configuration"]
                assert hashlib.sha256(np.ascontiguousarray(a["candidate_top200"]).tobytes()).hexdigest() == r["candidate_top200_sha256"]
                for i, target_row in enumerate(a["targets"]):
                    truth = set(map(int, target_row[target_row >= 3]))
                    count = int(a["target_counts"][i])
                    # Export contains reachable positives only; counts include all targets.
                    # Full-label equivalence is checked separately against source labels.
                    assert len(truth) <= count and count > 0, "invalid_supported_labels"
                    assert len(set(map(int, a["candidate_top200"][i]))) == 200
                    assert set(a["arrive_top10"][i]).issubset(set(a["candidate_top200"][i]))
                    for prefix in ("backbone", "rcce", "arrive"):
                        hits = np.isin(a[f"{prefix}_top10"][i], list(truth))
                        assert np.isclose(hits.sum()/count, a[f"{prefix}_recall"][i], atol=1e-12, rtol=0)
                        assert np.isclose((hits*discount).sum()/discount[:min(10,count)].sum(), a[f"{prefix}_ndcg"][i], atol=1e-12, rtol=0)
                    assert np.isclose(len(truth.intersection(a["candidate_top200"][i]))/count, a["pool_recall"][i], atol=1e-12, rtol=0)
                if reference is None:
                    reference = a
                for key in ("row_ids", "users", "bucket_ids", "target_counts"):
                    assert np.array_equal(a[key], reference[key]), (dataset, seed, key)
                for metric in values[mode]:
                    values[mode][metric].append(a["pool_recall" if metric == "pool_recall" else f"arrive_{metric}"])
                arms[mode], results[mode] = a, r
                audit.append({"dataset": dataset, "seed": seed, "mode": mode,
                              "rows": len(a["row_ids"]), "parameters": r["configuration"]["conversion_parameters"],
                              "fit_rows_sha256": r["fit_row_ids_sha256"], "selection": r["selection"],
                              "epochs": m["completed_epochs"], "updates": m["completed_updates"],
                              "prediction_sha256": m["predictions_sha256"]})
            for key in ("targets", "candidate_top200", "rcce_top10", "backbone_top10", "pool_recall"):
                assert np.array_equal(arms["pointwise"][key], arms["setwise"][key]), (dataset, seed, key)
            for key in ("binding", "fit_row_ids_sha256", "selection_row_ids_sha256"):
                assert results["pointwise"][key] == results["setwise"][key], key
            cfg = [{k:v for k,v in results[mode]["configuration"].items()
                    if k not in ("interaction_mode", "conversion_parameters")} for mode in values]
            assert cfg[0] == cfg[1], "unmatched_configuration"
        users = reference["users"]
        buckets = np.asarray(SCOPES[1:])[reference["bucket_ids"]]
        summary, contrasts = {}, {}
        values = {mode:{metric:np.stack(rows) for metric,rows in metrics.items()} for mode,metrics in values.items()}
        for mode, metrics in values.items():
            summary[mode] = {}
            for scope in SCOPES:
                mask = mask_for(buckets, scope)
                summary[mode][scope] = {}
                for metric, arrays in metrics.items():
                    samples = [float(user_values(a,users,mask).mean()) for a in arrays] if mask.any() else []
                    summary[mode][scope][metric] = {"mean":float(np.mean(samples)) if samples else None,
                        "std":float(np.std(samples,ddof=1)) if samples else None,"seeds":samples}
        for si, scope in enumerate(SCOPES):
            mask = mask_for(buckets,scope)
            contrasts[scope] = {}
            for mi, metric in enumerate(("recall","ndcg")):
                delta = values["setwise"][metric] - values["pointwise"][metric]
                estimate = paired(delta,users,mask,98100+si*10+mi)
                denominator = summary["pointwise"][scope][metric]["mean"]
                estimate["relative_percent"] = 100*estimate["delta"]/denominator if denominator else None
                estimate["seed_deltas"] = [float(user_values(a,users,mask).mean()) for a in delta] if mask.any() else []
                uv = user_values(delta.mean(axis=0),users,mask)
                estimate["user_coverage"] = {"improved":int((uv>0).sum()),"harmed":int((uv<0).sum()),"unchanged":int((uv==0).sum())}
                contrasts[scope][metric] = estimate
        longest = next(s for s in reversed(SCOPES[1:]) if mask_for(buckets,s).any())
        groups[dataset] = {"summary":summary,"setwise_minus_pointwise":contrasts,"longest_nonempty_scope":longest}
    atomic_json(output,{"schema":"arrive.candidate_interaction_analysis.v1","status":"pass",
        "integrity_status":"accepted", "scope":"supplementary_observed_validation_fixed_selector_SASRec_only",
        "plan_sha256":digest(root/"PLAN_TO_RUN_GATE.json"), "std_ddof":1,
        "groups":groups,"audit":audit,"test_opened":False,
        "limitations":["Conditional on frozen selector; does not compare joint-training alternatives.",
                       "Validation has been observed; not independent confirmation.",
                       "User bootstrap conditions on three fitted seeds, not population training variance.",
                       "Equal stopping rules, not forced identical realized update counts."]})
    print(json.dumps({d:g["setwise_minus_pointwise"]["overall"] for d,g in groups.items()},indent=2))


def paired_training(root: Path, output: Path, datasets: tuple[str, ...] = ("onion", "yambda")) -> None:
    """Audit the new same-parent Validation pairs; never mix in historical heads."""
    groups, audit = {}, []
    recipe_keys = {"training_rule", "proposal_optimizer", "selection_loss",
                   "ranking_loss", "gradient_update", "alignment_weight_effective"}
    discounts = 1.0 / np.log2(np.arange(2, 12))

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def load_cell(dataset: str, seed: int, arm: str):
        directory = root / dataset / "heads" / arm / f"s{seed}"
        result_path = directory / "VALIDATION_RESULT.json"
        manifest_path = directory / "training_manifest.json"
        predictions_path = directory / "validation_predictions.npz"
        checkpoint_path = directory / "joint_arrive_checkpoint.pt"
        result = json.loads(result_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        with np.load(predictions_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        require(result.get("status") == manifest.get("status") == "pass"
                and result.get("schema") == "arrive.joint_rcce_rgsc.v4"
                and result.get("seed") == manifest.get("seed") == seed
                and result.get("backbone") == manifest.get("backbone") == "sasrec"
                and result.get("objective_version") == manifest.get("objective_version") == "support_safe_v4"
                and result.get("component_variant") == manifest.get("component_variant") == "full"
                and result.get("binding") == manifest.get("binding")
                and result["binding"].get("test_accessed") is False
                and manifest.get("test_accessed") is False,
                f"cell_manifest_invalid:{directory}")
        require((arm == "ordinary") == (result.get("configuration", {}).get("training_rule") == "ordinary_multipositive")
                and result.get("configuration") == manifest.get("configuration")
                and result.get("configuration_sha256") == manifest.get("configuration_sha256")
                and result.get("predictions_sha256") == manifest.get("predictions_sha256") == digest(predictions_path)
                and result.get("checkpoint_sha256") == manifest.get("checkpoint_sha256") == digest(checkpoint_path),
                f"recipe_or_artifact_mismatch:{directory}")
        require(result.get("training", {}).get("completed_epochs") == manifest.get("completed_epochs")
                and int(manifest["completed_epochs"]) >= 15
                and manifest["configuration"]["epochs"] == 30
                and manifest["configuration"]["min_epochs"] == 15
                and manifest["configuration"]["patience"] == 5,
                f"training_insufficient:{directory}")
        require(len(arrays["row_ids"]) == len(arrays["users"]) == len(arrays["bucket_ids"])
                and arrays["candidate_top200"].shape == (len(arrays["row_ids"]), 200)
                and arrays["arrive_top10"].shape == (len(arrays["row_ids"]), 10)
                and hashlib.sha256(np.ascontiguousarray(arrays["candidate_top200"]).tobytes()).hexdigest()
                == result["candidate_top200_sha256"],
                f"prediction_shape_or_pool_hash:{directory}")
        targets = arrays["targets"]
        counts = arrays["target_counts"].astype(np.int64)
        require(np.all(counts > 0) and targets.shape[0] == len(counts), f"target_count:{directory}")
        for prefix in ("backbone", "rcce", "arrive"):
            top = arrays[f"{prefix}_top10"]
            hits = np.any(top[:, :, None] == targets[:, None, :], axis=2)
            recall = hits.sum(axis=1) / counts
            ideal = np.cumsum(discounts)[np.minimum(counts, 10) - 1]
            ndcg = (hits * discounts).sum(axis=1) / ideal
            require(np.allclose(recall, arrays[f"{prefix}_recall"], atol=1e-12, rtol=0)
                    and np.allclose(ndcg, arrays[f"{prefix}_ndcg"], atol=1e-12, rtol=0),
                    f"metric_recompute:{directory}:{prefix}")
        audit.append({"dataset": dataset, "seed": seed, "arm": arm,
                      "result": str(result_path), "result_sha256": digest(result_path),
                      "predictions_sha256": digest(predictions_path),
                      "checkpoint_sha256": digest(checkpoint_path),
                      "completed_epochs": manifest["completed_epochs"],
                      "selected_epoch": result["selection"]["epoch"],
                      "updates": result["training"]["updates"]})
        return result, arrays

    for dataset in datasets:
        values = {arm: {metric: [] for metric in ("recall", "ndcg", "pool_recall")}
                  for arm in ("arrive", "ordinary")}
        reference = None
        for seed in SEEDS:
            cells = {arm: load_cell(dataset, seed, arm) for arm in values}
            full_result, full = cells["arrive"]
            control_result, control = cells["ordinary"]
            require(full_result["binding"] == control_result["binding"],
                    f"paired_parent_or_feature_mismatch:{dataset}:{seed}")
            common_config = {key: value for key, value in full_result["configuration"].items()
                             if key not in recipe_keys}
            require(common_config == {key: value for key, value in control_result["configuration"].items()
                                      if key not in recipe_keys},
                    f"paired_budget_mismatch:{dataset}:{seed}")
            for key in ("row_ids", "users", "bucket_ids", "targets", "target_counts",
                        "backbone_top10", "backbone_recall", "backbone_ndcg"):
                require(np.array_equal(full[key], control[key]),
                        f"paired_label_or_backbone_mismatch:{dataset}:{seed}:{key}")
            if reference is None:
                reference = full
            else:
                # `targets` is the candidate-supported subset, so its contents
                # can differ with each seed's candidate pool even when the
                # underlying evaluation episodes and target counts are fixed.
                for key in ("row_ids", "users", "bucket_ids", "target_counts"):
                    require(np.array_equal(reference[key], full[key]),
                            f"seed_label_mismatch:{dataset}:{seed}:{key}")
            for arm, (_, arrays) in cells.items():
                for metric in values[arm]:
                    values[arm][metric].append(arrays[f"arrive_{metric}"] if metric != "pool_recall"
                                               else arrays["pool_recall"])
        users, buckets = reference["users"], reference["bucket_ids"]
        values = {arm: {metric: np.stack(rows) for metric, rows in metrics.items()}
                  for arm, metrics in values.items()}
        summaries, comparisons = {}, {}
        for scope_index, scope in enumerate(SCOPES):
            mask = np.ones(len(users), dtype=bool) if scope_index == 0 else buckets == scope_index - 1
            if not mask.any():
                continue
            summaries[scope] = {}
            comparisons[scope] = {}
            for arm in values:
                summaries[scope][arm] = {}
                for metric, array in values[arm].items():
                    per_seed = [float(user_values(row, users, mask).mean()) for row in array]
                    summaries[scope][arm][metric] = {
                        "mean": float(np.mean(per_seed)), "sample_sd": float(np.std(per_seed, ddof=1)),
                        "seeds": per_seed,
                    }
            for metric in ("recall", "ndcg", "pool_recall"):
                delta = values["arrive"][metric] - values["ordinary"][metric]
                estimate = paired(delta, users, mask, 98230 + 100 * scope_index + ("recall", "ndcg", "pool_recall").index(metric))
                denominator = summaries[scope]["ordinary"][metric]["mean"]
                estimate["relative_percent"] = 100 * estimate["delta"] / denominator if denominator else None
                estimate["seed_deltas"] = [float(user_values(row, users, mask).mean()) for row in delta]
                comparisons[scope][metric] = estimate
            a = values["arrive"]["recall"].mean(axis=0)[mask]
            b = values["ordinary"]["recall"].mean(axis=0)[mask]
            comparisons[scope]["episode_coverage"] = {
                "improved": int((a > b).sum()), "harmed": int((a < b).sum()),
                "unchanged": int((a == b).sum()),
                "rescued_zero_to_positive": int(((a > 0) & (b == 0)).sum()),
                "displaced_positive_to_zero": int(((a == 0) & (b > 0)).sum()),
            }
        groups[dataset] = {
            "rows": int(len(users)), "users": int(len(np.unique(users))),
            "longest_nonempty_scope": next(scope for scope in reversed(SCOPES[1:]) if scope in summaries),
            "summary": summaries, "comparison": comparisons,
        }
    atomic_json(output, {
        "schema": "arrive.paired_training_validation_analysis.v1", "status": "pass",
        "integrity_status": "provisional", "test_accessed": False,
        "seeds": list(SEEDS), "comparison": "ARRIVE training recipe versus ordinary two-head training on newly paired parents",
        "bootstrap": "5000 matched-user resamples after seed averaging; descriptive unadjusted 95% CI",
        "groups": groups, "integrity": audit,
        "limitations": ["Both arms use new paired parent assets; never compare against historical ARRIVE head scores.",
                        "Candidate-supported target arrays may differ across seeds; within each paired seed they must match exactly.",
                        "Training recipes change losses and optimizer/update rules together; this is not single-ingredient attribution.",
                        "User bootstrap conditions on three trained seeds; it does not include population training variance.",
                        "Deterministic hash and metric checks pass, but independent dataset-producer replay and parent-asset pullback are incomplete.",
                        "Validation is development evidence, not unopened Test confirmation."],
    })
    print(json.dumps({dataset: group["comparison"]["overall"] for dataset, group in groups.items()}, indent=2))


def industry_quality(root: Path, staged_root: Path, project: Path, output: Path) -> None:
    """Compare retained systems on identical Validation requests, without refitting.

    The historical staged and reconstructed systems have different backbone
    checkpoints. This measures system quality retention, not module attribution.
    """
    from run_music4all_candidate_support_a1 import target_lookup
    from run_music4all_component_c2_conversion_oof import row_metrics

    data = project / "data/music4all_onion/prepared_v1"
    truth = target_lookup(data)
    sessions = np.load(data / "sessions.npy", mmap_mode="r")
    values = {arm: {metric: [] for metric in ("recall", "ndcg")}
              for arm in ("staged_arrive", "rrf_lambdamart")}
    identity = None
    bindings = []
    for seed in SEEDS:
        old_dir, new_dir = staged_root / f"s{seed}", root / f"s{seed}"
        old_result = json.loads((old_dir / "VALIDATION_RESULT.json").read_text())
        new_result = json.loads((new_dir / "training_manifest.json").read_text())
        paths = {"staged_arrive": old_dir / "validation_predictions.npz",
                 "rrf_lambdamart": new_dir / "validation_predictions.npz"}
        arrays = {}
        for arm, manifest in (("staged_arrive", old_result), ("rrf_lambdamart", new_result)):
            prediction_key = "predictions_sha256" if arm == "staged_arrive" else "prediction_sha256"
            if (manifest.get("status") != "pass" or manifest.get("seed") != seed
                    or manifest.get("test_accessed") is not False
                    or manifest.get(prediction_key) != digest(paths[arm])):
                raise ValueError(f"industry_quality_manifest:{arm}:{seed}")
            with np.load(paths[arm], allow_pickle=False) as z:
                arrays[arm] = {k: z[k].copy() for k in z.files}
            current = arrays[arm]
            keys = ("row_ids", "users", "bucket_ids", "target_counts")
            if identity is None:
                identity = {k: current[k] for k in keys}
            elif any(not np.array_equal(identity[k], current[k]) for k in keys):
                raise ValueError(f"industry_quality_identity:{arm}:{seed}")
            if not np.all(sessions[current["row_ids"], 5] == 1):
                raise ValueError("industry_quality_non_validation_row")
            full_targets = np.full_like(current["targets"], -1)
            for at, rid in enumerate(current["row_ids"]):
                count = int(current["target_counts"][at])
                observed = list(map(int, current["targets"][at, :count]))
                expected = truth[int(rid)]
                # Legacy staged outputs retain pool-visible targets only and
                # pad the others with -1. Rebuild full labels for BOTH arms;
                # require retained positive labels to agree and metrics replay.
                if (count != len(expected) or count > full_targets.shape[1]
                        or not {x for x in observed if x >= 3}.issubset(set(expected))
                        or (arm == "rrf_lambdamart" and set(observed) != set(expected))):
                    raise ValueError("industry_quality_full_target_mismatch")
                full_targets[at, :count] = expected
            top = current["top10" if arm == "staged_arrive" else "rrf_top10"]
            if top.shape != (len(current["row_ids"]), 10) or any(len(set(row)) != 10 for row in top) or (top < 3).any():
                raise ValueError("industry_quality_invalid_top10")
            r, n = row_metrics(top, full_targets, current["target_counts"])
            for metric, actual, saved in (("recall", r, "method_r" if arm == "staged_arrive" else "rrf_recall"),
                                          ("ndcg", n, "method_n" if arm == "staged_arrive" else "rrf_ndcg")):
                if not np.allclose(actual, current[saved], rtol=0, atol=1e-14):
                    raise ValueError(f"industry_quality_metric_replay:{arm}:{seed}:{metric}")
                values[arm][metric].append(actual)
        bindings.append({"seed": seed, "predictions": {arm: {"path": str(path), "sha256": digest(path)}
                                                        for arm, path in paths.items()},
                         "staged_result_sha256": digest(old_dir / "VALIDATION_RESULT.json"),
                         "ranker_manifest_sha256": digest(new_dir / "training_manifest.json")})
    users, buckets = identity["users"], identity["bucket_ids"]
    summary, comparisons = {}, {}
    for scope_index, scope in enumerate(SCOPES):
        mask = np.ones(len(users), bool) if scope == "overall" else buckets == scope_index - 1
        summary[scope], comparisons[scope] = {}, {}
        for arm in values:
            summary[scope][arm] = {}
            for metric in ("recall", "ndcg"):
                means = [float(user_values(v, users, mask).mean()) for v in values[arm][metric]]
                summary[scope][arm][metric] = {"mean": float(np.mean(means)),
                                               "sample_sd": float(np.std(means, ddof=1)), "seed_means": means}
        for metric_index, metric in enumerate(("recall", "ndcg")):
            delta = np.asarray(values["rrf_lambdamart"][metric]) - np.asarray(values["staged_arrive"][metric])
            estimate = paired(delta, users, mask, 98400 + scope_index * 10 + metric_index)
            estimate["relative_percent"] = 100 * estimate["delta"] / summary[scope]["staged_arrive"][metric]["mean"]
            comparisons[scope][metric] = estimate
    atomic_json(output, {"schema": "arrive.industry_quality_retention.v1", "status": "pass",
                         "test_accessed": False, "rows": len(users), "users": len(np.unique(users)),
                         "seeds": list(SEEDS), "scope": "system_quality_retention_validation_only",
                         "same_backbone_checkpoints": False,
                         "not_a_module_causal_comparison": True, "serving_cost_measured": False,
                         "producer_sha256": digest(Path(__file__)),
                         "bindings": bindings, "ground_truth_sha256": digest(data / "metric_labels.npz"),
                         "summary": summary, "comparison": comparisons})
    print(json.dumps({"status": "pass", "overall": comparisons["overall"],
                      "over365d": comparisons[">365d"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--independent-components", action="store_true")
    parser.add_argument("--candidate-interaction", action="store_true")
    parser.add_argument("--paired-training", action="store_true")
    parser.add_argument("--industry-quality", action="store_true")
    parser.add_argument("--staged-root", type=Path)
    parser.add_argument("--paired-dataset", action="append", choices=("onion", "yambda"),
                        help="Analyze a completed dataset without waiting for the other; default: both")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--validation-subdir", default="")
    parser.add_argument("--std-ddof", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--backbone",
        choices=("sasrec", "ssd4rec", "muse", "tisasrec"),
        default="sasrec",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.industry_quality:
        if args.staged_root is None:
            parser.error("--industry-quality requires --staged-root")
        industry_quality(args.root.resolve(), args.staged_root.resolve(), args.project_root.resolve(), args.output)
        return
    if args.paired_training:
        paired_training(args.root.resolve(), args.output, tuple(args.paired_dataset or ("onion", "yambda")))
        return
    if args.candidate_interaction:
        candidate_interaction(args.root.resolve(), args.output)
        return
    if args.independent_components:
        independent_components(args.root.resolve(), args.project_root.resolve(), args.output)
        return
    validation_root = args.validation_root or args.root.parent / "remote_validation_v4_by_seed"

    values = {arm: {metric: [] for metric in ("recall", "ndcg")} for arm in ARMS}
    row_ids = users = buckets = None
    integrity = []
    common_bindings = None
    expected_rows = None
    for seed in SEEDS:
        directory = args.root / f"s{seed}"
        result_path = directory / "TEST_RESULT.json"
        prediction_path = directory / "test_predictions.npz"
        result = json.loads(result_path.read_text())
        validation_path = validation_root / f"s{seed}" / args.validation_subdir / "VALIDATION_RESULT.json"
        arrays = dict(np.load(prediction_path, allow_pickle=False))
        if (
            result.get("schema") != "arrive.joint_rcce_rgsc_test.v1"
            or result.get("status") != "pass"
            or result.get("seed") != seed
            or result.get("backbone") != args.backbone
            or result.get("objective_version") != "support_safe_v4"
            or result.get("inference_only") is not True
            or result.get("selection_or_tuning_performed") is not False
            or result.get("test_accessed") is not True
            or result.get("predictions_sha256") != digest(prediction_path)
            or result.get("bindings", {}).get("producer_sha256")
            not in ALLOWED_TEST_PRODUCERS
            or result.get("bindings", {}).get("joint_validation_result_sha256")
            != digest(validation_path)
        ):
            raise ValueError(f"invalid_joint_test_result:{seed}")
        if expected_rows is None:
            expected_rows = int(result["rows"])
        elif int(result["rows"]) != expected_rows:
            raise ValueError(f"cross_seed_joint_test_row_count_drift:{seed}")
        bindings = result["bindings"]
        architecture_contract = json.loads(json.dumps(bindings["architecture_contract"]))
        if isinstance(architecture_contract, dict) and isinstance(architecture_contract.get("configuration"), dict):
            architecture_contract["configuration"].pop("seed", None)
        current_common = {
            "test_manifest_sha256": bindings["test_manifest_sha256"],
            "test_feature_protocol": bindings["test_feature_protocol"],
            "audio_sha256": bindings["audio_sha256"],
            "genre_sha256": bindings["genre_sha256"],
            "forward_runner_family": REPORTING_FORWARD_FAMILY.get(
                bindings["forward_runner_sha256"], bindings["forward_runner_sha256"]
            ),
            "architecture_contract": architecture_contract,
            "objective_version": result["objective_version"],
            "rows": result["rows"],
        }
        if common_bindings is None:
            common_bindings = current_common
        elif current_common != common_bindings:
            raise ValueError(f"cross_seed_joint_test_binding_drift:{seed}")
        if row_ids is None:
            row_ids, users, buckets = arrays["row_ids"], arrays["users"], arrays["gap_buckets"]
        elif not (
            np.array_equal(row_ids, arrays["row_ids"])
            and np.array_equal(users, arrays["users"])
            and np.array_equal(buckets, arrays["gap_buckets"])
        ):
            raise ValueError(f"cross_seed_joint_test_row_drift:{seed}")
        for arm in ARMS:
            for metric in ("recall", "ndcg"):
                values[arm][metric].append(arrays[f"{arm}_{metric}"])
        integrity.append({
            "seed": seed,
            "rows": result["rows"],
            "checkpoint_sha256": result["bindings"]["joint_checkpoint_sha256"],
            "validation_result_sha256": result["bindings"]["joint_validation_result_sha256"],
            "test_feature_manifest_sha256": result["bindings"]["test_feature_manifest_sha256"],
            "feature_sha256": result["bindings"]["feature_sha256"],
            "test_producer_sha256": result["bindings"]["producer_sha256"],
            "forward_runner_sha256": bindings["forward_runner_sha256"],
            "forward_runner_compatibility": result["bindings"].get(
                "forward_runner_compatibility", "exact_pre_audit_field"
            ),
            "inference_only": True,
            "selection_or_tuning_performed": False,
        })

    assert row_ids is not None and users is not None and buckets is not None
    for arm in ARMS:
        for metric in ("recall", "ndcg"):
            values[arm][metric] = np.stack(values[arm][metric])

    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for scope in SCOPES:
            selected = mask_for(buckets, scope)
            summary[arm][scope] = {}
            for metric in ("recall", "ndcg"):
                if not np.any(selected):
                    summary[arm][scope][metric] = {
                        "mean": None,
                        "std": None,
                        "seeds": [None for _ in SEEDS],
                    }
                    continue
                per_seed = [
                    float(user_values(values[arm][metric][at], users, selected).mean())
                    for at in range(len(SEEDS))
                ]
                summary[arm][scope][metric] = {
                    "mean": float(np.mean(per_seed)),
                    "std": float(np.std(per_seed, ddof=args.std_ddof)),
                    "seeds": per_seed,
                }

    comparisons = {}
    for method, predecessor in (("rcce", "backbone"), ("arrive", "rcce"), ("arrive", "backbone")):
        key = f"{method}_vs_{predecessor}"
        comparisons[key] = {}
        for scope_index, scope in enumerate(SCOPES):
            selected = mask_for(buckets, scope)
            comparisons[key][scope] = {}
            for metric_index, metric in enumerate(("recall", "ndcg")):
                estimate = paired(
                    values[method][metric] - values[predecessor][metric],
                    users,
                    selected,
                    93000 + 100 * scope_index + metric_index,
                )
                denominator = summary[predecessor][scope][metric]["mean"]
                estimate["relative_percent"] = (
                    100.0 * estimate["delta"] / denominator
                    if estimate["delta"] is not None and denominator
                    else None
                )
                comparisons[key][scope][metric] = estimate
        method_recall = values[method]["recall"].mean(axis=0)
        predecessor_recall = values[predecessor]["recall"].mean(axis=0)
        comparisons[key]["coverage"] = {
            "better_rows": int((method_recall > predecessor_recall).sum()),
            "worse_rows": int((method_recall < predecessor_recall).sum()),
            "unchanged_rows": int((method_recall == predecessor_recall).sum()),
            "rescued_zero_to_positive_rows": int(((method_recall > 0) & (predecessor_recall == 0)).sum()),
            "displaced_positive_to_zero_rows": int(((method_recall == 0) & (predecessor_recall > 0)).sum()),
        }

    full = comparisons["arrive_vs_backbone"]
    rcce = comparisons["rcce_vs_backbone"]
    rgsc = comparisons["arrive_vs_rcce"]
    overall_supported = all(
        comparison["overall"][metric]["ci95_low"] > 0
        for comparison in (rcce, rgsc, full)
        for metric in ("recall", "ndcg")
    )
    nonempty_long_scopes = [
        scope for scope in reversed(SCOPES[1:]) if summary["backbone"][scope]["recall"]["mean"] is not None
    ]
    longest_scope = nonempty_long_scopes[0] if nonempty_long_scopes else "overall"
    long_positive = all(
        comparison[longest_scope][metric]["delta"] > 0
        for comparison in (rcce, rgsc, full)
        for metric in ("recall", "ndcg")
    )
    result = {
        "schema": "arrive.joint_rcce_rgsc_test_analysis.v1",
        "status": "pass",
        "decision": (
            f"{args.backbone.upper()}_JOINT_ARRIVE_TEST_COMPONENT_CHAIN_SUPPORTED"
            if overall_supported and long_positive
            else f"{args.backbone.upper()}_JOINT_ARRIVE_TEST_COMPONENT_CHAIN_PARTIAL"
        ),
        "rows": int(len(row_ids)),
        "users": int(len(np.unique(users))),
        "longest_nonempty_scope": longest_scope,
        "seeds": list(SEEDS),
        "std_ddof": args.std_ddof,
        "integrity": integrity,
        "test_producer_sha256s": sorted(
            {row["test_producer_sha256"] for row in integrity}
        ),
        "common_bindings": common_bindings,
        "summary": summary,
        "comparisons": comparisons,
        "claim_boundary": {
            "test_is_not_untouched": True,
            "test_was_not_used_for_method_or_hyperparameter_selection": True,
            "ablation_is_cumulative_inference_from_each_seed_owned_joint_checkpoint": True,
            "backbone_scope": args.backbone,
        },
    }
    atomic_json(args.output, result)
    print(json.dumps({"decision": result["decision"], "overall": full["overall"]}, indent=2))


if __name__ == "__main__":
    main()
