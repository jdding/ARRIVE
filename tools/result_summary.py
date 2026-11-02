"""User-macro, matched-seed summaries for saved predictions and source replays."""
from pathlib import Path

import numpy as np

from export_io import digest, npz_dict
from metrics import row_metrics

SEEDS = (2024, 2025, 2026)
IDENTITY = ("row_ids", "users", "targets", "target_counts", "bucket_ids")
GAPS = ("30-90d", "90-180d", "180-365d", ">365d")


def input_path(base, entry, field="path"):
    path = Path(entry[field])
    return path if path.is_absolute() else base / path


def user_values(values, users, mask=None):
    if mask is None:
        mask = np.ones(len(users), bool)
    selected = np.asarray(users)[mask]
    unique, inverse = np.unique(selected, return_inverse=True)
    return unique, np.bincount(inverse, weights=np.asarray(values)[mask]) / np.bincount(inverse)


def load_predictions(entry, base):
    path = input_path(base, entry)
    actual_hash = digest(path)
    if entry.get("sha256") is not None and entry["sha256"] != actual_hash:
        raise ValueError("prediction_hash_mismatch")
    raw = npz_dict(path)
    metadata = npz_dict(input_path(base, entry, "metadata")) if "metadata" in entry else raw
    if not set(IDENTITY) <= set(metadata) or not {"row_ids", "users", "top10"} <= set(raw):
        raise ValueError("prediction_and_complete_metadata_required")
    order = np.argsort(raw["row_ids"])
    meta_order = np.argsort(metadata["row_ids"])
    n = len(order)
    if not n or len(np.unique(raw["row_ids"])) != n:
        raise ValueError("nonempty_unique_request_ids_required")
    result = {key: metadata[key][meta_order] for key in IDENTITY}
    if (len(meta_order) != n or len(np.unique(result["row_ids"])) != n
            or result["targets"].ndim != 2
            or any(result[key].shape != (n,) for key in ("row_ids", "users", "target_counts", "bucket_ids"))
            or any(not np.issubdtype(result[key].dtype, np.integer) for key in IDENTITY)
            or np.any((result["bucket_ids"] < 0) | (result["bucket_ids"] > 3))):
        raise ValueError("complete_prediction_metadata_shape_or_type")
    if not (np.array_equal(raw["row_ids"][order], result["row_ids"])
            and np.array_equal(raw["users"][order], result["users"])):
        raise ValueError("prediction_metadata_request_identity")
    top = raw["top10"][order]
    if (top.shape != (n, 10) or not np.issubdtype(top.dtype, np.integer)
            or np.any(top < 3) or any(len(np.unique(row)) != 10 for row in top)):
        raise ValueError("ten_distinct_catalog_predictions_required")
    for truth, count in zip(result["targets"], result["target_counts"]):
        if not 1 <= count <= len(truth) or len(np.unique(truth[:count])) != count or np.isin(truth[:count], [-1, 0, 1, 2]).any():
            raise ValueError("complete_distinct_targets_required")
    result["recall"], result["ndcg"] = row_metrics(top, result["targets"], result["target_counts"])
    result["top10"] = top
    for name in ("recall", "ndcg"):
        if name in raw and not np.allclose(raw[name][order], result[name], rtol=0, atol=1e-12):
            raise ValueError("stored_metric_disagrees:" + name)
    if "cue_bundle" in entry:
        cue_path = input_path(base, entry, "cue_bundle")
        cue_hash = digest(cue_path)
        if entry.get("cue_bundle_sha256") is not None and entry["cue_bundle_sha256"] != cue_hash:
            raise ValueError("cue_bundle_hash_mismatch")
        cue = npz_dict(cue_path)
        result["cue_bundle_sha256"] = cue_hash
        cue_order = np.argsort(cue["row_ids"])
        if not np.array_equal(cue["row_ids"][cue_order], result["row_ids"]):
            raise ValueError("cue_bundle_request_identity")
        result["eligible"] = cue["eligible"][cue_order]
        if "cues" in cue:
            if (cue["cues"].shape != (n,) or not np.issubdtype(cue["cues"].dtype, np.integer)
                    or not np.array_equal(cue["users"][cue_order], result["users"])):
                raise ValueError("cue_bundle_shape_or_user_identity")
            result["cues"] = cue["cues"][cue_order]
    if "cues" in raw:
        if raw["cues"].shape != (n,) or not np.issubdtype(raw["cues"].dtype, np.integer):
            raise ValueError("prediction_cue_shape_or_type")
        actual = raw["cues"][order]
        if "cues" in result and not np.array_equal(actual, result["cues"]):
            raise ValueError("prediction_cue_disagreement")
        result["cues"] = actual
    result["input_sha256"] = actual_hash
    return result


def align(records, keys=IDENTITY):
    for record in records[1:]:
        for key in keys:
            if not np.array_equal(record[key], records[0][key]):
                raise ValueError("paired_cohort_difference:" + key)


def summary(values, users, mask=None):
    means = [float(user_values(row, users, mask)[1].mean()) for row in values]
    return dict(mean=float(np.mean(means)), sample_sd=float(np.std(means, ddof=1)), per_seed=means)


def contrast(left, right, users, mask, draws, random_seed):
    if not np.any(mask):
        return None
    left_users = np.asarray([user_values(row, users, mask)[1] for row in left])
    right_users = np.asarray([user_values(row, users, mask)[1] for row in right])
    unit = (left_users - right_users).mean(axis=0)
    rng = np.random.default_rng(random_seed)
    bootstrap = np.asarray([rng.choice(unit, len(unit), replace=True).mean() for _ in range(draws)])
    return dict(delta=float(unit.mean()), delta_pp=float(100 * unit.mean()),
        ci95_pp=(100 * np.quantile(bootstrap, [0.025, 0.975])).tolist(),
        ci98_75_pp=(100 * np.quantile(bootstrap, [0.00625, 0.99375])).tolist(),
        users=len(unit), requests=int(mask.sum()), bootstrap_draws=draws,
        bootstrap_seed=random_seed, resampling_unit="user_after_seed_averaging",
        per_seed_delta_pp=(100 * (left_users - right_users).mean(axis=1)).tolist())


def song_type_contributions(record):
    """Partition the original metrics, without changing ranks or denominators."""
    result = {kind: {metric: np.zeros(len(record["users"])) for metric in ("recall", "ndcg")}
              for kind in ("replay", "other")}
    discount = 1 / np.log2(np.arange(2, 12))
    for q, (top, count, cue) in enumerate(zip(record["top10"], record["target_counts"], record["cues"])):
        truth = record["targets"][q, :count]
        relevant = np.isin(top, truth)
        for kind, mask in (("replay", top == cue), ("other", top != cue)):
            hit = relevant & mask
            result[kind]["recall"][q] = hit.sum() / count
            result[kind]["ndcg"][q] = discount[hit].sum() / discount[:min(count, 10)].sum()
    return result


def other_song_rank_terms(a, b):
    """Exact other-song NDCG accounting: gained, lost, shared-hit rank changes."""
    align([a, b], IDENTITY + ("cues",))
    values = {key: np.zeros(len(a["users"])) for key in ("gained", "lost", "shared_rank")}
    discount = 1 / np.log2(np.arange(2, 12))
    for q, count in enumerate(a["target_counts"]):
        truth = set(map(int, a["targets"][q, :count])) - {int(a["cues"][q])}
        ranks = [{int(item): rank for rank, item in enumerate(r["top10"][q]) if item in truth}
                 for r in (a, b)]
        left, right = map(set, ranks)
        ideal = discount[:min(count, 10)].sum()
        values["gained"][q] = sum(discount[ranks[0][i]] for i in left - right) / ideal
        values["lost"][q] = -sum(discount[ranks[1][i]] for i in right - left) / ideal
        values["shared_rank"][q] = sum(discount[ranks[0][i]] - discount[ranks[1][i]]
                                        for i in left & right) / ideal
    return values


def summarize(config, base):
    random_seed = config.get("bootstrap_seed", 20261007)
    draws = config.get("bootstrap_draws", {"sasrec": 10000, "ssd4rec": 10000})
    if not isinstance(random_seed, int) or any(type(v) is not int or v < 1 for v in draws.values()):
        raise ValueError("bootstrap_configuration")
    groups = {}
    for entry in config.get("rankings", []):
        key = (entry["dataset"], entry["backbone"], entry["method"])
        seed = entry["seed"]
        if seed not in SEEDS or seed in groups.setdefault(key, {}):
            raise ValueError("duplicate_or_unsupported_seed")
        groups[key][seed] = load_predictions(entry, base)
    tables, comparisons, inputs, cdfs = [], [], [], {}
    for key, seeds in groups.items():
        if set(seeds) != set(SEEDS):
            raise ValueError("three_matched_seeds_required:" + str(key))
        records = [seeds[s] for s in SEEDS]
        align(records)
        row = dict(dataset=key[0], backbone=key[1], method=key[2])
        for metric in ("recall", "ndcg"):
            row[metric] = summary(np.asarray([r[metric] for r in records]), records[0]["users"])
        tables.append(row)
        inputs.extend(dict(dataset=key[0], backbone=key[1], method=key[2], seed=s,
                           predictions_sha256=seeds[s]["input_sha256"],
                           **({"cue_bundle_sha256": seeds[s]["cue_bundle_sha256"]}
                              if "cue_bundle_sha256" in seeds[s] else {})) for s in SEEDS)
    for dataset, backbone in sorted({key[:2] for key in groups}):
        akey = (dataset, backbone, "arrive")
        if akey not in groups:
            continue
        a = [groups[akey][s] for s in SEEDS]
        meta = a[0]
        users = meta["users"]
        all_rows = np.ones(len(users), bool)
        for control in ("backbone", "removed", "mismatched", "identity", "joint-removal"):
            bkey = (dataset, backbone, control)
            if bkey not in groups:
                continue
            b = [groups[bkey][s] for s in SEEDS]
            align(a + b)
            result = dict(dataset=dataset, backbone=backbone, contrast="arrive_minus_" + control)
            count = draws.get(backbone, 10000)
            for metric in ("recall", "ndcg"):
                left, right = [np.asarray([r[metric] for r in records]) for records in (a, b)]
                result[metric] = contrast(left, right, users, all_rows, count, random_seed)
                result[metric]["interval_role"] = ("Bonferroni_four_NDCG_contrasts" if
                    backbone in ("sasrec", "ssd4rec") and control in ("removed", "mismatched") and metric == "ndcg"
                    else "descriptive_95_percent")
                if result[metric]["interval_role"] == "Bonferroni_four_NDCG_contrasts":
                    result[metric]["interval_family"] = (
                        "sasrec_prespecified_four_matching_contrasts" if backbone == "sasrec"
                        else "ssd4rec_followup_four_matching_contrasts")
                if control in ("identity", "joint-removal") and metric == "ndcg":
                    result[metric]["interval_role"] = "98.75_percent_control_interval"
                    result[metric]["interval_family"] = control.replace("-", "_") + "_controls"
            if all("cues" in r for r in a + b):
                align(a + b, IDENTITY + ("cues",))
                parts = [[song_type_contributions(r) for r in records] for records in (a, b)]
                result["song_type"] = {
                    kind: {metric: contrast(np.asarray([r[kind][metric] for r in parts[0]]),
                        np.asarray([r[kind][metric] for r in parts[1]]), users, all_rows, count, random_seed)
                        for metric in ("recall", "ndcg")}
                    for kind in ("replay", "other")}
                terms = [other_song_rank_terms(x, y) for x, y in zip(a, b)]
                result["other_song_rank_terms"] = {
                    name: contrast(np.asarray([r[name] for r in terms]),
                        np.zeros((len(SEEDS), len(users))), users, all_rows, count, random_seed)
                    for name in ("gained", "lost", "shared_rank")}
                replay = np.asarray([int(cue) in truth[:n] for cue, truth, n in
                    zip(meta["cues"], meta["targets"], meta["target_counts"])])
                known_no_replay = (meta["cues"] >= 3) & ~replay
                result["no_replay_scope"] = "known_catalog_cue_absent_from_full_target_set"
                result["no_replay_requests"] = {metric: contrast(
                    np.asarray([r[metric] for r in a]), np.asarray([r[metric] for r in b]),
                    users, known_no_replay, count, random_seed) for metric in ("recall", "ndcg")}
            if control == "backbone":
                left, right = [np.asarray([r["ndcg"] for r in records]) for records in (a, b)]
                user_ids, delta = user_values((left - right).mean(axis=0), users)
                recall_delta = user_values(np.mean([x["recall"] - y["recall"] for x, y in zip(a, b)], axis=0), users)[1]
                harmed, unchanged, improved = delta < -1e-12, np.abs(delta) <= 1e-12, delta > 1e-12
                result["users"] = dict(harmed_fraction=float(harmed.mean()), unchanged_fraction=float(unchanged.mean()),
                    improved_fraction=float(improved.mean()),
                    harmed_ndcg_with_non_decreasing_recall=float((recall_delta[harmed] >= -1e-12).mean()) if harmed.any() else None)
                if "eligible" in meta:
                    if any(not np.array_equal(r.get("eligible"), meta["eligible"]) for r in a):
                        raise ValueError("cue_availability_changed_across_seeds")
                    any_unusable = user_values((~meta["eligible"]).astype(float), users)[1] > 0
                    result["users"]["harmed_by_cue_availability"] = {
                        name: float(harmed[mask].mean()) if mask.any() else None
                        for name, mask in (("any_unusable", any_unusable), ("all_usable", ~any_unusable))}
                order = np.argsort(delta, kind="stable")
                cdfs[dataset + "_" + backbone] = dict(user_ids=user_ids[order].tolist(),
                    ndcg_delta_pp=(100 * delta[order]).tolist(), ecdf=((np.arange(len(delta)) + 1) / len(delta)).tolist())
                masks = {"gap_" + str(i): meta["bucket_ids"] == i for i in range(4)}
                target = meta["target_counts"]
                masks.update(target_1=target == 1, target_2_3=(target >= 2) & (target <= 3),
                             target_4_10=(target >= 4) & (target <= 10), target_gt10=target > 10)
                result["strata_ndcg"] = {name: contrast(left, right, users, mask, count, random_seed)
                                          for name, mask in masks.items()}
            comparisons.append(result)
        if all((dataset, backbone, key) in groups for key in ("removed", "mismatched")):
            b = [groups[(dataset, backbone, "removed")][s] for s in SEEDS]
            c = [groups[(dataset, backbone, "mismatched")][s] for s in SEEDS]
            align(b + c)
            comparisons.append(dict(dataset=dataset, backbone=backbone, contrast="mismatched_minus_removed",
                ndcg=contrast(np.asarray([r["ndcg"] for r in c]), np.asarray([r["ndcg"] for r in b]),
                    users, all_rows, draws.get(backbone, 10000), random_seed)))
    coverage_groups = {}
    for entry in config.get("coverage", []):
        key, seed = entry["dataset"], entry["seed"]
        if seed not in SEEDS or seed in coverage_groups.setdefault(key, {}):
            raise ValueError("coverage_seed_identity")
        path = input_path(base, entry)
        if entry.get("sha256") is not None and entry["sha256"] != digest(path):
            raise ValueError("coverage_hash_mismatch")
        raw = npz_dict(path)
        n = len(raw["row_ids"])
        if (not n or raw["row_ids"].shape != (n,) or raw["users"].shape != (n,)
                or len(np.unique(raw["row_ids"])) != n
                or any(not np.issubdtype(raw[k].dtype, np.integer) for k in ("row_ids", "users"))
                or any(raw[k].shape != (n,) or not np.isfinite(raw[k]).all()
                       or np.any((raw[k] < 0) | (raw[k] > 1))
                       for k in ("full_coverage", "removed_coverage"))):
            raise ValueError("coverage_shape_identity_or_range")
        order = np.argsort(raw["row_ids"])
        coverage_groups[key][seed] = {k: v[order] for k, v in raw.items()}
    coverage = []
    for dataset, seeds in coverage_groups.items():
        if set(seeds) != set(SEEDS):
            raise ValueError("coverage_three_seeds_required")
        records = [seeds[s] for s in SEEDS]
        align(records, ("row_ids", "users"))
        users = records[0]["users"]
        result = dict(dataset=dataset)
        for name, transform in (("recall", lambda x: x), ("hit_rate", lambda x: (x > 0).astype(float))):
            full = np.asarray([transform(r["full_coverage"]) for r in records])
            removed = np.asarray([transform(r["removed_coverage"]) for r in records])
            result[name] = dict(full=summary(full, users), removed=summary(removed, users),
                paired=contrast(full, removed, users, np.ones(len(users), bool), draws.get("sasrec", 10000), random_seed))
        coverage.append(result)
    if not groups and not coverage_groups:
        raise ValueError("no_saved_results")
    return dict(schema="arrive.result_summary.v1", seeds=list(SEEDS), ranking_rows=tables,
        paired_comparisons=comparisons, coverage=coverage, user_gain_ecdf=cdfs, inputs=inputs,
        paper_metric_reproduction=False, new_training=False,
        inference_performed=False, metrics_unit="fraction; delta_pp and intervals in percentage points")
