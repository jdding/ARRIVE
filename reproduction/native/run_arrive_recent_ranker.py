#!/usr/bin/env python3
"""Matched recent-ranker comparison on frozen ARRIVE framework exports.

Only the ranker changes. This runner never exports data or opens Test.

Dataset/surface support (EXPERIMENT_PLAN.md gap 2): `--dataset` selects a
registry bundle (tools/arrive_framework_datasets.py) instead of hardcoded
Onion paths. Training still never opens Test; sealed-Test evaluation lives
in evaluate_arrive_yeti31_fusion_test.py and runs only after the freeze.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from arrive_framework_datasets import DATASETS, bundle


SEEDS = (2024, 2025, 2026)
ITERATIONS = (220, 440)
LOSS = "YetiRank:mode=NDCG;top=10;dcg_type=Base;dcg_denominator=LogPosition"
CONTEXT_MODES = ("base", "statistics", "relations")
CUE_MATCH_COLUMNS = (12, 13, 14, 16)
CUE_MATCH_MODES = ("true", "none", "wrong")


def apply_cue_match(block: np.ndarray, mode: str, treatment=None, ids=None, query_ids=None) -> np.ndarray:
    """Treat only a freshly allocated training/prediction block, never its source."""
    if mode not in CUE_MATCH_MODES:
        raise ValueError("unknown_cue_match_mode")
    if mode == "none":
        if block.ndim != 2 or block.shape[1] != 31 or block.dtype != np.float32 or not block.flags.writeable:
            raise ValueError("cue_match_ablation_requires_writable_statistics31")
        block[:, CUE_MATCH_COLUMNS] = 0
    elif mode == "wrong":
        if treatment is None or ids is None or query_ids is None:
            raise ValueError("wrong_cue_treatment_required")
        treatment.apply(block,ids,query_ids)
    return block


def request_relations(data, row_ids, items, audio, genre, tags, artists, x, h, frequency):
    """Request-only fields. Deliberately never read continuation targets."""
    from scipy import sparse
    from run_music4all_candidate_support_a0 import recent_anchors, ranked_sparse_row, ranked_history
    norm_items = items / np.maximum(np.linalg.norm(items, axis=1, keepdims=True), 1e-8)
    result = np.empty((len(row_ids), 18), np.float32)
    x_cache = {}
    for at, rid in enumerate(row_ids):
        session = data.sessions[int(rid)]
        start = int(session[2])
        cue = int(data.events[start])
        past = list(dict.fromkeys(map(int, data.events[max(int(session[1]), start - 128):start][::-1])))[:20]
        q = []
        for matrix in (audio, genre, tags, norm_items):
            if sparse.issparse(matrix):
                v = (matrix.getrow(cue) @ matrix[past].T).toarray().ravel() if past else np.zeros(1)
            else:
                v = matrix[past] @ matrix[cue] if past else np.zeros(1)
            q.extend((float(v.mean()), float(v.max()), float(v.std())))
        if cue not in x_cache:
            x_cache[cue] = ranked_sparse_row(x, cue, frequency, 0., 200)
        history = recent_anchors(data.events, session, 5)
        h_source = ranked_history(h, history, frequency, 0., 200)
        a = artists[past]
        trans = x.getrow(cue).data
        q.extend((float(np.mean((a == artists[cue]) & (a >= 0))) if past else 0.,
                  float(cue in past), len(past) / 20., float(np.log1p(trans.sum())),
                  float(trans.max() / max(float(trans.sum()), 1e-8)) if len(trans) else 0.,
                  len(set(x_cache[cue]) & set(h_source)) / 200.))
        result[at] = q
        if (at + 1) % 2000 == 0:
            print(json.dumps({"event": "context_rows", "rows": at+1, "total": len(row_ids)}), flush=True)
    if not np.isfinite(result).all():
        raise ValueError("request_relation_nonfinite")
    return result


def build_context(args: argparse.Namespace) -> None:
    import torch
    from scipy import sparse
    from types import SimpleNamespace
    from run_arrive_full_rcce_control import parent_paths
    from run_music4all_candidate_support_a1 import digest, atomic_json, atomic_npz, load_base_data, load_standard_base
    from run_music4all_artist_component_b import load_metadata
    if args.seed not in SEEDS or args.surface not in ("train_cv", "validation"):
        raise ValueError("invalid_context_seed_or_surface")
    output = args.output / f"s{args.seed}" / args.surface
    if output.exists():
        raise FileExistsError(str(output))
    began = time.monotonic()
    torch.set_num_threads(1)
    project = args.project
    ds = bundle(project, args.dataset)
    parent = (Path(args.parent_root) if args.parent_root else project / ds["parent_root"]) / f"s{args.seed}"
    if not parent.exists():
        raise FileNotFoundError(f"parent_export_root_missing:{parent} (dataset={args.dataset})")
    export_path = parent / args.surface / "EXPORT_MANIFEST.json"
    em = json.loads(export_path.read_text())
    if em.get("status") != "pass" or em.get("seed") != args.seed or em.get("surface") != args.surface or em.get("test_accessed") is not False or em.get("limit") != 0:
        raise ValueError("context_export_identity")
    paths = parent_paths(project, args.seed, args.surface)
    for name in ("feature", "feature_manifest", "base_checkpoint", "base_manifest"):
        if digest(paths[name]) != em["bindings"][name]:
            raise ValueError(f"context_parent_hash:{name}")
    with np.load(paths["feature"]) as z:
        row_ids, users = z["row_ids"].copy(), z["users"].copy()
    if args.canary_rows:
        if args.context_probe_asset:
            with np.load(args.context_probe_asset) as z:
                row_ids, users = z["row_ids"][:args.canary_rows].copy(), z["users"][:args.canary_rows].copy()
        else:
            row_ids, users = row_ids[:args.canary_rows], users[:args.canary_rows]
    else:
        # Reuse exact exported rows, never regenerate candidate pools.
        mp = parent / args.surface / "metadata.npz"
        if digest(mp) != em["output_sha256"]["metadata.npz"]:
            raise ValueError("context_metadata_hash")
        with np.load(mp) as z:
            if not (np.array_equal(row_ids, z["row_ids"]) and np.array_equal(users, z["users"])):
                raise ValueError("context_metadata_alignment")
    base_args = SimpleNamespace(data_dir=Path(ds["data_dir"]),
        event_times=Path(ds["event_times"]),
        ssd4rec_source=project / "vendor/ssd4rec_official/ssd4rec.py",
        muse_counts=Path(ds["muse_counts"]),
        device="cpu", max_context=129, horizon=64)
    base_manifest = json.loads(paths["base_manifest"].read_text())
    prepared_binding = base_manifest["binding"].get("prepared_artifacts", base_manifest["binding"].get("inputs", {}))
    prepared_hashes = {}
    for name in ("events.npy", "sessions.npy", "track_identity.tsv", "train_frequencies.npy"):
        prepared_hashes[name] = digest(base_args.data_dir / name)
        if prepared_hashes[name] != prepared_binding.get(name):
            raise ValueError(f"context_prepared_input_hash:{name}")
    data = load_base_data(base_args, paths["base_manifest"])
    expected_split = 0 if args.surface == "train_cv" else 1
    if not np.all(data.sessions[row_ids, 5] == expected_split):
        raise ValueError("context_nonmatching_split")
    if not np.array_equal(data.sessions[row_ids, 0], users):
        raise ValueError("context_prepared_users_mismatch")
    model, _ = load_standard_base(base_args, data, checkpoint_path=paths["base_checkpoint"], manifest_path=paths["base_manifest"])
    items = model.item_embedding.weight.detach().numpy().astype(np.float32)
    del model
    index = Path(ds["index"])
    im = json.loads((index / "index_manifest.json").read_text())
    if digest(index / "index_manifest.json") != em["index_manifest_sha256"]:
        raise ValueError("context_index_binding")
    suffix = "fit" if args.surface == "train_cv" else "full"
    index_hashes = {}
    for name in (f"x_{suffix}.npz", f"h_{suffix}.npz", f"{suffix}_target_frequency.npy"):
        index_hashes[name] = digest(index / name)
        if index_hashes[name] != im["artifacts"][name]["sha256"]:
            raise ValueError(f"context_index_hash:{name}")
    x, h = [sparse.load_npz(index / f"{k}_{suffix}.npz").tocsr() for k in ("x", "h")]
    frequency = np.load(index / f"{suffix}_target_frequency.npy")
    content = Path(ds["content_dir"])
    for name, expected in em["content_sha256"].items():
        if digest(content / name) != expected:
            raise ValueError(f"context_content_hash:{name}")
    audio = np.load(Path(ds["content_audio"]), mmap_mode="r")
    genre = sparse.load_npz(Path(ds["content_genre"])).tocsr()
    tag_path = Path(ds["tags"]) if ds["tags"] else base_args.data_dir / "id_tags.csv"
    artist_path = Path(ds["information"]) if ds["information"] else Path(ds["artist_mapping"])
    fm = json.loads(paths["feature_manifest"].read_text())
    if ds["tags"] and (digest(tag_path) != fm.get("tags_sha256") or digest(artist_path) != fm.get("information_sha256")):
        raise ValueError("context_tag_or_artist_binding")
    from arrive_framework_datasets import metadata_args
    tags, artists, _ = load_metadata(metadata_args(ds, base_args.data_dir), len(items))
    setup_seconds = time.monotonic()-began
    relation = request_relations(data, row_ids, items, audio, genre, tags, artists, x, h, frequency)
    parity = None
    if args.context_probe_asset:
        with np.load(args.context_probe_asset) as z:
            if not np.array_equal(row_ids, z["row_ids"][:len(row_ids)]):
                raise ValueError("context_probe_rows")
            parity = bool(np.array_equal(relation, z["relation"][:len(row_ids)]))
            if not parity:
                raise ValueError(f"context_probe_relation_difference:{np.max(np.abs(relation-z['relation'][:len(row_ids)]))}")
    output.mkdir(parents=True)
    p = output / "REQUEST_CONTEXT.npz"
    atomic_npz(p, row_ids=row_ids, users=users, relation=relation)
    rec = {"schema": "arrive.request_context.v1", "seed": args.seed, "split": args.surface,
        "status": "smoke" if args.canary_rows else "pass", "rows": len(row_ids),
        "availability_audit": "pass", "exact_export_alignment": not bool(args.canary_rows),
        "uses_suffix_labels": False, "context_sha256": digest(p), "export_sha256": digest(export_path),
        "index_sha256": index_hashes, "parent_sha256": em["bindings"], "content_sha256": em["content_sha256"],
        "prepared_sha256": prepared_hashes,
        "tag_sha256": digest(tag_path), "artist_sha256": digest(artist_path),
        "producer_sha256": digest(Path(__file__)), "prior_probe_exact_parity": parity,
        "availability": "Historical events stop before cue; cue itself is known. Train-CV uses x_fit/h_fit excluding Train-CV targets; Validation uses Train-only x_full/h_full. No suffix labels accessed by feature function.",
        "setup_seconds": setup_seconds, "feature_seconds": time.monotonic()-began-setup_seconds}
    atomic_json(p.with_suffix(".json"), rec)
    print(json.dumps(rec), flush=True)


def with_request_context(features: np.ndarray, query_ids: np.ndarray,
                         context: np.ndarray | None, mode: str) -> np.ndarray:
    """Append observable request fields; no base scores or second learner."""
    if mode not in CONTEXT_MODES:
        raise ValueError("unknown_request_context_mode")
    block = np.asarray(features, dtype=np.float32)
    if block.ndim != 2 or block.shape[1] != 28 or len(query_ids) != len(block):
        raise ValueError("invalid_candidate_feature_alignment")
    if mode == "base":
        return block
    if context is None or context.ndim != 2 or context.shape[1] != 18:
        raise ValueError("missing_18_request_fields")
    if not np.isfinite(context).all():
        raise ValueError("nonfinite_request_context")
    if len(query_ids) and (query_ids.min() < 0 or query_ids.max() >= len(context)):
        raise ValueError("invalid_request_context_index")
    columns = slice(14, 17) if mode == "statistics" else slice(None)
    return np.concatenate((block, context[query_ids, columns].astype(np.float32)), axis=1)


def load_request_context(root: Path | None, parent: Path, meta: dict,
                         seed: int, split: str, mode: str):
    from run_music4all_candidate_support_a1 import digest
    if mode == "base":
        return None, None
    if root is None:
        raise ValueError("request_context_root_required")
    path = root / f"s{seed}" / split / "REQUEST_CONTEXT.npz"
    receipt_path = path.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    expected = {"schema": "arrive.request_context.v1", "seed": seed, "split": split,
                "availability_audit": "pass", "exact_export_alignment": True,
                "uses_suffix_labels": False, "context_sha256": digest(path),
                "export_sha256": digest(parent / split / "EXPORT_MANIFEST.json")}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("request_context_binding_or_availability_failed")
    with np.load(path, allow_pickle=False) as z:
        if not (np.array_equal(z["row_ids"], meta["row_ids"]) and
                np.array_equal(z["users"], meta["users"])):
            raise ValueError("request_context_row_order_mismatch")
        context = np.asarray(z["relation"], dtype=np.float32)
    if context.shape != (len(meta["row_ids"]), 18) or not np.isfinite(context).all():
        raise ValueError("request_context_shape_or_finiteness")
    return context, {"receipt_sha256": digest(receipt_path), **expected}


def selected_rows(meta: dict, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = meta["offsets"]
    positions = np.concatenate([np.arange(int(offsets[r]), int(offsets[r + 1]), dtype=np.int64) for r in rows])
    group_id = np.repeat(np.arange(len(rows), dtype=np.int32), np.diff(offsets)[rows])
    return positions, group_id


def fit_pool(features, labels, meta, rows, context, mode, threads, chunk=65536,
             cue_match_mode="true", cue_treatment=None, candidate_ids=None):
    """One column-major input; bounded gathers, identical query/candidate order."""
    from catboost import Pool
    widths = np.diff(meta["offsets"])[rows]
    offsets = np.r_[0, np.cumsum(widths)]
    count = int(offsets[-1])
    columns = {"base": 28, "statistics": 31, "relations": 46}[mode]
    if chunk <= 0 or not count:
        raise ValueError("empty_fit_pool_or_invalid_chunk")
    if mode != "base" and (context is None or context.shape != (len(meta["offsets"])-1,18)
                            or not np.isfinite(context).all()):
        raise ValueError("invalid_fit_context")
    values = np.empty((count, columns), dtype=np.float32, order="F")
    targets = np.empty(count, dtype=np.int32)
    groups = np.empty(count, dtype=np.int32)
    for lo in range(0, count, chunk):
        hi = min(lo + chunk, count)
        local = np.arange(lo, hi, dtype=np.int64)
        query = np.searchsorted(offsets, local, side="right") - 1
        positions = meta["offsets"][rows[query]] + local - offsets[query]
        values[lo:hi, :28] = features[positions]
        if mode != "base":
            cols = slice(14,17) if mode == "statistics" else slice(None)
            values[lo:hi, 28:] = context[rows[query], cols]
        apply_cue_match(values[lo:hi], cue_match_mode, cue_treatment,
                        candidate_ids[positions] if candidate_ids is not None else None, rows[query])
        targets[lo:hi] = labels[positions]
        groups[lo:hi] = query
    print(json.dumps({"event":"fit_input_ready", "candidates":count, "features":columns,
                      "feature_bytes":values.nbytes, "layout":"F", "gather_chunk":chunk}), flush=True)
    return Pool(values, targets, group_id=groups, thread_count=threads)


def top10(model: CatBoostRanker, features: np.ndarray, ids: np.ndarray,
          meta: dict, rows: np.ndarray, trees: int,
          context: np.ndarray | None = None, context_mode: str = "base", threads: int = -1,
          batch_rows: int = 256, cue_match_mode: str = "true", cue_treatment=None) -> np.ndarray:
    from run_music4all_candidate_support_a1 import stable_topk

    offsets = meta["offsets"]
    source = meta["rrf_indices"]
    if batch_rows <= 0:
        raise ValueError("invalid_prediction_batch")
    result = np.empty((len(rows),10),dtype=np.int32)
    for lo in range(0,len(rows),batch_rows):
        current = rows[lo:lo+batch_rows]
        flat = (offsets[current,None] + source[current]).reshape(-1)
        qids = np.repeat(current,source.shape[1])
        block = with_request_context(np.asarray(features[flat]),qids,context,context_mode)
        apply_cue_match(block, cue_match_mode, cue_treatment, ids[flat], qids)
        scores = np.asarray(model.predict(block,ntree_end=trees,thread_count=threads),dtype=np.float64)
        chosen_ids = np.asarray(ids[flat],dtype=np.int32).reshape(len(current),-1)
        result[lo:lo+len(current)] = stable_topk(scores.reshape(len(current),-1),chosen_ids,10)
    return result


def check_train_cv_rows(meta: dict, expected: int | None) -> None:
    """Verify the prepared cohort; never silently resample a larger export."""
    if expected is None:
        return
    rows = meta["row_ids"]
    if not 0 < expected <= 20000 or len(rows) != expected or len(np.unique(rows)) != expected:
        raise ValueError("train_cv_row_contract_mismatch")


def train(args: argparse.Namespace) -> None:
    import catboost
    from catboost import CatBoostRanker, Pool

    from run_music4all_candidate_support_a1 import atomic_json, atomic_npz, digest, user_partition
    from run_music4all_component_c2_conversion_oof import row_metrics, user_macro

    if args.seed not in SEEDS or args.surface != "validation":
        raise ValueError("ranker_training_rejects_test; validation_only")
    expected_rows = getattr(args, "expected_train_rows", None)
    cue_match_mode = getattr(args, "cue_match_mode", "true")
    if cue_match_mode not in CUE_MATCH_MODES or (cue_match_mode != "true" and args.feature_mode != "statistics"):
        raise ValueError("cue_match_ablation_requires_statistics31")
    if expected_rows is not None and getattr(args, "full_train", False):
        if (args.dataset != "yambda500m" or expected_rows != 17370
                or cue_match_mode not in ("none","wrong") or not getattr(args, "source_code_bindings", None)):
            raise ValueError("bounded_train_cv_conflicts_with_full_train")
    if args.dataset not in DATASETS:
        raise ValueError(f"unknown_dataset:{args.dataset}")
    if not args.parent_root:
        raise ValueError("parent_root_required_for_full_train_folds")
    output = args.output / f"s{args.seed}"
    if output.exists():
        raise FileExistsError(f"existing_ranker_output:{output}")
    parent = Path(args.parent_root) / f"s{args.seed}"
    full_train = getattr(args,"full_train",False)
    extra = {}
    print(json.dumps({"event":"input_verification_start","seed":args.seed}),flush=True)
    if full_train:
        from arrive_train_fold_coverage import validate_coverage
        from arrive_framework_datasets import check_export, feature_schema, code_bindings
        if args.feature_mode != "statistics":
            raise ValueError("final_framework_requires_statistics31")
        compatibility = None
        if getattr(args,"source_code_bindings",None):
            compatibility = json.loads(args.source_code_bindings.read_text())
        if expected_rows is not None and (compatibility or {}).get("schema") != "arrive.matching_ablation_reader.v1":
            raise ValueError("bounded_oof_requires_matching_ablation_binding")
        if expected_rows is not None and compatibility.get('cue_match_mode')!=cue_match_mode:
            raise ValueError('matching_ablation_treatment_binding')
        coverage, exported = validate_coverage(parent/"train_cv",args.project,args.dataset,args.seed,
                                              compatibility=compatibility)
        x, train_ids, y, train_meta, train_manifest, context = exported
        vx, vids, _, val_meta, val_manifest, val_context = check_export(parent/"validation",args.dataset,args.seed,"validation",
            expected_code=coverage.get("source_code_sha256"))
        if val_manifest["serving_inputs"]["base_manifest"] != coverage["full_base_manifest_sha256"]:
            raise ValueError("validation_full_parent_mismatch")
        from arrive_framework_datasets import bundle, hashes
        from evaluate_arrive_yeti31_fusion_test import validate_cohort
        from lfm_timeconcat_control import LFMTimeConcatData
        prepared=Path(bundle(args.project,args.dataset)["data_dir"])
        if hashes({n:prepared/n for n in train_manifest["prepared_sha256"]})!=val_manifest["prepared_sha256"] or train_manifest["prepared_sha256"]!=val_manifest["prepared_sha256"]:
            raise ValueError("validation_prepared_binding")
        with np.load(prepared/"metric_labels.npz",allow_pickle=False) as z:
            labels={k:z[k] for k in z.files}
        validate_cohort(LFMTimeConcatData(prepared),labels,val_meta,split=1)
        context_receipt = val_context_receipt = None
        extra={"feature_schema":feature_schema(),"code_sha256":code_bindings(),
               "coverage_sha256":digest(parent/"train_cv/COVERAGE.json"),
               "serving_inputs":val_manifest["serving_inputs"],"eligible_train_rows":len(train_meta["row_ids"])}
        if compatibility is not None:
            binding_key = ("matching_ablation_reader_sha256" if compatibility.get("schema") ==
                           "arrive.matching_ablation_reader.v1" else "resource_upgrade_sha256")
            extra[binding_key] = digest(args.source_code_bindings)
    else:
        from run_arrive_full_rcce_control import load_export
        x, train_ids, y, train_meta, train_manifest = load_export(parent / "train_cv", args.seed, "train_cv")
        vx, vids, _, val_meta, val_manifest = load_export(parent / "validation", args.seed, "validation")
        context, context_receipt = load_request_context(args.request_context_root, parent, train_meta,
                                                       args.seed, "train_cv", args.feature_mode)
        val_context, val_context_receipt = load_request_context(args.request_context_root, parent, val_meta,
                                                               args.seed, "validation", args.feature_mode)
    check_train_cv_rows(train_meta, expected_rows)
    if train_manifest.get("limit") != 0 or val_manifest.get("limit") != 0:
        raise ValueError("limited_parent_export_forbidden")
    reference_manifest = json.loads((parent / "training_manifest.json").read_text()) if not full_train else {}
    if not full_train and (reference_manifest.get("status") != "pass" or reference_manifest.get("seed") != args.seed
            or reference_manifest.get("test_accessed") is not False
            or reference_manifest.get("train_export_sha256") != digest(parent / "train_cv/EXPORT_MANIFEST.json")
            or reference_manifest.get("validation_export_sha256") != digest(parent / "validation/EXPORT_MANIFEST.json")):
        raise ValueError("reference_manifest_binding_invalid")
    fit_partition = user_partition(train_meta["users"])
    positive = np.add.reduceat(np.asarray(y, dtype=np.int32), train_meta["offsets"][:-1]) > 0
    fit_rows = np.flatnonzero(fit_partition & positive)
    calib_rows = np.flatnonzero(~fit_partition)
    if args.canary_rows:
        fit_rows = fit_rows[:args.canary_rows]
        calib_rows = calib_rows[:min(args.canary_rows, len(calib_rows))]
    if not len(fit_rows) or not len(calib_rows):
        raise ValueError("empty_fit_or_calibration")
    if set(train_meta["users"][fit_rows].tolist()) & set(train_meta["users"][calib_rows].tolist()):
        raise ValueError("fit_calibration_user_overlap")
    train_treatment=val_treatment=donor_report=donor_mappings=None
    if cue_match_mode=="wrong":
        from arrive_cue_donors import build_treatments
        train_treatment,val_treatment,donor_report,donor_mappings=build_treatments(
            args.project,args.dataset,args.seed,train_meta,val_meta,x,vx,
            train_manifest,val_manifest,args.request_context_root)
    iterations = (5, 10) if args.canary_rows else ITERATIONS
    model = CatBoostRanker(
        iterations=max(iterations), depth=6, learning_rate=0.035,
        loss_function=LOSS, random_seed=args.seed + 1000, thread_count=args.threads,
        task_type="CPU", verbose=50, allow_writing_files=False,
    )
    began = time.monotonic()
    print(json.dumps({"event":"fit_input_start","seed":args.seed,"queries":len(fit_rows),
                      "full_train":full_train and not args.canary_rows,
                      "eligible_train_rows":len(train_meta["row_ids"]),
                      "coverage_sha256":extra.get("coverage_sha256")}),flush=True)
    pool = fit_pool(x,y,train_meta,fit_rows,context,args.feature_mode,args.threads,
                    cue_match_mode=cue_match_mode,cue_treatment=train_treatment,candidate_ids=train_ids)
    model.fit(pool)
    fit_seconds = time.monotonic() - began
    del pool
    print(json.dumps({"event":"fit_complete","seed":args.seed,"seconds":fit_seconds}),flush=True)
    records = []
    for trees in iterations:
        ranked = top10(model, x, train_ids, train_meta, calib_rows, trees,
                       context, args.feature_mode, threads=args.threads, cue_match_mode=cue_match_mode,
                       cue_treatment=train_treatment)
        _, ndcg = row_metrics(ranked, train_meta["targets"][calib_rows], train_meta["target_counts"][calib_rows])
        score = user_macro(ndcg, train_meta["users"][calib_rows], np.ones(len(calib_rows), bool))
        records.append({"trees": trees, "train_calibration_rrf_ndcg": float(score)})
    selected = max(records, key=lambda z: (z["train_calibration_rrf_ndcg"], -z["trees"]))["trees"]
    rows = np.arange(min(args.canary_rows, len(val_meta["row_ids"]))) if args.canary_rows else np.arange(len(val_meta["row_ids"]))
    began = time.monotonic()
    ranked = top10(model, vx, vids, val_meta, rows, selected, val_context, args.feature_mode,
                   threads=args.threads, cue_match_mode=cue_match_mode,cue_treatment=val_treatment)
    score_seconds = time.monotonic() - began
    recall, ndcg = row_metrics(ranked, val_meta["targets"][rows], val_meta["target_counts"][rows])
    output.mkdir(parents=True)
    if donor_report is not None:
        donor_report['matching_distribution']={'train_processed':train_treatment.distribution(),
                                              'validation_processed':val_treatment.distribution()}
        atomic_npz(output/"CUE_DONORS.npz",**donor_mappings)
        donor_report['mapping_file_sha256']=digest(output/"CUE_DONORS.npz")
        atomic_json(output/"CUE_DONORS.json",donor_report)
        extra['cue_donor_receipt_sha256']=digest(output/"CUE_DONORS.json")
        extra['cue_control_policy']=donor_report['unknown_cue_policy']
    model_path = output / "yeti_ndcg.cbm"
    model.save_model(str(model_path))
    prediction_path = output / "validation_predictions.npz"
    atomic_npz(prediction_path, row_ids=val_meta["row_ids"][rows], users=val_meta["users"][rows],
               bucket_ids=val_meta["bucket_ids"][rows], target_counts=val_meta["target_counts"][rows],
               targets=val_meta["targets"][rows], top10=ranked, recall=recall, ndcg=ndcg)
    prediction_receipt = {
        "schema": "arrive.recent_ranker.validation_receipt.v1",
        "rows": len(rows), "users": int(len(set(val_meta["users"][rows].tolist()))),
        "feature_mode": args.feature_mode, "test_accessed": False,
        "cue_match_mode": cue_match_mode,
        "prediction_sha256": digest(prediction_path),
    }
    receipt_path = output / "validation_predictions.npz.json"
    atomic_json(receipt_path, prediction_receipt)
    atomic_json(output / "training_manifest.json", {
        "schema": "arrive.recent_ranker.v1", "status": "canary" if args.canary_rows else "pass",
        "dataset": args.dataset, "surface": "validation", "parent_root": str(args.parent_root),
        "method": "CatBoost_YetiRank_metric_specific_NDCG", "canary_rows": args.canary_rows,
        "feature_mode": args.feature_mode, "feature_count": {"base": 28, "statistics": 31, "relations": 46}[args.feature_mode],
        "cue_match_mode": cue_match_mode,
        "cue_match_columns_zero_based": list(CUE_MATCH_COLUMNS),
        "cue_match_changed_columns": list(CUE_MATCH_COLUMNS) if cue_match_mode != "true" else [],
        "cue_availability_columns_preserved": [19,21,24,27],
        "train_request_context": context_receipt, "validation_request_context": val_context_receipt,
        "catboost_version": catboost.__version__, "threads": args.threads,
        "seed": args.seed, "test_accessed": False, "loss_function": LOSS,
        "selected_trees": selected, "calibration": records,
        "fit_rows": len(fit_rows), "calibration_rows": len(calib_rows), "validation_rows": len(rows),
        "train_cv_rows": len(train_meta["row_ids"]), "expected_train_rows": expected_rows,
        "zero_positive_train_rows": int((~positive).sum()),
        "fit_partition_zero_positive_rows": int((fit_partition & ~positive).sum()),
        "fit_seconds": fit_seconds, "score_seconds": score_seconds,
        "model_bytes": model_path.stat().st_size,
        "train_export_sha256": digest(parent / "train_cv/EXPORT_MANIFEST.json"),
        "validation_export_sha256": digest(parent / "validation/EXPORT_MANIFEST.json"),
        "reference_manifest_sha256": digest(parent / "training_manifest.json") if not full_train else None,
        "model_sha256": digest(model_path),
        "prediction_sha256": digest(prediction_path),
        "prediction_receipt_sha256": digest(receipt_path),
        "selected_feature_columns": "base28+context[14:17] (statistics)",
        "overall_recall": user_macro(recall, val_meta["users"][rows], np.ones(len(rows), bool)),
        "overall_ndcg": user_macro(ndcg, val_meta["users"][rows], np.ones(len(rows), bool)),
        "producer_sha256": digest(Path(__file__)), **extra,
    })
    print(json.dumps({"event": "recent_ranker_complete", "seed": args.seed,
                      "selected_trees": selected, "fit_seconds": fit_seconds,
                      "score_seconds": score_seconds}), flush=True)


def smoke() -> None:
    from catboost import CatBoostRanker, Pool

    from run_music4all_component_c2_conversion_oof import row_metrics

    rng = np.random.default_rng(7)
    x = rng.normal(size=(60, 28)).astype(np.float32)
    y = np.tile(np.asarray([1] + [0] * 11, dtype=np.int32), 5)
    group_id = np.repeat(np.arange(5), 12)
    model = CatBoostRanker(iterations=3, depth=3, learning_rate=.05, loss_function=LOSS,
                           random_seed=2024, thread_count=2, task_type="CPU", verbose=False,
                           allow_writing_files=False)
    model.fit(Pool(x, y, group_id=group_id))
    meta = {"offsets": np.arange(0, 61, 12),
            "rrf_indices": np.tile(np.arange(12, dtype=np.int32), (5, 1))}
    ids = np.arange(3, 63, dtype=np.int32)
    fit_positions, fit_groups = selected_rows(meta, np.arange(5))
    if (not np.array_equal(fit_positions, np.arange(60))
            or not np.array_equal(fit_groups, group_id)):
        raise ValueError("smoke_group_order_invalid")
    ranked = top10(model, x, ids, meta, np.arange(5), 3)
    if ranked.shape != (5, 10):
        raise ValueError("smoke_topk_shape_invalid")
    targets = ids[np.arange(0, 60, 12)][:, None]
    recall, ndcg = row_metrics(ranked, targets, np.ones(5, dtype=np.int32))
    if not (np.isfinite(recall).all() and np.isfinite(ndcg).all()):
        raise ValueError("smoke_eval_nonfinite")
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as d:
        path = Path(d) / "smoke.cbm"
        model.save_model(str(path))
        reloaded = CatBoostRanker()
        reloaded.load_model(str(path))
        if not np.array_equal(model.predict(x), reloaded.predict(x)):
            raise ValueError("save_load_prediction_mismatch")
    print(json.dumps({"event": "local_smoke_pass", "loss": LOSS, "groups": 5}), flush=True)


def context_smoke(args: argparse.Namespace) -> None:
    """Train-row plumbing check only; cannot report held-out recommendation gains."""
    from catboost import CatBoostRanker, Pool
    from run_music4all_candidate_support_a1 import digest, atomic_json
    from tempfile import TemporaryDirectory
    if args.context_probe_asset is None:
        raise ValueError("context_probe_asset_required")
    path = args.context_probe_asset
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt["sha256"] != digest(path) or receipt.get("validation_accessed") is not False:
        raise ValueError("probe_asset_hash_or_split")
    with np.load(path) as z:
        x, q = z["features"][:96], z["relation"][:96]
    n, width, _ = x.shape
    groups = np.repeat(np.arange(n, dtype=np.int32), width)
    # Artificial labels suffice for plumbing; do not inspect/re-score heldout targets.
    labels = np.tile(np.r_[1, np.zeros(width - 1)], n)
    rows = []
    for mode in CONTEXT_MODES:
        block = with_request_context(x.reshape(-1, 28), groups, q, mode)
        if mode == "base" and not np.array_equal(block, x.reshape(-1, 28)):
            raise ValueError("default_base_changed")
        began = time.monotonic()
        model = CatBoostRanker(iterations=3, depth=6, learning_rate=.035, loss_function=LOSS,
                              random_seed=3024, thread_count=2, verbose=False, allow_writing_files=False)
        model.fit(Pool(block, labels, group_id=groups))
        score = model.predict(block)
        with TemporaryDirectory() as d:
            p = Path(d) / "ranker.cbm"
            model.save_model(str(p))
            reload = CatBoostRanker()
            reload.load_model(str(p))
            if not np.array_equal(score, reload.predict(block)):
                raise ValueError("context_reload_parity")
        rows.append({"mode": mode, "features": block.shape[1], "groups": n,
                     "seconds": time.monotonic()-began, "reload_parity": True})
    result = {"status": "smoke_pass", "results": rows, "artificial_labels": True,
              "recommendation_effect_measured": False, "validation_evaluated": False, "test_evaluated": False,
              "asset_sha256": digest(path), "producer_sha256": digest(Path(__file__))}
    if args.smoke_receipt:
        atomic_json(args.smoke_receipt, result)
    print(json.dumps(result), flush=True)


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1].startswith("history_"):
        from arrive_history_reader import main as history_main
        history_main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "context_smoke", "context_build", "ilmart_smoke", "ilmart_train"))
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--dataset", choices=DATASETS, default="music4all_onion")
    parser.add_argument("--parent-root", type=Path,
                       help="Yeti31 parent EXPORT root; required for --dataset yambda500m and for full-Train fold exports")
    parser.add_argument("--output", type=Path, default=Path("runs/arrive_recent_ranker_20260924/onion"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--surface", default="validation")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--canary-rows", type=int, default=0)
    parser.add_argument("--full-train", action="store_true", help="Require complete user-disjoint OOF coverage")
    parser.add_argument("--expected-train-rows", type=int,
                        help="Exact prepared Train-CV cohort size (at most20000); rejects drift, never truncates")
    parser.add_argument("--source-code-bindings",type=Path,
                        help="Exact reviewed producer/reader bindings for the resource-only upgrade")
    parser.add_argument("--feature-mode", choices=CONTEXT_MODES, default="base")
    parser.add_argument("--cue-match-mode", choices=CUE_MATCH_MODES, default="true",
                        help="none: zero only four cue-candidate matching columns in fit/calibration/Validation; keeps31 inputs")
    parser.add_argument("--request-context-root", type=Path)
    parser.add_argument("--context-probe-asset", type=Path)
    parser.add_argument("--smoke-receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "context_build":
        build_context(args)
    elif args.mode == "context_smoke":
        context_smoke(args)
    elif args.mode.startswith("ilmart_"):
        from arrive_ilmart_backend import smoke as ilmart_smoke, train as ilmart_train
        ilmart_smoke() if args.mode == "ilmart_smoke" else ilmart_train(args)
    else:
        smoke() if args.mode == "smoke" else train(args)


if __name__ == "__main__":
    main()
