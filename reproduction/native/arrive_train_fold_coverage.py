#!/usr/bin/env python3
"""Full-Train OOF: plan, inventory, run one user-disjoint fold, merge exports.

run-fold reuses the existing index builder, fixed-epoch crossfit trainer and
shared exporter. It never starts Test. Existing fold outputs are preserved;
failed stages require inspection and the underlying runner's explicit resume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Measured SASRec-family per-epoch minutes on the shared cluster
# (handoff 14 / main-table runs 2026-09-18; 15-epoch formal runs 4-8h).
PER_EPOCH_MINUTES = {
    "music4all_onion": {"nominal": 24.0, "measured_range": [16.0, 32.0],
                        "provenance": "onion backbone formal 2026-09-18 (15 epochs, 4-8h)"},
    "yambda500m": {"nominal": 10.5, "measured_range": [10.0, 11.0],
                   "provenance": "yambda backbone formal 2026-09-18 (logs 10-11 min/epoch)"},
}


def deny_traversal(raw: Path, label: str) -> Path:
    # Check the RAW path first: resolve() collapses ".." and would silence
    # the traversal check (round-1 lesson).
    if ".." in Path(raw).parts:
        raise ValueError(f"{label}_traversal_denied:{raw}")
    return raw.expanduser().resolve()


def load_train_sessions(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Row indices and user ids of the eligible Train cohort (split column == 0)."""
    from lfm_timeconcat_control import LFMTimeConcatData
    data = LFMTimeConcatData(data_dir)
    rows = data.row_ids(0)
    users = np.asarray(data.sessions[rows, 0], dtype=np.int64)
    return rows, users


def fold_of_rows(data_dir: Path, folds: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic user-disjoint fold assignment for the eligible Train rows.

    Returns (rows, users, fold_of_row). Users are shuffled with a seeded RNG
    and split into contiguous K groups; every episode inherits its user's
    fold, so a user's whole history stays inside one fold.
    """
    if folds < 2:
        raise ValueError("fold_count_must_be_at_least_2")
    rows, users = load_train_sessions(data_dir)
    unique = np.unique(users)
    if len(unique) < folds:
        raise ValueError("fold_count_exceeds_user_count")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    fold_of_user = np.zeros(len(unique), dtype=np.int32)
    boundaries = np.linspace(0, len(unique), folds + 1).astype(int)
    user_index = {u: i for i, u in enumerate(unique)}
    for k in range(folds):
        fold_of_user[order[boundaries[k]:boundaries[k + 1]]] = k
    fold_of_row = np.fromiter((fold_of_user[user_index[u]] for u in users),
                              dtype=np.int32, count=len(users))
    return rows, users, fold_of_row


def run_fold(args):
    """Reuse the existing index, crossfit trainer and feature exporter, in order."""
    import subprocess
    import sys
    from arrive_framework_datasets import bundle
    ds = bundle(args.project, args.dataset)
    rows, _, assignments = fold_of_rows(Path(ds["data_dir"]), args.folds, args.assignment_seed)
    if not 0 <= args.fold < args.folds:
        raise ValueError("invalid_fold")
    root = args.output / f"fold{args.fold}"
    if root.exists():
        raise FileExistsError(root)
    full = json.loads(args.training_manifest.read_text())
    cfg = full["binding"]["configuration"]
    from run_music4all_candidate_support_a1 import base_method
    if cfg["seed"] != args.seed or full.get("status") != "pass" or base_method(full) not in ("sasrec", "ssd4rec"):
        raise ValueError("fold_requires_frozen_supported_seed")
    root.mkdir(parents=True)
    np.save(root / "excluded_rows.npy", rows[assignments == args.fold], allow_pickle=False)
    tools = Path(__file__).resolve().parent
    common = ["--data-dir", ds["data_dir"], "--manifest", str(args.prepared_manifest),
              "--index-dir", str(root / "index"), "--seed", str(args.seed), "--threads", str(args.threads)]
    commands = [
        [sys.executable, str(tools/"run_music4all_candidate_support_a0.py"), "build-index", *common,
         "--cv-rows", str(root/"excluded_rows.npy")],
        [sys.executable, str(tools/"run_music4all_candidate_support_a1.py"), "train-crossfit-base", *common,
         "--full-base-manifest", str(args.training_manifest), "--output", str(root/"base"),
         "--device", args.device, "--epochs", str(full["best_epoch"]),
         "--event-times", ds["event_times"],
         "--ssd4rec-source", str(args.project/"vendor/ssd4rec_official/ssd4rec.py"),
         "--final-checkpoint-only",
         *[v for k in ("batch_size", "learning_rate", "negative_samples", "max_context")
           for v in ("--"+k.replace("_","-"), str(cfg[k]))]],
        [sys.executable, str(tools/"run_arrive_full_rcce_control.py"), "framework_export",
         "--project", str(args.project), "--dataset", args.dataset, "--seed", str(args.seed),
         "--source-policy", getattr(args, "source_policy", "full"),
         "--surface", "train_cv", "--index-dir", str(root/"index"),
         "--base-checkpoint", str(root/"base/final_model.pt"),
         "--base-manifest", str(root/"base/training_manifest.json"),
         "--output", str(root/"export"), "--device", args.device, "--threads", str(args.threads)]]
    from run_music4all_candidate_support_a1 import atomic_json
    atomic_json(root/"COMMANDS.json", commands)
    for number, command in enumerate(commands):
        with (root/f"step{number}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    atomic_json(root/"EXIT.json", {"exit_code":0,"steps":len(commands)})


def checked_source_code(source, compatibility=None):
    """Allow the reviewed storage-reader upgrade, never a changed scientific producer."""
    from arrive_framework_datasets import code_bindings
    current = code_bindings()
    if compatibility is not None:
        if compatibility.get("schema") == "arrive.matching_ablation_reader.v1":
            allowed = {"arrive_train_fold_coverage.py", "run_arrive_recent_ranker.py"}
            prior = [compatibility.get(k, {}) for k in ("producer_code_sha256", "reader_code_sha256")]
            if (compatibility.get("code_sha256") != current or source not in prior
                    or compatibility.get("dataset") != "yambda500m"
                    or compatibility.get("expected_train_rows") != 17370
                    or compatibility.get("cue_match_mode") not in ("none","wrong")
                    or compatibility.get("changed_feature_columns") != [12, 13, 14, 16]
                    or any(set(p) != set(current)
                           or any(p[n] != h for n, h in current.items() if n not in allowed)
                           or any(not isinstance(h, str) or len(h) != 64
                                  or any(c not in "0123456789abcdef" for c in h)
                                  for h in p.values()) for p in prior)):
                raise ValueError("matching_ablation_reader_code_binding")
            if compatibility.get("cue_match_mode")=="wrong":
                from arrive_framework_datasets import file_hash
                import arrive_cue_donors
                if compatibility.get("cue_donor_helper_sha256")!=file_hash(Path(arrive_cue_donors.__file__)):
                    raise ValueError("wrong_cue_helper_code_binding")
            return source
        allowed = {"arrive_train_fold_coverage.py", "arrive_framework_datasets.py",
                   "run_arrive_recent_ranker.py", "run_arrive_full_rcce_control.py"}
        prior = [compatibility.get(k,{}) for k in ("producer_code_sha256","reader_code_sha256")]
        if (compatibility.get("schema") != "arrive.resource_upgrade.v1"
                or compatibility.get("code_sha256") != current or source not in prior
                or any(set(p)!=set(current) or any(p[n]!=h for n,h in current.items() if n not in allowed)
                       or any(not isinstance(h,str) or len(h)!=64 or any(c not in "0123456789abcdef" for c in h)
                              for h in p.values()) for p in prior)):
            raise ValueError("resource_upgrade_code_binding")
        return source
    reader_files = {"arrive_train_fold_coverage.py", "arrive_framework_datasets.py",
                    "run_arrive_recent_ranker.py"}
    if (set(source) != set(current) or
            any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
                for h in source.values()) or
            any(source[n] != h for n, h in current.items() if n not in reader_files)):
        raise ValueError("unreviewed_source_code_change")
    return source


class FoldArray:
    """Read a global candidate order from immutable fold memmaps, without a disk copy.

    Only first-axis integer indexing/slicing is needed by the existing ranker.
    Gather in bounded chunks so a full-Train index does not create several
    additional full-size routing arrays.
    """
    def __init__(self, arrays, offsets, fold_ids, starts):
        self.arrays, self.offsets = arrays, offsets
        self.fold_ids, self.starts = fold_ids, starts
        self.dtype = arrays[0].dtype
        self.shape = (int(offsets[-1]), *arrays[0].shape[1:])
        self.ndim = len(self.shape)
        if any(a.dtype != self.dtype or a.shape[1:] != self.shape[1:] for a in arrays):
            raise ValueError("inconsistent_fold_array_schema")

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, slice):
            key = np.arange(*key.indices(len(self)), dtype=np.int64)
        index = np.asarray(key)
        if index.dtype.kind not in "iu":
            raise TypeError("fold_array_requires_integer_index")
        flat = index.reshape(-1)
        result = np.empty((len(flat), *self.shape[1:]), dtype=self.dtype)
        for lo in range(0, len(flat), 65536):
            if flat.dtype.kind == "u" and np.any(flat[lo:lo+65536] >= len(self)):
                raise IndexError("fold_array_index_out_of_bounds")
            pos = flat[lo:lo+65536].astype(np.int64, copy=True)
            pos[pos < 0] += len(self)
            if np.any(pos < 0) or np.any(pos >= len(self)):
                raise IndexError("fold_array_index_out_of_bounds")
            query = np.searchsorted(self.offsets, pos, side="right") - 1
            fold = self.fold_ids[query]
            local = self.starts[query] + pos - self.offsets[query]
            block = result[lo:lo+len(pos)]
            for k, array in enumerate(self.arrays):
                take = fold == k
                block[take] = array[local[take]]
        return result.reshape((*index.shape, *self.shape[1:]))

    def __array__(self, dtype=None, copy=None):
        # Only the small uint8 labels are materialized by the training consumer.
        if self.ndim != 1:
            raise ValueError("use_indexed_feature_reads_not_full_materialization")
        result = np.empty(self.shape, dtype=self.dtype if dtype is None else dtype)
        for lo in range(0, len(self), 65536):
            result[lo:lo+65536] = self[lo:min(lo+65536,len(self))]
        return result


def fold_exports(project, dataset, seed, fold_root, folds, assignment_seed, source_code=None,
                 expected_policy=None):
    from arrive_framework_datasets import bundle, check_export, hashes, source_policy, file_hash as digest
    data_dir = Path(bundle(project,dataset)["data_dir"])
    actual_prepared=hashes({n:data_dir/n for n in ("events.npy","sessions.npy","train_frequencies.npy","track_identity.tsv","metric_labels.npz","PREP_STATS.json")})
    rows, users, assignment = fold_of_rows(data_dir,folds,assignment_seed)
    exports=[]; bindings={}; full_hashes=set()
    for k in range(folds):
        root=fold_root/f"fold{k}"
        part=check_export(root/"export",dataset,seed,"train_cv",expected_code=source_code)
        meta, em=part[3],part[4]
        policy = source_policy(em)
        if expected_policy is None:
            expected_policy = policy
        elif policy != expected_policy:
            raise ValueError("fold_source_policy_mismatch")
        if not (np.array_equal(meta["row_ids"],rows[assignment==k]) and np.array_equal(meta["users"],users[assignment==k])):
            raise ValueError("fold_export_coverage_identity")
        base_path=root/"base/training_manifest.json"; index_path=root/"index/index_manifest.json"
        bm=json.loads(base_path.read_text()); im=json.loads(index_path.read_text())
        excluded=root/"index/cv_rows.npy"
        if (bm.get("status")!="pass" or bm.get("test_accessed") is not False
                or bm["binding"]["configuration"]["seed"]!=seed
                or any(bm["binding"].get("prepared_artifacts",{}).get(n)!=h or im.get("prepared_artifacts",{}).get(n)!=h for n,h in actual_prepared.items())
                or bm.get("excluded_cv_rows")!=len(meta["row_ids"])
                or bm.get("train_rows")!=len(rows)-len(meta["row_ids"])
                or bm["binding"].get("cv_rows_sha256")!=digest(excluded)
                or bm["binding"].get("index_manifest_sha256")!=digest(index_path)
                or em["serving_inputs"]["base_manifest"]!=digest(base_path)
                or em["serving_inputs"]["base_checkpoint"]!=digest(root/"base/final_model.pt")
                or bm.get("final_model_sha256")!=digest(root/"base/final_model.pt")
                or not np.array_equal(np.load(excluded),meta["row_ids"])):
            raise ValueError("fold_carrier_provenance")
        for name,entry in im["artifacts"].items():
            if digest(root/"index"/name)!=entry["sha256"]:
                raise ValueError("fold_index_hash")
        full_hashes.add(bm["binding"]["full_base_manifest_sha256"])
        bindings[str(k)]={"export":digest(root/"export/EXPORT_MANIFEST.json"),
                           "base":digest(base_path),"index":digest(index_path)}
        exports.append(part)
    if len(full_hashes)!=1:
        raise ValueError("fold_full_parent_mismatch")
    return rows,users,assignment,exports,bindings,full_hashes.pop()


def merge_folds(args):
    from arrive_framework_datasets import EXPORT_FILES, code_bindings, feature_schema, source_policy
    from run_music4all_candidate_support_a1 import atomic_json, atomic_npz, digest
    if args.output.exists():
        raise FileExistsError(args.output)
    source_path = getattr(args, "source_code_bindings", None)
    source_code = checked_source_code(json.loads(source_path.read_text())["code_sha256"]) if source_path else code_bindings()
    sharded = getattr(args, "storage", "physical") == "sharded"
    if source_path and not sharded:
        raise ValueError("historical_source_bindings_require_sharded_storage")
    rows,users,assignment,parts,bindings,full_hash=fold_exports(
        args.project,args.dataset,args.seed,args.fold_root,args.folds,args.assignment_seed,source_code,
        source_policy(vars(args)))
    positions={int(r):(k,j) for k,p in enumerate(parts) for j,r in enumerate(p[3]["row_ids"])}
    widths=np.asarray([parts[k][3]["offsets"][j+1]-parts[k][3]["offsets"][j]
                       for k,j in (positions[int(r)] for r in rows)])
    offsets=np.r_[0,np.cumsum(widths)]
    target_width=max(p[3]["targets"].shape[1] for p in parts)
    meta={"row_ids":rows,"users":users,"offsets":offsets,
          "targets":np.full((len(rows),target_width),-1,np.int64),
          "target_counts":np.zeros(len(rows),np.int64),"bucket_ids":np.zeros(len(rows),np.int64),
          "rrf_indices":np.empty((len(rows),200),np.int64)}
    context=np.empty((len(rows),18),np.float32)
    source_parts = []
    if not sharded:
        for k, part in enumerate(parts):
            path = args.fold_root / f"fold{k}/export/sources.npz"
            expected = part[4].get("reproduction_sources_sha256")
            if expected is None:
                source_parts.append(None)
                continue
            if digest(path) != expected:
                raise ValueError("fold_complete_sources_hash")
            with np.load(path, allow_pickle=False) as z:
                values = {key: z[key] for key in z.files}
            n = len(part[3]["row_ids"])
            if (not np.array_equal(values["row_ids"], part[3]["row_ids"])
                    or not np.array_equal(values["users"], part[3]["users"])
                    or values["source_ids"].shape != (n, 4, 200)
                    or values["source_lengths"].shape != (n, 4) or values["cues"].shape != (n,)):
                raise ValueError("fold_complete_sources_alignment")
            source_parts.append(values)
        if any(p is not None for p in source_parts) and any(p is None for p in source_parts):
            raise ValueError("fold_complete_sources_inventory")
    complete_sources = (dict(source_ids=np.empty((len(rows),4,200),np.int32),
        source_lengths=np.empty((len(rows),4),np.int32), cues=np.empty(len(rows),np.int64))
        if source_parts and source_parts[0] is not None else None)
    args.output.mkdir(parents=True)
    outputs=[np.lib.format.open_memmap(args.output/name,mode="w+",dtype=parts[0][i].dtype,
                shape=(int(offsets[-1]),*parts[0][i].shape[1:]))
             for i,name in enumerate(EXPORT_FILES[:3])] if not sharded else []
    for at,row in enumerate(rows):
        k,j=positions[int(row)]; p=parts[k]; src=slice(*map(int,p[3]["offsets"][j:j+2])); dst=slice(offsets[at],offsets[at+1])
        for i,out in enumerate(outputs): out[dst]=p[i][src]
        for key in ("target_counts","bucket_ids","rrf_indices"): meta[key][at]=p[3][key][j]
        n=int(meta["target_counts"][at]); meta["targets"][at,:n]=p[3]["targets"][j,:n]
        context[at]=p[5][j]
        if complete_sources is not None:
            for key, out in complete_sources.items():
                out[at] = source_parts[k][key][j]
    for out in outputs: out.flush()
    atomic_npz(args.output/"metadata.npz",**meta)
    atomic_npz(args.output/"REQUEST_CONTEXT.npz",row_ids=rows,users=users,relation=context)
    files = EXPORT_FILES[3:] if sharded else EXPORT_FILES
    m=dict(parts[0][4]); m.update(rows=len(rows),source_policy=source_policy(vars(args)),
        output_sha256={n:digest(args.output/n) for n in files})
    if sharded:
        m.update(storage="fold_shards_v1", code_sha256=code_bindings(), source_code_sha256=source_code)
    for key in ("serving_inputs","base_manifest","index_dir","index_manifest_sha256"): m.pop(key,None)
    m.pop("reproduction_sources_sha256", None)
    if complete_sources is not None:
        atomic_npz(args.output/"sources.npz",row_ids=rows,users=users,**complete_sources)
        m["reproduction_sources_sha256"] = digest(args.output/"sources.npz")
    atomic_json(args.output/"EXPORT_MANIFEST.json",m)
    atomic_json(args.output/"COVERAGE.json",{
        "schema":"arrive.full_train_oof.v1","status":"pass","dataset":args.dataset,"seed":args.seed,
        "source_policy":source_policy(vars(args)),
        "folds":args.folds,"assignment_seed":args.assignment_seed,"fold_root":str(args.fold_root.resolve()),
        "rows":len(rows),"users":len(np.unique(users)),"full_base_manifest_sha256":full_hash,
        "fold_bindings":bindings,"export_sha256":digest(args.output/"EXPORT_MANIFEST.json"),
        "code_sha256":code_bindings(),"feature_schema":feature_schema(),"test_accessed":False,
        **({"storage":"fold_shards_v1","source_code_sha256":source_code} if sharded else {})})


def validate_coverage(root,project,dataset,seed,*,compatibility=None):
    from arrive_framework_datasets import check_export,code_bindings,source_policy,file_hash as digest
    receipt=json.loads((root/"COVERAGE.json").read_text())
    reader_code = (checked_source_code(compatibility["reader_code_sha256"],compatibility)
                   if compatibility is not None else code_bindings())
    expected={"schema":"arrive.full_train_oof.v1","status":"pass","dataset":dataset,"seed":seed,
              "test_accessed":False,"code_sha256":reader_code,"export_sha256":digest(root/"EXPORT_MANIFEST.json")}
    if any(receipt.get(k)!=v for k,v in expected.items()): raise ValueError("coverage_receipt_binding")
    sharded = receipt.get("storage") == "fold_shards_v1"
    source_code = checked_source_code(receipt["source_code_sha256"],compatibility) if sharded else None
    rows,users,_,parts,bindings,full=fold_exports(project,dataset,seed,Path(receipt["fold_root"]),receipt["folds"],receipt["assignment_seed"],source_code,
        source_policy(receipt))
    if sharded:
        from arrive_framework_datasets import feature_schema
        manifest = json.loads((root/"EXPORT_MANIFEST.json").read_text())
        identity = dict(parts[0][4])
        for key in ("serving_inputs","base_manifest","index_dir","index_manifest_sha256","reproduction_sources_sha256"):
            identity.pop(key, None)
        identity.update(rows=len(rows), storage="fold_shards_v1", code_sha256=reader_code,
                        source_code_sha256=source_code,
                        output_sha256={n:digest(root/n) for n in ("metadata.npz","REQUEST_CONTEXT.npz")})
        if "source_policy" in manifest:
            identity["source_policy"] = source_policy(receipt)
        if manifest != identity or receipt["feature_schema"] != feature_schema():
            raise ValueError("sharded_export_manifest_binding")
        with np.load(root/"metadata.npz", allow_pickle=False) as z:
            meta = {k:z[k] for k in z.files}
        with np.load(root/"REQUEST_CONTEXT.npz", allow_pickle=False) as z:
            if not np.array_equal(z["row_ids"],rows) or not np.array_equal(z["users"],users):
                raise ValueError("sharded_context_identity")
            context = z["relation"].copy()
        positions = {int(r):(k,j) for k,p in enumerate(parts) for j,r in enumerate(p[3]["row_ids"])}
        fold_ids, starts, widths = [], [], []
        for row in rows:
            k,j = positions[int(row)]
            lo,hi = parts[k][3]["offsets"][j:j+2]
            fold_ids.append(k); starts.append(lo); widths.append(hi-lo)
        offsets = np.r_[0,np.cumsum(widths)]
        if not np.array_equal(meta["offsets"],offsets) or context.shape != (len(rows),18):
            raise ValueError("sharded_offsets_or_context")
        arrays = [FoldArray([p[i] for p in parts],offsets,np.asarray(fold_ids),np.asarray(starts)) for i in range(3)]
        merged = (*arrays,meta,manifest,context)
    else:
        merged=check_export(root,dataset,seed,"train_cv")
    if source_policy(merged[4]) != source_policy(receipt):
        raise ValueError("coverage_source_policy_mismatch")
    if (bindings!=receipt["fold_bindings"] or full!=receipt["full_base_manifest_sha256"]
            or not np.array_equal(rows,merged[3]["row_ids"]) or not np.array_equal(users,merged[3]["users"])):
        raise ValueError("full_train_coverage_mismatch")
    # Verify merged values, not just counts or a producer's declaration.
    for p in parts:
        for j,row in enumerate(p[3]["row_ids"]):
            at=int(np.searchsorted(rows,row)); src=slice(*p[3]["offsets"][j:j+2]); dst=slice(*merged[3]["offsets"][at:at+2])
            # Shards refer directly to hash-checked source arrays; no copied candidate values exist.
            if (not sharded and any(not np.array_equal(p[i][src],merged[i][dst]) for i in range(3))) or not np.array_equal(p[5][j],merged[5][at]):
                raise ValueError("merged_fold_values_mismatch")
            n=int(p[3]["target_counts"][j])
            if (any(not np.array_equal(p[3][key][j],merged[3][key][at]) for key in ("rrf_indices","bucket_ids","target_counts"))
                    or not np.array_equal(p[3]["targets"][j,:n],merged[3]["targets"][at,:n])):
                raise ValueError("merged_fold_metadata_mismatch")
    return receipt,merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "inventory", "run-fold", "merge"))
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--prepared-manifest", type=Path)
    parser.add_argument("--fold-root", type=Path)
    parser.add_argument("--storage", choices=("physical","sharded"), default="sharded",
                        help="merge: sharded keeps candidate arrays in their original folds")
    parser.add_argument("--source-code-bindings", type=Path,
                        help="reviewed producer bindings for pre-storage-upgrade exports")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--assignment-seed", type=int, default=20260927)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    parser.add_argument("--source-policy", choices=("full","joint-removal"), default="full")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dataset", choices=("music4all_onion", "yambda500m"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path,
                        help="inventory mode: backbone training_manifest.json")
    parser.add_argument("--backbone-best-epochs",
                        help="Optional assertion against --backbone-manifests; comma-separated")
    parser.add_argument("--backbone-manifests", type=Path, nargs=3,
                        help="plan mode: frozen serving manifests in seed 2024,2025,2026 order")
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--crossfit-manifest", type=Path,
                        help="inventory mode: existing carrier training_manifest.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="plan mode: COVERAGE_PLAN.json path")
    args = parser.parse_args()

    if args.mode in ("run-fold","merge"):
        if args.seed not in (2024,2025,2026): parser.error("execution requires explicit model --seed 2024/2025/2026")
        if args.mode=="run-fold" and (args.fold is None or args.training_manifest is None or args.prepared_manifest is None):
            parser.error("run-fold requires --fold, --training-manifest and --prepared-manifest")
        if args.mode=="merge" and args.fold_root is None: parser.error("merge requires --fold-root")
        if args.mode == "run-fold":
            run_fold(args)
        else:
            merge_folds(args)
            receipt, exported = validate_coverage(args.output,args.project,args.dataset,args.seed)
            print(json.dumps({"event":"full_train_storage_verified", "status":"pass",
                "storage":receipt.get("storage","physical"), "rows":receipt["rows"],
                "candidate_rows":len(exported[1]),
                "output_bytes":sum(p.stat().st_size for p in args.output.iterdir()),
                "candidate_copy_bytes_avoided":sum(int(np.prod(a.shape))*a.dtype.itemsize+128
                    for a in exported[:3]) if receipt.get("storage") else 0}), flush=True)
        return

    from run_music4all_candidate_support_a1 import atomic_json

    data_dir = deny_traversal(args.data_dir, "data_dir")
    output_plan = deny_traversal(args.output, "output")
    rows, users = load_train_sessions(data_dir)

    if args.mode == "inventory":
        if not args.training_manifest or not args.crossfit_manifest:
            raise SystemExit("inventory requires --training-manifest and --crossfit-manifest")
        backbone_manifest = deny_traversal(args.training_manifest, "training_manifest")
        crossfit_manifest = deny_traversal(args.crossfit_manifest, "crossfit_manifest")
        cf = json.loads(crossfit_manifest.read_text())
        from run_music4all_candidate_support_a1 import digest
        if cf.get("binding",{}).get("full_base_manifest_sha256")!=digest(backbone_manifest):
            raise ValueError("inventory_full_parent_mismatch")
        excluded = int(cf["excluded_cv_rows"]) if "excluded_cv_rows" in cf else int(cf["excluded_rows"])
        payload = {
            "mode": "inventory", "dataset": args.dataset,
            "train_rows": int(len(rows)),
            "carrier_excluded_cv_rows": excluded,
            "carrier_excluded_fraction": round(excluded / len(rows), 4),
            "in_sample_rows": len(rows) - excluded,
            "note": "in-sample rows must not be exported through this carrier (EXPERIMENT_PLAN gap 1)",
        }
        from run_music4all_candidate_support_a1 import atomic_json
        atomic_json(output_plan, payload)
        print(json.dumps(payload), flush=True)
        return

    if not args.backbone_manifests: parser.error("plan requires --backbone-manifests for 2024/2025/2026")
    from run_music4all_candidate_support_a1 import digest
    best_epochs=[]; manifest_hashes={}
    for seed,path in zip((2024,2025,2026),args.backbone_manifests):
        m=json.loads(path.read_text())
        if m.get("status")!="pass" or m.get("test_accessed") is not False or m["binding"]["configuration"]["seed"]!=seed:
            raise ValueError("plan_frozen_backbone_manifest")
        best_epochs.append(int(m["best_epoch"])); manifest_hashes[str(seed)]=digest(path)
    if min(best_epochs)<=0: raise ValueError("plan_best_epoch_invalid")
    if args.backbone_best_epochs and best_epochs != [int(x) for x in args.backbone_best_epochs.split(",")]:
        raise ValueError("plan_best_epoch_assertion_mismatch")
    rows, users, fold_of_row = fold_of_rows(data_dir, args.folds, args.assignment_seed)
    rate = PER_EPOCH_MINUTES[args.dataset]["nominal"]
    train_rows = len(rows)
    per_fold = []
    unrounded = []
    for k in range(args.folds):
        excluded = int((fold_of_row == k).sum())
        fit_rows = train_rows - excluded
        hours_by_seed = [e * (fit_rows / train_rows) * rate / 60.0
                         for e in best_epochs]
        unrounded.append(hours_by_seed)
        per_fold.append({
            "fold": k,
            "excluded_rows": excluded,
            "fit_rows": fit_rows,
            "estimated_fit_hours_by_seed": [round(h, 2) for h in hours_by_seed],
        })

    seed_totals = [round(sum(f[i] for f in unrounded), 2)
                   for i in range(len(best_epochs))]
    aux_fits_total = args.folds * len(best_epochs)
    out = {
        "schema": "arrive.final_framework.fold_coverage.v1",
        "dataset": args.dataset,
        "folds": args.folds,
        "assignment_seed": args.assignment_seed,
        "frozen_backbone_manifest_sha256": manifest_hashes,
        "timing_status": "legacy rate assumption, requires target-machine canary",
        "backbone_best_epochs_by_seed": best_epochs,
        "backbone_best_epoch": best_epochs[0],
        "train_rows": int(train_rows),
        "users": int(len(np.unique(users))),
        "per_fold": per_fold,
        "auxiliary_fits_total_all_seeds": aux_fits_total,
        "portfolio_note": "aux fits multiply by dataset AND seed; totals below are THIS dataset, this fold plan",
        "estimated_fit_hours_per_seed": dict(zip(("2024", "2025", "2026"), seed_totals)),
        "per_epoch_minutes": rate,
        "user_disjoint": True,
        "assignment": "deterministic via tools/arrive_train_fold_coverage.py:fold_of_rows(data_dir, folds, seed)",
    }
    atomic_json(output_plan, out)
    print(json.dumps({"event": "coverage_plan_written", "path": str(output_plan),
                      "folds": args.folds,
                      "estimated_fit_hours_per_seed": out["estimated_fit_hours_per_seed"]}), flush=True)


if __name__ == "__main__":
    main()
