"""VISTA history features for the existing YetiRank; no neural serving scorer.

CD-SBRS adaptation: dominant genre (stable column tie-break), author window
boundary rule, then segment-local seed attention with a fixed total seed budget.
Unknown genre events remain in order and do not trigger content boundaries.
This is segmentation reuse, not a reproduction of CD-SBRS's full recommender.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch


def content_windows(categories):
    """Index-preserving form of author's process_item_set at commit2f264afb.

    Counts reset after the second category reaches three occurrences; the
    suffix is rescanned from the recorded boundary. Missing categories=-1
    are retained but excluded from boundary evidence (our task adaptation).
    """
    categories = np.asarray(categories, dtype=np.int64)
    if categories.ndim != 1 or np.any(categories < -1):
        raise ValueError("invalid_content_categories")
    counts, cuts, marker, i = {}, [], 0, 0
    while i < len(categories):
        category = int(categories[i])
        if category >= 0:
            counts[category] = counts.get(category, 0) + 1
            frequent = sum(v >= 3 for v in counts.values())
            if counts[category] >= 3 and frequent < 2:
                marker = i
            if frequent == 2:
                cuts.append(marker + 1)
                if marker + 1 < len(categories) - 1:
                    i = marker
                marker, counts = 0, {}
        i += 1
    if cuts and (cuts[0] <= 0 or any(a >= b for a, b in zip(cuts, cuts[1:]))):
        raise ValueError("nonmonotonic_content_boundary")
    return np.searchsorted(np.asarray(cuts), np.arange(len(categories)), side="right").astype(np.int64)


def category_map(genre):
    from scipy import sparse
    genre = sparse.csr_matrix(genre)
    if genre.shape[1] < 1 or not np.isfinite(genre.data).all() or np.any(genre.data < 0):
        raise ValueError("invalid_genre_matrix")
    result = np.asarray(genre.argmax(axis=1)).ravel().astype(np.int64)
    result[np.asarray(genre.sum(axis=1)).ravel() == 0] = -1
    return result


def group_histories(history, categories):
    history = np.asarray(history)
    if history.ndim != 2 or np.any(history < 0) or np.any(history >= len(categories)):
        raise ValueError("invalid_group_history")
    result = np.full(history.shape, -1, np.int64)
    for i, row in enumerate(history):
        valid = row >= 3
        result[i, valid] = content_windows(categories[row[valid]])
    return result


def attach_cd_groups(data, genre_path):
    from scipy import sparse
    from run_music4all_candidate_support_a1 import digest
    genre = sparse.load_npz(genre_path).tocsr()
    if (genre.shape[0] != len(data.items) or genre.shape[1] >= data.items.shape[1]
            or not np.array_equal(genre.toarray(), data.items[:, -genre.shape[1]:])):
        raise ValueError("genre_does_not_match_frozen_item_content")
    categories = category_map(genre)
    data.m["cd_groups"] = group_histories(data.m["history"], categories)
    return dict(genre_sha256=digest(genre_path), genre_dim=genre.shape[1],
                category_sha256=hashlib.sha256(categories.tobytes()).hexdigest(),
                rule="max_weight_genre_lowest_column_tie_unknown_no_boundary_v1",
                boundary="CD-SBRS_author_2f264afb_second_category_count3")


def segment_attention_mask(groups, valid, seeds, heads):
    """Keep all events and K seeds; distribute K by window length, >=1/window.

    Dense mask for the <=256-token pilot; no sparse-attention speed claim.
    Empty/padded queries can read group0 seeds, preventing all-masked softmax.
    """
    if groups is None or groups.shape != valid.shape or not torch.equal(groups >= 0, valid):
        raise ValueError("missing_or_misaligned_cd_groups")
    allocations = []
    for row, mask in zip(groups.detach().cpu().numpy(), valid.detach().cpu().numpy()):
        active = row[mask]
        if not len(active):
            allocations.append(np.zeros(seeds, np.int64))
            continue
        if active[0] != 0 or np.any(np.diff(active) < 0) or not np.array_equal(np.unique(active), np.arange(active[-1]+1)):
            raise ValueError("noncontiguous_cd_groups")
        lengths = np.bincount(active)
        if len(lengths) > seeds:
            raise ValueError("insufficient_seeds_to_preserve_all_content_windows")
        quota = (seeds-len(lengths))*lengths/lengths.sum()
        counts = 1 + np.floor(quota).astype(int)
        order = np.argsort(-(quota-np.floor(quota)), kind="stable")
        counts[order[:seeds-counts.sum()]] += 1
        allocations.append(np.repeat(np.arange(len(counts)), counts))
    seed_groups = torch.as_tensor(np.asarray(allocations), device=groups.device)
    tags = torch.cat((groups.clamp_min(0), seed_groups), 1)
    return (tags[:, :, None] != tags[:, None, :]).repeat_interleave(heads, dim=0)


def load_encoder(directory, data, device="cpu", genres=None):
    from arrive_history_reader import make_model, code_binding
    from run_music4all_candidate_support_a1 import digest
    manifest = json.loads((directory/"training_manifest.json").read_text())
    path = directory/"best_model.pt"
    config = manifest["config"]
    expected_status = "smoke" if data.receipt["smoke"] else "pass"
    if (manifest["status"] != expected_status or manifest["test_accessed"] is not False
            or manifest["seed"] != data.receipt["seed"] or config["arm"] != "vista_reconstruct"
            or config.get("reconstruction_discrimination_weight", 0.) != 0.
            or config["reconstruction_weight"] != 1.
            or digest(path) != manifest["output_sha256"]["best_model.pt"]):
        raise ValueError("encoder_identity_or_objective_mismatch")
    current_vista = digest(Path(__file__).with_name("arrive_vista.py"))
    # Audited legacy summary+MSE: read_memory extraction preserves its forward.
    legacy = "8a525b914adf7c73e553e53633d1749a3fe6450d36b52084fa712a61af894cbe"
    if manifest["binding"]["vista_code"] not in (current_vista, legacy):
        raise ValueError("unreviewed_encoder_implementation")
    train_path = data.path.parent/"train_cv/HISTORY_MANIFEST.json"
    train = json.loads(train_path.read_text())
    if (manifest["binding"]["train_manifest"] != digest(train_path)
            or train["output_sha256"]["items.npy"] != data.receipt["output_sha256"]["items.npy"]
            or train["reference_model_sha256"] != data.receipt["reference_model_sha256"]):
        raise ValueError("encoder_data_binding_mismatch")
    if config.get("history_grouping") == "cd_sbrs":
        if (genres is None or attach_cd_groups(data, genres) != config["cd_mapping"]
                or manifest["binding"].get("cd_code") != digest(Path(__file__))
                or manifest["binding"]["code"] != code_binding()):
            raise ValueError("cd_encoder_mapping_or_code_mismatch")
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved["binding"] != manifest["binding"] or saved["config"] != config:
        raise ValueError("encoder_checkpoint_contract_mismatch")
    model = make_model(data, config).to(device).eval()
    model.load_state_dict(saved["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, manifest


def export(args):
    """Materialize only64 learned columns, retaining original31 byte-for-byte."""
    from arrive_history_reader import Dataset
    from run_music4all_candidate_support_a1 import atomic_json, digest
    data = Dataset(args.bundle/f"s{args.seed}"/args.surface)
    out = args.output/f"s{args.seed}"/args.surface
    if out.exists():
        raise FileExistsError(out)
    if data.receipt["smoke"] != args.smoke or data.receipt["seed"] != args.seed:
        raise ValueError("export_smoke_or_seed_mismatch")
    model, manifest = load_encoder(args.encoder, data, args.device, args.cd_genres)
    required = len(data.ids)*model.hidden*4
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.smoke and shutil.disk_usage(out.parent).free < required+10_000_000_000:
        raise ValueError("export_would_violate_10GB_reserve")
    out.mkdir()
    values = np.lib.format.open_memmap(out/"read_features.npy", mode="w+", dtype=np.float32,
                                      shape=(len(data.ids), model.hidden))
    items = torch.as_tensor(np.array(data.items), device=args.device)
    start = time.perf_counter()
    with torch.inference_mode():
        for at in range(0, len(data.m["row_ids"]), args.batch_size):
            rows = np.arange(at, min(at+args.batch_size, len(data.m["row_ids"])))
            batch = data.batch(rows, args.device, union=True)
            memory = model.encode_history(batch["history"], items, batch.get("cd_groups"))
            read = model.read_memory(batch["ids"], items, *memory).cpu().numpy()
            if not np.isfinite(read).all():
                raise ValueError("nonfinite_learned_features")
            for i, row in enumerate(rows):
                a, b = data.m["offsets"][[row, row+1]]
                values[a:b] = read[i, :b-a]
    values.flush()
    del values
    atomic_json(out/"FEATURE_MANIFEST.json", dict(status="smoke" if args.smoke else "pass",
        seed=args.seed, surface=args.surface, test_accessed=False, rows=len(data.m["row_ids"]),
        columns=model.hidden, values=len(data.ids), elapsed_seconds=time.perf_counter()-start,
        data_manifest_sha256=digest(data.path/"HISTORY_MANIFEST.json"),
        encoder_sha256=digest(args.encoder/"best_model.pt"),
        encoder_manifest_sha256=digest(args.encoder/"training_manifest.json"),
        encoder_binding=manifest["binding"], history_grouping=model.history_grouping,
        feature_sha256=digest(out/"read_features.npy"), file_bytes=(out/"read_features.npy").stat().st_size,
        producer_sha256=digest(Path(__file__)), vista_code_sha256=digest(Path(__file__).with_name("arrive_vista.py")),
        online_neural_scorer=False, loss_changed=False))


def load_features(data, root, smoke):
    from run_music4all_candidate_support_a1 import digest
    path = root/f"s{data.receipt['seed']}"/data.receipt["surface"]
    receipt = json.loads((path/"FEATURE_MANIFEST.json").read_text())
    if (receipt["status"] != ("smoke" if smoke else "pass") or data.receipt["smoke"] != smoke
            or receipt["test_accessed"] is not False or receipt["seed"] != data.receipt["seed"]
            or receipt["surface"] != data.receipt["surface"]
            or receipt["data_manifest_sha256"] != digest(data.path/"HISTORY_MANIFEST.json")
            or receipt["feature_sha256"] != digest(path/"read_features.npy")):
        raise ValueError("learned_feature_binding_mismatch")
    values = np.load(path/"read_features.npy", mmap_mode="r")
    if values.shape != (len(data.ids), 64) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("learned_feature_shape_or_values")
    return values, receipt


def joined_features(data, values, positions):
    return np.concatenate((np.asarray(data.x[positions]), np.asarray(values[positions])), axis=-1)


def tree_predictions(model, data, values, rows, trees, threads):
    from run_music4all_candidate_support_a1 import stable_topk
    from run_music4all_component_c2_conversion_oof import row_metrics
    output = []
    for at in range(0, len(rows), 64):
        selected = rows[at:at+64]
        flat = data.m["offsets"][selected, None]+data.m["rrf_indices"][selected]
        scores = model.predict(joined_features(data, values, flat.reshape(-1)),
                               ntree_end=trees, thread_count=threads)
        output.append(stable_topk(np.asarray(scores).reshape(flat.shape), data.ids[flat], 10))
    top = np.concatenate(output)
    result = {k: data.m[k][rows] for k in ("row_ids", "users", "bucket_ids", "targets", "target_counts")}
    recall, ndcg = row_metrics(top, result["targets"], result["target_counts"])
    return dict(**result, top10=top, recall=recall, ndcg=ndcg)


def train_tree(args):
    from catboost import CatBoostRanker, Pool
    from arrive_history_reader import Dataset, macro, user_partition
    from run_arrive_recent_ranker import LOSS, ITERATIONS, selected_rows
    from run_music4all_candidate_support_a1 import atomic_json, digest
    data = Dataset(args.bundle/f"s{args.seed}"/"train_cv")
    if data.receipt["seed"] != args.seed:
        raise ValueError("tree_training_seed_mismatch")
    values, receipt = load_features(data, args.features, args.smoke)
    out = args.output/f"s{args.seed}"
    if out.exists():
        raise FileExistsError(out)
    positive = np.add.reduceat(np.asarray(data.y, np.int32), data.m["offsets"][:-1]) > 0
    partition = user_partition(data.m["users"])
    fit, cal = np.flatnonzero(partition & positive), np.flatnonzero(~partition)
    if not len(fit) or not len(cal) or set(data.m["users"][fit]) & set(data.m["users"][cal]):
        raise ValueError("tree_training_user_partition")
    positions, groups = selected_rows(data.m, fit)
    iterations = (5, 10) if args.smoke else ITERATIONS
    model = CatBoostRanker(iterations=max(iterations), depth=6, learning_rate=.035,
                          loss_function=LOSS, random_seed=args.seed+1000, thread_count=args.threads,
                          task_type="CPU", verbose=False, allow_writing_files=False)
    start = time.perf_counter()
    model.fit(Pool(joined_features(data, values, positions), np.asarray(data.y[positions]), group_id=groups))
    records = []
    for trees in iterations:
        prediction = tree_predictions(model, data, values, cal, trees, args.threads)
        records.append(dict(trees=trees, ndcg=macro(prediction, "ndcg")))
    selected = max(records, key=lambda r: (r["ndcg"], -r["trees"]))["trees"]
    out.mkdir(parents=True)
    model.save_model(str(out/"yeti_ndcg.cbm"))
    atomic_json(out/"training_manifest.json", dict(status="smoke" if args.smoke else "pass",
        seed=args.seed, test_accessed=False, validation_accessed=False, feature_count=95,
        method="YetiRank_original31_plus_frozen_VISTA_read64", loss_function=LOSS,
        selected_trees=selected, calibration=records, fit_rows=len(fit), calibration_rows=len(cal),
        training_seconds=time.perf_counter()-start, model_sha256=digest(out/"yeti_ndcg.cbm"),
        encoder_sha256=receipt["encoder_sha256"], history_grouping=receipt["history_grouping"],
        feature_manifest_sha256=digest(args.features/f"s{args.seed}"/"train_cv/FEATURE_MANIFEST.json"),
        data_manifest_sha256=digest(data.path/"HISTORY_MANIFEST.json"),
        model_bytes=(out/"yeti_ndcg.cbm").stat().st_size, producer_sha256=digest(Path(__file__))))


def validate_tree(args):
    from catboost import CatBoostRanker
    from arrive_history_reader import Dataset, macro
    from run_music4all_candidate_support_a1 import atomic_json, atomic_npz, digest
    if not args.allow_validation:
        raise ValueError("explicit_development_validation_required")
    out = args.output/f"s{args.seed}"
    if (out/"validation_predictions.npz").exists():
        raise FileExistsError("tree_validation_already_done")
    manifest = json.loads((out/"training_manifest.json").read_text())
    data = Dataset(args.bundle/f"s{args.seed}"/"validation")
    if data.receipt["seed"] != args.seed:
        raise ValueError("tree_validation_seed_mismatch")
    values, receipt = load_features(data, args.features, args.smoke)
    if (manifest["status"] != ("smoke" if args.smoke else "pass") or manifest["seed"] != args.seed
            or manifest["test_accessed"] is not False or manifest["encoder_sha256"] != receipt["encoder_sha256"]
            or manifest["history_grouping"] != receipt["history_grouping"]
            or manifest["model_sha256"] != digest(out/"yeti_ndcg.cbm")
            or manifest["producer_sha256"] != digest(Path(__file__))):
        raise ValueError("tree_validation_binding_mismatch")
    model = CatBoostRanker().load_model(str(out/"yeti_ndcg.cbm"))
    result = tree_predictions(model, data, values, np.arange(len(data.m["row_ids"])), manifest["selected_trees"], args.threads)
    atomic_npz(out/"validation_predictions.npz", **result)
    atomic_json(out/"validation_manifest.json", dict(status=manifest["status"], seed=args.seed,
        test_accessed=False, rows=len(result["row_ids"]), overall_recall=macro(result,"recall"),
        overall_ndcg=macro(result,"ndcg"), prediction_sha256=digest(out/"validation_predictions.npz"),
        data_manifest_sha256=digest(data.path/"HISTORY_MANIFEST.json"),
        feature_manifest_sha256=digest(args.features/f"s{args.seed}"/"validation/FEATURE_MANIFEST.json"),
        training_manifest_sha256=digest(out/"training_manifest.json")))


class VistaYetiServing:
    """Shared complete-request profiler adapter; neural scorer never executes.

    Cache keys: encoder+content version (instance-bound), user, exact prefix.
    File mode measures file loading too; repeated runs may hit the OS page cache.
    """
    def __init__(self, args, row_ids, request_data):
        from catboost import CatBoostRanker
        from arrive_history_reader import Dataset
        from run_music4all_candidate_support_a1 import digest
        self.directory = args.vista_yeti_model
        self.device, self.threads = args.device, args.threads
        self.request_data, self.row_ids = request_data, row_ids
        data = Dataset(args.history_bundle/f"s{args.seed}"/"validation")
        self.model, encoder_manifest = load_encoder(args.vista_encoder, data, self.device, args.cd_genres)
        # Training-only modules have no serving consumer, so release their weights.
        for name in ("scorer", "decoder", "decoder_input", "decoder_position", "decoder_output"):
            delattr(self.model, name)
        self.model_version = digest(args.vista_encoder/"best_model.pt")
        manifest = json.loads((self.directory/"training_manifest.json").read_text())
        if (manifest["status"] != "pass" or manifest["test_accessed"] is not False
                or manifest["seed"] != args.seed or manifest["encoder_sha256"] != self.model_version
                or manifest["model_sha256"] != digest(self.directory/"yeti_ndcg.cbm")):
            raise ValueError("serving_yeti_encoder_binding")
        validation = json.loads((self.directory/"validation_manifest.json").read_text())
        if (validation["status"] != "pass" or validation["test_accessed"] is not False
                or validation["training_manifest_sha256"] != digest(self.directory/"training_manifest.json")
                or validation["data_manifest_sha256"] != digest(data.path/"HISTORY_MANIFEST.json")):
            raise ValueError("serving_validation_binding")
        self.ranker = CatBoostRanker().load_model(str(self.directory/"yeti_ndcg.cbm"))
        self.tree_bytes = (self.directory/"yeti_ndcg.cbm").stat().st_size
        self.trees = manifest["selected_trees"]
        self.items = torch.as_tensor(np.array(data.items), device=self.device)
        self.categories = None
        if self.model.history_grouping == "cd_sbrs":
            from scipy import sparse
            self.categories = category_map(sparse.load_npz(args.cd_genres))
        self.mode, self.cache_dir = args.vista_cache, args.vista_cache_dir
        if self.mode == "file":
            if self.cache_dir is None or self.cache_dir.exists():
                raise ValueError("file_cache_requires_new_directory")
            self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
            needed = len(row_ids)*self.model.seed_count*self.model.hidden*4
            if shutil.disk_usage(self.cache_dir.parent).free < needed+10_000_000_000:
                raise ValueError("cache_would_violate_10GB_reserve")
            self.cache_dir.mkdir()
        self.cache, self.hits, self.misses = {}, 0, 0
        self.read_bytes = self.write_bytes = self.build_seconds = 0
        self.h2d_summary_bytes = self.d2h_read_bytes = 0
        self.cache_load_seconds = 0.
        index = {int(r):i for i,r in enumerate(data.m["row_ids"])}
        started = time.perf_counter()
        for start in range(0, len(row_ids), args.batch_size):
            rows = row_ids[start:start+args.batch_size]
            keys = [self.key(r) for r in rows]
            history = self.history(keys)
            if not np.array_equal(history, data.m["history"][[index[int(r)] for r in rows]]):
                raise ValueError("serving_prefix_does_not_match_export")
            summaries = self.encode(history)
            for key, memory in zip(keys, summaries):
                if key not in self.cache:
                    self.store(key, memory)
        self.build_seconds = time.perf_counter()-started
        self.encoder_binding = encoder_manifest["binding"]

    def key(self, row):
        s = self.request_data.sessions[int(row)]
        start = int(s[2])
        past = self.request_data.events[max(int(s[1]), start-128):start]
        return int(s[0]), tuple(int(x) for x in past if x >= 3)

    @staticmethod
    def history(keys):
        result = np.zeros((len(keys),128), np.int64)
        for i, (_, past) in enumerate(keys):
            result[i,:len(past)] = past
        return result

    def encode_tensor(self, history):
        groups = None if self.categories is None else torch.as_tensor(
            group_histories(history, self.categories), device=self.device)
        with torch.inference_mode():
            return self.model.encode_history(torch.as_tensor(history, device=self.device), self.items, groups)

    def encode(self, history):
        return self.encode_tensor(history)[0].cpu().numpy()

    def store(self, key, memory):
        if self.mode == "file":
            name = hashlib.sha256(repr((self.model_version,key)).encode()).hexdigest()+".npy"
            path = self.cache_dir/name
            if path.exists():
                raise FileExistsError(path)
            np.save(path, memory)
            self.cache[key] = path
            self.write_bytes += path.stat().st_size
        else:
            self.cache[key] = memory.copy()

    def __call__(self, feature_matrix, id_matrix, rows, cached):
        keys = [self.key(row) for row in rows]
        history = self.history(keys)
        with torch.inference_mode():
            if cached:
                memories = []
                for key in keys:
                    if key not in self.cache:
                        self.store(key, self.encode(self.history([key]))[0])
                        self.misses += 1
                    else:
                        self.hits += 1
                    start = time.perf_counter()
                    value = self.cache[key]
                    memory = np.load(value, allow_pickle=False) if self.mode == "file" else value
                    self.read_bytes += value.stat().st_size if self.mode == "file" else memory.nbytes
                    memories.append(memory)
                    self.cache_load_seconds += time.perf_counter()-start
                array = np.stack(memories)
                if str(self.device).startswith("cuda"):
                    self.h2d_summary_bytes += array.nbytes
                summary = (torch.as_tensor(array, device=self.device),
                           torch.ones((len(rows),self.model.seed_count),dtype=torch.bool,device=self.device),
                           torch.as_tensor((history>=3).any(1),device=self.device))
            else:
                summary = self.encode_tensor(history)
            read = self.model.read_memory(torch.as_tensor(id_matrix.astype(np.int64),device=self.device),
                                          self.items,*summary).cpu().numpy()
            if str(self.device).startswith("cuda"):
                self.d2h_read_bytes += read.nbytes
        matrix = np.concatenate((feature_matrix.reshape(len(rows),200,31), read),-1).reshape(-1,95)
        return self.ranker.predict(matrix,ntree_end=self.trees,thread_count=self.threads).reshape(len(rows),200)

    def receipt(self):
        return dict(encoder_sha256=self.model_version, encoder_binding=self.encoder_binding,
                    history_grouping=self.model.history_grouping, cache_mode=self.mode,
                    cache_entries=len(self.cache), cache_build_seconds=self.build_seconds,
                    summary_payload_bytes=len(self.cache)*self.model.seed_count*self.model.hidden*4,
                    item_content_device_bytes=self.items.numel()*self.items.element_size(),
                    serving_neural_parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters()),
                    tree_model_file_bytes=self.tree_bytes,
                    logical_cache_read_bytes=self.read_bytes, file_cache_write_bytes=self.write_bytes,
                    cache_load_seconds=self.cache_load_seconds, hits=self.hits, misses=self.misses,
                    h2d_summary_bytes=self.h2d_summary_bytes,d2h_read_bytes=self.d2h_read_bytes,
                    online_neural_scorer=False, neural_decoder_online=False,
                    io_scope="logical bytes incl warmup; file reads may use OS page cache; no network/storage-service latency",
                    cache_regime="prebuilt requested-prefix cache; not a production hit-rate estimate")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("history_yeti_export", "history_yeti_train", "history_yeti_validation"))
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(2024,2025,2026), required=True)
    p.add_argument("--surface", choices=("train_cv","validation"), default="train_cv")
    p.add_argument("--encoder", type=Path)
    p.add_argument("--features", type=Path)
    p.add_argument("--cd-genres", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-validation", action="store_true")
    args = p.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        p.error("positive threads and batch-size required")
    if args.mode == "history_yeti_export" and args.encoder is None:
        p.error("--encoder required")
    if args.mode != "history_yeti_export" and args.features is None:
        p.error("--features required")
    torch.set_num_threads(args.threads)
    {"history_yeti_export":export, "history_yeti_train":train_tree,
     "history_yeti_validation":validate_tree}[args.mode](args)


if __name__ == "__main__":
    main()
