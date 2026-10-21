"""Shared history-reader backend for run_arrive_recent_ranker.py.

Frozen exports and existing metrics are reused. No Test surface is supported.
All new arms share one scorer, training loop and candidate/feature contract.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from run_music4all_candidate_support_a1 import (
    atomic_json, atomic_npz, atomic_torch, digest, set_seed, stable_topk, user_partition,
)
from run_music4all_component_c2_conversion_oof import row_metrics, user_macro

ARMS = ("head_only", "full_history", "time", "content", "content_no_cue", "adaptive_content", "adaptive_time")
VISTA_ARMS = ("vista_full", "vista_summary", "vista_reconstruct")
SCHEMA = "arrive.history_reader.v1"
CODE_FILES = ("arrive_history_reader.py", "run_arrive_recent_ranker.py",
              "run_arrive_full_rcce_control.py", "run_music4all_candidate_support_a1.py",
              "run_music4all_component_c2_conversion_oof.py")


def code_binding():
    return {name: digest(Path(__file__).with_name(name)) for name in CODE_FILES}


def compatible_training_code(binding):
    """Only the frozen pre-UNK-guard revision is inference-compatible.

    Forward, fitting, inputs and weights are unchanged. Keep the original
    training receipt immutable; evaluation records its own producer separately.
    """
    current = code_binding()
    if binding == current:
        return True
    prior = dict(current)
    prior["arrive_history_reader.py"] = "4dae6c6c8adac73b9eae12f09c419cd19d185731224977377c4030b6b3e5033c"
    if binding == prior:
        return True
    # New adaptive arms and diagnostic CLI leave the five older arms' forward,
    # train/eval semantics unchanged; new arms separately bind their overlay.
    audited = dict(current)
    audited["arrive_history_reader.py"] = "a0d51c4da83c0f91ab0e663229d656ab0c5edebac7981b32806dfbcdc2dd4fa7"
    if binding == audited:
        return True
    original = dict(current)
    original["arrive_history_reader.py"] = "d4bb1ad503ba78b53c1ebc7da0fec7fce16f047913eee6ecd4047bbd251968e9"
    return binding == original


def groups_for(content: np.ndarray, k: int = 4, temporal: bool = False) -> np.ndarray:
    """Optimal nonempty contiguous SSE groups; earlier boundary wins ties."""
    n = len(content)
    if not n:
        return np.empty(0, np.int64)
    k = min(k, n)
    if temporal:
        return np.repeat(np.arange(k), [len(x) for x in np.array_split(np.arange(n), k)])
    x = np.asarray(content, np.float64)
    prefix = np.vstack((np.zeros((1, x.shape[1])), x.cumsum(0)))
    norm = (prefix * prefix).sum(1)
    lengths = np.arange(n + 1)[None, :] - np.arange(n + 1)[:, None]
    sums2 = np.r_[0., (x * x).sum(1).cumsum()]
    cost = sums2[None, :] - sums2[:, None] - (
        norm[:, None] + norm[None, :] - 2 * (prefix @ prefix.T)
    ) / np.maximum(lengths, 1)
    cost[lengths <= 0] = np.inf
    dp = np.full((k + 1, n + 1), np.inf)
    back = np.zeros((k + 1, n + 1), np.int64)
    dp[0, 0] = 0.
    for j in range(1, k + 1):
        for end in range(j, n + 1):
            start = np.arange(j - 1, end)
            values = dp[j - 1, start] + cost[start, end]
            b = int(start[np.argmin(values)])
            dp[j, end], back[j, end] = values.min(), b
    result = np.empty(n, np.int64)
    end = n
    for j in range(k, 0, -1):
        start = back[j, end]
        result[start:end] = j - 1
        end = start
    return result


def history_arrays(events, sessions, row_ids, users, content, split_id):
    """Read only pre-cue events; content is normalized audio + multigenre."""
    selected = sessions[row_ids]
    if not np.array_equal(selected[:, 0], users) or not np.all(selected[:, 5] == split_id):
        raise ValueError("history_users_or_split_mismatch")
    history = np.zeros((len(row_ids), 128), np.int32)
    cue = np.zeros(len(row_ids), np.int32)
    time_groups = np.full_like(history, -1)
    content_groups = np.full_like(history, -1)
    for i, s in enumerate(selected):
        user_begin, cue_at = map(int, s[1:3])
        if not 0 <= user_begin <= cue_at < len(events):
            raise ValueError("history_bad_event_boundary")
        h = np.asarray(events[max(user_begin, cue_at - 128):cue_at], np.int32)
        h = h[h >= 3]
        cue[i] = int(events[cue_at])
        # Prepared protocol: 0=padding, 1=unknown identity, 2=EOS.
        # An unseen return track remains a valid request, using frozen UNK.
        if cue[i] < 1 or cue[i] == 2 or cue[i] >= len(content) or np.any(h >= len(content)):
            raise ValueError("history_invalid_item")
        n = len(h)
        history[i, :n] = h
        time_groups[i, :n] = groups_for(content[h], temporal=True)
        content_groups[i, :n] = groups_for(content[h])
    return dict(history=history, cue=cue, time_groups=time_groups, content_groups=content_groups)


def prepare(args):
    """CPU-only immutable preparation; consumes exact existing union exports."""
    from scipy import sparse
    from run_arrive_full_rcce_control import load_export, parent_paths
    from run_arrive_recent_ranker import load_request_context, with_request_context, top10
    out = args.output / f"s{args.seed}" / args.surface
    if out.exists():
        raise FileExistsError(out)
    parent = args.project / "runs/arrive_full_rcce_control_20260924/onion" / f"s{args.seed}"
    features, ids, labels, meta, em = load_export(parent / args.surface, args.seed, args.surface)
    if em.get("limit") != 0:
        raise ValueError("limited_export_not_formal")
    context, context_binding = load_request_context(
        args.context_root, parent, meta, args.seed, args.surface, "statistics")
    refdir = args.project / "runs/arrive_yeti_request_context_20260925/statistics" / f"s{args.seed}"
    ref = json.loads((refdir / "training_manifest.json").read_text())
    if (ref.get("status") != "pass" or ref.get("feature_count") != 31
            or ref.get("seed") != args.seed or ref.get("test_accessed") is not False
            or ref.get("feature_mode") != "statistics"
            or ref["model_sha256"] != digest(refdir / "yeti_ndcg.cbm")):
        raise ValueError("invalid_31_feature_reference")
    key = "train" if args.surface == "train_cv" else "validation"
    if ref[key + "_export_sha256"] != digest(parent / args.surface / "EXPORT_MANIFEST.json"):
        raise ValueError("reference_export_mismatch")
    if ref[key + "_request_context"] != context_binding:
        raise ValueError("reference_context_mismatch")
    paths = parent_paths(args.project, args.seed, args.surface)
    for name in ("base_checkpoint", "base_manifest"):
        if digest(paths[name]) != em["bindings"][name]:
            raise ValueError("backbone_binding:" + name)
    # The existing features use split-specific backbones. The new reader must
    # not switch raw embedding coordinates between fitting and inference.
    history_paths = parent_paths(args.project, args.seed, "train_cv")
    history_export = json.loads((parent / "train_cv/EXPORT_MANIFEST.json").read_text())
    history_sha = digest(history_paths["base_checkpoint"])
    if (history_sha != history_export["bindings"]["base_checkpoint"]
            or digest(parent / "train_cv/EXPORT_MANIFEST.json") != ref["train_export_sha256"]):
        raise ValueError("history_embedding_binding")
    ck = torch.load(history_paths["base_checkpoint"], map_location="cpu", weights_only=False)
    items = ck["model"]["item_embedding.weight"].numpy().astype(np.float32)
    data = args.project / "data/music4all_onion/prepared_v1"
    bm = json.loads(paths["base_manifest"].read_text())
    declared = bm["binding"].get("prepared_artifacts", bm["binding"].get("inputs", {}))
    prepared = {}
    for name in ("events.npy", "sessions.npy", "track_identity.tsv"):
        prepared[name] = digest(data / name)
        if prepared[name] != declared.get(name):
            raise ValueError("prepared_binding:" + name)
    content_dir = args.project / "data/music4all_onion/component_b2_attainability_v1"
    for name, sha in em["content_sha256"].items():
        if digest(content_dir / name) != sha:
            raise ValueError("content_binding:" + name)
    audio = np.load(content_dir / "musicnn_normalized.npy")
    genre = sparse.load_npz(content_dir / "genre_normalized.npz").toarray()
    if not (len(audio) == len(genre) == len(items)):
        raise ValueError("catalog_alignment")
    # Each block has equal weight; retain missing (zero) content and multilabel genre.
    content = np.concatenate((audio, genre), axis=1).astype(np.float32) / np.sqrt(2.)
    began = time.monotonic()
    h = history_arrays(np.load(data / "events.npy", mmap_mode="r"),
                       np.load(data / "sessions.npy", mmap_mode="r"),
                       meta["row_ids"], meta["users"], content,
                       0 if args.surface == "train_cv" else 1)
    out.mkdir(parents=True)
    target = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float32,
                                     shape=(len(features), 31))
    # Chunk by query, preserving the entire fitting union, not only served C200.
    for lo in range(0, len(meta["row_ids"]), 256):
        hi = min(lo + 256, len(meta["row_ids"]))
        a, b = map(int, meta["offsets"][[lo, hi]])
        qids = np.repeat(np.arange(lo, hi), np.diff(meta["offsets"])[lo:hi])
        target[a:b] = with_request_context(features[a:b], qids, context, "statistics")
    target.flush()
    del target
    np.save(out / "candidate_ids.npy", np.asarray(ids))
    np.save(out / "labels.npy", np.asarray(labels))
    np.save(out / "items.npy", np.concatenate((items, audio, genre), axis=1).astype(np.float32))
    atomic_npz(out / "metadata.npz", **meta, **h)
    files = ["features.npy", "candidate_ids.npy", "labels.npy", "items.npy", "metadata.npz"]
    if args.surface == "train_cv":
        from catboost import CatBoostRanker
        reference = CatBoostRanker()
        reference.load_model(str(refdir / "yeti_ndcg.cbm"))
        rows = np.flatnonzero(~user_partition(meta["users"]))
        positive = np.add.reduceat(np.asarray(labels, np.int32), meta["offsets"][:-1]) > 0
        if (int((user_partition(meta["users"]) & positive).sum()) != ref["fit_rows"]
                or len(rows) != ref["calibration_rows"]):
            raise ValueError("reference_training_partition_mismatch")
        top = top10(reference, features, ids, meta, rows, ref["selected_trees"], context, "statistics")
        recall, ndcg = row_metrics(top, meta["targets"][rows], meta["target_counts"][rows])
        expected = next(v["train_calibration_rrf_ndcg"] for v in ref["calibration"] if v["trees"] == ref["selected_trees"])
        actual = user_macro(ndcg, meta["users"][rows], np.ones(len(rows), bool))
        if not np.isclose(actual, expected, atol=1e-12, rtol=0):
            raise ValueError("reference_calibration_replay_failed")
        atomic_npz(out / "reference_calibration.npz", row_ids=meta["row_ids"][rows], users=meta["users"][rows],
                   targets=meta["targets"][rows], target_counts=meta["target_counts"][rows],
                   bucket_ids=meta["bucket_ids"][rows], top10=top, recall=recall, ndcg=ndcg)
        files.append("reference_calibration.npz")
    receipt = dict(schema=SCHEMA, status="pass", smoke=False, seed=args.seed, surface=args.surface,
                   rows=len(meta["row_ids"]), embedding_dim=items.shape[1],
                   output_sha256={p: digest(out / p) for p in files}, test_accessed=False,
                   prepared_sha256=prepared, parent_export_sha256=digest(parent / args.surface / "EXPORT_MANIFEST.json"),
                   context_binding=context_binding, reference_manifest_sha256=digest(refdir / "training_manifest.json"),
                   reference_model_sha256=ref["model_sha256"], selected_trees=ref["selected_trees"],
                   backbone_sha256=em["bindings"]["base_checkpoint"], history_embedding_sha256=history_sha,
                   content_sha256=em["content_sha256"],
                   producer_sha256=code_binding(), preparation_seconds=time.monotonic() - began,
                   train_pool="exact exported fitting union", serving_pool="exact exported RRF C200")
    atomic_json(out / "HISTORY_MANIFEST.json", receipt)
    try:
        Dataset(out)
    except Exception:
        receipt["status"] = "failed_integrity"
        atomic_json(out / "HISTORY_MANIFEST.json", receipt)
        raise
    print(json.dumps({"event": "history_prepared", "rows": receipt["rows"],
                      "seconds": receipt["preparation_seconds"]}), flush=True)


class Dataset:
    def __init__(self, path):
        self.path = Path(path)
        self.receipt = json.loads((self.path / "HISTORY_MANIFEST.json").read_text())
        r = self.receipt
        if (r.get("schema") != SCHEMA or r.get("status") != "pass"
                or r.get("test_accessed") is not False or r.get("surface") not in ("train_cv", "validation")):
            raise ValueError("history_receipt_invalid")
        for name in ("features.npy", "candidate_ids.npy", "labels.npy", "items.npy", "metadata.npz"):
            if digest(self.path / name) != r["output_sha256"][name]:
                raise ValueError("history_artifact_hash:" + name)
        self.x = np.load(self.path / "features.npy", mmap_mode="r")
        self.ids = np.load(self.path / "candidate_ids.npy", mmap_mode="r")
        self.y = np.load(self.path / "labels.npy", mmap_mode="r")
        self.items = np.load(self.path / "items.npy", mmap_mode="r")
        with np.load(self.path / "metadata.npz", allow_pickle=False) as z:
            self.m = {k: z[k] for k in z.files}
        m, n = self.m, len(self.m["row_ids"])
        if (self.x.shape != (len(self.ids), 31) or len(self.y) != len(self.ids)
                or m["offsets"].shape != (n + 1,) or m["offsets"][0] != 0
                or m["offsets"][-1] != len(self.ids) or np.any(np.diff(m["offsets"]) < 200)
                or m["rrf_indices"].shape != (n, 200) or len(np.unique(m["row_ids"])) != n
                or m["history"].shape != (n, 128) or m["cue"].shape != (n,)
                or n != r["rows"] or not np.isfinite(self.x).all() or not np.isfinite(self.items).all()):
            raise ValueError("history_bundle_shape_or_values")
        for g in (m["time_groups"], m["content_groups"]):
            if (g.shape != (n, 128) or not np.array_equal(g >= 0, m["history"] >= 3)
                    or np.any(g >= 4)):
                raise ValueError("history_groups_invalid")
        if (np.any(self.ids < 3) or np.any(self.ids >= len(self.items))
                or np.any(m["history"] < 0) or np.any(m["history"] >= len(self.items))
                or np.any(m["cue"] < 1) or np.any(m["cue"] == 2)
                or np.any(m["cue"] >= len(self.items))):
            raise ValueError("history_catalog_id_invalid")
        for row in range(n):
            a, b = map(int, m["offsets"][row:row + 2])
            c = self.ids[a:b]
            rr = m["rrf_indices"][row]
            count = int(m["target_counts"][row])
            truth = m["targets"][row, :count]
            if (np.any(rr < 0) or np.any(rr >= len(c)) or len(np.unique(rr)) != 200
                    or len(np.unique(c)) != len(c) or not 0 <= count <= m["targets"].shape[1]
                    or len(np.unique(truth)) != count
                    or not np.array_equal(self.y[a:b], np.isin(c, truth))):
                raise ValueError("history_candidate_or_ground_truth_mismatch")

    def batch(self, rows, device, union=False):
        m = self.m
        if union:
            length = np.diff(m["offsets"])[rows]
            pos = np.arange(length.max())[None, :]
            valid = pos < length[:, None]
            flat = m["offsets"][rows, None] + np.minimum(pos, length[:, None] - 1)
        else:
            flat = m["offsets"][rows, None] + m["rrf_indices"][rows]
            valid = np.ones_like(flat, bool)
        values = dict(x=np.array(self.x[flat]), ids=np.array(self.ids[flat], np.int64),
                      labels=np.array(self.y[flat], np.float32), valid=valid,
                      history=m["history"][rows].astype(np.int64), cue=m["cue"][rows].astype(np.int64),
                      time_groups=m["time_groups"][rows].astype(np.int64),
                      content_groups=m["content_groups"][rows].astype(np.int64))
        for name in ("adaptive_content", "adaptive_time", "cd_groups"):
            if name in m:
                values[name] = m[name][rows].astype(np.int64)
        return {k: torch.as_tensor(v, device=device) for k, v in values.items()}


def masked_softmax(scores, mask):
    value = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = value.softmax(-1) * mask
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)


class HistoryScorer(nn.Module):
    def __init__(self, input_dim, embedding_dim, arm, hidden=64):
        super().__init__()
        self.arm, self.embedding_dim, self.hidden = arm, embedding_dim, hidden
        # All arms initialize the same parameter structure in the same order.
        self.encoder = nn.Linear(input_dim, hidden)
        self.position = nn.Embedding(128, hidden)
        self.pool = nn.Linear(hidden, 1, bias=False)
        self.item_query = nn.Linear(embedding_dim, hidden, bias=False)
        self.cue_query = nn.Linear(embedding_dim, hidden, bias=False)
        self.scorer = nn.Sequential(nn.Linear(31 + hidden, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer("mean", torch.zeros(31))
        self.register_buffer("scale", torch.ones(31))

    def forward(self, batch, items, read_mode=None):
        mode = read_mode or self.arm
        x = (batch["x"] - self.mean) / self.scale
        if mode == "head_only":
            r = x.new_zeros((*x.shape[:2], self.hidden))
        else:
            mask = batch["history"] >= 3
            h = torch.tanh(self.encoder(items[batch["history"]]) +
                           self.position(torch.arange(128, device=x.device))[None])
            b = self.pool(h).squeeze(-1)
            q = self.item_query(items[batch["ids"], :self.embedding_dim])
            if self.arm != "content_no_cue":
                q = q + self.cue_query(items[batch["cue"], :self.embedding_dim])[:, None]
            if mode == "full_history":
                w = masked_softmax(b[:, None] + torch.bmm(q, h.transpose(1, 2)) / self.hidden**.5,
                                   mask[:, None])
                r = torch.bmm(w, h)
            elif mode in ("time", "content", "content_no_cue", "adaptive_content", "adaptive_time"):
                adaptive = mode.startswith("adaptive_")
                group = batch[mode if adaptive else ("time_groups" if mode == "time" else "content_groups")]
                # Up to15 singleton summaries retain a short history exactly.
                # Fixed padding is training convenience, NOT a speedup claim.
                slots = 16 if adaptive else 4
                membership = (group[:, None] == torch.arange(slots, device=x.device)[None, :, None]) & mask[:, None]
                within = masked_softmax(b[:, None].expand(-1, slots, -1), membership)
                z = torch.bmm(within, h)
                logmass = b[:, None].expand(-1, slots, -1).masked_fill(~membership, -1e30).logsumexp(-1)
                present = membership.any(-1)
                weights = masked_softmax(logmass[:, None] + torch.bmm(q, z.transpose(1, 2)) / self.hidden**.5,
                                         present[:, None])
                r = torch.bmm(weights, z)
            else:
                raise ValueError("unknown_read_mode")
        return self.scorer(torch.cat((x, r), -1)).squeeze(-1)


def listwise_loss(scores, labels, valid):
    positive = labels * valid
    keep = positive.sum(-1) > 0
    if not keep.any():
        raise ValueError("no_positive_training_batch")
    target = positive[keep] / positive[keep].sum(-1, keepdim=True)
    logp = scores[keep].masked_fill(~valid[keep], -1e30).log_softmax(-1)
    return -(target * logp).sum(-1).mean()


def make_model(data, config):
    if config["arm"] in VISTA_ARMS:
        from arrive_vista import VistaScorer
        return VistaScorer(data.items.shape[1], data.receipt["embedding_dim"], config["arm"],
                           seeds=config["summary_seeds"], reconstruction_weight=config["reconstruction_weight"],
                           reconstruction_discrimination_weight=config.get("reconstruction_discrimination_weight", 0.0),
                           history_grouping=config.get("history_grouping", "none"))
    return HistoryScorer(data.items.shape[1], data.receipt["embedding_dim"], config["arm"])


def evaluate(model, data, rows, items, batch_size, device, read_mode=None):
    model.eval()
    outputs, margins = [], []
    with torch.no_grad():
        for at in range(0, len(rows), batch_size):
            batch = data.batch(rows[at:at + batch_size], device)
            scores = model(batch, items, read_mode).cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError("nonfinite_eval_scores")
            outputs.append(stable_topk(scores, batch["ids"].cpu().numpy(), 10))
            order = np.sort(scores, axis=1)
            margins.append(order[:, -10] - order[:, -11])
    top = np.concatenate(outputs)
    m = data.m
    recall, ndcg = row_metrics(top, m["targets"][rows], m["target_counts"][rows])
    return dict(row_ids=m["row_ids"][rows], users=m["users"][rows], bucket_ids=m["bucket_ids"][rows],
                target_counts=m["target_counts"][rows], targets=m["targets"][rows], top10=top,
                recall=recall, ndcg=ndcg, boundary_margin=np.concatenate(margins))


def macro(result, key):
    return user_macro(result[key], result["users"], np.ones(len(result[key]), bool))


def normalize_fit(model, data, rows):
    sums, squares, count = np.zeros(31), np.zeros(31), 0
    for row in rows:
        a, b = data.m["offsets"][row:row + 2]
        x = np.asarray(data.x[a:b], np.float64)
        sums += x.sum(0)
        squares += (x * x).sum(0)
        count += len(x)
    mean = sums / count
    scale = np.sqrt(np.maximum(squares / count - mean**2, 0)).clip(1e-6)
    model.mean.copy_(torch.tensor(mean, dtype=torch.float32))
    model.scale.copy_(torch.tensor(scale, dtype=torch.float32))


def train(args):
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    began = time.monotonic()
    data = Dataset(args.bundle / f"s{args.seed}" / "train_cv")
    if (data.receipt["seed"] != args.seed or data.receipt["surface"] != "train_cv"
            or (data.receipt["smoke"] and not args.smoke)):
        raise ValueError("training_seed_or_smoke_bundle")
    out = args.output / args.arm / f"s{args.seed}"
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    config = {k: getattr(args, k) for k in ("arm", "seed", "epochs", "patience", "min_delta",
              "batch_size", "lr", "weight_decay", "threads", "smoke")}
    vista = args.arm in VISTA_ARMS
    if vista:
        config.update(summary_seeds=getattr(args, "summary_seeds", 128),
                      reconstruction_weight=getattr(args, "reconstruction_weight", 1.0))
        config["reconstruction_discrimination_weight"] = getattr(args, "reconstruction_discrimination_weight", 0.0)
        if getattr(args, "history_grouping", "none") == "cd_sbrs":
            from arrive_vista_yeti import attach_cd_groups
            if args.cd_genres is None or args.arm != "vista_reconstruct" or config["reconstruction_discrimination_weight"]:
                raise ValueError("cd_requires_original_reconstruction_and_frozen_genres")
            config.update(history_grouping="cd_sbrs", cd_mapping=attach_cd_groups(data, args.cd_genres))
    binding = {"train_manifest": digest(data.path / "HISTORY_MANIFEST.json"), "code": code_binding()}
    if vista:
        binding["vista_code"] = digest(Path(__file__).with_name("arrive_vista.py"))
        if config.get("history_grouping") == "cd_sbrs":
            binding["cd_code"] = digest(Path(__file__).with_name("arrive_vista_yeti.py"))
    if args.arm.startswith("adaptive_"):
        from arrive_history_grouping import attach_overlay
        binding["adaptive"] = attach_overlay(data, args.adaptive_root)
    if args.resume and (out / "training_manifest.json").exists():
        raise ValueError("terminal_run_no_resume")
    fit_partition = user_partition(data.m["users"])
    positive = np.add.reduceat(np.asarray(data.y, np.int32), data.m["offsets"][:-1]) > 0
    fit = np.flatnonzero(fit_partition & positive)
    cal = np.flatnonzero(~fit_partition)
    if not len(fit) or not len(cal) or set(data.m["users"][fit]) & set(data.m["users"][cal]):
        raise ValueError("training_user_partition")
    model = make_model(data, config)
    normalize_fit(model, data, fit)
    model.to(args.device)
    items = torch.tensor(np.array(data.items), device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, best, stop_best, bad, curves, total_seconds = 0, -float("inf"), -float("inf"), 0, [], 0.
    if args.resume:
        saved = torch.load(out / "last_state.pt", map_location=args.device, weights_only=False)
        if saved["config"] != config or saved["binding"] != binding:
            raise ValueError("resume_contract_mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best, bad = saved["epoch"], saved["best"], saved["bad"]
        stop_best = saved["stop_best"]
        curves, total_seconds = saved["curves"], saved["total_seconds"]
    out.mkdir(parents=True, exist_ok=True)
    setup_seconds = time.monotonic() - began
    stopped = False
    for epoch in range(start, args.epochs):
        if bad >= args.patience:
            break
        if total_seconds >= args.max_seconds:
            atomic_json(out / "PROGRESS.json", {"status": "budget_paused", "epoch": epoch,
                        "total_seconds": total_seconds, "not_a_negative_result": True})
            return
        t = time.monotonic()
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(fit)
        losses = []
        ranking_losses, reconstruction_sum, reconstruction_pairs = [], 0., 0
        mse_sum, discrimination_sum = 0., 0.
        for at in range(0, len(order), args.batch_size):
            batch = data.batch(order[at:at + args.batch_size], args.device, union=True)
            optimizer.zero_grad(set_to_none=True)
            if vista:
                scores, reconstruction, pairs = model.training_outputs(batch, items)
                ranking = listwise_loss(scores, batch["labels"], batch["valid"])
                loss = ranking + model.reconstruction_weight * reconstruction
                ranking_losses.append((float(ranking.detach()), len(batch["x"])))
                n_pairs = int(pairs.detach())
                reconstruction_sum += float(reconstruction.detach()) * n_pairs
                reconstruction_pairs += n_pairs
                mse_sum += float(model.reconstruction_metrics.get("mse", 0.)) * n_pairs
                discrimination_sum += float(model.reconstruction_metrics.get("discrimination_ce", 0.)) * n_pairs
            else:
                loss = listwise_loss(model(batch, items), batch["labels"], batch["valid"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite_training_loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append((float(loss.detach()), len(batch["x"])))
        train_seconds = time.monotonic() - t
        t = time.monotonic()
        result = evaluate(model, data, cal, items, args.batch_size, args.device)
        fit_result = evaluate(model, data, fit[:1024], items, args.batch_size, args.device)
        ndcg = macro(result, "ndcg")
        improved = ndcg > best
        significant = ndcg > stop_best + args.min_delta
        bad = 0 if significant else bad + 1
        if significant:
            stop_best = ndcg
        if improved:
            best = ndcg
            atomic_torch(out / "best_model.pt", {"model": model.state_dict(), "epoch": epoch + 1,
                         "config": config, "binding": binding, "calibration_ndcg": ndcg})
        eval_seconds = time.monotonic() - t
        record = dict(epoch=epoch + 1, loss=sum(v*n for v, n in losses)/sum(n for _, n in losses),
                      calibration_ndcg=ndcg, calibration_recall=macro(result, "recall"),
                      fit_probe_ndcg=macro(fit_result, "ndcg"), fit_probe_rows=len(fit_result["row_ids"]),
                      train_seconds=train_seconds, eval_seconds=eval_seconds, bad_epochs=bad)
        if vista:
            record.update(ranking_loss=sum(v*n for v, n in ranking_losses)/sum(n for _, n in ranking_losses),
                          reconstruction_mse=mse_sum / max(1, reconstruction_pairs),
                          reconstruction_auxiliary=reconstruction_sum / max(1, reconstruction_pairs),
                          reconstruction_discrimination_ce=discrimination_sum / max(1, reconstruction_pairs),
                          reconstruction_pairs=reconstruction_pairs)
        curves.append(record)
        total_seconds += train_seconds + eval_seconds
        atomic_torch(out / "last_state.pt", dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     epoch=epoch + 1, best=best, stop_best=stop_best, bad=bad, curves=curves, config=config,
                     binding=binding, total_seconds=total_seconds))
        atomic_json(out / "learning_curve.json", curves)
        print(json.dumps({"event": "history_epoch", "arm": args.arm, **record}), flush=True)
        if args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch:
            stopped = True
            break
    if stopped:
        return
    selected = torch.load(out / "best_model.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(selected["model"])
    result = evaluate(model, data, cal, items, args.batch_size, args.device)
    atomic_npz(out / "calibration_predictions.npz", **result)
    # Pure diagnostic: no new fits; both directions retained where defined.
    if args.arm in ("full_history", "content"):
        for mode in ("full_history", "time", "content"):
            probe = evaluate(model, data, cal, items, args.batch_size, args.device, mode)
            atomic_npz(out / f"calibration_read_{mode}.npz", **probe)
    status = "smoke" if args.smoke else ("pass" if bad >= args.patience else "epoch_cap_review_required")
    files = [p.name for p in out.iterdir() if p.is_file() and p.suffix in (".npz", ".pt", ".json")]
    atomic_json(out / "training_manifest.json", dict(schema=SCHEMA, status=status, seed=args.seed,
                arm=args.arm, config=config, binding=binding, test_accessed=False, validation_accessed=False,
                fit_rows=len(fit), calibration_rows=len(cal), skipped_no_positive=int((fit_partition & ~positive).sum()),
                selected_epoch=selected["epoch"], best_calibration_ndcg=best, setup_seconds=setup_seconds,
                total_train_eval_seconds=total_seconds, parameter_count=sum(p.numel() for p in model.parameters()),
                max_gpu_bytes=torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
                max_train_eval_seconds=args.max_seconds,
                output_sha256={name: digest(out / name) for name in files}))


def validation(args):
    if not args.allow_validation:
        raise ValueError("explicit_frozen_validation_required")
    out = args.output / args.arm / f"s{args.seed}"
    if (out / "validation_predictions.npz").exists():
        raise FileExistsError("validation_already_evaluated")
    manifest = json.loads((out / "training_manifest.json").read_text())
    if manifest["status"] != "pass" and not (args.smoke and manifest["status"] == "smoke"):
        raise ValueError("training_not_complete_for_validation")
    data = Dataset(args.bundle / f"s{args.seed}" / "validation")
    if manifest["config"].get("history_grouping") == "cd_sbrs":
        from arrive_vista_yeti import attach_cd_groups
        if args.cd_genres is None or attach_cd_groups(data, args.cd_genres) != manifest["config"]["cd_mapping"]:
            raise ValueError("cd_validation_mapping_mismatch")
        if manifest["binding"]["cd_code"] != digest(Path(__file__).with_name("arrive_vista_yeti.py")):
            raise ValueError("cd_validation_code_mismatch")
    adaptive_binding = None
    if args.arm.startswith("adaptive_"):
        from arrive_history_grouping import attach_overlay
        adaptive_binding = attach_overlay(data, args.adaptive_root)
        if (manifest["binding"]["adaptive"]["policy"] != adaptive_binding["policy"]
                or manifest["binding"]["adaptive"]["producer"] != adaptive_binding["producer"]):
            raise ValueError("adaptive_validation_policy_mismatch")
    train_data = json.loads((args.bundle / f"s{args.seed}" / "train_cv/HISTORY_MANIFEST.json").read_text())
    if (data.receipt["seed"] != args.seed or data.receipt["surface"] != "validation"
            or manifest["seed"] != args.seed or manifest["arm"] != args.arm
            or data.receipt["smoke"] != args.smoke or train_data["reference_model_sha256"] != data.receipt["reference_model_sha256"]
            or train_data["output_sha256"]["items.npy"] != data.receipt["output_sha256"]["items.npy"]
            or not compatible_training_code(manifest["binding"]["code"])
            or manifest["binding"]["train_manifest"] != digest(args.bundle / f"s{args.seed}" / "train_cv/HISTORY_MANIFEST.json")):
        raise ValueError("validation_identity_mismatch")
    if digest(out / "best_model.pt") != manifest["output_sha256"]["best_model.pt"]:
        raise ValueError("checkpoint_hash_mismatch")
    saved = torch.load(out / "best_model.pt", map_location=args.device, weights_only=False)
    if saved["binding"] != manifest["binding"] or saved["config"] != manifest["config"]:
        raise ValueError("checkpoint_training_contract_mismatch")
    if args.arm in VISTA_ARMS:
        if (manifest["binding"]["code"] != code_binding()
                or manifest["binding"].get("vista_code") != digest(Path(__file__).with_name("arrive_vista.py"))):
            raise ValueError("vista_validation_code_mismatch")
    model = make_model(data, saved["config"]).to(args.device)
    model.load_state_dict(saved["model"])
    items = torch.tensor(np.array(data.items), device=args.device)
    result = evaluate(model, data, np.arange(len(data.m["row_ids"])), items, args.batch_size, args.device)
    atomic_npz(out / "validation_predictions.npz", **result)
    atomic_json(out / "validation_manifest.json", dict(schema=SCHEMA, status="smoke" if args.smoke else "pass",
                seed=args.seed, arm=args.arm, test_accessed=False, rows=len(result["row_ids"]),
                evaluation_code=code_binding(), training_code=manifest["binding"]["code"],
                unknown_cue_rows=int((data.m["cue"] == 1).sum()),
                training_manifest_sha256=digest(out / "training_manifest.json"),
                data_manifest_sha256=digest(data.path / "HISTORY_MANIFEST.json"),
                adaptive_binding=adaptive_binding,
                prediction_sha256=digest(out / "validation_predictions.npz"),
                overall_recall=macro(result, "recall"), overall_ndcg=macro(result, "ndcg")))


def paired_stats(delta, users, mask, seed=2026, resamples=5000):
    from analyze_arrive_joint_test import user_values
    values = user_values(delta.mean(0), users, mask)
    if not len(values):
        return dict(delta=None, ci95=None, p_two_sided=None, users=0)
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(resamples)])
    observed = values.mean()
    p = (1 + np.count_nonzero(np.abs(means - observed) >= abs(observed))) / (resamples + 1)
    return dict(delta=float(observed), ci95=np.quantile(means, [.025, .975]).tolist(),
                p_two_sided=float(p), users=len(values), seed_deltas=[float(user_values(d, users, mask).mean()) for d in delta])


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    result, running = {}, 0.
    for rank, key in enumerate(order):
        running = max(running, min(1., (len(order) - rank) * pvalues[key]))
        result[key] = running
    return result


def analyze(args):
    """Consume all stage-one seeds; refuse incomplete or mismatched evidence."""
    vista = getattr(args, "vista", False)
    if vista and args.include_no_cue:
        raise ValueError("vista_and_content_no_cue_are_separate_experiments")
    dest = args.output / ("VISTA_ANALYSIS.json" if vista else
                          "STAGE2_ANALYSIS.json" if args.include_no_cue else "STAGE1_ANALYSIS.json")
    if dest.exists():
        raise FileExistsError(dest)
    arms = ("A",) + VISTA_ARMS if vista else ("A", "head_only", "full_history", "time", "content")
    if args.include_no_cue:
        arms += ("content_no_cue",)
    values, bindings = {arm: [] for arm in arms}, []
    canonical = None
    for seed in (2024, 2025, 2026):
        common_training = None
        for arm in arms:
            if arm == "A":
                directory = args.project / "runs/arrive_yeti_request_context_20260925/statistics" / f"s{seed}"
                receipt = json.loads((directory / "training_manifest.json").read_text())
                pred = directory / "validation_predictions.npz"
                if (receipt.get("status") != "pass" or receipt.get("seed") != seed
                        or receipt.get("test_accessed") is not False or receipt.get("feature_count") != 31
                        or receipt["prediction_sha256"] != digest(pred)
                        or receipt["model_sha256"] != digest(directory / "yeti_ndcg.cbm")):
                    raise ValueError("analysis_reference_identity")
                reference_hash = receipt["model_sha256"]
            else:
                directory = args.output / arm / f"s{seed}"
                receipt = json.loads((directory / "validation_manifest.json").read_text())
                training = json.loads((directory / "training_manifest.json").read_text())
                data_manifest = args.bundle / f"s{seed}" / "validation/HISTORY_MANIFEST.json"
                data_receipt = json.loads(data_manifest.read_text())
                pred = directory / "validation_predictions.npz"
                if (receipt.get("status") != "pass" or training.get("status") != "pass"
                        or receipt.get("seed") != seed or receipt.get("arm") != arm
                        or receipt.get("test_accessed") is not False or training.get("test_accessed") is not False
                        or receipt["prediction_sha256"] != digest(pred)
                        or receipt["training_manifest_sha256"] != digest(directory / "training_manifest.json")
                        or receipt["data_manifest_sha256"] != digest(data_manifest)
                        or data_receipt["reference_model_sha256"] != reference_hash):
                    raise ValueError("analysis_run_identity")
                contract = {k: v for k, v in training["config"].items() if k != "arm"}
                comparison = (contract, training["binding"])
                if common_training is not None and comparison != common_training:
                    raise ValueError("analysis_training_contract_drift")
                common_training = comparison
            with np.load(pred) as z:
                row = {key: z[key].copy() for key in z.files}
            for key in ("row_ids", "users", "bucket_ids", "targets", "target_counts"):
                if canonical is not None and not np.array_equal(row[key], canonical[key]):
                    raise ValueError("analysis_alignment:" + key)
            if canonical is None:
                canonical = row
            r, n = row_metrics(row["top10"], row["targets"], row["target_counts"])
            if not np.allclose(r, row["recall"], atol=1e-12, rtol=0) or not np.allclose(n, row["ndcg"], atol=1e-12, rtol=0):
                raise ValueError("analysis_metric_replay")
            values[arm].append(row)
            bindings.append(dict(arm=arm, seed=seed, path=str(pred), sha256=digest(pred)))
    summary = {}
    for arm in arms:
        summary[arm] = {}
        for metric in ("recall", "ndcg"):
            seed_values = [macro(v, metric) for v in values[arm]]
            summary[arm][metric] = dict(mean=float(np.mean(seed_values)), sample_sd=float(np.std(seed_values, ddof=1)), seeds=seed_values)
    comparisons = {}
    pairs = (("head_only", "A"), ("full_history", "head_only"), ("content", "time"),
             ("content", "A"), ("content", "head_only"), ("time", "head_only"), ("full_history", "content"))
    if vista:
        pairs = (("vista_reconstruct", "vista_summary"), ("vista_reconstruct", "vista_full"),
                 ("vista_reconstruct", "A"), ("vista_summary", "A"), ("vista_full", "A"),
                 ("vista_summary", "vista_full"))
    if args.include_no_cue:
        pairs += (("content", "content_no_cue"),)
    longest = canonical["bucket_ids"].max()
    for left, right in pairs:
        key = left + "-" + right
        comparisons[key] = {}
        for metric in ("recall", "ndcg"):
            delta = np.stack([a[metric] - b[metric] for a, b in zip(values[left], values[right])])
            comparisons[key][metric] = {
                scope: paired_stats(delta, canonical["users"], mask)
                for scope, mask in (("overall", np.ones(len(delta[0]), bool)), ("longest_nonempty", canonical["bucket_ids"] == longest))}
    primary = [a + "-" + b for a, b in pairs[:2 if vista else 4]]
    adjusted = holm({key: comparisons[key]["ndcg"]["overall"]["p_two_sided"] for key in primary})
    report = dict(schema=SCHEMA, status="complete_conditional_validation_analysis" if args.include_no_cue else "complete_stage1_validation_analysis", test_accessed=False,
                  rows=len(canonical["row_ids"]), summary=summary, comparisons=comparisons,
                  primary_ndcg_holm=adjusted, longest_bucket=int(longest), bindings=bindings,
                  interpretation="Conditional on three fitted models; diagnostics do not automatically select a repair or close the history hypothesis.")
    atomic_json(dest, report)


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1].startswith("history_yeti_"):
        from arrive_vista_yeti import main as yeti_main
        yeti_main()
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("history_prepare", "history_train", "history_validation", "history_analyze", "history_group_audit", "history_adaptive_prepare"))
    p.add_argument("--project", type=Path, default=Path("."))
    p.add_argument("--bundle", type=Path, default=Path("runs/arrive_content_history_20260926/data"))
    p.add_argument("--context-root", type=Path, default=Path("runs/arrive_yeti_request_context_20260925/context"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True, choices=(2024, 2025, 2026))
    p.add_argument("--surface", choices=("train_cv", "validation"), default="train_cv")
    p.add_argument("--arm", choices=ARMS + VISTA_ARMS, default="head_only")
    p.add_argument("--summary-seeds", type=int, default=128)
    p.add_argument("--reconstruction-weight", type=float, default=1.0)
    p.add_argument("--reconstruction-discrimination-weight", type=float, default=0.0)
    p.add_argument("--history-grouping", choices=("none", "cd_sbrs"), default="none")
    p.add_argument("--cd-genres", type=Path, help="Frozen genre_normalized.npz, checked against item content")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--max-seconds", type=float, default=5400., help="Train/eval compute cap per fit; pauses at epoch boundary, never declares failure")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-validation", action="store_true")
    p.add_argument("--include-no-cue", action="store_true", help="Analyze conditional D only after its separate decision")
    p.add_argument("--vista", action="store_true", help="Analyze learned-summary/reconstruction arms instead of legacy pooling")
    p.add_argument("--adaptive-root", type=Path)
    p.add_argument("--group-policy", type=Path)
    p.add_argument("--stop-after-epoch", type=int, default=0)
    args = p.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.max_seconds <= 0:
        p.error("positive epochs, patience and batch size required")
    if args.summary_seeds < 1 or not np.isfinite(args.reconstruction_weight) or args.reconstruction_weight < 0:
        p.error("positive summary seeds and finite nonnegative reconstruction weight required")
    if not np.isfinite(args.reconstruction_discrimination_weight) or args.reconstruction_discrimination_weight < 0:
        p.error("finite nonnegative reconstruction discrimination weight required")
    if args.reconstruction_discrimination_weight and args.arm != "vista_reconstruct":
        p.error("discrimination objective requires vista_reconstruct")
    if args.history_grouping != "none" and args.arm != "vista_reconstruct":
        p.error("CD grouping requires vista_reconstruct")
    if args.mode == "history_group_audit":
        from arrive_history_grouping import audit
        audit(args)
        return
    if args.mode == "history_adaptive_prepare":
        from arrive_history_grouping import prepare_overlay
        if args.group_policy is None:
            p.error("--group-policy required")
        prepare_overlay(args)
        return
    {"history_prepare": prepare, "history_train": train, "history_validation": validation,
     "history_analyze": analyze}[args.mode](args)
