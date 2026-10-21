"""Opt-in research branch of the shared ARRIVE runner. Train/Validation only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from run_music4all_candidate_support_a1 import (
    atomic_json, atomic_npz, atomic_torch, digest, load_base_data,
    load_standard_base, set_seed, stable_topk, user_partition,
)
from run_music4all_component_c2_conversion_oof import (
    deterministic_shuffle, row_metrics, user_macro, paired_bootstrap,
)
from run_music4all_component_c2_validation import supported_target_values
from run_lfm_timeconcat_control import early_stop_reached

SCHEMA = "arrive.return_query.v1"


class StructuredValue(torch.nn.Module):
    """Pool-independent dense score plus explicit sparse relation channels.

    Diagnostic student, NOT the previous moment-query or an exact ARRIVE rewrite.
    Static item vectors and relation definitions are frozen; query coefficients
    are unconstrained and may be signed. No per-candidate neural forward.
    """
    def __init__(self, query_width, item_width, relation_width, hidden=128):
        super().__init__()
        self.item_width = item_width
        self.coefficients = nn.Sequential(
            nn.LayerNorm(query_width), nn.Linear(query_width, hidden), nn.GELU(),
            nn.Linear(hidden, item_width + relation_width))
        self.bias = nn.Linear(item_width, 1, bias=False)
        nn.init.zeros_(self.coefficients[-1].weight)
        nn.init.zeros_(self.coefficients[-1].bias)
        nn.init.zeros_(self.bias.weight)

    def forward(self, query, items, relations):
        co = self.coefficients(query)
        return (torch.einsum("bd,bnd->bn", co[:, :self.item_width], items)
                + self.bias(items).squeeze(-1)
                + torch.einsum("bm,bnm->bn", co[:, self.item_width:], relations))


def certified_sparse_topk(base, channels, coefficients, k=10, block=32):
    """Conservative float64 threshold search; random access completes each score.

    base is an ALREADY computed eligible-catalog vector; caller must charge its
    dense scan and ordering. Each channel is (positions, values), unique indices.
    No full dense final score is computed to make the stopping decision.
    """
    base = np.asarray(base, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if len(base) < k or not np.isfinite(base).all():
        raise ValueError("invalid_search_base")
    ids = np.arange(len(base))
    prefix_size = min(len(base), 8 * block + k + 1)
    prefix = np.argpartition(-base, prefix_size - 1)[:prefix_size]
    base_order = prefix[np.lexsort((prefix, -base[prefix]))]
    orders, lookups = [], []
    for idx, values in channels:
        idx, values = np.asarray(idx), np.asarray(values, dtype=np.float64)
        if len(np.unique(idx)) != len(idx) or not np.isfinite(values).all():
            raise ValueError("invalid_sparse_channel")
        order = np.lexsort((idx, -np.abs(values)))
        orders.append((idx[order], values[order]))
        lookups.append(dict(zip(map(int, idx), map(float, values))))
    seen, positions = set(), np.zeros(len(channels), dtype=np.int64)
    length, iterations, sorted_reads = 0, 0, 0
    scores = {}
    while True:
        iterations += 1
        if iterations > 8:
            support = set().union(*(set(x) for x in lookups))
            masked = base.copy()
            if support:
                masked[np.fromiter(support,dtype=np.int64)] = -np.inf
            threshold = np.partition(masked, -k)[-k]
            outside = np.flatnonzero(masked > threshold).tolist()
            if np.isfinite(threshold):
                outside += np.flatnonzero(masked == threshold)[:k-len(outside)].tolist()
            exact = {}
            for i in support | set(outside):
                val = base[i]
                for a, lookup in zip(coefficients, lookups):
                    val += a * lookup.get(i, 0.0)
                exact[i] = val
            chosen = sorted(exact, key=lambda i: (-exact[i], i))[:k]
            return np.asarray(chosen), dict(unique_scored=len(seen | set(exact)),
                sorted_reads=sorted_reads + sum(len(x) for x in lookups), iterations=iterations,
                base_prefix=length, complete_catalog_fallback=False, support_fallback=True,
                channel_entries=sum(len(x[0]) for x in channels))
        next_length = min(len(base_order), length + block)
        fresh = set(map(int, base_order[length:next_length])) - seen
        sorted_reads += next_length - length
        length = next_length
        for j, (idx, values) in enumerate(orders):
            end = min(len(idx), int(positions[j]) + block)
            fresh.update(set(map(int, idx[positions[j]:end])) - seen)
            sorted_reads += end - int(positions[j])
            positions[j] = end
        for i in fresh:
            score = base[i]
            for a, lookup in zip(coefficients, lookups):
                score += a * lookup.get(i, 0.0)
            scores[i] = score
        seen.update(fresh)
        chosen = sorted(seen, key=lambda i: (-scores[i], i))[:k]
        bound = base[base_order[length]] if length < len(base) else -np.inf
        for a, pos, (_, values) in zip(coefficients, positions, orders):
            if pos < len(values):
                bound += abs(a) * abs(values[pos])
        # Rounding envelope for summation/product arithmetic; ties continue.
        max_term = max([abs(base).max()] + [abs(a) * max(lookup.values(), key=abs)
                       if lookup else 0.0 for a, lookup in zip(coefficients, lookups)], key=abs)
        envelope = 32 * np.finfo(np.float64).eps * (len(channels) + 1) * max(1., abs(max_term))
        if length == len(base) or (len(chosen) == k and scores[chosen[-1]] > bound + envelope):
            return np.asarray(chosen), dict(unique_scored=len(seen), sorted_reads=sorted_reads,
                iterations=iterations, base_prefix=length, complete_catalog_fallback=length == len(base), support_fallback=False,
                channel_entries=sum(len(x[0]) for x in channels))


class ReturnQuery(nn.Module):
    def __init__(self, width: int, dim: int = 64):
        super().__init__()
        def encoder():
            return nn.Sequential(nn.Linear(width, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.history = encoder()
        self.cue = encoder()
        self.item = encoder()
        self.z = nn.Sequential(nn.Linear(3 * dim + 1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.beta = nn.Linear(3 * dim + 1, 1)
        self.bias = nn.Linear(width, 1)
        self.log_scale = nn.Parameter(torch.tensor(2.3))

    def state(self, features, mask):
        x = F.normalize(self.history(features), dim=-1)
        weights = mask / mask.sum(-1, keepdim=True).clamp_min(1)
        mu = (x * weights[..., None]).sum(1)
        moment = x.transpose(1, 2) @ (x * weights[..., None])
        return mu, moment

    def query(self, mu, moment, cue_features, gap, rank=None):
        cue = F.normalize(self.cue(cue_features), dim=-1)
        context = torch.cat((mu, cue, mu * cue, torch.log1p(gap.clamp_min(0))[:, None] / 8), -1)
        scale = self.log_scale.exp().clamp(max=100)
        beta = F.softplus(self.beta(context)) * scale
        z = self.z(context) * scale
        eps = torch.zeros(len(mu), device=mu.device)
        if rank is None:
            read = (moment @ cue[..., None]).squeeze(-1)
        else:
            # Diagnostic only: production stores these factors offline.
            eig, vec = torch.linalg.eigh(moment.double())
            eig, vec = eig.flip(-1).float().clamp_min(0), vec.flip(-1).float()
            top = vec[:, :, :rank]
            projected = (top.transpose(1, 2) @ cue[..., None]).squeeze(-1)
            read = (top @ (eig[:, :rank] * projected)[..., None]).squeeze(-1)
            tail = (cue.square().sum(-1) - projected.square().sum(-1)).clamp_min(0).sqrt()
            eps = beta.squeeze(-1) * eig[:, rank] * tail
        return z + beta * read, eps

    def item_vectors(self, features):
        return F.normalize(self.item(features), dim=-1), self.bias(features).squeeze(-1)

    def forward(self, history, mask, cue, gap, candidates, rank=None):
        mu, moment = self.state(history, mask)
        query, eps = self.query(mu, moment, cue, gap, rank)
        vectors, bias = self.item_vectors(candidates)
        return torch.einsum("bd,bnd->bn", query, vectors) + bias, eps


def value_loss(scores, positives, counts, user_weights):
    discounts = 1 / torch.log2(torch.arange(2, 12, device=scores.device, dtype=scores.dtype))
    ideal = discounts.cumsum(0)[counts.long().clamp(1, 10) - 1]
    per_row = -(F.log_softmax(scores, -1) * positives).sum(-1) / ideal
    return (per_row * user_weights).mean()


def history_ids(data, row_ids, width=128):
    ids = np.zeros((len(row_ids), width), dtype=np.int64)
    mask = np.zeros_like(ids, dtype=np.float32)
    cue = np.empty(len(row_ids), dtype=np.int64)
    for at, row_id in enumerate(row_ids):
        _, first, start, _, _, _ = map(int, data.sessions[int(row_id)])
        begin = max(first, start - width)
        values = np.asarray(data.events[begin:start])
        ids[at, :len(values)] = values
        mask[at, :len(values)] = 1
        cue[at] = data.events[start]
    return ids, mask, cue


def _hash_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def static_features(args, data, checkpoint, manifest):
    """Offline, aligned landmark coordinates; no online sequence forward."""
    from run_arrive_joint_objective import deterministic_landmark_ids
    model, _ = load_standard_base(args, data, checkpoint_path=checkpoint, manifest_path=manifest)
    with torch.no_grad():
        emb = F.normalize(model.item_embedding.weight.detach(), dim=-1)
        landmarks = emb[deterministic_landmark_ids(data.frequencies, 32)]
        coords = (emb @ landmarks.T).cpu().numpy()
    del model, emb
    audio = np.load(args.content_dir / args.audio_file, mmap_mode="r")
    artist = sparse.load_npz(args.content_dir / args.categorical_file).tocsr()
    rng = np.random.default_rng(90121)
    projection = (rng.integers(0, 2, size=(artist.shape[1], 32), dtype=np.int8).astype(np.float32) * 2 - 1) / np.sqrt(32)
    artist_sketch = np.asarray(artist @ projection, dtype=np.float32)
    freq = np.log1p(np.asarray(data.frequencies, dtype=np.float32))[:, None] / 20
    values = np.concatenate((coords, np.asarray(audio), artist_sketch, freq), axis=1).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite_static_features")
    return torch.as_tensor(values, device=args.device)


def parent_pool(args, arrays, parent):
    from run_arrive_joint_objective import JointARRIVE, joint_universe, select_pool
    universe = joint_universe(arrays)
    selector = JointARRIVE(parent["proposal_width"], parent["behavior_width"], parent["hidden_dim"]).to(args.device)
    selector.load_state_dict(parent["model"], strict=True)
    selector.eval()
    pools, positives, selected_scores = [], [], []
    with torch.no_grad():
        for start in range(0, len(arrays["users"]), args.eval_batch_size):
            sl = slice(start, start + args.eval_batch_size)
            c = torch.as_tensor(universe["candidates"][sl], device=args.device)
            valid = torch.as_tensor(universe["valid"][sl], device=args.device)
            raw = torch.as_tensor(universe["raw_scores"][sl], device=args.device)
            features = torch.as_tensor(universe["proposal_features"][sl], device=args.device)
            proposal_scores = selector.proposal_scores(raw, features)
            selected = select_pool(proposal_scores, valid, c)
            selected_scores.append(proposal_scores.gather(1, selected).cpu().numpy())
            pools.append(c.gather(1, selected).cpu().numpy())
            p = torch.as_tensor(universe["positives"][sl], device=args.device)
            positives.append(p.gather(1, selected).cpu().numpy())
    universe["selected_scores"] = np.concatenate(selected_scores)
    return np.concatenate(pools), np.concatenate(positives), universe


def prepare_surface(args, data, arrays, parent):
    pools, positives, universe = parent_pool(args, arrays, parent)
    hist, mask, cue = history_ids(data, arrays["row_ids"], args.max_context - 1)
    if not np.array_equal(cue, arrays["observed"]):
        raise ValueError("cue_identity_mismatch")
    targets = supported_target_values(universe["candidates"], universe["positives"], arrays["target_counts"])
    return dict(arrays, pools=pools, rcce_top10=stable_topk(universe["selected_scores"], pools), positives=positives.astype(np.float32), history=hist,
                mask=mask, cue=cue, targets=targets, fit_eligible=universe["positives"].any(1))


def batch(surface, rows, static, device):
    def t(key):
        return torch.as_tensor(surface[key][rows], device=device)
    return (static[t("history")], t("mask"), static[t("cue")], t("gaps").float(), static[t("pools")])


@torch.no_grad()
def predict(args, model, surface, static, rows, rank=None):
    model.eval()
    predictions, epsilons = [], []
    for start in range(0, len(rows), args.eval_batch_size):
        idx = rows[start:start + args.eval_batch_size]
        scores, eps = model(*batch(surface, idx, static, args.device), rank=rank)
        predictions.append(stable_topk(scores.cpu().numpy(), surface["pools"][idx]))
        epsilons.append(eps.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(epsilons)


def summarize(surface, predictions, reference, seed):
    r, n = row_metrics(predictions, surface["targets"], surface["target_counts"])
    br, bn = row_metrics(reference, surface["targets"], surface["target_counts"])
    result = {}
    for scope, selected in (("overall", np.ones(len(r), bool)),
                            ("longest", surface["bucket_ids"] == surface["bucket_ids"].max())):
        result[scope] = {"recall": user_macro(r, surface["users"], selected),
                         "ndcg": user_macro(n, surface["users"], selected),
                         "delta_recall": paired_bootstrap(r-br, surface["users"], selected, seed),
                         "delta_ndcg": paired_bootstrap(n-bn, surface["users"], selected, seed+1)}
    return result, r, n


def run(args):
    from run_arrive_joint_objective import load_surface, validate_backbone_manifests, validate_feature_backbone_binding
    if args.return_query_eval_only:
        evaluate_saved(args)
        return
    if args.frozen_proposal_dir is None or args.component_variant != "full":
        raise ValueError("return_query_requires_frozen_full_parent")
    if args.output.exists() and not args.resume_if_present:
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    started = time.monotonic()
    _, _, method = validate_backbone_manifests(args)
    if method != "sasrec":
        raise ValueError("first_pilot_sasrec_only")
    train, train_manifest = load_surface(args, "train_cv")
    validate_feature_backbone_binding(train_manifest, args.base_checkpoint, args.base_training_manifest, method, "train")
    serving_args = argparse.Namespace(**vars(args))
    serving_args.base_checkpoint = args.serving_base_checkpoint
    serving_args.base_training_manifest = args.serving_base_training_manifest
    data = load_base_data(serving_args, args.serving_base_training_manifest)
    parent_path = args.frozen_proposal_dir / "joint_arrive_checkpoint.pt"
    parent_manifest = json.loads((args.frozen_proposal_dir / "training_manifest.json").read_text())
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent_manifest.get("status") != "pass" or parent_manifest.get("checkpoint_sha256") != digest(parent_path) or parent["seed"] != args.seed:
        raise ValueError("invalid_frozen_parent")
    binding = {"parent": digest(parent_path), "train_features": train_manifest["feature_sha256"],
               "train_base": digest(args.base_checkpoint), "serve_base": digest(args.serving_base_checkpoint),
               "code": digest(Path(__file__)), "shared_runner": digest(Path(__file__).with_name("run_arrive_joint_objective.py")),
               "audio": digest(args.content_dir / args.audio_file), "artist": digest(args.content_dir / args.categorical_file),
               "sessions": digest(args.data_dir / "sessions.npy"), "events": digest(args.data_dir / "events.npy"),
               "frequencies": digest(args.data_dir / "train_frequencies.npy")}
    for key, value in (("train_feature_sha256", binding["train_features"]),
                       ("train_base_checkpoint_sha256", binding["train_base"]),
                       ("serving_base_checkpoint_sha256", binding["serve_base"]),
                       ("audio_sha256", binding["audio"]), ("genre_sha256", binding["artist"])):
        if parent_manifest["binding"].get(key) != value:
            raise ValueError(f"parent_binding:{key}")
    surface = prepare_surface(args, data, train, parent)
    if not np.all(data.sessions[train["row_ids"], 5] == 0):
        raise ValueError("nontrain_fit_rows")
    fit = np.flatnonzero(user_partition(train["users"]) & surface["fit_eligible"])
    heldout = np.flatnonzero(~user_partition(train["users"]))
    if not len(fit) or not len(heldout) or set(train["users"][fit]) & set(train["users"][heldout]):
        raise ValueError("fit_selection_partition")
    static = static_features(args, data, args.base_checkpoint, args.base_training_manifest)
    model = ReturnQuery(static.shape[1], args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    config = {key: getattr(args, key) for key in ("seed", "epochs", "min_epochs", "patience", "min_delta", "batch_size", "eval_batch_size", "hidden_dim", "learning_rate", "weight_decay", "max_context")}
    config.update(selection="Train-heldout user-macro NDCG@10", history_weights="uniform", artist_projection_seed=90121)
    binding["config"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    binding["train_pool"] = _hash_array(surface["pools"])
    binding["fit_rows"] = _hash_array(train["row_ids"][fit])
    args.output.mkdir(parents=True, exist_ok=True)
    records, best, epoch0 = [], None, 0
    state_path = args.output / "last_state.pt"
    if args.resume_if_present and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["binding"] != binding:
            raise ValueError("resume_binding_changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        records, best, epoch0 = state["records"], state["best"], state["epoch"]
    _, inv, user_counts = np.unique(train["users"][fit], return_inverse=True, return_counts=True)
    weights = np.zeros(len(train["users"]), np.float32)
    weights[fit] = len(fit) / len(user_counts) / user_counts[inv]
    print(json.dumps({"event": "query_ready", "train_rows": len(train["users"]), "fit": len(fit), "selection": len(heldout), "binding": binding, "config": config}), flush=True)
    for epoch in range(epoch0 + 1, args.epochs + 1):
        model.train()
        start_epoch = time.monotonic()
        order = deterministic_shuffle(fit, args.seed + epoch, "return-query-fit")
        losses = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            scores, _ = model(*batch(surface, idx, static, args.device))
            loss = value_loss(scores, torch.as_tensor(surface["positives"][idx], device=args.device),
                              torch.as_tensor(surface["target_counts"][idx], device=args.device),
                              torch.as_tensor(weights[idx], device=args.device))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite_training_loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        preds, _ = predict(args, model, surface, static, heldout)
        rr, nn_ = row_metrics(preds, surface["targets"][heldout], surface["target_counts"][heldout])
        score = user_macro(nn_, surface["users"][heldout], np.ones(len(heldout), bool))
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "selection_score": score,
               "selection_recall": user_macro(rr, surface["users"][heldout], np.ones(len(heldout), bool)), "seconds": time.monotonic()-start_epoch}
        records.append(rec)
        if best is None or score > best["score"]:
            best = {"epoch": epoch, "score": score, "model": {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        atomic_torch(state_path, {"binding": binding, "epoch": epoch, "records": records, "best": best,
                                 "model": model.state_dict(), "optimizer": optimizer.state_dict()})
        atomic_json(args.output / "LEARNING_CURVE.json", records)
        print(json.dumps({"event": "return_query_epoch", **rec}), flush=True)
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            atomic_json(args.output / "SMOKE.json", {"status": "pass", "epoch": epoch, "binding": binding, "validation_accessed": False})
            return
        if early_stop_reached(records, patience=args.patience, min_epochs=args.min_epochs, min_delta=args.min_delta):
            break
    model.load_state_dict(best["model"], strict=True)
    checkpoint = args.output / "return_query_checkpoint.pt"
    atomic_torch(checkpoint, {"schema": SCHEMA, "model": model.state_dict(), "config": config,
                              "binding": binding, "width": static.shape[1], "best_epoch": best["epoch"]})
    loaded = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(loaded["model"], strict=True)
    del static
    finish_evaluation(args, model, data, parent, config, binding, best, records, started, checkpoint, method)


def evaluate_saved(args):
    from run_arrive_joint_objective import validate_backbone_manifests
    torch.set_num_threads(args.threads)
    checkpoint = args.output / "return_query_checkpoint.pt"
    saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
    state = torch.load(args.output / "last_state.pt", map_location="cpu", weights_only=False)
    if state["binding"] != saved["binding"]:
        raise ValueError("evaluation_checkpoint_state_binding")
    for key,value in state["best"]["model"].items():
        if not torch.equal(value, saved["model"][key].cpu()):
            raise ValueError("selected_model_changed")
    _,_,method = validate_backbone_manifests(args)
    for key,path in (("train_base",args.base_checkpoint),("serve_base",args.serving_base_checkpoint),
                     ("audio",args.content_dir/args.audio_file),("artist",args.content_dir/args.categorical_file),
                     ("sessions",args.data_dir/"sessions.npy"),("events",args.data_dir/"events.npy"),
                     ("frequencies",args.data_dir/"train_frequencies.npy"),
                     ("parent",args.frozen_proposal_dir/"joint_arrive_checkpoint.pt")):
        if saved["binding"][key] != digest(path):
            raise ValueError(f"evaluation_inputs_changed:{key}")
    model = ReturnQuery(saved["width"], saved["config"]["hidden_dim"]).to(args.device)
    model.load_state_dict(saved["model"],strict=True)
    ns=argparse.Namespace(**vars(args)); ns.base_checkpoint=args.serving_base_checkpoint
    ns.base_training_manifest=args.serving_base_training_manifest
    data=load_base_data(ns,args.serving_base_training_manifest)
    parent=torch.load(args.frozen_proposal_dir/"joint_arrive_checkpoint.pt",map_location="cpu",weights_only=False)
    finish_evaluation(args, model, data, parent, saved["config"], saved["binding"], state["best"],
                      state["records"], time.monotonic(), checkpoint, method)


def finish_evaluation(args, model, data, parent, config, binding, best, records, started, checkpoint, method):
    from run_arrive_joint_objective import load_surface, validate_feature_backbone_binding
    serving_args=argparse.Namespace(**vars(args))
    serving_args.base_checkpoint=args.serving_base_checkpoint
    serving_args.base_training_manifest=args.serving_base_training_manifest
    validation, vm = load_surface(args, "validation")
    validate_feature_backbone_binding(vm, args.serving_base_checkpoint, args.serving_base_training_manifest, method, "validation")
    surface = prepare_surface(args, data, validation, parent)
    static = static_features(serving_args, data, args.serving_base_checkpoint, args.serving_base_training_manifest)
    old = np.load(args.frozen_proposal_dir / "validation_predictions.npz", allow_pickle=False)
    for key in ("row_ids", "users", "target_counts"):
        if not np.array_equal(old[key], surface[key]):
            raise ValueError(f"reference_row_mismatch:{key}")
    if "candidate_top200" in old and not np.array_equal(old["candidate_top200"], surface["pools"]):
        raise ValueError("reference_pool_mismatch")
    if not np.array_equal(old["rcce_top10"],surface["rcce_top10"]):
        raise ValueError("parent_selector_top10_mismatch")
    if not np.all([np.isin(old["arrive_top10"][i],pool).all() for i,pool in enumerate(surface["pools"])]):
        raise ValueError("parent_predictions_outside_reconstructed_pool")
    pool_r,_=row_metrics(surface["pools"],surface["targets"],surface["target_counts"])
    if not np.allclose(pool_r,old["pool_recall"],rtol=0,atol=1e-12):
        raise ValueError("parent_pool_recall_mismatch")
    rows = np.arange(len(validation["users"]))
    full, _ = predict(args, model, surface, static, rows)
    compressed, eps = predict(args, model, surface, static, rows, rank=8)
    metrics, r, n = summarize(surface, full, old["arrive_top10"], args.seed)
    cm, cr, cn = summarize(surface, compressed, full, args.seed)
    atomic_npz(args.output / "validation_predictions.npz", row_ids=surface["row_ids"], users=surface["users"],
               bucket_ids=surface["bucket_ids"], target_counts=surface["target_counts"], targets=surface["targets"],
               candidate_top200=surface["pools"], full_top10=full, rank8_top10=compressed,
               arrive_top10=old["arrive_top10"], full_recall=r, full_ndcg=n, rank8_recall=cr, rank8_ndcg=cn, epsilon=eps)
    result = {"schema": SCHEMA, "status": "pass", "seed": args.seed, "config": config, "binding": binding,
              "validation_features": vm["feature_sha256"], "checkpoint_sha256": digest(checkpoint),
              "predictions_sha256": digest(args.output / "validation_predictions.npz"),
              "best_epoch": best["epoch"], "completed_epochs": len(records), "test_accessed": False,
              "fixed_pool_vs_arrive": metrics, "rank8_vs_full": cm,
              "rank8_changed_rows": int(np.any(full != compressed, axis=1).sum()),
              "seconds": time.monotonic()-started,
              "evaluation_producer_sha256": digest(Path(__file__)),
              "parent_pool_audit": "reconstructed from exact parent weights/features; rcce_top10 exact, arrive list membership and saved pool recall exact; historical Top200 file absent",
              "cost_status": "pending_complete_catalog_request_measurement_not_fixed_pool_timing"}
    atomic_json(args.output / "VALIDATION_RESULT.json", result)
    atomic_json(args.output / "training_manifest.json", result)
    print(json.dumps({"event": "return_query_complete", **result}), flush=True)
