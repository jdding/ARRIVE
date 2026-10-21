"""Train-only adaptive-history diagnostic, shared by the history runner.

No recommendation labels enter partition selection. This is a bounded
processing-mechanism check, not a fitted ranking model or a VISTA reproduction.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse


def partition_frontier(x, minimum=8, maximum=4):
    """Globally optimal contiguous SSE partitions for each feasible group count."""
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or minimum < 1 or maximum < 1:
        raise ValueError("invalid_partition_input")
    n = len(x)
    if not n:
        return []
    minimum = min(n, minimum)
    prefix = np.vstack([np.zeros((1, x.shape[1])), x.cumsum(0)])
    norm = (prefix * prefix).sum(1)
    length = np.arange(n + 1)[None] - np.arange(n + 1)[:, None]
    squares = np.r_[0., (x * x).sum(1).cumsum()]
    cost = squares[None] - squares[:, None] - (
        norm[:, None] + norm[None] - 2 * prefix @ prefix.T) / np.maximum(length, 1)
    cost = np.maximum(cost, 0.)
    cost[length < minimum] = np.inf
    kmax = min(maximum, n // minimum)
    dp = np.full((kmax + 1, n + 1), np.inf)
    back = np.full((kmax + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.
    result = []
    for k in range(1, kmax + 1):
        for end in range(k * minimum, n + 1):
            start = np.arange((k-1) * minimum, end-minimum+1)
            values = dp[k-1, start] + cost[start, end]
            at = int(np.argmin(values))
            dp[k, end], back[k, end] = values[at], start[at]
        group = np.empty(n, dtype=np.int64)
        end = n
        for j in range(k, 0, -1):
            begin = back[j, end]
            if begin < 0:
                raise ValueError("infeasible_partition")
            group[begin:end] = j-1
            end = begin
        result.append((float(dp[k, n]), group))
    return result


def complexity_statistic(x):
    """Largest SSE gain per additional complexity unit under shuffled order."""
    frontier = partition_frontier(x)
    if len(frontier) <= 1 or frontier[0][0] <= 1e-12:
        return 0.
    base = frontier[0][0]
    return max((1-cost/base) * len(x) / ((k-1)*np.log(len(x)))
               for k, (cost, _) in enumerate(frontier[1:], 2))


def adaptive_groups(x, penalty, equal_count=False):
    """Short histories retain individual events; long histories select1–4 groups."""
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("invalid_group_penalty")
    n = len(x)
    if n < 16:
        return np.arange(n, dtype=np.int64)
    frontier = partition_frontier(x)
    base = frontier[0][0]
    objective = [cost/base + penalty*(k-1)*np.log(n)/n
                 for k, (cost, _) in enumerate(frontier, 1)] if base > 1e-12 else [0.]
    at = int(np.argmin(objective))  # exact ties favor fewer groups
    if equal_count:
        return np.repeat(np.arange(at+1), [len(v) for v in np.array_split(np.arange(n), at+1)])
    return frontier[at][1]


def frozen_read(model, batch, items, groups=None):
    """Existing reader algebra, allowing a variable number of summary slots."""
    from arrive_history_reader import masked_softmax
    h = torch.tanh(model.encoder(items[batch["history"]]) +
                   model.position(torch.arange(128, device=items.device))[None])
    mask = batch["history"] >= 3
    b = model.pool(h).squeeze(-1)
    q = model.item_query(items[batch["ids"], :model.embedding_dim])
    q = q + model.cue_query(items[batch["cue"], :model.embedding_dim])[:, None]
    if groups is None:
        z, mass, present = h, b, mask
    else:
        slots = max(4, int(groups.max()) + 1)
        membership = (groups[:, None] == torch.arange(slots, device=items.device)[None, :, None]) & mask[:, None]
        within = masked_softmax(b[:, None].expand(-1, slots, -1), membership)
        z = torch.bmm(within, h)
        mass = b[:, None].expand(-1, slots, -1).masked_fill(~membership, -1e30).logsumexp(-1)
        present = membership.any(-1)
    weights = masked_softmax(mass[:, None] + torch.bmm(q, z.transpose(1, 2))/model.hidden**.5, present[:, None])
    r = torch.bmm(weights, z)
    scores = model.scorer(torch.cat(((batch["x"]-model.mean)/model.scale, r), -1)).squeeze(-1)
    return r, scores


def prepare_overlay(args):
    """Write only grouped-history metadata; share immutable full fitting assets."""
    from arrive_history_reader import Dataset
    from run_music4all_candidate_support_a1 import atomic_json, atomic_npz, digest
    from threadpoolctl import threadpool_limits
    policy = json.loads(args.group_policy.read_text())
    if (policy.get("schema") != "arrive.adaptive_history.diagnostic.v1"
            or policy.get("pilot_gate") != "ready_for_matched_fit"
            or policy.get("test_accessed") is not False
            or policy.get("validation_accessed") is not False):
        raise ValueError("adaptive_policy_not_frozen_train_only")
    data = Dataset(args.bundle/f"s{args.seed}"/args.surface)
    if data.receipt["seed"] != args.seed or data.receipt["surface"] != args.surface:
        raise ValueError("overlay_data_identity")
    out = args.output/f"s{args.seed}"/args.surface
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    history = data.m["history"]
    cg = np.full_like(history,-1); tg = np.full_like(history,-1)
    content = np.asarray(data.items[:, data.receipt["embedding_dim"]:])/np.sqrt(2.)
    with threadpool_limits(limits=1):
        for i,h in enumerate(history):
            valid = h >= 3
            x = content[h[valid]]
            g = adaptive_groups(x,policy["penalty"])
            cg[i,valid] = g
            # Same count and short-history route; only boundaries differ.
            if len(g):
                tg[i,valid] = np.repeat(np.arange(int(g.max())+1),
                                       [len(z) for z in np.array_split(np.arange(len(g)),int(g.max())+1)])
    out.mkdir(parents=True)
    atomic_npz(out/"ADAPTIVE_GROUPS.npz",row_ids=data.m["row_ids"],users=data.m["users"],
               history=history,adaptive_content=cg,adaptive_time=tg)
    atomic_json(out/"GROUP_MANIFEST.json",dict(schema="arrive.adaptive_groups.v1",status="pass",
        seed=args.seed,surface=args.surface,rows=len(history),test_accessed=False,
        smoke=data.receipt["smoke"], source_manifest_sha256=digest(data.path/"HISTORY_MANIFEST.json"),
        policy_sha256=digest(args.group_policy),penalty=policy["penalty"],
        producer_sha256=digest(Path(__file__)),output_sha256=digest(out/"ADAPTIVE_GROUPS.npz"),
        elapsed_seconds=time.monotonic()-start))
    print(json.dumps(dict(event="adaptive_overlay_ready",seed=args.seed,surface=args.surface,seconds=time.monotonic()-start)),flush=True)


def attach_overlay(data, root):
    from run_music4all_candidate_support_a1 import digest
    if root is None:
        raise ValueError("adaptive_overlay_required")
    path = root/f"s{data.receipt['seed']}"/data.receipt["surface"]
    receipt = json.loads((path/"GROUP_MANIFEST.json").read_text())
    if (receipt.get("schema") != "arrive.adaptive_groups.v1" or receipt.get("status") != "pass"
            or receipt.get("test_accessed") is not False
            or receipt["source_manifest_sha256"] != digest(data.path/"HISTORY_MANIFEST.json")
            or receipt["output_sha256"] != digest(path/"ADAPTIVE_GROUPS.npz")
            or receipt["producer_sha256"] != digest(Path(__file__))
            or any(receipt[k] != data.receipt[k] for k in ("seed","surface","rows","smoke"))):
        raise ValueError("adaptive_overlay_binding_mismatch")
    with np.load(path/"ADAPTIVE_GROUPS.npz",allow_pickle=False) as z:
        for k in ("row_ids","users","history"):
            if not np.array_equal(z[k],data.m[k]):
                raise ValueError("adaptive_overlay_row_mismatch")
        for k in ("adaptive_content","adaptive_time"):
            g = z[k]
            if g.shape != data.m["history"].shape or not np.array_equal(g >= 0,data.m["history"] >= 3):
                raise ValueError("adaptive_overlay_coverage")
            for row,h in zip(g,data.m["history"]):
                active = row[h >= 3]
                if not len(active):
                    continue
                if not np.array_equal(np.unique(active),np.arange(active.max()+1)) or np.any(np.diff(active) < 0):
                    raise ValueError("adaptive_overlay_contiguity")
                if len(active) < 16:
                    if not np.array_equal(active,np.arange(len(active))):
                        raise ValueError("adaptive_short_not_direct")
                elif active.max() >= 4 or np.bincount(active).min() < 8:
                    raise ValueError("adaptive_long_budget")
            data.m[k] = g.copy()
    if not np.array_equal(data.m['adaptive_content'].max(1),data.m['adaptive_time'].max(1)):
        raise ValueError("adaptive_control_count_mismatch")
    return {"manifest":digest(path/"GROUP_MANIFEST.json"),"policy":receipt["policy_sha256"],
            "producer":receipt["producer_sha256"]}


def audit(args):
    """Local deterministic no-fit check; only existing Train inputs are loaded."""
    from arrive_history_reader import HistoryScorer, groups_for, user_partition
    from run_music4all_candidate_support_a1 import atomic_json, digest, stable_topk
    from run_music4all_component_c2_conversion_oof import row_metrics, user_macro
    from threadpoolctl import threadpool_limits
    root, out = args.project, args.output
    if out.exists():
        raise FileExistsError(out)
    began = time.monotonic()
    torch.set_num_threads(1)
    prepared = root / "data/music4all_onion/prepared_v1"
    events = np.load(prepared/"events.npy", mmap_mode="r")
    sessions = np.load(prepared/"sessions.npy", mmap_mode="r")
    folder = root/"data/music4all_onion/component_b2_attainability_v1"
    audio = np.load(folder/"musicnn_normalized.npy")
    genre = sparse.load_npz(folder/"genre_normalized.npz").toarray()
    content = np.concatenate([audio, genre], 1).astype(np.float32)/np.sqrt(2.)
    fit_mask = user_partition(sessions[:,0])
    eligible = np.flatnonzero((sessions[:,5] == 0) & fit_mask)
    # One earliest eligible prefix per user, ranked by a fixed identifier hash.
    _, first = np.unique(sessions[eligible,0], return_index=True)
    ordered = sorted(eligible[first], key=lambda r: hashlib.sha256(
        f"arrive-adaptive-fit-v1:{sessions[r,0]}".encode()).digest())
    calibration, held = np.asarray(ordered[:512]), np.asarray(ordered[512:1024])
    assert len(calibration) == len(held) == 512
    assert not set(sessions[calibration,0]) & set(sessions[held,0])

    def past(row):
        s = sessions[row]
        assert s[5] == 0
        h = np.asarray(events[int(s[1]):int(s[2])]); h = h[h >= 3]
        assert len(h) <= 128
        return h

    rng = np.random.default_rng(92626)
    with threadpool_limits(limits=1):
        null = [complexity_statistic(content[rng.permutation(past(r))]) for r in calibration]
        penalty = float(np.percentile(null,95))
        counts, short, minimums = [], 0, []
        for row in held:
            h = past(row); g = adaptive_groups(content[h], penalty)
            counts.append(len(np.unique(g))); short += len(h) < 16
            if len(h) >= 16:
                minimums.extend(np.bincount(g).tolist())
        print(json.dumps(dict(event="adaptive_partition_diagnostic",penalty=penalty,
                              mean_groups=np.mean(counts),short_direct_rows=short)),flush=True)

        asset = root/"refine-logs/arrive_cue_residual_20260925/current_assets_a6e10d63_s2024.npz"
        with np.load(asset) as z:
            names = ("ids","features","relation","targets","target_counts","row_ids","users","buckets")
            source = {k:z[k] for k in names}
        assert np.all(~fit_mask[source["row_ids"]])
        assert np.all(sessions[source["row_ids"],5] == 0)
        assert np.array_equal(sessions[source["row_ids"],0],source["users"])
        _, first = np.unique(source["users"], return_index=True)
        selected = sorted(first,key=lambda r: hashlib.sha256(
            f"arrive-adaptive-cal-v1:{source['users'][r]}".encode()).digest())[:512]
        source = {k:v[selected] for k,v in source.items()}
        checkpoint = root/"runs/paired_20260923/onion/crossfit/s2024/final_model.pt"
        embedding = torch.load(checkpoint,map_location="cpu",weights_only=False)["model"]["item_embedding.weight"].numpy()
        items = torch.tensor(np.concatenate([embedding,audio,genre],1).astype(np.float32))
        modeldir = root/"refine-logs/arrive_content_history_20260926/remote_results/models/content/s2024"
        receipt = json.loads((modeldir/"training_manifest.json").read_text())
        assert receipt["status"] == "pass" and not receipt["test_accessed"]
        assert digest(modeldir/"best_model.pt") == receipt["output_sha256"]["best_model.pt"]
        model = HistoryScorer(items.shape[1],embedding.shape[1],"content")
        saved = torch.load(modeldir/"best_model.pt",map_location="cpu",weights_only=False)
        assert saved["binding"] == receipt["binding"]
        model.load_state_dict(saved["model"]); model.eval()
        data = {name:dict(read_mse=[],relative_read_error=[],score_mse=[],top10_overlap=[],top=[],groups=[])
                for name in ("fixed4","adaptive_content","adaptive_time")}
        full_top = []
        for i,row in enumerate(source["row_ids"]):
            h = past(row); history = np.zeros((1,128),np.int64); history[0,:len(h)] = h
            x = np.concatenate([source["features"][i],np.broadcast_to(source["relation"][i,14:17],(200,3))],1)
            batch = dict(history=torch.tensor(history),ids=torch.tensor(source["ids"][i:i+1].astype(np.int64)),
                         cue=torch.tensor([int(events[int(sessions[row,2])])]),x=torch.tensor(x[None],dtype=torch.float32))
            with torch.no_grad():
                full_r,full_s = frozen_read(model,batch,items)
                ref = stable_topk(full_s.numpy(),source["ids"][i:i+1],10)[0]
                full_top.append(ref)
                for name in data:
                    g = groups_for(content[h]) if name == "fixed4" else adaptive_groups(content[h],penalty,name == "adaptive_time")
                    padded = np.full((1,128),-1,np.int64); padded[0,:len(h)] = g
                    r,s = frozen_read(model,batch,items,torch.tensor(padded))
                    if name == "fixed4":
                        batch["content_groups"] = torch.tensor(padded)
                        torch.testing.assert_close(s,model(batch,items),atol=0,rtol=0)
                    top = stable_topk(s.numpy(),source["ids"][i:i+1],10)[0]
                    d = data[name]; d["top"].append(top); d["groups"].append(len(np.unique(g)))
                    d["read_mse"].append(float((r-full_r).square().mean()))
                    d["relative_read_error"].append(float((r-full_r).square().sum().sqrt()/full_r.square().sum().sqrt().clamp_min(1e-12)))
                    d["score_mse"].append(float((s-full_s).square().mean()))
                    d["top10_overlap"].append(len(set(top)&set(ref))/10)
            if (i+1)%128 == 0:
                print(json.dumps(dict(event="adaptive_fidelity_rows",rows=i+1)),flush=True)
        metrics = {}
        for name,d in list(data.items())+[("full_same_weight",dict(top=full_top))]:
            recall,ndcg = row_metrics(np.asarray(d["top"]),source["targets"],source["target_counts"])
            metrics[name] = {k:dict(mean=float(np.mean(v)),median=float(np.median(v))) for k,v in d.items() if k != "top"}
            metrics[name].update(recall=user_macro(recall,source["users"],np.ones(len(recall),bool)),
                                 ndcg=user_macro(ndcg,source["users"],np.ones(len(ndcg),bool)))
        fixed,adaptive = metrics["fixed4"],metrics["adaptive_content"]
        ratio = adaptive["relative_read_error"]["mean"]/max(fixed["relative_read_error"]["mean"],1e-12)
        median_ratio = adaptive["relative_read_error"]["median"]/max(fixed["relative_read_error"]["median"],1e-12)
        pilot = ratio <= 1.10 and median_ratio <= 1.10 and adaptive["groups"]["mean"] < fixed["groups"]["mean"]
        result = dict(schema="arrive.adaptive_history.diagnostic.v1",status="complete",test_accessed=False,
                      validation_accessed=False,training_performed=False,rows=512,
                      penalty=penalty,penalty_null_quantile=.95,min_segment=8,max_segments=4,short_direct_threshold=16,
                      group_count_histogram={str(k):counts.count(k) for k in sorted(set(counts))},
                      short_direct_rows=short,long_segment_min=min(minimums),
                      calibration_users=512,heldout_fit_users=512,train_cal_probe_users=512,
                      calibration_row_ids=calibration.tolist(),heldout_fit_row_ids=held.tolist(),
                      train_cal_probe_row_ids=source["row_ids"].tolist(), metrics=metrics,
                      fixed4_forward_exact=True,mean_read_distortion_ratio=ratio,median_read_distortion_ratio=median_ratio,
                      pilot_gate="ready_for_matched_fit" if pilot else "revise_compression_before_fit",
                      limitations=["Existing model trained on fixed4; altered reading is an intervention, not fitted adaptive performance.",
                                   "C200 Train-cal diagnostic only; formal fitting must retain the full merged training pool.",
                                   "Split penalty is a heuristic fitted to shuffled Train-fit histories, not a significance test.",
                                   "No proof of novelty, better ranking, or complete-request speed."],
                      sources={str(p.relative_to(root)):digest(p) for p in [asset,checkpoint,modeldir/"best_model.pt",
                               prepared/"sessions.npy",folder/"musicnn_normalized.npy",folder/"genre_normalized.npz"]},
                      elapsed_seconds=time.monotonic()-began)
        out.mkdir(parents=True)
        atomic_json(out/"DIAGNOSTIC.json",result)
        print(json.dumps(dict(event="adaptive_diagnostic_complete",gate=result["pilot_gate"],ratio=ratio,metrics=metrics)),flush=True)
