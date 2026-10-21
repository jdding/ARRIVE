"""Matched sequence/parallel probability models for one exploratory music pilot."""
from __future__ import annotations

import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

PAD, UNK, EOS = 0, 1, 2


class MusicData:
    def __init__(self, root, horizon=64):
        self.root, self.horizon = Path(root), horizon
        self.events = np.load(self.root / "events.npy", mmap_mode="r")
        self.sessions = np.load(self.root / "sessions.npy", mmap_mode="r")
        self.vocab_size = len(np.load(self.root / "train_frequencies.npy", mmap_mode="r"))

    def row_ids(self, split, returns_only=False):
        ok = self.sessions[:, 5] == split
        if returns_only:
            ok &= self.sessions[:, 4] >= 30 * 86400
        return np.flatnonzero(ok)

    def batch(self, row_ids, device):
        histories, previous, targets, gaps, lengths = [], [], [], [], []
        for index in row_ids:
            _, first_user, start, end, gap, _ = map(int, self.sessions[index])
            history = np.asarray(self.events[max(first_user, start - 128):start + 1])
            suffix = list(map(int, self.events[start + 1:min(end, start + 1 + self.horizon)]))
            # At a crop there is no observed EOS; never invent one.
            if end - start - 1 < self.horizon:
                suffix.append(EOS)
            assert history.size > 0 and 0 < len(suffix) <= self.horizon
            histories.append(history)
            lengths.append(len(history))
            targets.append(suffix)
            previous.append([int(self.events[start])] + suffix[:-1])
            gaps.append([math.log1p(max(gap, 0) / 86400) / math.log(366), float(gap < 0)])
        h = np.zeros((len(row_ids), max(lengths)), dtype=np.int64)
        y = np.zeros((len(row_ids), max(map(len, targets))), dtype=np.int64)
        p = np.zeros_like(y)
        for b in range(len(row_ids)):
            h[b, :lengths[b]] = histories[b]
            y[b, :len(targets[b])] = targets[b]
            p[b, :len(previous[b])] = previous[b]
        return {"history": torch.as_tensor(h, device=device),
                "lengths": torch.tensor(lengths),
                "previous": torch.as_tensor(p, device=device),
                "target": torch.as_tensor(y, device=device),
                "gap": torch.tensor(gaps, dtype=torch.float32, device=device)}


class MusicModel(nn.Module):
    def __init__(self, vocab_size, variant, horizon=64, embedding=64, hidden=128, dropout=.1):
        super().__init__()
        assert variant in {"sequence", "parallel"}
        self.variant, self.horizon = variant, horizon
        self.embedding = nn.Embedding(vocab_size, embedding, padding_idx=PAD)
        self.encoder = nn.GRU(embedding, hidden, batch_first=True)
        self.query = nn.Linear(hidden + 2, hidden)
        self.positions = nn.Embedding(horizon, embedding)
        self.decoder = nn.GRU(embedding, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # Output class 0 is EOS; item k>=3 maps to k-2. PAD/UNK have no output mass.
        classes = vocab_size - 2
        cutoffs = [c for c in (10000, 100000) if c < classes]
        if not cutoffs:
            cutoffs = [max(1, classes // 2)]
        self.output = nn.AdaptiveLogSoftmaxWithLoss(hidden, classes, cutoffs, div_value=2)

    def context(self, batch):
        x = self.dropout(self.embedding(batch["history"]))
        packed = pack_padded_sequence(x, batch["lengths"].cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        return torch.tanh(self.query(torch.cat([hidden[-1], batch["gap"]], dim=-1)))

    def hidden_states(self, batch):
        prev = batch["previous"]
        if self.variant == "parallel":
            # Same observed first item at every position: no teacher-target input.
            prev = prev[:, :1].expand_as(prev)
        positions = torch.arange(prev.shape[1], device=prev.device)
        x = self.dropout(self.embedding(prev) + self.positions(positions)[None])
        hidden, _ = self.decoder(x, self.context(batch)[None])
        return hidden

    def loss(self, batch):
        states = self.hidden_states(batch)
        valid = batch["target"] >= EOS
        return self.output(states[valid], batch["target"][valid] - 2).loss

    def step(self, previous, hidden, position):
        pos = torch.full_like(previous, position)
        x = self.embedding(previous) + self.positions(pos)
        y, hidden = self.decoder(x[:, None], hidden)
        return y[:, 0], hidden


def set_metrics(prediction, truth):
    relevant = set(map(int, truth))
    if not relevant:
        return None, None
    found = set()
    hits = []
    for token in prediction[:10]:
        hit = token >= 3 and token in relevant and token not in found
        hits.append(float(hit))
        found.add(token)
    recall = sum(hits) / len(relevant)
    # Fixed K=10 IDCG: early stopping must not shrink its own denominator.
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    return recall, sum(hit / math.log2(i + 2) for i, hit in enumerate(hits)) / ideal


def edit_error(predicted, target, cap):
    a, b = list(predicted[:cap]), list(target[:cap])
    prior = list(range(len(b) + 1))
    for i, token in enumerate(a, 1):
        row = [i]
        for j, other in enumerate(b, 1):
            row.append(min(row[-1] + 1, prior[j] + 1, prior[j - 1] + (token != other)))
        prior = row
    return prior[-1] / max(len(a), len(b), 1)


@torch.no_grad()
def evaluate(model, data, split, device, limit=None, direct=True):
    """Model inputs never load the separate label identities; metrics do."""
    model.eval()
    identity = np.load(data.root / "metric_labels.npz", allow_pickle=False)
    lookup = {int(row): i for i, row in enumerate(identity["row_ids"])}
    indices = data.row_ids(split, True)
    if limit is not None:
        indices = indices[:limit]  # allowed only for labeled engineering smoke.
    result, start_time = [], time.monotonic()
    for at in range(0, len(indices), 4):
        ids = indices[at:at + 4]
        b = data.batch(ids, device)
        states = model.hidden_states(b)
        valid = b["target"] >= EOS
        nll = torch.zeros_like(b["target"], dtype=torch.float32)
        nll[valid] = -model.output(states[valid], b["target"][valid] - 2).output
        first_lp = model.output.log_prob(states[:, 0])
        first_eos = first_lp[:, 0].exp().cpu().tolist()
        first_top = first_lp[:, 1:].topk(min(10, first_lp.shape[1] - 1), dim=-1).indices + 3
        first_top = first_top.cpu().tolist()
        del first_lp, states

        seed = b["previous"][:, 0]
        previous = seed.clone()
        hidden = model.context(b)[None]
        generated = [[] for _ in ids]
        active = [True] * len(ids)
        total_presence = no_item_yet = None
        for t in range(model.horizon):
            h, hidden = model.step(previous, hidden, t)
            chosen = (model.output.predict(h) + 2).cpu().tolist()
            if model.variant == "parallel" and direct:
                probabilities = model.output.log_prob(h).exp()
                item_p = probabilities[:, 1:]
                eos_p = probabilities[:, :1]
                if total_presence is None:
                    total_presence = torch.zeros_like(item_p)
                    no_item_yet = torch.ones_like(item_p)
                # Exact item-inclusion marginal of the independent positional
                # law with EOS: no use of generated choices or future labels.
                total_presence.add_(no_item_yet * item_p)
                no_item_yet.mul_((1 - eos_p - item_p).clamp_min(0))
                del probabilities, item_p, eos_p
            for j, token in enumerate(chosen):
                if active[j]:
                    if token == EOS:
                        active[j] = False
                    else:
                        generated[j].append(token)
            previous = seed if model.variant == "parallel" else torch.tensor(chosen, device=device)
            if not any(active) and model.variant == "sequence":
                break
        direct_rank = (total_presence.topk(min(10, total_presence.shape[1]), dim=-1).indices + 3).cpu().tolist() if total_presence is not None else generated
        for j, row_id in enumerate(ids):
            ix = lookup[int(row_id)]
            truth = identity["values"][identity["offsets"][ix]:identity["offsets"][ix + 1]].tolist()
            user, _, _, _, gap, _ = map(int, data.sessions[row_id])
            pred = generated[j]
            recall, ndcg = set_metrics(pred, truth)
            dr, dn = set_metrics(direct_rank[j], truth)
            first_rank = first_top[j].index(truth[0]) + 1 if truth and truth[0] in first_top[j] else None
            item_mask = b["target"][j] >= 3
            eos_mask = b["target"][j] == EOS
            result.append({
                "row_id": int(row_id), "user": user, "gap_days": gap / 86400,
                "suffix_length": len(truth), "reachable_suffix_events": sum(v >= 3 for v in truth),
                "generated_length": len(pred), "eos_emitted": not active[j],
                "length_abs_error": abs(len(pred) - len(truth)),
                "empty_suffix": not truth, "first_eos_probability": first_eos[j],
                "empty_brier": (first_eos[j] - float(not truth)) ** 2,
                "next_hit10": float(first_rank is not None) if truth else None,
                "next_ndcg10": 1 / math.log2(first_rank + 1) if first_rank else 0.0 if truth else None,
                "free_recall10": recall, "free_ndcg10": ndcg,
                "direct_recall10": dr, "direct_ndcg10": dn,
                "known_item_nll_sum": float(nll[j][item_mask].sum()),
                "known_item_nll_count": int(item_mask.sum()),
                "eos_nll_sum": float(nll[j][eos_mask].sum()), "eos_nll_count": int(eos_mask.sum()),
                "duplicate_fraction": 1 - len(set(pred)) / len(pred) if pred else 0.0,
                "capped_edit_error": edit_error(pred, truth, model.horizon),
                "generated_tokens": pred, "direct_top10": direct_rank[j][:10],
            })
        if at % 100 == 0:
            print(f"evaluation split={split} rows={min(at + 4, len(indices))}/{len(indices)} seconds={time.monotonic()-start_time:.1f}", flush=True)
    metrics = {}
    for key in ("next_hit10", "next_ndcg10", "free_recall10", "free_ndcg10", "direct_recall10", "direct_ndcg10", "length_abs_error", "empty_brier", "capped_edit_error"):
        by_user = {}
        for row in result:
            if row[key] is not None:
                by_user.setdefault(row["user"], []).append(row[key])
        metrics[key] = float(np.mean([np.mean(v) for v in by_user.values()])) if by_user else None
    metrics["known_item_nll"] = sum(r["known_item_nll_sum"] for r in result) / max(sum(r["known_item_nll_count"] for r in result), 1)
    return {"rows": result, "user_macro_metrics": metrics, "split": split,
            "evaluation_seconds": time.monotonic() - start_time,
            "engineering_limit": limit, "development_is_reused": split == 2}
