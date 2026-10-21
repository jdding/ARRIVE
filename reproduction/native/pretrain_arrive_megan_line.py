#!/usr/bin/env python3
"""Train the shared Train-only LINE initialization required by MEGAN.

The input is the frozen heterogeneous-music-graph (HMG) neighborhood asset.
Those neighborhoods were sampled once in proportion to the complete Train edge
weights; duplicate sampled edges therefore retain their multiplicity here and
must not be weighted a second time.  Metadata edges are added from the frozen
user/music token tables.  No Validation or Test labels are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


RELATIONS = (
    ("user_music", "user", "music"),
    ("user_session", "user", "session"),
    ("music_user", "music", "user"),
    ("music_music", "music", "music"),
    ("music_session", "music", "session"),
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def graph_edges(cache: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, dict]:
    counts = {
        "user": int(cache["user_tokens"].shape[0]),
        "music": int(cache["music_tokens"].shape[0]),
        "session": int(cache["session_count"]),
        "feature": int(cache["feature_count"]),
    }
    offsets, total = {}, 0
    for name in ("user", "music", "session", "feature"):
        offsets[name] = total
        total += counts[name]

    sources, targets = [], []
    relation_counts: dict[str, int] = {}
    for name, source_type, target_type in RELATIONS:
        ids = np.asarray(cache[name + "_ids"], dtype=np.int64)
        valid = ids >= 0
        source = np.broadcast_to(
            np.arange(len(ids), dtype=np.int64)[:, None], ids.shape
        )[valid]
        target = ids[valid]
        sources.append(source + offsets[source_type])
        targets.append(target + offsets[target_type])
        relation_counts[name] = int(valid.sum())
        # The stored downstream views do not query session nodes as sources,
        # but LINE must train a usable session representation.  HMG membership
        # edges are undirected, so materialize their missing reverse direction.
        if target_type == "session":
            sources.append(target + offsets[target_type])
            targets.append(source + offsets[source_type])
            relation_counts[name + "_reverse"] = int(valid.sum())

    # Profile/attribute feature edges are explicit HMG relations.  Add both
    # directions so feature nodes receive a LINE source representation too.
    for owner, node_type in (("user_tokens", "user"), ("music_tokens", "music")):
        tokens = np.asarray(cache[owner], dtype=np.int64)
        valid = tokens > 0
        node = np.broadcast_to(
            np.arange(len(tokens), dtype=np.int64)[:, None], tokens.shape
        )[valid] + offsets[node_type]
        feature = tokens[valid] + offsets["feature"]
        sources.extend((node, feature))
        targets.extend((feature, node))
        relation_counts[node_type + "_feature_bidirectional"] = int(2 * valid.sum())

    source = np.concatenate(sources)
    target = np.concatenate(targets)
    if len(source) == 0 or source.min() < 0 or target.min() < 0:
        raise ValueError("empty_or_invalid_hmg_edges")
    if source.max() >= total or target.max() >= total:
        raise ValueError("hmg_edge_out_of_range")
    return source, target, {
        "counts": counts,
        "offsets": offsets,
        "nodes": total,
        "edges": int(len(source)),
        "relation_sample_counts": relation_counts,
    }


class LINE(nn.Module):
    """Independent first- and second-order LINE objectives."""

    def __init__(self, nodes: int, half_dim: int) -> None:
        super().__init__()
        self.first = nn.Embedding(nodes, half_dim, sparse=True)
        self.second_source = nn.Embedding(nodes, half_dim, sparse=True)
        self.second_context = nn.Embedding(nodes, half_dim, sparse=True)
        scale = 1.0 / max(1, half_dim)
        for embedding in (self.first, self.second_source, self.second_context):
            nn.init.uniform_(embedding.weight, -scale, scale)

    def loss(self, source: torch.Tensor, target: torch.Tensor, negative: torch.Tensor):
        first_source, first_target = self.first(source), self.first(target)
        first_positive = (first_source * first_target).sum(-1)
        first_negative = torch.einsum("bd,bkd->bk", first_source, self.first(negative))
        second_source = self.second_source(source)
        second_positive = (second_source * self.second_context(target)).sum(-1)
        second_negative = torch.einsum(
            "bd,bkd->bk", second_source, self.second_context(negative)
        )
        first_per_edge = -(F.logsigmoid(first_positive) + F.logsigmoid(-first_negative).sum(-1))
        second_per_edge = -(F.logsigmoid(second_positive) + F.logsigmoid(-second_negative).sum(-1))
        # LINE is trained with edge-wise SGD.  Summing the independently sampled
        # edge objectives preserves that update scale when edges are grouped into
        # a large efficiency batch; averaging here would divide every sparse node
        # update by the batch size and leave the initialization effectively frozen.
        optimization_loss = first_per_edge.sum() + second_per_edge.sum()
        return optimization_loss, first_per_edge.mean().detach(), second_per_edge.mean().detach()


def train(args: argparse.Namespace) -> None:
    if args.dim % 2 or args.dim < 2 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("invalid_LINE_configuration")
    args.output.mkdir(parents=True, exist_ok=True)
    input_manifest = json.loads(args.hmg_manifest.read_text())
    if (
        input_manifest.get("schema") != "genrec.arrive_megan_hmg_assets.v1"
        or input_manifest.get("status") != "pass"
        or input_manifest.get("test_accessed") is not False
        or input_manifest.get("artifact_sha256") != digest(args.hmg_assets)
    ):
        raise ValueError("unbound_hmg_asset")

    cache = np.load(args.hmg_assets, allow_pickle=False)
    source_np, target_np, graph = graph_edges(cache)
    degree = np.bincount(target_np, minlength=graph["nodes"]).astype(np.float64)
    distribution = np.power(degree, 0.75)
    if not np.isfinite(distribution).all() or distribution.sum() <= 0:
        raise ValueError("invalid_negative_distribution")
    distribution /= distribution.sum()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    source = torch.as_tensor(source_np, dtype=torch.long, device=device)
    target = torch.as_tensor(target_np, dtype=torch.long, device=device)
    noise = torch.as_tensor(distribution, dtype=torch.float32, device=device)
    model = LINE(graph["nodes"], args.dim // 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    updates_per_epoch = (graph["edges"] + args.batch_size - 1) // args.batch_size
    target_updates = args.epochs * updates_per_epoch
    if args.stop_after:
        target_updates = min(target_updates, args.stop_after)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    started = time.time()
    completed = 0
    last = None
    while completed < target_updates:
        positive_indices = torch.randint(
            graph["edges"], (args.batch_size,), generator=generator, device=device
        )
        negative = torch.multinomial(
            noise, args.batch_size * args.negatives, replacement=True, generator=generator
        ).reshape(args.batch_size, args.negatives)
        fraction = completed / max(1, target_updates)
        learning_rate = max(args.learning_rate * (1.0 - fraction), args.learning_rate * 1e-4)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        loss, first, second = model.loss(
            source[positive_indices], target[positive_indices], negative
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite_LINE_loss")
        loss.backward()
        optimizer.step()
        completed += 1
        last = (float(loss.detach()) / args.batch_size, float(first), float(second))
        if completed == 1 or completed % args.log_interval == 0 or completed == target_updates:
            print(json.dumps({
                "event": "update", "completed_updates": completed,
                "target_updates": target_updates, "loss": last[0],
                "first_order_loss": last[1], "second_order_loss": last[2],
                "updates_per_second": completed / max(time.time() - started, 1e-9),
            }), flush=True)

    embeddings = torch.cat((model.first.weight, model.second_source.weight), dim=1)
    embeddings = embeddings.detach().cpu().numpy().astype(np.float32)
    output = args.output / "megan_line_embeddings.npy"
    np.save(output, embeddings)
    manifest = {
        "schema": "genrec.arrive_megan_line_initialization.v1",
        "status": "engineering_pass" if args.stop_after else "pass",
        "hmg_assets_sha256": digest(args.hmg_assets),
        "hmg_manifest_sha256": digest(args.hmg_manifest),
        "producer_sha256": digest(Path(__file__)),
        "artifact_sha256": digest(output),
        "artifact": output.name,
        "training_split": "Train",
        "validation_accessed": False,
        "test_accessed": False,
        "objective": "LINE first-order plus second-order negative sampling; independent half-spaces concatenated",
        "optimization_reduction": "sum_of_sampled_edge_objectives_matching_edge_wise_SGD",
        "edge_semantics": "frozen weight-proportional HMG edge samples plus bidirectional metadata edges",
        "negative_distribution": "target_degree_power_0.75",
        "seed": args.seed,
        "dimension": args.dim,
        "negative_samples": args.negatives,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "completed_updates": completed,
        "target_updates": target_updates,
        "learning_rate": args.learning_rate,
        "elapsed_seconds": time.time() - started,
        "final_loss": last[0] if last else None,
        **graph,
    }
    atomic_json(args.output / "MEGAN_LINE.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("data/music4all_onion/faithful_baseline_assets_v1")
    parser.add_argument("--hmg-assets", type=Path, default=root / "megan_hmg_assets.npz")
    parser.add_argument("--hmg-manifest", type=Path, default=root / "MEGAN_ASSETS.json")
    parser.add_argument("--output", type=Path, default=root / "megan_line_v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--stop-after", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
