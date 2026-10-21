#!/usr/bin/env python3
"""Task-matched closest-prior and backbone controls for ARRIVE.

The runner deliberately keeps the Music4All protocol, sampled-softmax
correction, catalog and evaluator shared with the frozen SASRec predecessor.
It implements only the missing scientific controls:

* ``car``: gated fusion of a durable-history state and the current return item;
* ``hybrid``: ordinary identity-plus-audio/genre feature fusion;
* ``gru4rec``: an alternate recurrent sequential backbone.

The already completed ``timeconcat_oneshot`` run is the matched ordinary
multi-positive session-set control and is therefore not retrained here.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy import sparse
import torch
from torch import nn

import run_lfm_timeconcat_control as training
from lfm_timeconcat_control import (
    FIRST_ITEM,
    PAD,
    LFMTimeConcatData,
    LFMTimeConcatSASRec,
    gap_bucket,
    popularity_noise,
)
from music_sequence_core import set_metrics


STATE_SCHEMA = "genrec.music4all_onion.arrive_confirmation_state.v1"
TRAINING_SCHEMA = "genrec.music4all_onion.arrive_confirmation_training.v1"
VARIANTS = ("car", "hybrid", "gru4rec")
OBJECTIVE = {
    "car": "oneshot_only",
    "hybrid": "oneshot_only",
    "gru4rec": "continuation_only",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def deterministic_content_features(audio_path: Path, genre_path: Path, seed: int = 1701) -> np.ndarray:
    """Create a fixed, label-free audio plus projected-genre representation."""

    audio = np.asarray(np.load(audio_path, mmap_mode="r"), dtype=np.float32)
    genre = sparse.load_npz(genre_path).tocsr().astype(np.float32)
    if len(audio) != genre.shape[0]:
        raise ValueError("content_row_mismatch")
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((genre.shape[1], 32), dtype=np.float32) / np.sqrt(32.0)
    projected = np.asarray(genre @ projection, dtype=np.float32)
    features = np.concatenate((audio, projected), axis=1)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = np.divide(features, np.maximum(norms, 1e-8), dtype=np.float32)
    features[:FIRST_ITEM] = 0.0
    if not np.isfinite(features).all():
        raise ValueError("nonfinite_content_features")
    return features


class CARControl(LFMTimeConcatSASRec):
    """General/current preference gate adapted to the frozen return task."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        dimension = self.item_embedding.embedding_dim
        self.preference_gate = nn.Sequential(
            nn.Linear(dimension * 2 + 1, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(dimension)

    def encode_all(self, sequence: torch.Tensor, gap_days: torch.Tensor) -> torch.Tensor:
        # The causal transformer is the durable/general preference encoder.
        general_including_current = super().encode_all(sequence, gap_days)
        general = torch.zeros_like(general_including_current)
        general[:, 1:] = general_including_current[:, :-1]
        current = self.item_embedding(sequence)
        logged_gap = torch.log1p(gap_days.clamp_min(0)).view(-1, 1, 1)
        logged_gap = logged_gap.expand(-1, sequence.shape[1], -1)
        gate = self.preference_gate(torch.cat((general, current, logged_gap), dim=-1))
        fused = self.fusion_norm(gate * current + (1.0 - gate) * general)
        return fused.masked_fill(sequence.eq(PAD).unsqueeze(-1), 0.0)


class GRU4RecControl(LFMTimeConcatSASRec):
    """Capacity-matched recurrent carrier with the same tied retrieval head."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        layers = int(kwargs.get("num_layers", 2))
        dropout = float(kwargs.get("dropout", 0.2))
        super().__init__(*args, **kwargs)
        dimension = self.item_embedding.embedding_dim
        del self.transformer
        self.gru = nn.GRU(
            dimension,
            dimension,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )

    def encode_all(self, sequence: torch.Tensor, gap_days: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2 or sequence.shape[1] != self.max_sequence:
            raise ValueError("sequence shape does not match frozen width")
        time_shift = self.time_mlp(torch.log1p(gap_days.clamp_min(0)).unsqueeze(-1)).unsqueeze(1)
        hidden = self.dropout(self.item_embedding(sequence) + time_shift)
        output, _ = self.gru(hidden)
        output = self.layer_norm(output)
        return output.masked_fill(sequence.eq(PAD).unsqueeze(-1), 0.0)


class HybridControl(LFMTimeConcatSASRec):
    """Ordinary early fusion of collaborative identity and item content."""

    def __init__(self, *args: Any, content_features: np.ndarray, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if len(content_features) != self.vocab_size:
            raise ValueError("content_vocabulary_mismatch")
        self.register_buffer(
            "content_features", torch.as_tensor(content_features, dtype=torch.float32), persistent=True
        )
        dimension = self.item_embedding.embedding_dim
        self.content_projection = nn.Sequential(
            nn.Linear(content_features.shape[1], dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.content_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def item_vectors(self, indices: torch.Tensor) -> torch.Tensor:
        content = self.content_projection(self.content_features[indices])
        return self.item_embedding(indices) + self.content_scale * content

    def encode_all(self, sequence: torch.Tensor, gap_days: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2 or sequence.shape[1] != self.max_sequence:
            raise ValueError("sequence shape does not match frozen width")
        positions = torch.arange(self.max_sequence, device=sequence.device).unsqueeze(0)
        time_shift = self.time_mlp(torch.log1p(gap_days.clamp_min(0)).unsqueeze(-1)).unsqueeze(1)
        hidden = self.item_vectors(sequence) + self.position_embedding(positions) + time_shift
        hidden = self.dropout(hidden).masked_fill(sequence.eq(PAD).unsqueeze(-1), 0.0)
        causal = torch.triu(
            torch.full((self.max_sequence, self.max_sequence), float("-inf"), device=sequence.device),
            diagonal=1,
        )
        return self.layer_norm(self.transformer(hidden, mask=causal))

    def sampled_loss(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        sessions: torch.Tensor,
        weights: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        positive = (states * self.item_vectors(targets)).sum(-1, keepdim=True)
        negative = states @ self.item_vectors(negatives).T
        count = float(negatives.numel())
        positive -= torch.log(count * self.noise_probabilities[targets].unsqueeze(1)).clamp_min(-30)
        negative -= torch.log(count * self.noise_probabilities[negatives]).clamp_min(-30).unsqueeze(0)
        negative = negative.masked_fill(negatives.unsqueeze(0).eq(targets.unsqueeze(1)), float("-inf"))
        logits = torch.cat((positive, negative), dim=1)
        losses = torch.nn.functional.cross_entropy(
            logits, torch.zeros(states.shape[0], dtype=torch.long, device=states.device), reduction="none"
        )
        per_session = torch.zeros(int(sessions.max()) + 1, device=states.device)
        per_session.scatter_add_(0, sessions, losses * weights)
        return per_session.mean()


def item_vectors(model: nn.Module, indices: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "item_vectors"):
        return model.item_vectors(indices)
    return model.item_embedding(indices)


@torch.no_grad()
def exact_topk(model: nn.Module, states: torch.Tensor, k: int, chunk_size: int = 8192) -> torch.Tensor:
    scores_out = torch.full((len(states), k), float("-inf"), device=states.device)
    items_out = torch.full((len(states), k), -1, dtype=torch.long, device=states.device)
    for start in range(FIRST_ITEM, model.vocab_size, chunk_size):
        stop = min(start + chunk_size, model.vocab_size)
        ids = torch.arange(start, stop, device=states.device)
        scores = states @ item_vectors(model, ids).T
        local_k = min(k, stop - start)
        local_scores, local_items = scores.topk(local_k, dim=1)
        local_items += start
        joined_scores = torch.cat((scores_out, local_scores), dim=1)
        joined_items = torch.cat((items_out, local_items), dim=1)
        scores_out, selected = joined_scores.topk(k, dim=1)
        items_out = joined_items.gather(1, selected)
    if (items_out < FIRST_ITEM).any():
        raise ValueError("catalog_topk_not_filled")
    return items_out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scope in ("overall", "30-90d", "90-180d", "180-365d", ">365d"):
        selected = rows if scope == "overall" else [row for row in rows if row["gap_bucket"] == scope]
        by_user: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"recall": [], "ndcg": []})
        for row in selected:
            by_user[row["user"]]["recall"].append(row["recall_at_10"])
            by_user[row["user"]]["ndcg"].append(row["ndcg_at_10"])
        output[f"oneshot:{scope}"] = {
            "rows": len(selected),
            "users": len(by_user),
            "recall_at_10": (
                float(np.mean([np.mean(value["recall"]) for value in by_user.values()]))
                if by_user else None
            ),
            "ndcg_at_10": (
                float(np.mean([np.mean(value["ndcg"]) for value in by_user.values()]))
                if by_user else None
            ),
        }
    return output


@torch.no_grad()
def evaluate(model: nn.Module, data: LFMTimeConcatData, *, split: int, device, batch_size=32, limit=0):
    model.eval()
    row_ids = data.row_ids(split, returns_only=True)
    if limit:
        row_ids = row_ids[:limit]
    labels = np.load(data.root / "metric_labels.npz", allow_pickle=False)
    lookup = {int(row): at for at, row in enumerate(labels["row_ids"])}
    rows: list[dict[str, Any]] = []
    for start in range(0, len(row_ids), batch_size):
        ids = row_ids[start : start + batch_size]
        serving = data.serving_batch(
            [data.context(int(row)) for row in ids],
            [data.gap_days(int(row)) for row in ids],
            device,
        )
        hidden = model.encode_all(serving["sequence"], serving["gap_days"])
        states = model.gather_states(hidden, serving["state_positions"])
        top = exact_topk(model, states, 10).cpu().numpy()
        for offset, row_id in enumerate(map(int, ids)):
            at = lookup[row_id]
            truth = labels["values"][labels["offsets"][at] : labels["offsets"][at + 1]].tolist()
            recall, ndcg = set_metrics(top[offset].tolist(), truth)
            user = int(data.sessions[row_id, 0])
            gap = data.gap_days(row_id)
            rows.append({
                "row_id": row_id,
                "user": user,
                "gap_bucket": gap_bucket(gap),
                "prediction": top[offset].tolist(),
                "recall_at_10": recall,
                "ndcg_at_10": ndcg,
            })
        print(json.dumps({"event": "confirmation_validation", "rows": min(start + len(ids), len(row_ids)), "total": len(row_ids)}), flush=True)
    return {
        "schema": "genrec.music4all_onion.arrive_confirmation_evaluation.v1",
        "split": split,
        "rows": rows,
        "metrics": aggregate(rows),
        "future_labels_enter_serving": False,
        "engineering_limit": limit,
    }


def validate_variant(variant: str, objective: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported_variant:{variant}")
    if objective != OBJECTIVE[variant]:
        raise ValueError(f"variant_objective_mismatch:{variant}:expected={OBJECTIVE[variant]}:actual={objective}")


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    validate_variant(args.variant, args.objective)
    prepared = json.loads(args.manifest.read_text())
    labels = json.loads(args.labels_manifest.read_text())
    if prepared.get("schema") != "genrec.music4all_onion.prepared.v1":
        raise ValueError("prepared_schema_mismatch")
    if prepared.get("test_accessed") is not False or labels.get("test_accessed") is not False:
        raise ValueError("test_accessed_before_confirmation_freeze")
    hashes = {}
    for name, entry in prepared["artifacts"].items():
        path = args.data_dir / name
        if digest(path) != entry["sha256"]:
            raise ValueError(f"input_hash_mismatch:{name}")
        hashes[name] = entry["sha256"]
    if digest(args.data_dir / "metric_labels.npz") != labels["sha256"]:
        raise ValueError("metric_label_hash_mismatch")
    if args.variant == "hybrid":
        content_manifest = json.loads(args.content_manifest.read_text())
        if (
            content_manifest.get("schema") != "genrec.music4all_onion.component_b2_feature_cache.v1"
            or content_manifest.get("status") != "pass"
            or content_manifest.get("test_accessed") is not False
            or int(content_manifest.get("vocab_size", -1)) != len(np.load(args.data_dir / "train_frequencies.npy", mmap_mode="r"))
            or content_manifest.get("source_hashes", {}).get("identity") != prepared.get("vocabulary_sha256")
            or content_manifest.get("artifacts", {}).get("audio") != digest(args.audio_features)
            or content_manifest.get("artifacts", {}).get("genre") != digest(args.genre_features)
        ):
            raise ValueError("content_feature_cache_binding_mismatch")
        hashes["content_feature_cache_manifest.json"] = digest(args.content_manifest)
        hashes["musicnn_normalized.npy"] = content_manifest["artifacts"]["audio"]
        hashes["genre_normalized.npz"] = content_manifest["artifacts"]["genre"]
    return {
        "schema": "genrec.music4all_onion.arrive_confirmation_binding.v1",
        "variant": args.variant,
        "inputs": hashes,
        "vocabulary": prepared["vocabulary_sha256"],
        "source_hashes": {
            "runner": digest(Path(__file__)),
            "core": digest(Path(__file__).with_name("lfm_timeconcat_control.py")),
            "training_controller": digest(Path(__file__).with_name("run_lfm_timeconcat_control.py")),
        },
        "configuration": {
            key: getattr(args, key)
            for key in ("variant", "seed", "objective", "epochs", "batch_size", "horizon", "max_context", "learning_rate", "negative_samples", "embed_dim", "num_heads", "num_layers", "dropout")
        },
        "estimand": "distinct_later_tracks_in_same_return_session",
        "selection_metric": "validation_user_macro_ndcg_at_10",
        "validation_only": True,
        "test_accessed": False,
    }


def create_model(args: argparse.Namespace, data: LFMTimeConcatData) -> nn.Module:
    common = dict(
        vocab_size=data.vocab_size,
        noise_probabilities=popularity_noise(data.frequencies),
        embed_dim=args.embed_dim,
        max_sequence=args.max_context + args.horizon - 1,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    if args.variant == "car":
        return CARControl(**common)
    if args.variant == "gru4rec":
        return GRU4RecControl(**common)
    content = deterministic_content_features(args.audio_features, args.genre_features)
    return HybridControl(**common, content_features=content)


def normalize_outputs(output: Path, variant: str) -> None:
    manifest_path = output / "training_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest["validation_only"] = bool(manifest.pop("calibration_only", True))
    manifest.pop("confirmation_accessed", None)
    manifest["test_accessed"] = False
    manifest["variant"] = variant
    training.atomic_json(manifest_path, manifest)
    for source in output.glob("calibration_epoch*.json"):
        destination = output / source.name.replace("calibration_", "validation_", 1)
        if destination.exists():
            raise ValueError("validation_artifact_collision")
        source.rename(destination)


def run(args: argparse.Namespace) -> None:
    validate_variant(args.variant, args.objective)
    if args.resume:
        raise ValueError("confirmation_resume_disabled_rng_not_checkpointed")
    training.STATE_SCHEMA = STATE_SCHEMA
    training.TRAINING_SCHEMA = TRAINING_SCHEMA
    training.validate_inputs = validate_inputs
    training.create_model = create_model
    training.evaluate_shared_checkpoint = evaluate
    result = training.train(args)
    if result.get("status") == "engineering_pass":
        smoke_path = args.output / "SMOKE.json"
        smoke = json.loads(smoke_path.read_text())
        smoke["same_checkpoint_decoders"] = False
        smoke["decoder_contract"] = "one_shot_full_catalog_only"
        training.atomic_json(smoke_path, smoke)
    elif result.get("status") == "pass":
        normalize_outputs(args.output, args.variant)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("refine-logs/music4all_onion/PREPARED_DATASET.json"))
    parser.add_argument("--labels-manifest", type=Path, default=Path("refine-logs/music4all_onion/METRIC_LABELS.json"))
    parser.add_argument("--audio-features", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1/musicnn_normalized.npy"))
    parser.add_argument("--genre-features", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1/genre_normalized.npz"))
    parser.add_argument("--content-manifest", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1/feature_cache_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--objective", choices=("oneshot_only", "continuation_only"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=129)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--negative-samples", type=int, default=1024)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    run(args)


if __name__ == "__main__":
    main()
