#!/usr/bin/env python3
"""Train missing Music4All B0 baselines and export exact Validation Top-200 ranks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

import run_lfm_timeconcat_control as training
from lfm_timeconcat_control import (
    LFMTimeConcatData,
    LFMTimeConcatSASRec,
    exact_catalog_topk,
    gap_bucket,
    popularity_noise,
)
from music_sequence_core import set_metrics


STATE_SCHEMA = "genrec.music4all_onion.baseline_state.v1"
TRAINING_SCHEMA = "genrec.music4all_onion.baseline_training.v1"
BASELINES = ("sasrec_standard", "timeconcat_oneshot")
OBJECTIVE_BY_BASELINE = {
    "sasrec_standard": "continuation_only",
    "timeconcat_oneshot": "oneshot_only",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


class Music4AllSASRec(LFMTimeConcatSASRec):
    """Capacity-matched SASRec without the hiatus-duration embedding."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.time_mlp = None

    def encode_all(self, sequence: torch.Tensor, gap_days: torch.Tensor) -> torch.Tensor:
        del gap_days
        if sequence.ndim != 2 or sequence.shape[1] != self.max_sequence:
            raise ValueError("sequence shape does not match frozen width")
        positions = torch.arange(self.max_sequence, device=sequence.device).unsqueeze(0)
        hidden = self.item_embedding(sequence) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        hidden = hidden.masked_fill(sequence.eq(0).unsqueeze(-1), 0.0)
        causal = torch.triu(
            torch.full(
                (self.max_sequence, self.max_sequence),
                float("-inf"),
                device=sequence.device,
            ),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal)
        return self.layer_norm(hidden)


def validate_baseline_contract(baseline: str, objective: str) -> None:
    if baseline not in BASELINES:
        raise ValueError(f"unsupported_baseline:{baseline}")
    expected = OBJECTIVE_BY_BASELINE[baseline]
    if objective != expected:
        raise ValueError(
            f"baseline_objective_mismatch:{baseline}:expected={expected}:actual={objective}"
        )


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    validate_baseline_contract(args.baseline, args.objective)
    prepared = json.loads(args.manifest.read_text())
    labels = json.loads(args.labels_manifest.read_text())
    # Two complete dataset suites share this runner: the Music4All-Onion
    # original and the Yambda-500M counterpart.
    if prepared.get("schema") not in (
        "genrec.music4all_onion.prepared.v1", "genrec.yambda500m.prepared.v2"
    ):
        raise ValueError("prepared_schema_mismatch")
    if prepared.get("test_accessed") is not False or labels.get("test_accessed") is not False:
        raise ValueError("test_accessed_before_method_freeze")
    hashes: dict[str, str] = {}
    for name, entry in prepared["artifacts"].items():
        path = args.data_dir / name
        if digest(path) != entry["sha256"]:
            raise ValueError(f"input_hash_mismatch:{name}")
        hashes[name] = entry["sha256"]
    label_path = args.data_dir / "metric_labels.npz"
    if digest(label_path) != labels["sha256"]:
        raise ValueError("metric_label_hash_mismatch")
    source_root = Path(__file__).resolve().parent
    return {
        "schema": "genrec.music4all_onion.baseline_bindings.v1",
        "inputs": hashes,
        "vocabulary": prepared["vocabulary_sha256"],
        "source_hashes": {
            "core": digest(source_root / "lfm_timeconcat_control.py"),
            "training_controller": digest(source_root / "run_lfm_timeconcat_control.py"),
            "runner": digest(Path(__file__)),
        },
        "configuration": {
            key: getattr(args, key)
            for key in (
                "baseline",
                "seed",
                "objective",
                "epochs",
                "batch_size",
                "horizon",
                "max_context",
                "learning_rate",
                "negative_samples",
                "embed_dim",
                "num_heads",
                "num_layers",
                "dropout",
            )
        },
        "architecture_contract": (
            "standard_sasrec_without_gap_embedding"
            if args.baseline == "sasrec_standard"
            else "timeconcat_sasrec_with_gap_embedding"
        ),
        "convergence_policy": {
            key: getattr(args, key, default) for key, default in
            (("min_epochs", 0), ("patience", 0), ("min_delta", 0.0))
        },
        "resume_policy": "full_rng_optimizer_mid_epoch_v1",
        "objective_contract": (
            "teacher_forced_next_item_continuation"
            if args.baseline == "sasrec_standard"
            else "multi_positive_initial_query_only"
        ),
        "negative_sampling": "train_frequency_power_0.75_logq_corrected",
        "selection_metric": "oneshot_user_macro_ndcg_at_10_on_validation",
        "validation_only": True,
        "test_accessed": False,
    }


def create_model(args: argparse.Namespace, data: LFMTimeConcatData) -> LFMTimeConcatSASRec:
    model_type = Music4AllSASRec if args.baseline == "sasrec_standard" else LFMTimeConcatSASRec
    return model_type(
        data.vocab_size,
        popularity_noise(data.frequencies),
        embed_dim=args.embed_dim,
        max_sequence=args.max_context + args.horizon - 1,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )


def normalize_completed_artifacts(output: Path, baseline: str) -> None:
    manifest_path = output / "training_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest["validation_only"] = bool(manifest.pop("calibration_only", True))
    manifest.pop("confirmation_accessed", None)
    manifest["test_accessed"] = False
    manifest["baseline"] = baseline
    training.atomic_json(manifest_path, manifest)
    for source in output.glob("calibration_epoch*.json"):
        destination = output / source.name.replace("calibration_", "validation_", 1)
        if destination.exists():
            raise ValueError("validation_artifact_collision")
        source.rename(destination)


def train(args: argparse.Namespace) -> None:
    validate_baseline_contract(args.baseline, args.objective)
    if args.resume and not getattr(args, "require_exact_resume", False):
        raise ValueError("baseline_resume_requires_rng_checkpoint_contract")
    training.STATE_SCHEMA = STATE_SCHEMA
    training.TRAINING_SCHEMA = TRAINING_SCHEMA
    training.validate_inputs = validate_inputs
    training.create_model = create_model
    result = training.train(args)
    if result.get("status") == "pass":
        normalize_completed_artifacts(args.output, args.baseline)


def classify_targets(top: list[int], truth: list[int]) -> tuple[dict[str, int], list[int | None]]:
    positions = {int(item): rank + 1 for rank, item in enumerate(top)}
    groups = {"top10": 0, "rank11_200": 0, "train_seen_beyond_200": 0, "future_unseen": 0}
    ranks: list[int | None] = []
    for target in truth:
        if target < 3:
            groups["future_unseen"] += 1
            ranks.append(None)
        elif target in positions:
            rank = positions[target]
            groups["top10" if rank <= 10 else "rank11_200"] += 1
            ranks.append(rank)
        else:
            groups["train_seen_beyond_200"] += 1
            ranks.append(201)
    return groups, ranks


@torch.no_grad()
def export_rank_depth(args: argparse.Namespace) -> None:
    destination = args.output / "validation_rank_depth_200.json.gz"
    if destination.exists():
        raise FileExistsError("rank_depth_output_exists")
    binding = validate_inputs(args)
    manifest = json.loads((args.output / "training_manifest.json").read_text())
    if manifest.get("status") != "pass" or manifest.get("binding") != binding:
        raise ValueError("training_manifest_binding_mismatch")
    checkpoint_path = args.output / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    if digest(checkpoint_path) != manifest["best_model_sha256"]:
        raise ValueError("checkpoint_hash_mismatch")
    if checkpoint.get("binding") != binding:
        raise ValueError("checkpoint_binding_mismatch")
    data = LFMTimeConcatData(args.data_dir, max_context=args.max_context, horizon=args.horizon)
    model = create_model(args, data).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    identities = np.load(args.data_dir / "metric_labels.npz", allow_pickle=False)
    lookup = {int(row): index for index, row in enumerate(identities["row_ids"])}
    row_ids = data.row_ids(1, returns_only=True)
    rows: list[dict[str, Any]] = []
    for at in range(0, len(row_ids), args.eval_batch_size):
        ids = row_ids[at : at + args.eval_batch_size]
        contexts = [data.context(int(row_id)) for row_id in ids]
        gaps = [data.gap_days(int(row_id)) for row_id in ids]
        serving = data.serving_batch(contexts, gaps, torch.device(args.device))
        hidden = model.encode_all(serving["sequence"], serving["gap_days"])
        states = model.gather_states(hidden, serving["state_positions"])
        top = exact_catalog_topk(model, states, k=200).cpu().numpy()
        for offset, row_id in enumerate(ids):
            label_index = lookup[int(row_id)]
            truth = identities["values"][
                identities["offsets"][label_index] : identities["offsets"][label_index + 1]
            ].tolist()
            truth = list(dict.fromkeys(map(int, truth)))
            ranked = list(map(int, top[offset]))
            groups, ranks = classify_targets(ranked, truth)
            recall, ndcg = set_metrics(ranked[:10], truth)
            user, _, _, _, gap, _ = map(int, data.sessions[int(row_id)])
            rows.append(
                {
                    "row_id": int(row_id),
                    "user": user,
                    "gap_days": gap / 86400.0,
                    "gap_bucket": gap_bucket(gap / 86400.0),
                    "targets": len(truth),
                    "target_groups": groups,
                    "target_ranks": ranks,
                    "recall_at_10": recall,
                    "ndcg_at_10": ndcg,
                }
            )
        print(
            json.dumps(
                {
                    "event": "validation_rank_depth",
                    "baseline": args.baseline,
                    "rows": min(at + len(ids), len(row_ids)),
                    "total": len(row_ids),
                }
            ),
            flush=True,
        )
    payload = {
        "schema": "genrec.music4all_onion.baseline_validation_rank_depth.v1",
        "status": "pass",
        "baseline": args.baseline,
        "surface": "validation",
        "rank_depth": 200,
        "checkpoint_sha256": manifest["best_model_sha256"],
        "binding": binding,
        "test_accessed": False,
        "rows": rows,
    }
    handle = tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as compressed:
            json.dump(payload, compressed, separators=(",", ":"), allow_nan=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"status": "pass", "baseline": args.baseline, "rows": len(rows)}))


def validate_rank_depth_artifact(args: argparse.Namespace) -> None:
    destination = args.output / "validation_rank_depth_200.json.gz"
    with gzip.open(destination, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    binding = validate_inputs(args)
    manifest = json.loads((args.output / "training_manifest.json").read_text())
    expected_rows = len(LFMTimeConcatData(
        args.data_dir, max_context=args.max_context, horizon=args.horizon
    ).row_ids(1, returns_only=True))
    if (
        payload.get("status") != "pass"
        or payload.get("baseline") != args.baseline
        or payload.get("surface") != "validation"
        or payload.get("rank_depth") != 200
        or payload.get("binding") != binding
        or payload.get("checkpoint_sha256") != manifest.get("best_model_sha256")
        or payload.get("test_accessed") is not False
        or len(payload.get("rows", [])) != expected_rows
    ):
        raise ValueError("rank_depth_artifact_validation_failed")
    print(json.dumps({"status": "pass", "baseline": args.baseline, "rows": expected_rows}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "rank-depth", "validate-rank-depth"))
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("refine-logs/music4all_onion/PREPARED_DATASET.json"))
    parser.add_argument("--labels-manifest", type=Path, default=Path("refine-logs/music4all_onion/METRIC_LABELS.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--objective", choices=("continuation_only", "oneshot_only"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--require-exact-resume", action="store_true")
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
    if args.mode == "train":
        train(args)
    elif args.mode == "rank-depth":
        export_rank_depth(args)
    else:
        validate_rank_depth_artifact(args)


if __name__ == "__main__":
    main()
