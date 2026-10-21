#!/usr/bin/env python3
"""Train and evaluate the shared LFM TimeConcat-SASRec control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from lfm_timeconcat_control import (
    LFMTimeConcatData,
    LFMTimeConcatSASRec,
    evaluate_shared_checkpoint,
    popularity_noise,
)


STATE_SCHEMA = "genrec.music_lfm1b_timeconcat_state.v2"
TRAINING_SCHEMA = "genrec.music_lfm1b_timeconcat_training.v1"
OBJECTIVES = (
    "shared",
    "oneshot_only",
    "continuation_only",
    "shared_shuffled",
    "shuffled_continuation_only",
)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    prepared = json.loads(args.manifest.read_text())
    hashes = {}
    for name, entry in prepared["artifacts"].items():
        path = args.data_dir / Path(entry["path"]).name
        if file_sha(path) != entry["sha256"]:
            raise ValueError(f"input_hash_mismatch:{name}")
        hashes[name] = entry["sha256"]
    labels = json.loads(args.labels_manifest.read_text())
    label_sha = file_sha(args.data_dir / "metric_labels.npz")
    if label_sha != labels["sha256"]:
        raise ValueError("metric_label_hash_mismatch")
    hashes["metric_labels"] = label_sha
    source_root = Path(__file__).resolve().parent
    return {
        "schema": "genrec.music_lfm1b_timeconcat_bindings.v1",
        "inputs": hashes,
        "vocabulary": prepared["vocabulary_sha256"],
        "source_hashes": {
            "core": file_sha(source_root / "lfm_timeconcat_control.py"),
            "runner": file_sha(Path(__file__)),
        },
        "configuration": {
            key: getattr(args, key)
            for key in (
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
        "objective_contract": {
            "shared": "equal_weight_multi_positive_initial_query_plus_teacher_forced_continuation",
            "oneshot_only": "multi_positive_initial_query_only",
            "continuation_only": "teacher_forced_continuation_only",
            "shared_shuffled": (
                "equal_weight_multi_positive_initial_query_plus_deterministically_"
                "shuffled_teacher_forced_continuation"
            ),
            "shuffled_continuation_only": (
                "deterministically_shuffled_teacher_forced_continuation_only"
            ),
        }[args.objective],
        "negative_sampling": "train_frequency_power_0.75_logq_corrected",
        "selection_metric": (
            "mean_of_oneshot_and_autoregressive_user_macro_ndcg_at_10"
            if args.objective == "shared"
            else "oneshot_user_macro_ndcg_at_10"
        ),
        "test_accessed": False,
    }


def create_model(args: argparse.Namespace, data: LFMTimeConcatData) -> LFMTimeConcatSASRec:
    return LFMTimeConcatSASRec(
        data.vocab_size,
        popularity_noise(data.frequencies),
        embed_dim=args.embed_dim,
        max_sequence=args.max_context + args.horizon - 1,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )


def sample_negative_schedule(
    probabilities: torch.Tensor,
    *,
    batches: int,
    samples_per_batch: int,
    deterministic_cpu: bool = False,
) -> torch.Tensor:
    """Draw a fresh popularity-noise sample for every optimizer update."""

    if batches < 1 or samples_per_batch < 1:
        raise ValueError("invalid_negative_schedule_shape")
    # CUDA multinomial's floating-point cumulative distribution can vary
    # across processes for nonuniform noise. Exact-resume runs draw once on
    # CPU and persist the complete schedule; the sampling law is unchanged.
    sampling_probabilities = probabilities.detach().cpu() if deterministic_cpu else probabilities
    return torch.multinomial(
        sampling_probabilities,
        batches * samples_per_batch,
        replacement=True,
    ).reshape(batches, samples_per_batch).to(probabilities.device)


def early_stop_reached(records, *, patience: int, min_epochs: int, min_delta: float) -> bool:
    """Validation-only patience; checkpoint selection still retains exact best."""
    if patience <= 0 or len(records) < min_epochs:
        return False
    best, stale = -float("inf"), 0
    for record in records:
        score = float(record["selection_score"])
        if score > best + min_delta:
            best, stale = score, 0
        else:
            stale += 1
    return stale >= patience


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "training_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("completed_run_must_not_repeat")
    binding = validate_inputs(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    data = LFMTimeConcatData(args.data_dir, max_context=args.max_context, horizon=args.horizon)
    model = create_model(args, data).to(device)
    optimizer_type = getattr(torch.optim, getattr(args, "optimizer", "AdamW"))
    optimizer = optimizer_type(model.parameters(), lr=args.learning_rate,
                               betas=(.9, getattr(args, "beta2", .999)),
                               weight_decay=getattr(args, "weight_decay", 1e-4))
    train_rows = data.row_ids(0)
    state_path = args.output / "last_state.pt"
    completed_epochs = 0
    completed_updates = 0
    records: list[dict[str, Any]] = []
    best_score = None
    pending_calibration: dict[str, Any] | None = None
    in_epoch = None
    resumed = False
    if state_path.exists():
        if not args.resume:
            raise FileExistsError("checkpoint_exists_use_resume")
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("schema") != STATE_SCHEMA or state.get("binding") != binding:
            raise ValueError("resume_binding_mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        completed_epochs = int(state["completed_epochs"])
        completed_updates = int(state["completed_updates"])
        records = list(state["epoch_records"])
        best_score = state["best_score"]
        pending_calibration = state.get("pending_calibration")
        in_epoch = state.get("in_epoch")
        if getattr(args, "require_exact_resume", False) and "rng" not in state:
            raise ValueError("exact_resume_requires_rng_checkpoint")
        if "rng" in state:
            restore_rng(state["rng"])
        resumed = True
    elif args.resume:
        raise FileNotFoundError("resume_requested_without_checkpoint")

    started = time.monotonic()

    def persist_state(status: str) -> None:
        atomic_torch(
            state_path,
            {
                "schema": STATE_SCHEMA,
                "binding": binding,
                "completed_epochs": completed_epochs,
                "completed_updates": completed_updates,
                "epoch_records": records,
                "best_score": best_score,
                "pending_calibration": pending_calibration,
                "in_epoch": in_epoch,
                "rng": capture_rng(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
        )
        atomic_json(
            args.output / "state.json",
            {
                "schema": STATE_SCHEMA,
                "status": status,
                "completed_epochs": completed_epochs,
                "completed_updates": completed_updates,
                "pending_calibration_epoch": (
                    int(pending_calibration["epoch"]) if pending_calibration is not None else None
                ),
                "last_state_sha256": file_sha(state_path),
                "best_score": best_score,
                "in_epoch_next_offset": None if in_epoch is None else in_epoch["next_offset"],
            },
        )

    def finish_pending_calibration() -> None:
        nonlocal best_score, completed_epochs, pending_calibration
        if pending_calibration is None:
            return
        epoch = int(pending_calibration["epoch"])
        if epoch != completed_epochs + 1:
            raise ValueError("pending_calibration_epoch_discontinuous")
        calibration_path = args.output / f"calibration_epoch{epoch}.json"
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text())
        else:
            calibration = evaluate_shared_checkpoint(
                model, data, split=1, device=device, batch_size=args.eval_batch_size
            )
            atomic_json(calibration_path, calibration)
        score = (
            0.5
            * (
                calibration["metrics"]["oneshot:overall"]["ndcg_at_10"]
                + calibration["metrics"]["autoregressive:overall"]["ndcg_at_10"]
            )
            if args.objective == "shared"
            else calibration["metrics"]["oneshot:overall"]["ndcg_at_10"]
        )
        record = dict(pending_calibration["training_record"])
        record["selection_score"] = score
        records.append(record)
        completed_epochs = epoch
        if best_score is None or score > best_score:
            best_score = score
            atomic_torch(
                args.output / "best_model.pt",
                {"schema": TRAINING_SCHEMA, "binding": binding, "epoch": epoch, "model": model.state_dict()},
            )
        pending_calibration = None
        persist_state("complete" if completed_epochs == args.epochs else "checkpointed")

    print(
        json.dumps(
            {
                "event": "started",
                "seed": args.seed,
                "device": args.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "train_sessions": len(train_rows),
                "completed_epochs": completed_epochs,
            }
        ),
        flush=True,
    )
    if pending_calibration is not None:
        print(
            json.dumps({"event": "resume_pending_calibration", "epoch": pending_calibration["epoch"]}),
            flush=True,
        )
        finish_pending_calibration()
    stop_now = False
    def should_stop():
        return early_stop_reached(records, patience=getattr(args, "patience", 0),
                                  min_epochs=getattr(args, "min_epochs", 0),
                                  min_delta=getattr(args, "min_delta", 0.0))

    while completed_epochs < args.epochs and not stop_now and not should_stop():
        epoch = completed_epochs + 1
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        if hasattr(data, "set_epoch"):
            data.set_epoch(epoch, args.seed)
        order = np.random.default_rng(args.seed + epoch).permutation(train_rows)
        batches_in_epoch = (len(order) + args.batch_size - 1) // args.batch_size
        # Each update receives an independent draw. Reusing a 256-row pool here
        # lets the model memorize a small negative support and produces a sharp
        # loss reset when the pool changes at the next epoch.
        if in_epoch is None:
            negative_schedule = sample_negative_schedule(
                model.noise_probabilities, batches=batches_in_epoch,
                samples_per_batch=args.negative_samples,
                deterministic_cpu=getattr(args, "require_exact_resume", False),
            )
            in_epoch = {"epoch": epoch, "next_offset": 0, "negatives": negative_schedule,
                        "losses": [], "set_losses": [], "ar_losses": []}
        if in_epoch["epoch"] != epoch:
            raise ValueError("in_epoch_state_discontinuous")
        negative_schedule = in_epoch["negatives"].to(device)
        losses, set_losses, ar_losses = (in_epoch[key] for key in ("losses", "set_losses", "ar_losses"))
        for at in range(in_epoch["next_offset"], len(order), args.batch_size):
            if hasattr(data, "set_batch_offset"):
                data.set_batch_offset(at)
            batch = data.training_batch(
                order[at : at + args.batch_size],
                device,
                target_order=(
                    "shuffled"
                    if args.objective in {"shared_shuffled", "shuffled_continuation_only"}
                    else "observed"
                ),
                shuffle_seed=args.seed,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            negatives = negative_schedule[at // args.batch_size]
            loss, parts = model.loss(
                batch,
                negative_samples=args.negative_samples,
                objective=args.objective,
                negatives=negatives,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite_loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                model.item_embedding.weight[0].zero_()
            completed_updates += 1
            losses.append(float(loss.detach()))
            set_losses.append(parts["set_loss"])
            ar_losses.append(parts["ar_loss"])
            in_epoch["next_offset"] = min(at + args.batch_size, len(order))
            if completed_updates <= 2 or completed_updates % args.log_interval == 0:
                print(
                    json.dumps(
                        {
                            "event": "update",
                            "epoch": epoch,
                            "completed_updates": completed_updates,
                            "loss": losses[-1],
                            "set_loss": set_losses[-1],
                            "ar_loss": ar_losses[-1],
                            "gradient_norm": float(gradient_norm),
                        }
                    ),
                    flush=True,
                )
            if args.stop_after and completed_updates >= args.stop_after:
                stop_now = True
                break
            interval = getattr(args, "checkpoint_interval", 0)
            if interval and completed_updates % interval == 0:
                persist_state("training_checkpointed")
        if stop_now:
            persist_state("engineering_smoke")
            engineering = evaluate_shared_checkpoint(
                model, data, split=1, device=device, batch_size=2, limit=args.validation_limit
            )
            atomic_json(
                args.output / "SMOKE.json",
                {
                    "status": "engineering_pass",
                    "grounding_mode": getattr(args, "grounding_mode", "observed"),
                    "audio_file": getattr(args, "audio_file", "musicnn_normalized.npy"),
                    "categorical_file": getattr(args, "categorical_file", "genre_normalized.npz"),
                    "categorical_sparse": bool(getattr(args, "categorical_sparse", False)),
                    "scientific_result": False,
                    "updates": completed_updates,
                    "validation_rows": len(engineering["rows"]),
                    "finite_loss": bool(np.isfinite(losses[-1])),
                    "same_checkpoint_decoders": True,
                    "seconds": time.monotonic() - started,
                },
            )
            return {"status": "engineering_pass"}

        pending_calibration = {
            "epoch": epoch,
            "training_record": {
                "epoch": epoch,
                "completed_updates": completed_updates,
                "mean_loss": float(np.mean(losses)),
                "mean_set_loss": float(np.mean(set_losses)),
                "mean_ar_loss": float(np.mean(ar_losses)),
            },
        }
        in_epoch = None
        # Persist the trained epoch before the comparatively long evaluator.
        # A calibration failure can now resume without repeating the epoch.
        persist_state("trained_pending_calibration")
        finish_pending_calibration()

    checkpoint_path = args.output / "best_model.pt"
    persist_state("complete")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    strict = create_model(args, data)
    strict.load_state_dict(checkpoint["model"], strict=True)
    manifest = {
        "schema": TRAINING_SCHEMA,
        "status": "pass",
        "binding": binding,
        "train_sessions": len(train_rows),
        "completed_epochs": completed_epochs,
        "completed_updates": completed_updates,
        "epoch_records": records,
        "best_epoch": checkpoint["epoch"],
        "best_score": best_score,
        "checkpoint_strict_reload": True,
        "best_model_sha256": file_sha(checkpoint_path),
        "last_state_sha256": file_sha(state_path),
        "calibration_only": True,
        "confirmation_accessed": False,
        "test_accessed": False,
        "resumed": resumed,
        "stop_reason": "validation_patience" if should_stop() else "max_epochs",
        "convergence_policy": {
            "max_epochs": args.epochs, "min_epochs": getattr(args, "min_epochs", 0),
            "patience": getattr(args, "patience", 0), "min_delta": getattr(args, "min_delta", 0.0),
        },
        "seconds": time.monotonic() - started,
    }
    atomic_json(manifest_path, manifest)
    print("TIMECONCAT_TRAINING_COMPLETE_CONFIRMATION_DEFERRED", flush=True)
    return manifest


def confirm(args: argparse.Namespace) -> dict[str, Any]:
    destination = args.output / "confirmation.json"
    if destination.exists():
        raise FileExistsError("confirmation_already_exists")
    for objective in OBJECTIVES:
        for seed in (2024, 2025):
            other = args.output.parent / f"{objective}_s{seed}" / "training_manifest.json"
            manifest = json.loads(other.read_text())
            if manifest.get("status") != "pass" or manifest.get("completed_epochs") != args.epochs:
                raise ValueError(f"matched_run_not_complete:{objective}:{seed}")
    binding = validate_inputs(args)
    manifest = json.loads((args.output / "training_manifest.json").read_text())
    if manifest["binding"] != binding:
        raise ValueError("training_binding_mismatch")
    checkpoint_path = args.output / "best_model.pt"
    if file_sha(checkpoint_path) != manifest["best_model_sha256"]:
        raise ValueError("checkpoint_hash_mismatch")
    device = torch.device(args.device)
    data = LFMTimeConcatData(args.data_dir, max_context=args.max_context, horizon=args.horizon)
    model = create_model(args, data).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["binding"] != binding:
        raise ValueError("checkpoint_binding_mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    result = evaluate_shared_checkpoint(
        model, data, split=2, device=device, batch_size=args.eval_batch_size
    )
    result["binding"] = binding
    result["checkpoint_sha256"] = manifest["best_model_sha256"]
    atomic_json(destination, result)
    return result


def self_test(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    frequencies = np.zeros(40, dtype=np.float64)
    frequencies[3:] = np.arange(1, 38)
    model = LFMTimeConcatSASRec(
        40, popularity_noise(frequencies), embed_dim=8, max_sequence=8, num_heads=2, num_layers=1, dropout=0
    )
    from lfm_timeconcat_control import build_training_batch, exact_catalog_topk

    batch = build_training_batch(
        [[3, 4, 5], [6, 7]], [[8, 9, 8], [10, 11]], [31.0, 120.0], max_sequence=8, device=torch.device("cpu")
    )
    model.eval()
    loss, parts = model.loss(batch, negative_samples=12, objective="shared")
    assert torch.isfinite(loss) and all(np.isfinite(value) for value in parts.values())
    one_shot_loss, one_shot_parts = model.loss(batch, negative_samples=12, objective="oneshot_only")
    assert torch.isfinite(one_shot_loss) and one_shot_parts["ar_loss"] == 0.0
    continuation_loss, continuation_parts = model.loss(
        batch, negative_samples=12, objective="continuation_only"
    )
    assert torch.isfinite(continuation_loss) and continuation_parts["ar_loss"] > 0.0
    hidden = model.encode_all(batch["sequence"], batch["gap_days"])
    initial = model.gather_states(hidden, batch["initial_positions"])
    top = exact_catalog_topk(model, initial, k=10, chunk_size=7)
    assert top.shape == (2, 10) and (top >= 3).all()
    original = model.gather_states(hidden, batch["initial_positions"])
    changed = batch["sequence"].clone()
    changed[0, batch["initial_positions"][0] + 1] = 20
    altered = model.encode_all(changed, batch["gap_days"])
    # Causality: a later teacher-forced target cannot alter the initial query.
    assert torch.equal(original[0], model.gather_states(altered, batch["initial_positions"])[0])
    # But it must alter the state used for the following autoregressive target.
    position = batch["initial_positions"][0] + 1
    assert not torch.equal(hidden[0, position], altered[0, position])
    atomic_json(
        args.output / "SELF_TEST.json",
        {
            "status": "pass",
            "checks": [
                "finite_popularity_corrected_hybrid_loss",
                "exact_chunked_catalog_top10",
                "future_target_causal_separation",
                "generated_prefix_state_dependence",
            ],
            "scientific_result": False,
        },
    )
    print("LFM_TIMECONCAT_CONTROL_SELF_TEST_PASS", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("self-test", "train", "confirm"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/lfm_music_confirmation_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("refine-logs/music_lfm1b_confirmation/PREPARED_DATASET.json"))
    parser.add_argument("--labels-manifest", type=Path, default=Path("refine-logs/music_lfm1b_confirmation/METRIC_LABELS.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--objective", choices=OBJECTIVES, default="shared")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=16)
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
    if args.mode == "self-test":
        self_test(args)
    elif args.mode == "train":
        train(args)
    else:
        confirm(args)


if __name__ == "__main__":
    main()
