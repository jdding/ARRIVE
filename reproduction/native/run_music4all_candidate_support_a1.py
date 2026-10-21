#!/usr/bin/env python3
"""Learned Top-10 conversion for Music4All candidate-support component A.

The program has three deliberately separated stages:

* ``train-crossfit-base`` fits the frozen SASRec architecture after excluding
  the A0 Train-CV rows.  It never opens Validation and exists only to create
  leakage-free A1 training features.
* ``build-features`` materializes the fixed 200-candidate A0 union together
  with SASRec, first-return (X), and history-anchor (H) evidence.
* ``train-head`` learns a small setwise residual on Train-CV, calibrates on a
  user-disjoint Train-CV subset, and evaluates the untouched Validation rows.

The frozen SASRec Top-10 remains a structural member of every candidate set.
No Test identity or label is materialized or accessed.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Iterable

import numpy as np
from scipy import sparse
import torch
from torch import nn

from lfm_timeconcat_control import LFMTimeConcatData, exact_catalog_topk, gap_bucket
from run_lfm_timeconcat_control import sample_negative_schedule
from run_music4all_baselines import create_model
from run_arrive_main_table_baseline import (
    MainTableData,
    create_model as create_main_table_model,
    exact_topk as exact_main_table_topk,
)
from arrive_main_table_models import MainTableModel
from music_sequence_core import set_metrics
from run_music4all_candidate_support_a0 import (
    FIRST_ITEM,
    digest,
    load_prepared,
    merge_candidates,
    ranked_history,
    ranked_sparse_row,
    recent_anchors,
    validate_index,
)


CROSSFIT_SCHEMA = "genrec.music4all_onion.a1_crossfit_sasrec.v1"
CROSSFIT_SSD_SCHEMA = "genrec.music4all_onion.a1_crossfit_ssd4rec.v1"
CROSSFIT_MAIN_TABLE_SCHEMA = "genrec.arrive.a1_crossfit_main_table.v1"
MAIN_TABLE_BASE_METHODS = {"ssd4rec", "muse", "tisasrec"}
FEATURE_SCHEMA = "genrec.music4all_onion.a1_feature_bundle.v1"
HEAD_SCHEMA = "genrec.music4all_onion.a1_setwise_head.v1"
SCOPES = ("overall", "30-90d", "90-180d", "180-365d", ">365d")


def crossfit_muse_transition_counts(
    events: np.ndarray,
    times: np.ndarray,
    sessions: np.ndarray,
    vocab_size: int,
    excluded_rows: set[int],
    *,
    session_gap_seconds: int = 1800,
) -> sparse.csr_matrix:
    """MUSE transitions from split-0 rows after exact Train-CV exclusion."""

    train_ids = np.asarray(
        [
            int(row_id)
            for row_id in np.flatnonzero(sessions[:, 5] == 0)
            if int(row_id) not in excluded_rows
        ],
        dtype=np.int64,
    )
    train = sessions[train_ids]
    train = train[np.argsort(train[:, 0], kind="stable")]
    boundaries = np.r_[0, np.flatnonzero(np.diff(train[:, 0])) + 1, len(train)]
    counts: dict[tuple[int, int], int] = {}
    for start, stop in zip(boundaries, boundaries[1:]):
        timeline: set[tuple[int, int]] = set()
        for _, begin, _, end, _, _ in train[start:stop]:
            for item, timestamp in zip(events[int(begin) : int(end)], times[int(begin) : int(end)]):
                if FIRST_ITEM <= int(item) < vocab_size:
                    timeline.add((int(timestamp), int(item)))
        ordered = sorted(timeline)
        for left, right in zip(ordered, ordered[1:]):
            if right[0] - left[0] <= session_gap_seconds:
                edge = (left[1], right[1])
                counts[edge] = counts.get(edge, 0) + 1
    if not counts:
        return sparse.csr_matrix((vocab_size, vocab_size), dtype=np.float64)
    edges = np.asarray(list(counts), dtype=np.int64)
    return sparse.csr_matrix(
        (
            np.asarray(list(counts.values()), dtype=np.float64),
            (edges[:, 0], edges[:, 1]),
        ),
        shape=(vocab_size, vocab_size),
    )


def validate_full_carrier_inputs(
    args: argparse.Namespace, full: dict[str, Any], prepared: dict[str, Any]
) -> None:
    """Bind every shared prepared artifact to the selected frozen carrier."""

    frozen = full.get("binding", {}).get("inputs", {})
    required = (
        "PREP_STATS.json",
        "events.npy",
        "sessions.npy",
        "train_frequencies.npy",
        "track_identity.tsv",
        "metric_labels.npz",
    )
    for name in required:
        actual = prepared.get("artifacts", {}).get(name, {}).get("sha256")
        if actual is None or frozen.get(name) != actual:
            raise ValueError(f"full_carrier_prepared_input_mismatch:{name}")


def class_source_sha256(path: Path, class_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise ValueError(f"class_source_missing:{class_name}")
            return hashlib.sha256(segment.encode()).hexdigest()
    raise ValueError(f"class_missing:{class_name}")


def base_method(manifest: dict[str, Any]) -> str:
    """Resolve the carrier identity without relying on directory names."""

    if manifest.get("schema") in {
        CROSSFIT_SCHEMA, CROSSFIT_SSD_SCHEMA, CROSSFIT_MAIN_TABLE_SCHEMA
    }:
        return str(manifest["binding"].get("base_method", "sasrec"))
    if manifest.get("baseline") == "sasrec_standard":
        return "sasrec"
    method = manifest.get("binding", {}).get("method")
    if method in MAIN_TABLE_BASE_METHODS:
        return str(method)
    raise ValueError("base_training_variant_invalid")


def load_base_data(args: argparse.Namespace, manifest_path: Path) -> LFMTimeConcatData:
    manifest = json.loads(manifest_path.read_text())
    method = base_method(manifest)
    if method in MAIN_TABLE_BASE_METHODS:
        data = MainTableData(
            args.data_dir,
            event_times=args.event_times,
            max_context=args.max_context,
            horizon=args.horizon,
        )
        if method == "muse":
            from arrive_muse_repaired import MUSETransitionAugmenter

            if not args.muse_counts.is_file():
                raise ValueError("muse_counts_missing")
            data.augmented_width = 2 * args.max_context - 1
            data.muse_augmenter = MUSETransitionAugmenter(
                sparse.load_npz(args.muse_counts), max_sequence=data.augmented_width
            )
        return data
    return LFMTimeConcatData(args.data_dir, max_context=args.max_context, horizon=args.horizon)


def build_base_model(
    args: argparse.Namespace,
    data: LFMTimeConcatData,
    manifest: dict[str, Any],
) -> nn.Module:
    configuration = dict(manifest["binding"]["configuration"])
    binding = manifest["binding"]
    configuration.update(
        method=base_method(manifest),
        adapter_revision=binding.get(
            "adapter_revision",
            binding.get("architecture_contract", {}).get("adapter_revision", "original")
            if isinstance(binding.get("architecture_contract"), dict)
            else "original",
        ),
        ssd4rec_source=args.ssd4rec_source,
    )
    model_args = argparse.Namespace(**configuration)
    if configuration["method"] in MAIN_TABLE_BASE_METHODS:
        return create_main_table_model(model_args, data)
    return create_model(model_args, data)


def encode_base_states(
    args: argparse.Namespace,
    data: LFMTimeConcatData,
    model: nn.Module,
    row_ids: Iterable[int],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rows = list(map(int, row_ids))
    device = torch.device(args.device)
    if isinstance(data, MainTableData):
        serving = data.serving_rows(rows, device)
        hidden = model.encode_batch(serving)
        return model.gather_states(hidden, serving["initial_positions"]), serving
    serving = data.serving_batch(
        [data.context(row) for row in rows],
        [data.gap_days(row) for row in rows],
        device,
    )
    hidden = model.encode_all(serving["sequence"], serving["gap_days"])
    return model.gather_states(hidden, serving["state_positions"]), serving


def base_topk(
    model: nn.Module,
    states: torch.Tensor,
    serving: dict[str, torch.Tensor],
    *,
    k: int,
) -> torch.Tensor:
    if isinstance(model, MainTableModel):
        return exact_main_table_topk(model, states, serving, k=k, chunk_size=4096)
    return exact_catalog_topk(model, states, k=k)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.close()
        os.replace(temporary, path)
    finally:
        try:
            handle.close()
        except Exception:
            pass
        temporary.unlink(missing_ok=True)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def user_partition(users: np.ndarray, fraction: float = 0.8) -> np.ndarray:
    """Return a stable fit mask; rows from one user never cross the boundary."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("invalid_user_partition_fraction")
    unique = np.unique(users.astype(np.int64, copy=False))
    fit_users = {
        int(user)
        for user in unique
        if int.from_bytes(hashlib.sha256(f"a1-fit:{int(user)}".encode()).digest()[:8], "big")
        < int(fraction * (1 << 64))
    }
    mask = np.fromiter((int(user) in fit_users for user in users), dtype=bool, count=len(users))
    if not mask.any() or mask.all():
        raise ValueError("degenerate_user_partition")
    return mask


def load_a0_configuration(args: argparse.Namespace) -> dict[str, Any]:
    result = json.loads(args.a0_result.read_text())
    configuration = result.get("frozen_configuration")
    fixed_recipe = result.get("schema") == "arrive.reproduction.fixed_recipe.v1"
    if not configuration or (not fixed_recipe and (result.get("status") != "pass" or result.get("decision") != "PASS")):
        raise ValueError("a0_result_not_pass")
    expected = {
        "budget": 200,
        "protected_base_prefix": 10,
        "relation_slots": 40,
        "x_fraction": 0.75,
        "anchor_count": 5,
        "beta_x": 0.0,
        "beta_h": 0.0,
    }
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise ValueError(f"a0_configuration_mismatch:{key}")
    return configuration


def load_standard_base(
    args: argparse.Namespace,
    data: LFMTimeConcatData,
    *,
    checkpoint_path: Path,
    manifest_path: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "pass" or manifest.get("test_accessed") is not False:
        raise ValueError("base_training_manifest_invalid")
    method = base_method(manifest)
    if method in MAIN_TABLE_BASE_METHODS and getattr(args, "surface", None) != "test":
        expected_times = manifest["binding"].get("inputs", {}).get("event_times.npy")
        if expected_times is None:
            expected_times = manifest["binding"].get("event_times_sha256")
        if expected_times is None or digest(args.event_times) != expected_times:
            raise ValueError("base_event_times_hash_mismatch")
    if method == "muse":
        expected_counts = manifest["binding"].get("inputs", {}).get("muse_transition_counts.npz")
        if expected_counts is None:
            expected_counts = manifest["binding"].get("muse_counts_sha256")
        if expected_counts is None or digest(args.muse_counts) != expected_counts:
            raise ValueError("base_muse_counts_hash_mismatch")
    if manifest.get("schema") in {
        CROSSFIT_SCHEMA, CROSSFIT_SSD_SCHEMA, CROSSFIT_MAIN_TABLE_SCHEMA
    }:
        checkpoint_key = "final_model_sha256"
    elif manifest.get("baseline") == "sasrec_standard":
        checkpoint_key = "best_model_sha256"
    elif method in MAIN_TABLE_BASE_METHODS:
        checkpoint_key = "best_model_sha256"
    else:
        raise ValueError("base_training_variant_invalid")
    if digest(checkpoint_path) != manifest.get(checkpoint_key):
        raise ValueError("base_checkpoint_hash_mismatch")
    model = build_base_model(args, data, manifest).to(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    if checkpoint.get("binding") != manifest.get("binding"):
        raise ValueError("base_checkpoint_binding_mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, manifest


def crossfit_binding(args: argparse.Namespace) -> dict[str, Any]:
    prepared = load_prepared(args)
    index = validate_index(args)
    full = json.loads(args.full_base_manifest.read_text())
    if full.get("status") != "pass":
        raise ValueError("full_base_manifest_invalid")
    method = base_method(full)
    validate_full_carrier_inputs(args, full, prepared)
    if method not in {"sasrec", *MAIN_TABLE_BASE_METHODS}:
        raise ValueError("full_base_method_invalid")
    if method == "ssd4rec" and digest(args.ssd4rec_source) != full["binding"]["source_hashes"].get("official_ssd4rec"):
        raise ValueError("ssd4rec_source_hash_mismatch")
    if method == "muse":
        expected = full["binding"].get("inputs", {}).get("muse_transition_counts.npz")
        if expected is None or digest(args.muse_counts) != expected:
            raise ValueError("muse_counts_hash_mismatch")
    if method in MAIN_TABLE_BASE_METHODS:
        expected = full["binding"].get("inputs", {}).get("event_times.npy")
        if expected is None or digest(args.event_times) != expected:
            raise ValueError("event_times_hash_mismatch")
    if int(full["binding"]["configuration"].get("seed", -1)) != int(args.seed):
        raise ValueError("full_base_seed_mismatch")
    if int(args.epochs) != int(full.get("best_epoch", args.epochs)):
        raise ValueError("crossfit_epoch_must_match_frozen_full_selection")
    configuration = dict(full["binding"]["configuration"])
    configuration["seed"] = int(args.seed)
    configuration["epochs"] = int(args.epochs)
    return {
        "schema": "genrec.music4all_onion.a1_crossfit_binding.v1",
        "prepared_sha256": digest(args.manifest),
        "prepared_artifacts": {name: item["sha256"] for name, item in prepared["artifacts"].items()},
        "index_manifest_sha256": digest(args.index_dir / "index_manifest.json"),
        "cv_rows_sha256": index["artifacts"]["cv_rows.npy"]["sha256"],
        "full_base_manifest_sha256": digest(args.full_base_manifest),
        "event_times_sha256": digest(args.event_times) if method in MAIN_TABLE_BASE_METHODS else None,
        "full_muse_counts_sha256": digest(args.muse_counts) if method == "muse" else None,
        "base_method": method,
        "adapter_revision": full["binding"].get("adapter_revision", "original"),
        "architecture_contract": full["binding"].get(
            "architecture_contract",
            {"method": method, "adapter_revision": full["binding"].get("adapter_revision", "original")},
        ),
        "objective_contract": configuration.get("objective"),
        "selection": "final_epoch_without_validation_access",
        "configuration": configuration,
        "test_accessed": False,
    }


def train_crossfit_base(args: argparse.Namespace) -> None:
    final_only = getattr(args, "final_checkpoint_only", False)
    if final_only and (args.resume_if_present or args.stop_after_epoch):
        raise ValueError("final_checkpoint_only_incompatible_with_resume")
    if args.output.exists() and not args.resume_if_present:
        raise FileExistsError("crossfit_output_exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    binding = crossfit_binding(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    full_manifest = json.loads(args.full_base_manifest.read_text())
    cv_rows = set(map(int, np.load(args.index_dir / "cv_rows.npy", allow_pickle=False)))
    data = load_base_data(args, args.full_base_manifest)
    args.output.mkdir(parents=True, exist_ok=args.resume_if_present)
    if binding["base_method"] == "muse":
        from arrive_muse_repaired import MUSETransitionAugmenter

        counts = crossfit_muse_transition_counts(
            data.events,
            data.event_times,
            data.sessions,
            data.vocab_size,
            cv_rows,
        )
        counts_path = args.output / "muse_transition_counts.npz"
        sparse.save_npz(counts_path, counts)
        binding["muse_counts_sha256"] = digest(counts_path)
        binding["muse_counts_contract"] = "split0_rows_excluding_exact_train_cv_rows"
        data.muse_augmenter = MUSETransitionAugmenter(
            counts, max_sequence=data.augmented_width
        )
    model = build_base_model(args, data, full_manifest).to(device)
    optimizer_contract = dict(
        full_manifest["binding"].get(
            "optimizer",
            {"optimizer": "AdamW", "beta2": 0.999, "weight_decay": 1e-4},
        )
    )
    optimizer_name = optimizer_contract.get("optimizer", "AdamW")
    optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}.get(
        optimizer_name
    )
    if optimizer_class is None:
        raise ValueError("unsupported_frozen_optimizer")
    optimizer = optimizer_class(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, float(optimizer_contract.get("beta2", 0.999))),
        weight_decay=float(optimizer_contract.get("weight_decay", 1e-4)),
    )
    binding["optimizer"] = optimizer_contract
    if not args.stop_after:
        expected_configuration = full_manifest["binding"]["configuration"]
        exact = {
            "batch_size": args.batch_size,
            "negative_samples": args.negative_samples,
            "learning_rate": args.learning_rate,
            "max_context": args.max_context,
        }
        for key, actual in exact.items():
            if key in expected_configuration and actual != expected_configuration[key]:
                raise ValueError(f"crossfit_training_contract_mismatch:{key}")
    train_rows = np.asarray(
        [int(row) for row in data.row_ids(0) if int(row) not in cv_rows], dtype=np.int64
    )
    if len(train_rows) + len(cv_rows) != len(data.row_ids(0)):
        raise ValueError("crossfit_exclusion_count_mismatch")
    updates = 0
    records = []
    completed_epoch = 0
    started = time.monotonic()
    print(
        json.dumps(
            {
                "event": "crossfit_started",
                "train_rows": len(train_rows),
                "excluded_cv_rows": len(cv_rows),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        ),
        flush=True,
    )
    crossfit_schema = (
        CROSSFIT_SCHEMA if binding["base_method"] == "sasrec"
        else CROSSFIT_SSD_SCHEMA if binding["base_method"] == "ssd4rec"
        else CROSSFIT_MAIN_TABLE_SCHEMA
    )
    state_path = args.output / "last_state.pt"

    def save_resume_state(epoch: int) -> None:
        if final_only:
            return
        atomic_torch(
            state_path,
            {
                "schema": crossfit_schema,
                "binding": binding,
                "target_epochs": args.epochs,
                "epoch": epoch,
                "updates": updates,
                "records": records,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        )

    if args.resume_if_present and state_path.exists():
        resume = torch.load(state_path, map_location="cpu", weights_only=False)
        if (
            resume.get("schema") != crossfit_schema
            or resume.get("binding") != binding
            or int(resume.get("target_epochs", -1)) != args.epochs
        ):
            raise ValueError("crossfit_resume_binding_mismatch")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        completed_epoch = int(resume["epoch"])
        updates = int(resume["updates"])
        records = list(resume["records"])
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"])
        if torch.cuda.is_available() and resume.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
        print(
            json.dumps(
                {"event": "crossfit_resumed", "epoch": completed_epoch, "updates": updates}
            ),
            flush=True,
        )
    else:
        # A zero-update checkpoint makes even an interruption during the first
        # epoch safely restartable without overwriting scientific artifacts.
        save_resume_state(0)

    for epoch in range(completed_epoch + 1, args.epochs + 1):
        if isinstance(data, MainTableData):
            data.set_epoch(epoch, args.seed)
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        order = np.random.default_rng(args.seed + epoch).permutation(train_rows)
        schedule = sample_negative_schedule(
            model.noise_probabilities,
            batches=(len(order) + args.batch_size - 1) // args.batch_size,
            samples_per_batch=args.negative_samples,
        )
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = data.training_batch(order[start : start + args.batch_size], device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.loss(
                batch,
                negative_samples=args.negative_samples,
                objective=(
                    "continuation_only" if binding["base_method"] == "sasrec"
                    else "oneshot_only"
                ),
                negatives=schedule[start // args.batch_size],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite_crossfit_loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                model.item_embedding.weight[0].zero_()
            updates += 1
            losses.append(float(loss.detach()))
            if updates <= 2 or updates % args.log_interval == 0:
                print(json.dumps({"event": "update", "updates": updates, "loss": losses[-1]}), flush=True)
            if args.stop_after and updates >= args.stop_after:
                atomic_json(
                    args.output / "SMOKE.json",
                    {
                        "status": "engineering_pass",
                        "scientific_result": False,
                        "updates": updates,
                        "finite_loss": True,
                        "excluded_cv_rows": len(cv_rows),
                        "validation_accessed": False,
                        "test_accessed": False,
                    },
                )
                return
        records.append({"epoch": epoch, "completed_updates": updates, "mean_loss": float(np.mean(losses))})
        save_resume_state(epoch)
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            print(
                json.dumps({"event": "crossfit_interrupted_for_resume_smoke", "epoch": epoch}),
                flush=True,
            )
            return

    checkpoint_path = args.output / "final_model.pt"
    atomic_torch(
        checkpoint_path,
        {"schema": crossfit_schema, "binding": binding, "epoch": args.epochs, "model": model.state_dict()},
    )
    strict = build_base_model(args, data, full_manifest)
    strict.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"], strict=True)
    manifest = {
        "schema": crossfit_schema,
        "status": "pass",
        "binding": binding,
        "train_rows": len(train_rows),
        "excluded_cv_rows": len(cv_rows),
        "completed_epochs": args.epochs,
        "checkpoint_policy": "final_only" if final_only else "epoch_resume",
        "completed_updates": updates,
        "epoch_records": records,
        "final_model_sha256": digest(checkpoint_path),
        "checkpoint_strict_reload": True,
        "validation_accessed": False,
        "test_accessed": False,
        "seconds": time.monotonic() - started,
    }
    atomic_json(args.output / "training_manifest.json", manifest)
    print(json.dumps({"event": "crossfit_complete", "updates": updates}), flush=True)


@torch.no_grad()
def verify_base_parity(args: argparse.Namespace) -> None:
    """Replay a bounded frozen Validation prefix against stored row metrics."""

    reference = json.loads(args.reference_evaluation.read_text())
    if (
        reference.get("split") != 1
        or reference.get("test_accessed") is not False
        or not reference.get("rows")
    ):
        raise ValueError("reference_evaluation_invalid")
    rows = reference["rows"][: args.limit or 64]
    row_ids = np.asarray([int(row["row_id"]) for row in rows], dtype=np.int64)
    manifest_value = json.loads(args.base_training_manifest.read_text())
    prepared = load_prepared(args)
    validate_full_carrier_inputs(args, manifest_value, prepared)
    method = base_method(manifest_value)
    source_root = Path(__file__).resolve().parent
    source_contract: dict[str, Any]
    if method == "muse":
        expected = manifest_value["binding"].get("source_hashes", {}).get("repaired_operators")
        actual = digest(source_root / "arrive_muse_repaired.py")
        if expected != actual:
            raise ValueError("frozen_muse_operator_source_mismatch")
        source_contract = {"kind": "exact_file", "sha256": actual}
    elif method == "tisasrec":
        expected_source = "8f2605266583d8e42cd7a81dc6c0ef534f858f8937e60392cfb9f35cef83c40b"
        actual_source = class_source_sha256(
            source_root / "arrive_main_table_models.py", "TiSASRecAdapter"
        )
        if actual_source != expected_source:
            raise ValueError("frozen_tisasrec_class_source_mismatch")
        source_contract = {
            "kind": "class_source_lineage",
            "class": "TiSASRecAdapter",
            "sha256": actual_source,
            "accepted_full_file_sha256": [
                "755e52c64bee75a767fcbea14475c106d53bf0ebb77ca8f5c2a92792d9826955",
                "e95035d6b1fd9f68674f11d88a781318c29748ddb7d5a14fa21946d5ce36f300",
            ],
        }
        frozen_model_source = manifest_value["binding"].get("source_hashes", {}).get("models")
        if frozen_model_source not in source_contract["accepted_full_file_sha256"]:
            raise ValueError("frozen_tisasrec_model_lineage_mismatch")
    else:
        raise ValueError("portability_parity_method_invalid")
    data = load_base_data(args, args.base_training_manifest)
    model, manifest = load_standard_base(
        args,
        data,
        checkpoint_path=args.base_checkpoint,
        manifest_path=args.base_training_manifest,
    )
    states, serving = encode_base_states(args, data, model, row_ids)
    predictions = base_topk(model, states, serving, k=10).cpu().numpy()
    labels = np.load(args.data_dir / "metric_labels.npz", allow_pickle=False)
    lookup = {int(row_id): at for at, row_id in enumerate(labels["row_ids"])}
    for at, (row_id, expected) in enumerate(zip(row_ids, rows)):
        label_at = lookup[int(row_id)]
        truth = labels["values"][
            labels["offsets"][label_at] : labels["offsets"][label_at + 1]
        ].tolist()
        recall, ndcg = set_metrics(predictions[at].tolist(), truth)
        if (
            int(data.sessions[int(row_id), 0]) != int(expected["user"])
            or gap_bucket(data.gap_days(int(row_id))) != expected["gap_bucket"]
            or not math.isclose(recall, float(expected["recall_at_10"]), abs_tol=1e-12)
            or not math.isclose(ndcg, float(expected["ndcg_at_10"]), abs_tol=1e-12)
        ):
            raise ValueError(f"frozen_carrier_parity_mismatch:{int(row_id)}")
    atomic_json(
        args.output,
        {
            "schema": "genrec.arrive.frozen_carrier_parity.v1",
            "status": "pass",
            "base_method": base_method(manifest),
            "rows": len(rows),
            "metric_contract": "exact_per_row_recall_ndcg_at_10",
            "source_contract": source_contract,
            "row_ids_sha256": hashlib.sha256(row_ids.astype("<i8").tobytes()).hexdigest(),
            "checkpoint_sha256": digest(args.base_checkpoint),
            "manifest_sha256": digest(args.base_training_manifest),
            "reference_sha256": digest(args.reference_evaluation),
            "event_times_sha256": digest(args.event_times),
            "muse_counts_sha256": (
                digest(args.muse_counts) if base_method(manifest) == "muse" else None
            ),
            "test_accessed": False,
        },
    )


def target_lookup(data_dir: Path) -> dict[int, list[int]]:
    values = np.load(data_dir / "metric_labels.npz", allow_pickle=False)
    return {
        int(row): list(
            dict.fromkeys(
                map(int, values["values"][values["offsets"][at] : values["offsets"][at + 1]])
            )
        )
        for at, row in enumerate(values["row_ids"])
    }


def rank_map(items: list[int]) -> dict[int, int]:
    return {int(item): rank + 1 for rank, item in enumerate(items)}


@torch.no_grad()
def build_features(args: argparse.Namespace) -> None:
    if args.output.exists() or args.feature_manifest.exists():
        raise FileExistsError("feature_output_exists")
    if args.surface == "test":
        prepared = json.loads(args.manifest.read_text())
        # Two complete dataset suites share this loader (Onion original +
        # Yambda-500M counterpart).
        if prepared.get("schema") not in (
            "genrec.music4all_onion.prepared.v1", "genrec.yambda500m.prepared.v2"
        ) or prepared.get("test_accessed") is not False:
            raise ValueError("frozen_train_manifest_invalid_for_test")
        train_data_dir = args.train_data_dir or args.data_dir
        for name, entry in prepared["artifacts"].items():
            if digest(train_data_dir / name) != entry["sha256"]:
                raise ValueError(f"frozen_train_artifact_mismatch:{name}")
        sealed_manifest_path = args.data_dir / "SEALED_TEST_MANIFEST.json"
        sealed = json.loads(sealed_manifest_path.read_text())
        if (
            sealed.get("status") != "pass"
            or sealed.get("test_accessed") is not True
            or sealed.get("selection_or_tuning_permitted") is not False
        ):
            raise ValueError("sealed_test_manifest_invalid_for_features")
        for name, entry in sealed["artifacts"].items():
            if digest(args.data_dir / name) != entry["sha256"]:
                raise ValueError(f"sealed_test_artifact_mismatch:{name}")
    else:
        prepared = load_prepared(args)
    index = validate_index(args)
    configuration = load_a0_configuration(args)
    data = load_base_data(args, args.base_training_manifest)
    model, base_manifest = load_standard_base(
        args, data, checkpoint_path=args.base_checkpoint, manifest_path=args.base_training_manifest
    )
    if args.surface == "train_cv":
        row_ids_path = args.row_ids or (args.index_dir / "cv_rows.npy")
        row_ids = np.load(row_ids_path, allow_pickle=False)
        x_matrix = sparse.load_npz(args.index_dir / "x_fit.npz").tocsr()
        h_matrix = sparse.load_npz(args.index_dir / "h_fit.npz").tocsr()
        frequency = np.load(args.index_dir / "fit_target_frequency.npy", allow_pickle=False)
        labels = None
        if base_manifest.get("schema") not in {
            CROSSFIT_SCHEMA, CROSSFIT_SSD_SCHEMA, CROSSFIT_MAIN_TABLE_SCHEMA
        }:
            raise ValueError("train_cv_requires_crossfit_base")
    else:
        row_ids = np.load(args.data_dir / "metric_labels.npz", allow_pickle=False)["row_ids"]
        x_matrix = sparse.load_npz(args.index_dir / "x_full.npz").tocsr()
        h_matrix = sparse.load_npz(args.index_dir / "h_full.npz").tocsr()
        frequency = np.load(args.index_dir / "full_target_frequency.npy", allow_pickle=False)
        labels = target_lookup(args.data_dir)
        if (
            base_manifest.get("baseline") != "sasrec_standard"
            and base_method(base_manifest) not in MAIN_TABLE_BASE_METHODS
        ):
            raise ValueError("validation_requires_frozen_full_base")
    if args.limit:
        row_ids = row_ids[: args.limit]

    # The A0 merge fallback is all-Train event frequency, not continuation-
    # target frequency.  Keep the exact frozen candidate-union contract.
    train_frequency = np.asarray(np.load(args.data_dir / "train_frequencies.npy", mmap_mode="r"))
    popularity = list(map(int, np.argsort(-train_frequency, kind="stable")))
    popularity = [item for item in popularity if item >= FIRST_ITEM]
    n, budget = len(row_ids), int(configuration["budget"])
    candidates = np.empty((n, budget), dtype=np.int32)
    target_mask = np.zeros((n, budget), dtype=np.uint8)
    base_scores = np.empty((n, budget), dtype=np.float32)
    base_ranks = np.full((n, budget), 201, dtype=np.uint16)
    x_ranks = np.full((n, budget), 201, dtype=np.uint16)
    h_ranks = np.full((n, budget), 201, dtype=np.uint16)
    users = np.empty(n, dtype=np.int64)
    gaps = np.empty(n, dtype=np.float32)
    bucket_ids = np.empty(n, dtype=np.uint8)
    target_counts = np.empty(n, dtype=np.uint16)
    log_popularity = np.empty((n, budget), dtype=np.float32)
    x_cache: dict[int, list[int]] = {}
    bucket_map = {value: at for at, value in enumerate(SCOPES[1:])}

    for start in range(0, n, args.eval_batch_size):
        ids = row_ids[start : start + args.eval_batch_size]
        states, serving = encode_base_states(args, data, model, ids)
        base_top = base_topk(model, states, serving, k=200).cpu().numpy()
        batch_candidates = []
        batch_metadata = []
        for offset, row_id in enumerate(ids):
            row_id = int(row_id)
            row = data.sessions[row_id]
            user, _, event_start, _, gap, split = map(int, row)
            expected_split = {"train_cv": 0, "validation": 1, "test": 2}[args.surface]
            if split != expected_split:
                raise ValueError("feature_surface_split_mismatch")
            observed = int(data.events[event_start])
            if observed not in x_cache:
                x_cache[observed] = ranked_sparse_row(
                    x_matrix, observed, frequency, configuration["beta_x"], 200
                )
            x_items = x_cache[observed]
            anchors = recent_anchors(data.events, row, configuration["anchor_count"])
            h_items = ranked_history(
                h_matrix, anchors, frequency, configuration["beta_h"], 200
            )
            base = list(map(int, base_top[offset]))
            merged = merge_candidates(
                base,
                [] if args.exclude_x_source else x_items,
                h_items,
                popularity,
                relation_slots=configuration["relation_slots"],
                x_fraction=0.0 if args.exclude_x_source else configuration["x_fraction"],
                budget=budget,
            )
            truth = (
                list(dict.fromkeys(data.known_suffix(row_id)))
                if labels is None
                else labels[row_id]
            )
            batch_candidates.append(merged)
            batch_metadata.append((row_id, user, gap, truth, rank_map(base), rank_map(x_items), rank_map(h_items)))
        candidate_tensor = torch.as_tensor(batch_candidates, dtype=torch.long, device=args.device)
        scores = torch.einsum("bd,bkd->bk", states, model.item_embedding(candidate_tensor)).cpu().numpy()
        for offset, metadata in enumerate(batch_metadata):
            row_id, user, gap, truth, base_map, x_map, h_map = metadata
            output_at = start + offset
            merged = batch_candidates[offset]
            candidates[output_at] = merged
            base_scores[output_at] = scores[offset]
            base_ranks[output_at] = [base_map.get(item, 201) for item in merged]
            x_ranks[output_at] = [x_map.get(item, 201) for item in merged]
            h_ranks[output_at] = [h_map.get(item, 201) for item in merged]
            truth_set = set(map(int, truth))
            target_mask[output_at] = [int(item in truth_set) for item in merged]
            users[output_at] = user
            gaps[output_at] = gap / 86400.0
            bucket_ids[output_at] = bucket_map[gap_bucket(gap / 86400.0)]
            target_counts[output_at] = len(truth_set)
            log_popularity[output_at] = np.log1p(train_frequency[np.asarray(merged)])
        print(json.dumps({"event": "features", "surface": args.surface, "rows": min(start + len(ids), n), "total": n}), flush=True)

    atomic_npz(
        args.output,
        row_ids=np.asarray(row_ids, dtype=np.int64),
        users=users,
        gaps=gaps,
        bucket_ids=bucket_ids,
        target_counts=target_counts,
        candidates=candidates,
        target_mask=target_mask,
        base_scores=base_scores,
        base_ranks=base_ranks,
        x_ranks=x_ranks,
        h_ranks=h_ranks,
        log_popularity=log_popularity,
    )
    payload = {
        "schema": FEATURE_SCHEMA,
        "status": "pass",
        "surface": args.surface,
        "rows": n,
        "budget": budget,
        "positive_rows": int(target_mask.any(axis=1).sum()),
        "candidate_target_hits": int(target_mask.sum()),
        "producer_sha256": digest(Path(__file__)),
        "feature_sha256": digest(args.output),
        "prepared_sha256": digest(args.manifest),
        "prepared_artifacts": {name: item["sha256"] for name, item in prepared["artifacts"].items()},
        "serving_manifest_sha256": (
            digest(args.data_dir / "SEALED_TEST_MANIFEST.json")
            if args.surface == "test" else None
        ),
        "index_manifest_sha256": digest(args.index_dir / "index_manifest.json"),
        "a0_result_sha256": digest(args.a0_result),
        "a0_configuration": configuration,
        "source_intervention": "exclude_first_track_transition" if args.exclude_x_source else "none",
        "base_checkpoint_sha256": digest(args.base_checkpoint),
        "base_training_manifest_sha256": digest(args.base_training_manifest),
        "base_method": base_method(base_manifest),
        "row_ids_source_sha256": digest(row_ids_path) if args.surface == "train_cv" else None,
        "test_accessed": args.surface == "test",
        "selection_or_tuning_permitted": False if args.surface == "test" else None,
    }
    atomic_json(args.feature_manifest, payload)
    print(json.dumps({"event": "feature_complete", "surface": args.surface, "rows": n}), flush=True)


def load_features(path: Path, manifest_path: Path, surface: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != FEATURE_SCHEMA
        or manifest.get("status") != "pass"
        or manifest.get("surface") != surface
        or manifest.get("feature_sha256") != digest(path)
        or manifest.get("test_accessed") is not (surface == "test")
        or (surface == "test" and manifest.get("selection_or_tuning_permitted") is not False)
    ):
        raise ValueError(f"feature_manifest_invalid:{surface}")
    loaded = np.load(path, allow_pickle=False)
    arrays = {key: loaded[key] for key in loaded.files}
    n = len(arrays["row_ids"])
    if n != manifest["rows"] or arrays["candidates"].shape != (n, 200):
        raise ValueError(f"feature_shape_invalid:{surface}")
    if len(set(map(int, arrays["row_ids"]))) != n:
        raise ValueError(f"feature_row_duplicate:{surface}")
    return arrays, manifest


def numeric_features(arrays: dict[str, np.ndarray], variant: str) -> tuple[np.ndarray, np.ndarray]:
    base = arrays["base_scores"].astype(np.float32)
    base_z = (base - base.mean(axis=1, keepdims=True)) / (base.std(axis=1, keepdims=True) + 1e-6)
    base_rank = arrays["base_ranks"].astype(np.float32)
    x_rank = arrays["x_ranks"].astype(np.float32)
    h_rank = arrays["h_ranks"].astype(np.float32)
    base_rr = np.where(base_rank <= 200, 1.0 / np.log2(base_rank + 1.0), 0.0)
    x_rr = np.where(x_rank <= 200, 1.0 / np.log2(x_rank + 1.0), 0.0)
    h_rr = np.where(h_rank <= 200, 1.0 / np.log2(h_rank + 1.0), 0.0)
    x_flag = (x_rank <= 200).astype(np.float32)
    h_flag = (h_rank <= 200).astype(np.float32)
    if variant == "x_only":
        h_rr.fill(0.0)
        h_flag.fill(0.0)
    # Fixed transforms keep Train-CV and Validation on the identical feature
    # scale without estimating any normalization statistic from Validation.
    pop = arrays["log_popularity"].astype(np.float32) / 20.0
    gap = np.log1p(arrays["gaps"].astype(np.float32)) / 10.0
    gap = np.broadcast_to(gap[:, None], base.shape)
    features = np.stack((base_z, base_rr, x_rr, h_rr, x_flag, h_flag, pop, gap), axis=-1)
    relation_mask = x_flag if variant == "x_only" else np.maximum(x_flag, h_flag)
    return features.astype(np.float32), relation_mask.astype(np.float32)


class SetwiseResidual(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, features: torch.Tensor, relation_mask: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1) * relation_mask


def setwise_loss(scores: torch.Tensor, positives: torch.Tensor, margin: float) -> torch.Tensor:
    valid = positives.any(dim=1)
    if not valid.any():
        raise ValueError("batch_has_no_positive_candidate")
    scores = scores[valid]
    positives = positives[valid]
    positive_scores = scores.masked_fill(~positives, float("-inf"))
    mass = torch.logsumexp(scores, dim=1) - torch.logsumexp(positive_scores, dim=1)
    best_positive = positive_scores.max(dim=1).values
    negative_scores = scores.masked_fill(positives, float("-inf"))
    tenth_negative = negative_scores.topk(10, dim=1).values[:, -1]
    boundary = torch.nn.functional.softplus(margin + tenth_negative - best_positive)
    return (mass + boundary).mean()


def stable_topk(scores: np.ndarray, candidates: np.ndarray, k: int = 10) -> np.ndarray:
    output = np.empty((len(scores), k), dtype=np.int32)
    for row in range(len(scores)):
        order = np.lexsort((candidates[row], -scores[row]))[:k]
        output[row] = candidates[row, order]
    return output


def per_row_metrics(predictions: np.ndarray, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    recalls = np.zeros(len(predictions), dtype=np.float64)
    ndcgs = np.zeros(len(predictions), dtype=np.float64)
    for row, prediction in enumerate(predictions):
        relevant = set(
            map(int, arrays["candidates"][row][arrays["target_mask"][row].astype(bool)])
        )
        total = max(int(arrays["target_counts"][row]), 1)
        hits = [int(int(item) in relevant) for item in prediction]
        recalls[row] = sum(hits) / total
        dcg = sum(hit / math.log2(rank + 2.0) for rank, hit in enumerate(hits))
        ideal = sum(1.0 / math.log2(rank + 2.0) for rank in range(min(total, len(prediction))))
        ndcgs[row] = dcg / ideal if ideal else 0.0
    return recalls, ndcgs


def user_macro(values: np.ndarray, users: np.ndarray, selected: np.ndarray) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for value, user, keep in zip(values, users, selected):
        if keep:
            grouped[int(user)].append(float(value))
    return float(np.mean([np.mean(items) for items in grouped.values()])) if grouped else 0.0


def metric_table(predictions: np.ndarray, arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    recalls, ndcgs = per_row_metrics(predictions, arrays)
    output = {}
    for scope_index, scope in enumerate(SCOPES):
        selected = np.ones(len(recalls), dtype=bool) if scope == "overall" else arrays["bucket_ids"] == scope_index - 1
        output[scope] = {
            "rows": int(selected.sum()),
            "users": int(len(np.unique(arrays["users"][selected]))),
            "recall_at_10": user_macro(recalls, arrays["users"], selected),
            "ndcg_at_10": user_macro(ndcgs, arrays["users"], selected),
        }
    return output, recalls, ndcgs


def calibration_score(metrics: dict[str, Any], base_metrics: dict[str, Any]) -> float:
    short_safe = metrics["30-90d"]["ndcg_at_10"] >= base_metrics["30-90d"]["ndcg_at_10"] - 0.005
    if not short_safe:
        return -1e9
    long_scope = next(
        scope for scope in reversed(SCOPES[1:]) if metrics[scope]["rows"] > 0
    )
    return sum(
        metrics[scope][name]
        for scope in ("overall", long_scope)
        for name in ("recall_at_10", "ndcg_at_10")
    )


def bootstrap_delta(
    model_values: np.ndarray,
    base_values: np.ndarray,
    users: np.ndarray,
    selected: np.ndarray,
    *,
    seed: int,
    draws: int = 2000,
) -> dict[str, Any]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for model, base, user, keep in zip(model_values, base_values, users, selected):
        if keep:
            grouped[int(user)].append(float(model - base))
    values = np.asarray([np.mean(items) for items in grouped.values()], dtype=np.float64)
    if len(values) == 0:
        return {
            "delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "users": 0,
        }
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [values[rng.integers(0, len(values), size=len(values))].mean() for _ in range(draws)]
    )
    return {
        "delta": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "users": int(len(values)),
    }


def relation_shuffle(features: np.ndarray, masks: np.ndarray, row_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    output = features.copy()
    output_mask = masks.copy()
    for row, row_id in enumerate(row_ids):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(row_id)]))
        permutation = rng.permutation(features.shape[1])
        output[row, :, 2:6] = features[row, permutation, 2:6]
        output_mask[row] = masks[row, permutation]
    return output, output_mask


@torch.no_grad()
def predict_residual(model: nn.Module, features: np.ndarray, masks: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    output = np.empty(features.shape[:2], dtype=np.float32)
    model.eval()
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        output[start:stop] = model(
            torch.as_tensor(features[start:stop], device=device),
            torch.as_tensor(masks[start:stop], device=device),
        ).cpu().numpy()
    return output


def train_head(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError("head_output_exists")
    train, train_manifest = load_features(args.train_features, args.train_feature_manifest, "train_cv")
    set_seed(args.seed)
    fit_mask = user_partition(train["users"])
    calibration_mask = ~fit_mask
    if set(train["users"][fit_mask]) & set(train["users"][calibration_mask]):
        raise ValueError("user_partition_overlap")
    train_features, train_relation_mask = numeric_features(train, args.variant)
    device = torch.device(args.device)
    model = SetwiseResidual(train_features.shape[-1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    fit_rows = np.flatnonzero(fit_mask & train["target_mask"].any(axis=1))
    # merge_candidates stores the frozen exact SASRec Top-10 first.  Use that
    # identity directly so alpha=0 is exact even under floating-point score ties.
    base_calibration_predictions = train["candidates"][calibration_mask, :10].copy()
    calibration_arrays = {key: value[calibration_mask] for key, value in train.items()}
    base_calibration_metrics, _, _ = metric_table(base_calibration_predictions, calibration_arrays)
    best = None
    best_state = None
    records = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(args.seed + epoch).permutation(fit_rows)
        losses = []
        for start in range(0, len(order), args.batch_size):
            rows = order[start : start + args.batch_size]
            base = torch.as_tensor(train["base_scores"][rows], device=device)
            features = torch.as_tensor(train_features[rows], device=device)
            relation = torch.as_tensor(train_relation_mask[rows], device=device)
            positives = torch.as_tensor(train["target_mask"][rows].astype(bool), device=device)
            optimizer.zero_grad(set_to_none=True)
            residual = model(features, relation)
            loss = setwise_loss(base + residual, positives, args.margin)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite_head_loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if args.stop_after and sum(len(record.get("batches", [])) for record in records) + len(losses) >= args.stop_after:
                args.output.mkdir(parents=True, exist_ok=False)
                atomic_json(
                    args.output / "SMOKE.json",
                    {"status": "engineering_pass", "scientific_result": False, "finite_loss": True, "batches": len(losses), "test_accessed": False},
                )
                return
        calibration_residual = predict_residual(
            model,
            train_features[calibration_mask],
            train_relation_mask[calibration_mask],
            args.device,
            args.eval_batch_size,
        )
        alpha_records = []
        for alpha in args.alpha_grid:
            predictions = (
                base_calibration_predictions
                if alpha == 0
                else stable_topk(
                    train["base_scores"][calibration_mask] + alpha * calibration_residual,
                    train["candidates"][calibration_mask],
                )
            )
            metrics, _, _ = metric_table(predictions, calibration_arrays)
            score = calibration_score(metrics, base_calibration_metrics)
            alpha_records.append({"alpha": alpha, "score": score, "metrics": metrics})
            key = (score, -abs(alpha), -alpha)
            if best is None or key > best:
                best = key
                best_state = {
                    "epoch": epoch,
                    "alpha": float(alpha),
                    "score": float(score),
                    "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                }
        records.append({"epoch": epoch, "mean_loss": float(np.mean(losses)), "calibration": alpha_records})
        print(json.dumps({"event": "head_epoch", "epoch": epoch, "mean_loss": records[-1]["mean_loss"], "best": {k: v for k, v in best_state.items() if k != "model"}}), flush=True)
    if best_state is None:
        raise RuntimeError("head_selection_failed")
    model.load_state_dict(best_state["model"], strict=True)
    # Validation is deliberately opened only after epoch and alpha selection
    # are frozen on the user-disjoint Train-CV calibration rows.
    validation, validation_manifest = load_features(
        args.validation_features, args.validation_feature_manifest, "validation"
    )
    if train_manifest["a0_configuration"] != validation_manifest["a0_configuration"]:
        raise ValueError("feature_configuration_mismatch")
    if train_manifest["index_manifest_sha256"] != validation_manifest["index_manifest_sha256"]:
        raise ValueError("feature_index_mismatch")
    if train_manifest.get("base_method") != validation_manifest.get("base_method"):
        raise ValueError("feature_base_method_mismatch")
    validation_features, validation_relation_mask = numeric_features(validation, args.variant)
    validation_residual = predict_residual(
        model, validation_features, validation_relation_mask, args.device, args.eval_batch_size
    )
    base_predictions = validation["candidates"][:, :10].copy()
    model_predictions = (
        base_predictions
        if best_state["alpha"] == 0
        else stable_topk(
            validation["base_scores"] + best_state["alpha"] * validation_residual,
            validation["candidates"],
        )
    )
    shuffled_features, shuffled_mask = relation_shuffle(
        validation_features, validation_relation_mask, validation["row_ids"], args.seed
    )
    shuffled_residual = predict_residual(
        model, shuffled_features, shuffled_mask, args.device, args.eval_batch_size
    )
    shuffled_predictions = (
        base_predictions
        if best_state["alpha"] == 0
        else stable_topk(
            validation["base_scores"] + best_state["alpha"] * shuffled_residual,
            validation["candidates"],
        )
    )
    base_metrics, base_recall, base_ndcg = metric_table(base_predictions, validation)
    model_metrics, model_recall, model_ndcg = metric_table(model_predictions, validation)
    shuffled_metrics, _, _ = metric_table(shuffled_predictions, validation)
    inference = {}
    for scope_index, scope in enumerate(SCOPES):
        selected = np.ones(len(model_recall), dtype=bool) if scope == "overall" else validation["bucket_ids"] == scope_index - 1
        inference[scope] = {
            "recall_delta": bootstrap_delta(model_recall, base_recall, validation["users"], selected, seed=args.seed),
            "ndcg_delta": bootstrap_delta(model_ndcg, base_ndcg, validation["users"], selected, seed=args.seed + 1),
        }
    base_hits = base_recall > 0
    model_hits = model_recall > 0
    result = {
        "schema": HEAD_SCHEMA,
        "status": "pass",
        "variant": args.variant,
        "seed": args.seed,
        "selection": {key: value for key, value in best_state.items() if key != "model"},
        "train_fit_rows": int(fit_mask.sum()),
        "train_calibration_rows": int(calibration_mask.sum()),
        "validation_rows": len(validation["row_ids"]),
        "metrics": {"sasrec_base": base_metrics, "a1": model_metrics, "shuffled_relation": shuffled_metrics},
        "paired_bootstrap": inference,
        "rescued_rows": int((model_hits & ~base_hits).sum()),
        "displaced_rows": int((base_hits & ~model_hits).sum()),
        "changed_top10_rows": int(np.any(model_predictions != base_predictions, axis=1).sum()),
        "candidate_positive_rows": int(validation["target_mask"].any(axis=1).sum()),
        "test_accessed": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output / "head_checkpoint.pt"
    binding = {
        "producer_sha256": digest(Path(__file__)),
        "train_feature_sha256": train_manifest["feature_sha256"],
        "validation_feature_sha256": validation_manifest["feature_sha256"],
        "variant": args.variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "margin": args.margin,
        "alpha_grid": args.alpha_grid,
        "test_accessed": False,
    }
    atomic_torch(
        checkpoint_path,
        {"schema": HEAD_SCHEMA, "binding": binding, "model": best_state["model"], "selection": result["selection"]},
    )
    strict = SetwiseResidual(train_features.shape[-1], args.hidden_dim)
    strict.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"], strict=True)
    atomic_npz(
        args.output / "validation_predictions.npz",
        row_ids=validation["row_ids"],
        users=validation["users"],
        bucket_ids=validation["bucket_ids"],
        base_top10=base_predictions,
        a1_top10=model_predictions,
        shuffled_top10=shuffled_predictions,
        base_recall=base_recall,
        a1_recall=model_recall,
        base_ndcg=base_ndcg,
        a1_ndcg=model_ndcg,
    )
    result["bindings"] = binding
    result["checkpoint_sha256"] = digest(checkpoint_path)
    result["predictions_sha256"] = digest(args.output / "validation_predictions.npz")
    result["checkpoint_strict_reload"] = True
    result["seconds"] = time.monotonic() - started
    atomic_json(args.output / "A1_RESULT.json", result)
    atomic_json(
        args.output / "training_manifest.json",
        {
            "schema": HEAD_SCHEMA,
            "status": "pass",
            "binding": binding,
            "selection": result["selection"],
            "epoch_records": records,
            "checkpoint_sha256": result["checkpoint_sha256"],
            "predictions_sha256": result["predictions_sha256"],
            "checkpoint_strict_reload": True,
            "validation_accessed_once_after_freeze": True,
            "test_accessed": False,
        },
    )
    print(json.dumps({"event": "head_complete", "variant": args.variant, "seed": args.seed, "metrics": model_metrics["overall"]}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("train-crossfit-base", "verify-base-parity", "build-features", "train-head"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--train-data-dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("refine-logs/music4all_onion/PREPARED_DATASET.json"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/music4all_onion/a0_candidate_index_v1"))
    parser.add_argument("--a0-result", type=Path, default=Path("refine-logs/music4all_onion/candidate_support_a0/A0_RESULT.json"))
    parser.add_argument("--event-times", type=Path, default=Path("data/music4all_onion/baseline_features_v1/event_times.npy"))
    parser.add_argument("--ssd4rec-source", type=Path, default=Path("vendor/ssd4rec_official/ssd4rec.py"))
    parser.add_argument(
        "--muse-counts",
        type=Path,
        default=Path("data/music4all_onion/main_table_repair_v1/muse_transition_counts.npz"),
    )
    parser.add_argument("--full-base-manifest", type=Path, default=Path("runs/music4all_baselines/sasrec_standard_s2024/training_manifest.json"))
    parser.add_argument("--base-training-manifest", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--reference-evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--surface", choices=("train_cv", "validation", "test"))
    parser.add_argument("--row-ids", type=Path,
                        help="Optional deterministic Train row selection for feature export.")
    parser.add_argument("--train-features", type=Path)
    parser.add_argument("--train-feature-manifest", type=Path)
    parser.add_argument("--validation-features", type=Path)
    parser.add_argument("--validation-feature-manifest", type=Path)
    parser.add_argument("--variant", choices=("full", "x_only"), default="full")
    parser.add_argument("--exclude-x-source", action="store_true",
                        help="Attribution only: remove X from both quota and fallback; retain X feature values for admitted items.")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
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
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    parser.add_argument("--resume-if-present", action="store_true")
    parser.add_argument("--final-checkpoint-only", action="store_true",
                        help="Save only the final model, without epoch resume backups.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0, 2.0))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "build-features" and not all((args.base_training_manifest, args.base_checkpoint, args.feature_manifest, args.surface)):
        parser.error("build-features requires base manifest/checkpoint, feature manifest, and surface")
    if args.exclude_x_source and args.mode != "build-features":
        parser.error("exclude-x-source is only defined for feature construction")
    if args.mode == "verify-base-parity" and not all(
        (args.base_training_manifest, args.base_checkpoint, args.reference_evaluation)
    ):
        parser.error("verify-base-parity requires base manifest/checkpoint and reference evaluation")
    if args.mode == "train-head" and not all((args.train_features, args.train_feature_manifest)):
        parser.error("train-head requires the Train-CV feature bundle and manifest")
    if args.mode == "train-head" and not args.stop_after and not all((args.validation_features, args.validation_feature_manifest)):
        parser.error("formal train-head requires the Validation feature bundle and manifest")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.mode == "train-crossfit-base":
        train_crossfit_base(args)
    elif args.mode == "verify-base-parity":
        verify_base_parity(args)
    elif args.mode == "build-features":
        build_features(args)
    else:
        train_head(args)


if __name__ == "__main__":
    main()
