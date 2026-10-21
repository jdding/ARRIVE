#!/usr/bin/env python3
"""Unified Train/Validation runner for ARRIVE paper-facing baselines.

One data adapter, objective, evaluator, artifact schema and CLI serve every
method and seed.  A new dataset is admitted by adding a JSON dataset contract;
a new method adds only a model adapter in ``arrive_main_table_models.py``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
from scipy import sparse
import torch

import run_lfm_timeconcat_control as training
from arrive_main_table_models import MODEL_CLASSES, MainTableModel
from lfm_timeconcat_control import FIRST_ITEM, PAD, LFMTimeConcatData, gap_bucket, popularity_noise
from music_sequence_core import set_metrics


STATE_SCHEMA = "genrec.arrive_main_table.state.v1"
TRAINING_SCHEMA = "genrec.arrive_main_table.training.v1"
SMOKE_SCHEMA = "genrec.arrive_main_table.gpu_smoke.v1"
METHODS = (*MODEL_CLASSES, "car", "deltagate")
OBJECTIVE = "oneshot_only"
SESSION_GAP_SECONDS = 1800


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text())
    if contract.get("schema") != "genrec.arrive_dataset_contract.v1":
        raise ValueError("dataset_contract_schema_mismatch")
    if contract.get("test_accessed") is not False:
        raise ValueError("test_accessed_before_baseline_freeze")
    return contract


class MainTableData(LFMTimeConcatData):
    """Shared efficient batch builder with aligned observed event times."""

    def __init__(
        self, root: Path, *, event_times: Path, max_context: int, horizon: int,
        transition_cache: Path | None = None,
    ) -> None:
        super().__init__(root, max_context=max_context, horizon=horizon)
        self.event_times = np.load(event_times, mmap_mode="r")
        if self.event_times.shape != self.events.shape:
            raise ValueError("event_time_alignment_shape_mismatch")
        self.transition_cache = None
        self.muse_augmenter = None
        self.augmentation_seed = 0
        self.augmentation_epoch = 0
        self.augmentation_offset = 0
        self.augmented_width = self.max_context
        if transition_cache is not None:
            cache = np.load(transition_cache, allow_pickle=False)
            self.transition_cache = {name: cache[name] for name in cache.files}

    def set_epoch(self, epoch, seed):
        self.augmentation_epoch, self.augmentation_seed = int(epoch), int(seed)

    def set_batch_offset(self, offset):
        self.augmentation_offset = int(offset)

    def context_with_time(self, row_id: int) -> tuple[list[int], list[int]]:
        _, first_user, start, _, _, _ = map(int, self.sessions[row_id])
        begin = max(first_user, start - self.max_context + 1)
        # Materialize each contiguous mmap slice in one NumPy operation.  The
        # previous ``map(int, memmap_slice)`` crossed the Python/mmap boundary
        # once per event (roughly sixty thousand calls for a 256-row MUSE
        # batch) even though the resulting integer lists are identical.
        items = np.asarray(self.events[begin : start + 1]).tolist()
        times = np.asarray(self.event_times[begin : start + 1]).tolist()
        if not items or len(items) != len(times):
            raise ValueError("empty_or_misaligned_context")
        return items, times

    @staticmethod
    def _set_targets(targets: list[list[int]], device: torch.device) -> dict[str, torch.Tensor]:
        values, sessions, weights = [], [], []
        for session, suffix in enumerate(targets):
            unique = list(dict.fromkeys(suffix))
            if not unique:
                raise ValueError("empty known target set")
            for target in unique:
                values.append(target)
                sessions.append(session)
                weights.append(1.0 / len(unique))
        return {
            "set_targets": torch.as_tensor(values, dtype=torch.long, device=device),
            "set_sessions": torch.as_tensor(sessions, dtype=torch.long, device=device),
            "set_weights": torch.as_tensor(weights, dtype=torch.float32, device=device),
        }

    def _session_features(self, times: list[int]) -> tuple[list[int], np.ndarray, int]:
        session_index = [0]
        for at in range(1, len(times)):
            session_index.append(
                session_index[-1] + int(times[at] - times[at - 1] > SESSION_GAP_SECONDS)
            )
        count = session_index[-1] + 1
        features = np.zeros((self.max_context, 5), dtype=np.float32)
        starts = [0]
        starts.extend(at for at in range(1, len(times)) if session_index[at] != session_index[at - 1])
        for slot, start in enumerate(starts):
            timestamp = times[start]
            gap_days = 0.0 if start == 0 else (timestamp - times[start - 1]) / 86400.0
            hour = (timestamp % 86400) / 86400.0
            weekday = ((timestamp // 86400) + 3) % 7 / 7.0
            features[slot] = (
                np.log1p(max(gap_days, 0.0)),
                np.sin(2 * np.pi * hour), np.cos(2 * np.pi * hour),
                np.sin(2 * np.pi * weekday), np.cos(2 * np.pi * weekday),
            )
        return session_index, features, count

    def _transition_augment(self, items: list[int], times: list[int]) -> tuple[list[int], list[int]]:
        if self.transition_cache is None:
            return items, list(range(len(items)))
        outgoing_items = self.transition_cache["outgoing_items"]
        outgoing_weights = self.transition_cache["outgoing_weights"]
        incoming_items = self.transition_cache["incoming_items"]
        incoming_weights = self.transition_cache["incoming_weights"]
        augmented: list[int] = []
        positions: list[int] = []
        for at, item in enumerate(items):
            augmented.append(item)
            positions.append(at)
            if at + 1 == len(items) or times[at + 1] - times[at] > SESSION_GAP_SECONDS:
                continue
            target = items[at + 1]
            out_ids, in_ids = outgoing_items[item], incoming_items[target]
            matches = out_ids[:, None] == in_ids[None, :]
            matches &= out_ids[:, None] >= FIRST_ITEM
            if not matches.any():
                continue
            scores = outgoing_weights[item][:, None] + incoming_weights[target][None, :]
            scores = np.where(matches, scores, -np.inf)
            bridge = int(out_ids[np.unravel_index(int(np.argmax(scores)), scores.shape)[0]])
            if bridge not in (item, target):
                augmented.append(bridge)
                positions.append(-1)
        return augmented[-self.max_context :], positions[-self.max_context :]

    def training_batch(
        self,
        row_ids: Iterable[int],
        device: torch.device,
        *,
        target_order: str = "observed",
        shuffle_seed: int = 0,
    ) -> dict[str, torch.Tensor]:
        del shuffle_seed
        if target_order != "observed":
            raise ValueError("main-table objective preserves observed labels")
        rows = list(map(int, row_ids))
        sequence = torch.full((len(rows), self.max_context), PAD, dtype=torch.long)
        event_times = torch.zeros((len(rows), self.max_context), dtype=torch.long)
        event_session_index = torch.zeros((len(rows), self.max_context), dtype=torch.long)
        session_features = torch.zeros((len(rows), self.max_context, 5), dtype=torch.float32)
        session_positions = torch.empty(len(rows), dtype=torch.long)
        augmented_sequence = torch.full((len(rows), self.augmented_width), PAD, dtype=torch.long)
        augmented_positions = torch.full((len(rows), self.augmented_width), -1, dtype=torch.long)
        positions = torch.empty(len(rows), dtype=torch.long)
        targets: list[list[int]] = []
        gaps: list[float] = []
        for out, row_id in enumerate(rows):
            items, times = self.context_with_time(row_id)
            items, times = items[-self.max_context :], times[-self.max_context :]
            sequence[out, : len(items)] = torch.as_tensor(items)
            event_times[out, : len(times)] = torch.as_tensor(times)
            indices, features, session_count = self._session_features(times)
            event_session_index[out, : len(indices)] = torch.as_tensor(indices)
            session_features[out] = torch.as_tensor(features)
            session_positions[out] = session_count - 1
            if self.muse_augmenter is None:
                augmented, source_positions = self._transition_augment(items, times)
            else:
                rng = np.random.default_rng(np.random.SeedSequence(
                    [self.augmentation_seed, self.augmentation_epoch, row_id]))
                augmented, source_positions = self.muse_augmenter(items, times, rng)
            augmented_sequence[out, : len(augmented)] = torch.as_tensor(augmented)
            augmented_positions[out, : len(source_positions)] = torch.as_tensor(source_positions)
            positions[out] = len(items) - 1
            targets.append(self.known_suffix(row_id))
            gaps.append(self.gap_days(row_id))
        batch = {
            "sequence": sequence.to(device),
            "event_times": event_times.to(device),
            "event_session_index": event_session_index.to(device),
            "session_features": session_features.to(device),
            "session_positions": session_positions.to(device),
            "augmented_sequence": augmented_sequence.to(device),
            "augmented_positions": augmented_positions.to(device),
            "initial_positions": positions.to(device),
            "gap_days": torch.as_tensor(gaps, dtype=torch.float32, device=device),
            "user_ids": torch.as_tensor(
                [int(self.sessions[row_id, 0]) for row_id in rows],
                dtype=torch.long, device=device,
            ),
        }
        batch.update(self._set_targets(targets, device))
        return batch

    def serving_rows(self, row_ids: Iterable[int], device: torch.device) -> dict[str, torch.Tensor]:
        rows = list(map(int, row_ids))
        sequence = torch.full((len(rows), self.max_context), PAD, dtype=torch.long)
        event_times = torch.zeros((len(rows), self.max_context), dtype=torch.long)
        event_session_index = torch.zeros((len(rows), self.max_context), dtype=torch.long)
        session_features = torch.zeros((len(rows), self.max_context, 5), dtype=torch.float32)
        session_positions = torch.empty(len(rows), dtype=torch.long)
        positions = torch.empty(len(rows), dtype=torch.long)
        gaps = []
        for out, row_id in enumerate(rows):
            items, times = self.context_with_time(row_id)
            sequence[out, : len(items)] = torch.as_tensor(items)
            event_times[out, : len(times)] = torch.as_tensor(times)
            indices, features, session_count = self._session_features(times)
            event_session_index[out, : len(indices)] = torch.as_tensor(indices)
            session_features[out] = torch.as_tensor(features)
            session_positions[out] = session_count - 1
            positions[out] = len(items) - 1
            gaps.append(self.gap_days(row_id))
        return {
            "sequence": sequence.to(device),
            "event_times": event_times.to(device),
            "event_session_index": event_session_index.to(device),
            "session_features": session_features.to(device),
            "session_positions": session_positions.to(device),
            "initial_positions": positions.to(device),
            "gap_days": torch.as_tensor(gaps, dtype=torch.float32, device=device),
            "user_ids": torch.as_tensor(
                [int(self.sessions[row_id, 0]) for row_id in rows],
                dtype=torch.long, device=device,
            ),
        }


def deterministic_content(audio_path: Path, genre_path: Path, seed: int = 1701) -> np.ndarray:
    audio = np.asarray(np.load(audio_path, mmap_mode="r"), dtype=np.float32)
    genre = sparse.load_npz(genre_path).tocsr().astype(np.float32)
    if len(audio) != genre.shape[0]:
        raise ValueError("content_feature_row_mismatch")
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((genre.shape[1], 32), dtype=np.float32) / np.sqrt(32.0)
    projected = np.asarray(genre @ projection, dtype=np.float32)
    features = np.concatenate((audio, projected), axis=1)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    features[:FIRST_ITEM] = 0
    if not np.isfinite(features).all():
        raise ValueError("nonfinite_content_features")
    return features


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    repaired = getattr(args, "adapter_revision", "original") == "audit20260912"
    faithful = getattr(args, "adapter_revision", "original") == "faithful20260912"
    deltagate = getattr(args, "adapter_revision", "original") == "faithful20260917"
    warm_head = getattr(args, "adapter_revision", "original") == "warmhead20260917"
    warm_full = getattr(args, "adapter_revision", "original") == "warmfull20260917"
    if repaired and args.method not in {"car", "muse", "tisasrec", "ssd4rec"}:
        raise ValueError("method_fidelity_unresolved_no_repaired_formal_launch")
    if faithful and args.method not in {"megan", "cosernn"}:
        raise ValueError("faithful_revision_is_only_for_megan_or_cosernn")
    if deltagate and args.method != "deltagate":
        raise ValueError("deltagate_revision_is_only_for_deltagate")
    if args.method == "deltagate" and not (deltagate or warm_head or warm_full):
        raise ValueError("deltagate_requires_admitted_revision")
    if (warm_head or warm_full) and args.method != "deltagate":
        raise ValueError("warm_revisions_are_only_for_deltagate")
    if warm_head or warm_full:
        backbone = getattr(args, "deltagate_backbone", None)
        if not backbone or not Path(backbone).is_file():
            raise ValueError("warm_deltagate_requires_backbone_checkpoint")
    if args.method in {"megan", "cosernn", "factubr"} and not faithful:
        raise ValueError("paper_method_name_requires_faithful_admitted_revision")
    if args.method == "car" and not repaired:
        raise ValueError("CAR_requires_source_verified_revision")
    contract = load_contract(args.dataset_config)
    prepared = json.loads(args.manifest.read_text())
    labels = json.loads(args.labels_manifest.read_text())
    if prepared.get("schema") != contract["prepared_schema"]:
        raise ValueError("prepared_schema_mismatch")
    if prepared.get("test_accessed") is not False or labels.get("test_accessed") is not False:
        raise ValueError("test_accessed_before_baseline_freeze")
    hashes: dict[str, str] = {}
    for name, entry in prepared["artifacts"].items():
        path = args.data_dir / name
        actual = digest(path)
        if actual != entry["sha256"]:
            raise ValueError(f"input_hash_mismatch:{name}")
        hashes[name] = actual
    if digest(args.data_dir / "metric_labels.npz") != labels["sha256"]:
        raise ValueError("metric_label_hash_mismatch")
    timestamp_manifest = json.loads(args.timestamp_manifest.read_text())
    if timestamp_manifest.get("schema") == "genrec.yambda500m.prepared.v2":
        event_time_hash = timestamp_manifest.get("artifacts", {}).get("event_times.npy", {}).get("sha256")
        timestamp_ok = (
            timestamp_manifest.get("status") == "finalized"
            and timestamp_manifest.get("test_accessed") is False
            and timestamp_manifest.get("artifacts", {}).get("events.npy", {}).get("sha256") == hashes["events.npy"]
            and event_time_hash == digest(args.event_times)
            and timestamp_manifest.get("protocol", {}).get("axis_unit_seconds") == 1
        )
    else:
        event_time_hash = timestamp_manifest.get("event_times_sha256")
        timestamp_ok = (
            timestamp_manifest.get("schema") in (
                "genrec.music4all_onion.event_times.v1", "genrec.yambda500m.event_times.v1"
            )
            and timestamp_manifest.get("status") == "pass"
            and timestamp_manifest.get("prepared_events_sha256") == hashes["events.npy"]
            and event_time_hash == digest(args.event_times)
            and timestamp_manifest.get("test_accessed") is False
        )
    if not timestamp_ok:
        raise ValueError("event_timestamp_binding_mismatch")
    hashes["event_times.npy"] = event_time_hash
    if args.method in {"megan", "factubr"} and not faithful:
        content_manifest = json.loads(args.content_manifest.read_text())
        if (
            content_manifest.get("status") != "pass"
            or content_manifest.get("test_accessed") is not False
            or content_manifest.get("source_hashes", {}).get("identity") != prepared["vocabulary_sha256"]
            or content_manifest.get("artifacts", {}).get("audio") != digest(args.audio_features)
            or content_manifest.get("artifacts", {}).get("genre") != digest(args.genre_features)
        ):
            raise ValueError("content_feature_cache_binding_mismatch")
        hashes["musicnn_normalized.npy"] = digest(args.audio_features)
        hashes["genre_normalized.npz"] = digest(args.genre_features)
    if repaired and args.method == "muse":
        receipt = json.loads(args.muse_counts_manifest.read_text())
        if receipt.get("status") != "pass" or receipt.get("test_accessed") is not False:
            raise ValueError("muse_count_cache_not_admitted")
        expected = {"events": hashes["events.npy"], "sessions": hashes["sessions.npy"],
                    "times": hashes["event_times.npy"]}
        if receipt["inputs"] != expected or receipt["counts_sha256"] != digest(args.muse_counts):
            raise ValueError("muse_raw_counts_binding_mismatch")
        hashes["muse_transition_counts.npz"] = receipt["counts_sha256"]
    elif args.method in {"muse", "megan", "factubr"} and not faithful:
        transition_manifest = json.loads(args.transition_manifest.read_text())
        if (
            transition_manifest.get("schema") != "genrec.arrive_transition_cache.v1"
            or transition_manifest.get("status") != "pass"
            or transition_manifest.get("validation_accessed") is not False
            or transition_manifest.get("test_accessed") is not False
            or transition_manifest.get("source_hashes", {}).get("events") != hashes["events.npy"]
            or transition_manifest.get("source_hashes", {}).get("sessions") != hashes["sessions.npy"]
            or transition_manifest.get("source_hashes", {}).get("event_times") != hashes["event_times.npy"]
            or transition_manifest.get("artifact", {}).get("sha256") != digest(args.transition_cache)
        ):
            raise ValueError("transition_cache_binding_mismatch")
        hashes["transition_topk.npz"] = digest(args.transition_cache)
    source_root = Path(__file__).resolve().parent
    provenance = contract["methods"][args.method]
    source_hashes = {
        "models": digest(source_root / "arrive_main_table_models.py"),
        "runner": digest(Path(__file__)),
        "training_controller": digest(source_root / "run_lfm_timeconcat_control.py"),
    }
    if args.method == "ssd4rec":
        source_hashes["official_ssd4rec"] = digest(args.ssd4rec_source)
    if args.method == "deltagate":
        source_hashes["deltagate_operators"] = digest(source_root / "arrive_deltagate_repaired.py")
        source_hashes["vendor_deltagate"] = digest(
            source_root.parent / "vendor" / "deltagate_corrected" / "src" / "deltagate.py"
        )
        if warm_head or warm_full:
            hashes["deltagate_backbone"] = digest(Path(args.deltagate_backbone))
            # stash for the create_model warm route: fail closed if the file
            # changes between validation and model construction (TOCTOU), and
            # require the backbone to have been trained on this exact data.
            args._deltagate_backbone_sha = hashes["deltagate_backbone"]
            args._deltagate_backbone_data_hashes = {
                key: hashes[key] for key in ("events.npy", "sessions.npy", "event_times.npy")
            }
    if repaired and args.method in {"muse", "car"}:
        source_hashes["repaired_operators"] = digest(source_root / (
            "arrive_muse_repaired.py" if args.method == "muse" else "arrive_context_repaired.py"))
    if faithful:
        asset_manifest = json.loads(args.faithful_asset_manifest.read_text())
        metadata_information = args.metadata_information
        metadata_tags = args.metadata_tags
        asset_producer = source_root / "prepare_arrive_faithful_baseline_assets.py"
        expected_asset_sources = {
            "events": hashes["events.npy"],
            "sessions": hashes["sessions.npy"],
            "event_times": hashes["event_times.npy"],
            "track_identity": digest(args.data_dir / "track_identity.tsv"),
            "information": digest(metadata_information),
            "tags": digest(metadata_tags),
        }
        expected_schema = {
            "megan": "genrec.arrive_megan_hmg_assets.v1",
            "cosernn": "genrec.arrive_cosernn_track_space.v1",
        }[args.method]
        if (
            asset_manifest.get("schema") != expected_schema
            or asset_manifest.get("status") != "pass"
            or asset_manifest.get("test_accessed") is not False
            or asset_manifest.get("source_hashes") != expected_asset_sources
            or asset_manifest.get("producer_sha256") != digest(asset_producer)
        ):
            raise ValueError("faithful_asset_manifest_binding_mismatch")
        asset_path = args.megan_assets if args.method == "megan" else args.cosernn_embeddings
        if asset_manifest.get("artifact_sha256") != digest(asset_path):
            raise ValueError("faithful_asset_hash_mismatch")
        hashes[asset_path.name] = asset_manifest["artifact_sha256"]
        source_hashes["faithful_operators"] = digest(source_root / (
            "arrive_megan_repaired.py" if args.method == "megan" else "arrive_context_repaired.py"
        ))
        source_hashes["faithful_asset_producer"] = asset_manifest["producer_sha256"]
        if args.method == "megan":
            line_manifest = json.loads(args.megan_line_manifest.read_text())
            line_producer = source_root / "pretrain_arrive_megan_line.py"
            if (
                line_manifest.get("schema") != "genrec.arrive_megan_line_initialization.v1"
                or line_manifest.get("status") != "pass"
                or line_manifest.get("training_split") != "Train"
                or line_manifest.get("validation_accessed") is not False
                or line_manifest.get("test_accessed") is not False
                or line_manifest.get("hmg_assets_sha256") != asset_manifest["artifact_sha256"]
                or line_manifest.get("hmg_manifest_sha256") != digest(args.faithful_asset_manifest)
                or line_manifest.get("producer_sha256") != digest(line_producer)
                or line_manifest.get("artifact_sha256") != digest(args.megan_line_embeddings)
                or int(line_manifest.get("dimension", -1)) != args.embed_dim
            ):
                raise ValueError("MEGAN_LINE_initialization_binding_mismatch")
            hashes[args.megan_line_embeddings.name] = line_manifest["artifact_sha256"]
            source_hashes["megan_line_producer"] = line_manifest["producer_sha256"]
    return {
        "schema": "genrec.arrive_main_table.binding.v1",
        "dataset": contract["dataset_id"],
        "method": args.method,
        "adapter_revision": getattr(args, "adapter_revision", "original"),
        "inputs": hashes,
        "vocabulary": prepared["vocabulary_sha256"],
        "source_hashes": source_hashes,
        "provenance": provenance,
        "configuration": {
            key: getattr(args, key, {"cosernn_fc_dim": 200, "cosernn_hidden_dim": 400}.get(key))
            for key in (
                "seed", "objective", "epochs", "batch_size", "max_context", "horizon",
                "learning_rate", "negative_samples", "embed_dim", "num_heads",
                "num_layers", "dropout", "cosernn_fc_dim", "cosernn_hidden_dim",
                "deltagate_backbone",
            )
        },
        "convergence_policy": {
            key: getattr(args, key, default) for key, default in
            (("min_epochs", 0), ("patience", 0), ("min_delta", 0.0))
        },
        "resume_policy": "full_rng_optimizer_mid_epoch_v1",
        "optimizer": {key: getattr(args, key, default) for key, default in
                      (("optimizer", "AdamW"), ("beta2", .999), ("weight_decay", 1e-4))},
        "muse_synthetic_capacity": 2 * args.max_context - 1 if repaired and args.method == "muse" else None,
        "estimand": "distinct_later_tracks_in_same_return_session",
        "selection_metric": "validation_user_macro_ndcg_at_10",
        "shared_runner": True,
        "validation_only": True,
        "test_accessed": False,
    }


def create_model(args: argparse.Namespace, data: MainTableData) -> MainTableModel:
    common: dict[str, Any] = {
        "vocab_size": data.vocab_size,
        "noise_probabilities": popularity_noise(data.frequencies),
        "embed_dim": args.embed_dim,
        "max_sequence": args.max_context,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    if getattr(args, "adapter_revision", "original") in ("faithful20260917", "warmhead20260917", "warmfull20260917"):
        if args.method != "deltagate":
            raise ValueError("deltagate_revisions_route_only_deltagate")
        from arrive_deltagate_repaired import DeltaGateRepairedAdapter, load_warm_backbone
        model = DeltaGateRepairedAdapter(**common)
        revision = getattr(args, "adapter_revision", "original")
        if revision in ("warmhead20260917", "warmfull20260917"):
            expected_sha = getattr(args, "_deltagate_backbone_sha", None)
            if expected_sha is not None and digest(Path(args.deltagate_backbone)) != expected_sha:
                raise ValueError("deltagate_backbone_changed_after_validation")
            payload = torch.load(args.deltagate_backbone, map_location="cpu", weights_only=False)
            backbone_binding = payload.get("binding", {})
            if int(backbone_binding.get("configuration", {}).get("seed", -1)) != int(args.seed):
                raise ValueError("deltagate_backbone_seed_mismatch")
            expected_data = getattr(args, "_deltagate_backbone_data_hashes", {})
            if expected_data:
                backbone_inputs = backbone_binding.get("inputs", {})
                checked = 0
                for data_key, expected in expected_data.items():
                    # the backbone records only the artifacts its own runner
                    # consumed; SASRec-standard never read event_times.npy,
                    # so absent keys are skipped rather than treated as drift
                    if data_key not in backbone_inputs:
                        continue
                    if backbone_inputs[data_key] != expected:
                        raise ValueError(f"deltagate_backbone_data_mismatch:{data_key}")
                    checked += 1
                if checked < 2:
                    raise ValueError("deltagate_backbone_shares_too_few_data_bindings")
            state = payload["model_state_dict"] if "model_state_dict" in payload else payload["model"]
            load_warm_backbone(
                model, state, "head_only" if revision == "warmhead20260917" else "full"
            )
        return model
    if getattr(args, "adapter_revision", "original") == "audit20260912":
        if args.method == "muse":
            from arrive_muse_repaired import MUSERepairedAdapter
            return MUSERepairedAdapter(**common, max_augmented_sequence=2 * args.max_context - 1)
        if args.method == "car":
            from arrive_context_repaired import CARRepairedAdapter
            uniform = np.zeros(data.vocab_size, dtype=np.float32)
            uniform[FIRST_ITEM:] = 1 / (data.vocab_size - FIRST_ITEM)
            common["noise_probabilities"] = uniform
            return CARRepairedAdapter(**common)
    if getattr(args, "adapter_revision", "original") == "faithful20260912":
        manifest = json.loads(args.faithful_asset_manifest.read_text())
        if args.method == "cosernn":
            from arrive_context_repaired import CoSeRNNPlayedOnlyAdapter
            fixed = np.load(args.cosernn_embeddings, mmap_mode="r")
            return CoSeRNNPlayedOnlyAdapter(
                **common,
                fixed_item_embeddings=fixed,
                embedding_provenance={
                    "training_split": "train",
                    "sha256": manifest["artifact_sha256"],
                    "producer": manifest["producer_sha256"],
                },
                fc_dim=getattr(args, "cosernn_fc_dim", 200),
                hidden_dim=getattr(args, "cosernn_hidden_dim", 400),
            )
        from arrive_megan_repaired import MEGANVerifiedRepresentation, NeighborTable
        cache = np.load(args.megan_assets, allow_pickle=False)
        line_manifest = json.loads(args.megan_line_manifest.read_text())
        line_embeddings = np.load(args.megan_line_embeddings, mmap_mode="r")
        relations = {
            name: NeighborTable(
                torch.as_tensor(cache[name + "_ids"]),
                torch.as_tensor(cache[name + "_weights"]),
            )
            for name in ("user_music", "user_session", "music_user", "music_music", "music_session")
        }
        return MEGANVerifiedRepresentation(
            vocab_size=common["vocab_size"],
            noise_probabilities=common["noise_probabilities"],
            embed_dim=common["embed_dim"],
            user_tokens=cache["user_tokens"], music_tokens=cache["music_tokens"],
            feature_count=int(cache["feature_count"]),
            session_count=int(cache["session_count"]), relations=relations,
            line_embeddings=line_embeddings, line_offsets=line_manifest["offsets"],
            feature_dim=args.embed_dim, hidden_dim=args.embed_dim,
        )
    if args.method == "megan":
        common["content_features"] = deterministic_content(args.audio_features, args.genre_features)
        cache = np.load(args.transition_cache, allow_pickle=False)
        common["transition_items"] = np.concatenate(
            (cache["outgoing_items"], cache["incoming_items"]), axis=1
        )
        common["transition_weights"] = np.concatenate(
            (cache["outgoing_weights"], cache["incoming_weights"]), axis=1
        )
    if args.method == "factubr":
        common["content_features"] = deterministic_content(args.audio_features, args.genre_features)
    if args.method == "ssd4rec":
        common["official_source"] = args.ssd4rec_source
    return MODEL_CLASSES[args.method](**common)


@torch.no_grad()
def exact_topk(model: MainTableModel, states: torch.Tensor, batch: dict[str, torch.Tensor], *, k: int, chunk_size: int) -> torch.Tensor:
    best_scores = torch.full((len(states), k), float("-inf"), device=states.device)
    best_items = torch.full((len(states), k), -1, dtype=torch.long, device=states.device)
    scoring_batch = dict(batch)
    scoring_batch["candidate_sessions"] = torch.arange(len(states), device=states.device)
    for start in range(FIRST_ITEM, model.vocab_size, chunk_size):
        stop = min(start + chunk_size, model.vocab_size)
        items = torch.arange(start, stop, device=states.device)
        scores = model.score_items(states, items, batch=scoring_batch)
        local_k = min(k, stop - start)
        local_scores, local_items = scores.topk(local_k, dim=1)
        local_items += start
        joined_scores = torch.cat((best_scores, local_scores), dim=1)
        joined_items = torch.cat((best_items, local_items), dim=1)
        best_scores, selected = joined_scores.topk(k, dim=1)
        best_items = joined_items.gather(1, selected)
    if (best_items < FIRST_ITEM).any():
        raise ValueError("catalog_topk_not_filled")
    return best_items


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
            "recall_at_10": float(np.mean([np.mean(v["recall"]) for v in by_user.values()])) if by_user else None,
            "ndcg_at_10": float(np.mean([np.mean(v["ndcg"]) for v in by_user.values()])) if by_user else None,
        }
    return output


@torch.no_grad()
def evaluate(model: MainTableModel, data: MainTableData, *, split: int, device, batch_size=16, limit=0):
    model.eval()
    catalog_vectors = model.item_vectors(
        torch.arange(model.vocab_size, device=device)
    ).detach()
    row_ids = data.row_ids(split, returns_only=True)
    if limit:
        row_ids = row_ids[:limit]
    labels = np.load(data.root / "metric_labels.npz", allow_pickle=False)
    lookup = {int(row): at for at, row in enumerate(labels["row_ids"])}
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for at in range(0, len(row_ids), batch_size):
        ids = row_ids[at : at + batch_size]
        batch = data.serving_rows(ids, torch.device(device))
        batch["_catalog_vectors"] = catalog_vectors
        hidden = model.encode_batch(batch)
        states = model.gather_states(hidden, batch["initial_positions"])
        top = exact_topk(model, states, batch, k=10, chunk_size=4096).cpu().numpy()
        for offset, row_id in enumerate(map(int, ids)):
            label_at = lookup[row_id]
            truth = labels["values"][labels["offsets"][label_at] : labels["offsets"][label_at + 1]].tolist()
            recall, ndcg = set_metrics(top[offset].tolist(), truth)
            gap = data.gap_days(row_id)
            rows.append({
                "row_id": row_id,
                "user": int(data.sessions[row_id, 0]),
                "gap_bucket": gap_bucket(gap),
                "recall_at_10": recall,
                "ndcg_at_10": ndcg,
            })
    return {
        "schema": "genrec.arrive_main_table.validation.v1",
        "split": split,
        "rows": rows,
        "metrics": aggregate(rows),
        "future_labels_enter_serving": False,
        "engineering_limit": limit,
        "evaluation_seconds": time.monotonic() - started,
        "test_accessed": False,
    }


def normalize_smoke(args: argparse.Namespace, binding: dict[str, Any]) -> None:
    path = args.output / "SMOKE.json"
    payload = json.loads(path.read_text())
    elapsed = float(payload["seconds"])
    payload.pop("same_checkpoint_decoders", None)
    payload.update({
        "schema": SMOKE_SCHEMA,
        "status": "pass",
        "method": args.method,
        "dataset": binding["dataset"],
        "seed": args.seed,
        "device": args.device,
        "cuda_visible_devices": (
            os.environ.get("CUDA_VISIBLE_DEVICES")
            if args.device == "cuda" else None
        ),
        "protocol_binding": binding,
        "finite_first_loss": payload.pop("finite_loss"),
        "minimum_validation_artifact": payload.get("validation_rows", 0) > 0,
        "updates_per_second": float(payload["updates"]) / max(elapsed, 1e-9),
        "decoder_contract": "one_shot_full_catalog_topk",
        "test_accessed": False,
    })
    training.atomic_json(path, payload)


def run(args: argparse.Namespace) -> None:
    if args.objective != OBJECTIVE:
        raise ValueError("main_table_objective_must_be_oneshot_only")
    binding = validate_inputs(args)
    training.STATE_SCHEMA = STATE_SCHEMA
    training.TRAINING_SCHEMA = TRAINING_SCHEMA
    def data_factory(root, max_context, horizon):
        repaired_muse = args.method == "muse" and getattr(args, "adapter_revision", "original") == "audit20260912"
        data = MainTableData(root, event_times=args.event_times, max_context=max_context, horizon=horizon,
                             transition_cache=args.transition_cache if args.method == "muse" and not repaired_muse else None)
        if repaired_muse:
            from arrive_muse_repaired import MUSETransitionAugmenter
            data.augmented_width = 2 * max_context - 1
            data.muse_augmenter = MUSETransitionAugmenter(sparse.load_npz(args.muse_counts), max_sequence=data.augmented_width)
        return data
    training.LFMTimeConcatData = data_factory
    training.validate_inputs = validate_inputs
    training.create_model = create_model
    training.evaluate_shared_checkpoint = evaluate
    result = training.train(args)
    if result.get("status") == "engineering_pass":
        normalize_smoke(args, binding)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--adapter-revision", choices=("original", "audit20260912", "faithful20260912", "faithful20260917", "warmhead20260917", "warmfull20260917"), default="original")
    parser.add_argument("--deltagate-backbone", type=Path,
                        help="matched frozen SASRec checkpoint for warm-start deltagate revisions")
    parser.add_argument("--faithful-asset-manifest", type=Path)
    parser.add_argument("--metadata-information", type=Path,
                        default=Path("data/music4all_onion/raw/id_information.csv"))
    parser.add_argument("--metadata-tags", type=Path,
                        default=Path("data/music4all_onion/raw/id_tags.csv"))
    parser.add_argument("--megan-assets", type=Path)
    parser.add_argument("--megan-line-manifest", type=Path)
    parser.add_argument("--megan-line-embeddings", type=Path)
    parser.add_argument("--cosernn-embeddings", type=Path)
    parser.add_argument("--cosernn-fc-dim", type=int, default=200)
    parser.add_argument("--cosernn-hidden-dim", type=int, default=400)
    parser.add_argument("--muse-counts", type=Path, default=Path("data/music4all_onion/main_table_repair_v1/muse_transition_counts.npz"))
    parser.add_argument("--muse-counts-manifest", type=Path, default=Path("data/music4all_onion/main_table_repair_v1/MUSE_COUNTS.json"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/arrive_main_table/music4all_onion.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("refine-logs/music4all_onion/PREPARED_DATASET.json"))
    parser.add_argument("--labels-manifest", type=Path, default=Path("refine-logs/music4all_onion/METRIC_LABELS.json"))
    parser.add_argument("--event-times", type=Path, default=Path("data/music4all_onion/baseline_features_v1/event_times.npy"))
    parser.add_argument("--timestamp-manifest", type=Path, default=Path("data/music4all_onion/baseline_features_v1/event_times_manifest.json"))
    parser.add_argument("--audio-features", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1/musicnn_normalized.npy"))
    parser.add_argument("--genre-features", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1/genre_normalized.npz"))
    parser.add_argument("--content-manifest", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1/feature_cache_manifest.json"))
    parser.add_argument("--transition-cache", type=Path, default=Path("data/music4all_onion/main_table_shared_v1/transition_topk.npz"))
    parser.add_argument("--transition-manifest", type=Path, default=Path("data/music4all_onion/main_table_shared_v1/MANIFEST.json"))
    parser.add_argument("--ssd4rec-source", type=Path, default=Path("vendor/ssd4rec_official/ssd4rec.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--objective", choices=(OBJECTIVE,), default=OBJECTIVE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--require-exact-resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=129)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=("Adam", "AdamW"), default="AdamW")
    parser.add_argument("--beta2", type=float, default=.999)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    args = parser.parse_args()
    if getattr(args, "deltagate_backbone", None) is not None:
        # keep the binding manifest JSON-serializable
        args.deltagate_backbone = str(args.deltagate_backbone)
    return args


if __name__ == "__main__":
    args = parse_args()
    torch.set_num_threads(args.threads)
    run(args)
