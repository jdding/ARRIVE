#!/usr/bin/env python3
"""Train the post-A1 fixed-support candidate-value component.

The runner reuses the frozen A1 score as its exact predecessor.  Three matched
variants isolate the two changes selected by the residual analysis:

* ``dense_old`` removes A1's sparse relation mask but keeps its old loss;
* ``sparse_value`` keeps the mask but uses the normalized multi-positive loss;
* ``full`` combines dense support and normalized multi-positive value learning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from run_music4all_candidate_support_a1 import (
    HEAD_SCHEMA,
    SCOPES,
    SetwiseResidual,
    atomic_json,
    atomic_npz,
    atomic_torch,
    bootstrap_delta,
    digest,
    load_features,
    metric_table,
    numeric_features,
    predict_residual,
    set_seed,
    setwise_loss,
    stable_topk,
    user_partition,
)


SCHEMA = "genrec.music4all_onion.component_b_value.v1"
VARIANTS = {
    "dense_old": ("dense", "a1"),
    "sparse_value": ("sparse", "value"),
    "full": ("dense", "value"),
}


def load_a1_model(
    checkpoint_path: Path,
    result_path: Path,
    prediction_path: Path,
    train_feature_sha256: str,
    validation_feature_sha256: str,
    device: str,
) -> tuple[SetwiseResidual, dict[str, Any], dict[str, Any]]:
    result = json.loads(result_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        result.get("schema") != HEAD_SCHEMA
        or result.get("status") != "pass"
        or result.get("variant") != "full"
        or result.get("test_accessed") is not False
        or result.get("checkpoint_sha256") != digest(checkpoint_path)
        or result.get("predictions_sha256") != digest(prediction_path)
        or checkpoint.get("schema") != HEAD_SCHEMA
        or checkpoint.get("binding") != result.get("bindings")
        or result["bindings"].get("train_feature_sha256") != train_feature_sha256
        or result["bindings"].get("validation_feature_sha256") != validation_feature_sha256
    ):
        raise ValueError("frozen_a1_binding_invalid")
    model = SetwiseResidual(8, int(result["bindings"]["hidden_dim"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint, result


def a1_scores(
    model: SetwiseResidual,
    checkpoint: dict[str, Any],
    arrays: dict[str, np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    features, relation = numeric_features(arrays, "full")
    residual = predict_residual(model, features, relation, device, batch_size)
    return arrays["base_scores"].astype(np.float64) + float(checkpoint["selection"]["alpha"]) * residual


def normalized_value_loss(
    scores: torch.Tensor,
    positives: torch.Tensor,
    target_counts: torch.Tensor,
    row_weights: torch.Tensor,
    *,
    margin: float,
    quota_weight: float,
    k: int = 10,
) -> torch.Tensor:
    valid = positives.any(dim=1)
    if not valid.any():
        raise ValueError("batch_has_no_positive_candidate")
    scores = scores[valid]
    positives = positives[valid]
    target_counts = target_counts.to(torch.int64)[valid].to(scores.dtype)
    row_weights = row_weights[valid].to(scores.dtype)
    log_probabilities = torch.log_softmax(scores, dim=1)
    value = -(log_probabilities * positives.to(scores.dtype)).sum(dim=1) / target_counts

    positive_scores = scores.masked_fill(~positives, float("-inf")).sort(dim=1, descending=True).values[:, :k]
    negative_scores = scores.masked_fill(positives, float("-inf")).topk(k, dim=1).values
    positive_counts = positives.sum(dim=1).clamp(max=k)
    quota_terms = []
    for rank in range(k):
        active = positive_counts > rank
        threshold = negative_scores[:, k - rank - 1]
        term = torch.nn.functional.softplus(margin + threshold - positive_scores[:, rank])
        quota_terms.append(torch.where(active, term / target_counts, torch.zeros_like(term)))
    quota = torch.stack(quota_terms, dim=1).sum(dim=1)
    combined = value + quota_weight * quota
    return (combined * row_weights).sum() / row_weights.sum()


def calibration_score(metrics: dict[str, Any], base: dict[str, Any]) -> tuple[float, float]:
    primary = sum(
        metrics[scope][metric]
        for scope in ("overall", ">365d")
        for metric in ("recall_at_10", "ndcg_at_10")
    )
    short_decline = max(
        base["30-90d"][metric] - metrics["30-90d"][metric]
        for metric in ("recall_at_10", "ndcg_at_10")
    )
    return primary, -max(short_decline, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--train-feature-manifest", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--validation-feature-manifest", type=Path, required=True)
    parser.add_argument("--a1-checkpoint", type=Path, required=True)
    parser.add_argument("--a1-result", type=Path, required=True)
    parser.add_argument("--a1-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--quota-weight", type=float, default=1.0)
    parser.add_argument("--beta-grid", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0, 2.0))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("component_b_output_exists")
    set_seed(args.seed)
    train, train_manifest = load_features(args.train_features, args.train_feature_manifest, "train_cv")
    validation_manifest_metadata = json.loads(args.validation_feature_manifest.read_text())
    if validation_manifest_metadata.get("surface") != "validation" or validation_manifest_metadata.get("test_accessed") is not False:
        raise ValueError("validation_feature_manifest_invalid")
    a1_model, a1_checkpoint, a1_result = load_a1_model(
        args.a1_checkpoint,
        args.a1_result,
        args.a1_predictions,
        train_manifest["feature_sha256"],
        validation_manifest_metadata["feature_sha256"],
        args.device,
    )
    train_a1 = a1_scores(a1_model, a1_checkpoint, train, args.device, args.eval_batch_size)
    feature_values, relation_mask = numeric_features(train, "full")
    mask_mode, loss_mode = VARIANTS[args.variant]
    if mask_mode == "dense":
        relation_mask = np.ones_like(relation_mask)

    fit_mask = user_partition(train["users"])
    calibration_mask = ~fit_mask
    fit_rows = np.flatnonzero(fit_mask & train["target_mask"].any(axis=1))
    user_counts: dict[int, int] = {}
    for user in train["users"][fit_rows]:
        user_counts[int(user)] = user_counts.get(int(user), 0) + 1
    row_weights = np.asarray([1.0 / user_counts.get(int(user), 1) for user in train["users"]], dtype=np.float32)
    calibration_arrays = {key: value[calibration_mask] for key, value in train.items()}
    base_calibration_predictions = stable_topk(train_a1[calibration_mask], calibration_arrays["candidates"])
    base_calibration_metrics, _, _ = metric_table(base_calibration_predictions, calibration_arrays)

    device = torch.device(args.device)
    model = SetwiseResidual(feature_values.shape[-1], args.hidden_dim).to(device)
    nn.init.zeros_(model.network[-1].weight)
    nn.init.zeros_(model.network[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_key = None
    best_state = None
    epoch_records = []
    batches_seen = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(args.seed + epoch).permutation(fit_rows)
        losses = []
        for start in range(0, len(order), args.batch_size):
            rows = order[start : start + args.batch_size]
            features = torch.as_tensor(feature_values[rows], device=device)
            mask = torch.as_tensor(relation_mask[rows], device=device)
            scores = torch.as_tensor(train_a1[rows], device=device) + model(features, mask)
            positives = torch.as_tensor(train["target_mask"][rows].astype(bool), device=device)
            if loss_mode == "a1":
                loss = setwise_loss(scores, positives, args.margin)
            else:
                loss = normalized_value_loss(
                    scores,
                    positives,
                    torch.as_tensor(train["target_counts"][rows], device=device),
                    torch.as_tensor(row_weights[rows], device=device),
                    margin=args.margin,
                    quota_weight=args.quota_weight,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite_component_b_loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            batches_seen += 1
            if args.stop_after and batches_seen >= args.stop_after:
                args.output.mkdir(parents=True, exist_ok=False)
                atomic_json(
                    args.output / "SMOKE.json",
                    {
                        "status": "engineering_pass",
                        "scientific_result": False,
                        "variant": args.variant,
                        "batches": batches_seen,
                        "finite_loss": True,
                        "validation_accessed": False,
                        "test_accessed": False,
                    },
                )
                return
        residual = predict_residual(
            model,
            feature_values[calibration_mask],
            relation_mask[calibration_mask],
            args.device,
            args.eval_batch_size,
        )
        choices = []
        for beta in args.beta_grid:
            predictions = (
                base_calibration_predictions
                if beta == 0
                else stable_topk(
                    train_a1[calibration_mask] + beta * residual,
                    calibration_arrays["candidates"],
                )
            )
            metrics, _, _ = metric_table(predictions, calibration_arrays)
            score = calibration_score(metrics, base_calibration_metrics)
            choices.append({"beta": beta, "score": score, "metrics": metrics})
            key = (score, -abs(beta), -beta)
            if best_key is None or key > best_key:
                best_key = key
                best_state = {
                    "epoch": epoch,
                    "beta": float(beta),
                    "score": score,
                    "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                }
        epoch_records.append({"epoch": epoch, "mean_loss": float(np.mean(losses)), "calibration": choices})
        print(json.dumps({"event": "component_b_epoch", "variant": args.variant, "epoch": epoch, "loss": epoch_records[-1]["mean_loss"], "best_beta": best_state["beta"]}), flush=True)

    if best_state is None:
        raise RuntimeError("component_b_selection_failed")
    model.load_state_dict(best_state["model"], strict=True)
    validation, validation_manifest = load_features(
        args.validation_features, args.validation_feature_manifest, "validation"
    )
    validation_a1 = a1_scores(a1_model, a1_checkpoint, validation, args.device, args.eval_batch_size)
    prior_predictions = np.load(args.a1_predictions, allow_pickle=False)
    if not np.array_equal(prior_predictions["row_ids"], validation["row_ids"]):
        raise ValueError("a1_prediction_row_mismatch")
    if not np.array_equal(stable_topk(validation_a1, validation["candidates"]), prior_predictions["a1_top10"]):
        raise ValueError("a1_score_reconstruction_failed")
    validation_values, validation_relation_mask = numeric_features(validation, "full")
    if mask_mode == "dense":
        validation_relation_mask = np.ones_like(validation_relation_mask)
    validation_residual = predict_residual(
        model, validation_values, validation_relation_mask, args.device, args.eval_batch_size
    )
    base_predictions = stable_topk(validation_a1, validation["candidates"])
    model_predictions = (
        base_predictions
        if best_state["beta"] == 0
        else stable_topk(
            validation_a1 + best_state["beta"] * validation_residual,
            validation["candidates"],
        )
    )
    base_metrics, base_recall, base_ndcg = metric_table(base_predictions, validation)
    model_metrics, model_recall, model_ndcg = metric_table(model_predictions, validation)
    inference = {}
    for scope_index, scope in enumerate(SCOPES):
        selected = np.ones(len(model_recall), dtype=bool) if scope == "overall" else validation["bucket_ids"] == scope_index - 1
        inference[scope] = {
            "recall_delta": bootstrap_delta(model_recall, base_recall, validation["users"], selected, seed=args.seed),
            "ndcg_delta": bootstrap_delta(model_ndcg, base_ndcg, validation["users"], selected, seed=args.seed + 1),
        }

    binding = {
        "schema": SCHEMA,
        "variant": args.variant,
        "mask_mode": mask_mode,
        "loss_mode": loss_mode,
        "seed": args.seed,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "margin": args.margin,
        "quota_weight": args.quota_weight,
        "beta_grid": list(args.beta_grid),
        "producer_sha256": digest(Path(__file__)),
        "train_feature_sha256": train_manifest["feature_sha256"],
        "validation_feature_sha256": validation_manifest["feature_sha256"],
        "a1_checkpoint_sha256": digest(args.a1_checkpoint),
        "a1_result_sha256": digest(args.a1_result),
        "test_accessed": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output / "component_b_checkpoint.pt"
    atomic_torch(
        checkpoint_path,
        {
            "schema": SCHEMA,
            "binding": binding,
            "model": best_state["model"],
            "selection": {key: value for key, value in best_state.items() if key != "model"},
        },
    )
    strict = SetwiseResidual(feature_values.shape[-1], args.hidden_dim)
    strict.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"], strict=True)
    prediction_path = args.output / "validation_predictions.npz"
    atomic_npz(
        prediction_path,
        row_ids=validation["row_ids"],
        users=validation["users"],
        bucket_ids=validation["bucket_ids"],
        a1_top10=base_predictions,
        component_b_top10=model_predictions,
        a1_recall=base_recall,
        component_b_recall=model_recall,
        a1_ndcg=base_ndcg,
        component_b_ndcg=model_ndcg,
    )
    result = {
        "schema": SCHEMA,
        "status": "pass",
        "variant": args.variant,
        "seed": args.seed,
        "selection": {key: value for key, value in best_state.items() if key != "model"},
        "metrics": {"a1": base_metrics, "component_b": model_metrics},
        "paired_bootstrap": inference,
        "rescued_rows": int(((model_recall > 0) & (base_recall == 0)).sum()),
        "displaced_rows": int(((base_recall > 0) & (model_recall == 0)).sum()),
        "changed_top10_rows": int(np.any(model_predictions != base_predictions, axis=1).sum()),
        "binding": binding,
        "checkpoint_sha256": digest(checkpoint_path),
        "predictions_sha256": digest(prediction_path),
        "checkpoint_strict_reload": True,
        "seconds": time.monotonic() - started,
        "test_accessed": False,
    }
    atomic_json(args.output / "COMPONENT_B_RESULT.json", result)
    atomic_json(
        args.output / "training_manifest.json",
        {
            "schema": SCHEMA,
            "status": "pass",
            "binding": binding,
            "selection": result["selection"],
            "epoch_records": epoch_records,
            "fit_rows": int(fit_mask.sum()),
            "calibration_rows": int(calibration_mask.sum()),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "predictions_sha256": result["predictions_sha256"],
            "validation_accessed_after_selection": True,
            "test_accessed": False,
        },
    )
    print(json.dumps({"status": "pass", "variant": args.variant, "selection": result["selection"], "metrics": model_metrics["overall"]}), flush=True)


if __name__ == "__main__":
    main()
