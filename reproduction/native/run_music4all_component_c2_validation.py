#!/usr/bin/env python3
"""Three-seed Validation transfer test for fixed-support C2 conversion."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lfm_timeconcat_control import LFMTimeConcatData
from run_music4all_artist_component_b import (
    HEAD_SCHEMA as ARTIST_SCHEMA,
    ProposalResidual,
    candidate_view,
    load_artist_features,
    proposal_numeric,
    residual_scores as artist_residual_scores,
)
from run_music4all_candidate_support_a1 import (
    HEAD_SCHEMA as A1_SCHEMA,
    SetwiseResidual,
    atomic_json,
    atomic_npz,
    atomic_torch,
    digest,
    stable_topk,
    user_partition,
)
from run_music4all_component_b import a1_scores, load_a1_model
from run_music4all_component_c_oof import context_states
from run_music4all_component_c2_conversion_oof import (
    SCHEMA as OOF_SCHEMA,
    VARIANTS,
    ConversionRanker,
    conversion_loss,
    deterministic_shuffle,
    paired_bootstrap,
    predict_residual,
    row_metrics,
    selection_score,
    set_seed,
    token_features,
    user_disjoint_derangement,
    user_macro,
)


SCHEMA = "genrec.music4all_onion.component_c2_validation.v1"
SCOPES = ("overall", "30-90d", "90-180d", "180-365d", ">365d")


def frozen_predecessor(args: argparse.Namespace, arrays: dict[str, np.ndarray], surface: str):
    feature_manifest = json.loads((args.features_root / f"{surface}_features_manifest.json").read_text())
    training_features_root = getattr(args, "training_features_root", None) or args.features_root
    training_feature_manifest = json.loads(
        (training_features_root / "train_cv_features_manifest.json").read_text()
    )
    rcce = getattr(args, "rcce_dir", None)
    a1_dir = rcce / "a1" if rcce is not None else args.a1_root / f"full_s{args.seed}"
    a1_result_path = a1_dir / "A1_RESULT.json"
    a1_checkpoint_path = a1_dir / "head_checkpoint.pt"
    a1_predictions_path = a1_dir / "validation_predictions.npz"
    a1_result = json.loads(a1_result_path.read_text())
    if int(a1_result.get("seed", -1)) != args.seed:
        raise ValueError("a1_seed_mismatch")
    if surface == "train_cv":
        a1_checkpoint = torch.load(a1_checkpoint_path, map_location="cpu", weights_only=False)
        if (
            a1_result.get("schema") != A1_SCHEMA
            or a1_result.get("status") != "pass"
            or a1_result.get("variant") != "full"
            or a1_result.get("checkpoint_sha256") != digest(a1_checkpoint_path)
            or a1_checkpoint.get("schema") != A1_SCHEMA
            or a1_checkpoint.get("binding") != a1_result.get("bindings")
            or a1_result["bindings"].get("train_feature_sha256") != training_feature_manifest["a1_feature_sha256"]
        ):
            raise ValueError("frozen_a1_train_binding_invalid")
        a1_model = SetwiseResidual(8, int(a1_result["bindings"]["hidden_dim"])).to(args.device)
        a1_model.load_state_dict(a1_checkpoint["model"], strict=True)
        a1_model.eval()
    elif surface == "validation":
        a1_model, a1_checkpoint, _ = load_a1_model(
            a1_checkpoint_path,
            a1_result_path,
            a1_predictions_path,
            training_feature_manifest["a1_feature_sha256"],
            feature_manifest["a1_feature_sha256"],
            args.device,
        )
    elif surface == "test":
        a1_checkpoint = torch.load(a1_checkpoint_path, map_location="cpu", weights_only=False)
        train_feature_sha = training_feature_manifest["a1_feature_sha256"]
        if (
            a1_result.get("schema") != A1_SCHEMA
            or a1_result.get("status") != "pass"
            or a1_result.get("variant") != "full"
            or a1_result.get("test_accessed") is not False
            or a1_result.get("checkpoint_sha256") != digest(a1_checkpoint_path)
            or a1_checkpoint.get("schema") != A1_SCHEMA
            or a1_checkpoint.get("binding") != a1_result.get("bindings")
            or a1_result["bindings"].get("train_feature_sha256") != train_feature_sha
        ):
            raise ValueError("frozen_a1_test_binding_invalid")
        a1_model = SetwiseResidual(8, int(a1_result["bindings"]["hidden_dim"])).to(args.device)
        a1_model.load_state_dict(a1_checkpoint["model"], strict=True)
        a1_model.eval()
    else:
        raise ValueError(f"unsupported_surface:{surface}")
    original_proxy = {key: arrays[key] for key in ("base_scores", "base_ranks", "x_ranks", "h_ranks", "log_popularity", "gaps")}
    original_scores = a1_scores(a1_model, a1_checkpoint, original_proxy, args.device, args.eval_batch_size)
    proposal_features, _ = proposal_numeric(arrays, 0)
    proposal_proxy = {
        "base_scores": arrays["proposal_base_scores"][:, 0],
        "base_ranks": arrays["proposal_base_ranks"][:, 0],
        "x_ranks": arrays["proposal_x_ranks"][:, 0],
        "h_ranks": arrays["proposal_h_ranks"][:, 0],
        "log_popularity": arrays["proposal_log_popularity"][:, 0],
        "gaps": arrays["gaps"],
    }
    proposal_scores = a1_scores(a1_model, a1_checkpoint, proposal_proxy, args.device, args.eval_batch_size)
    if surface == "validation":
        cached_a1 = dict(np.load(a1_predictions_path, allow_pickle=False))["a1_top10"]
    else:
        cached_a1 = stable_topk(original_scores, arrays["candidates"])
    view = candidate_view(arrays, original_scores, proposal_scores, proposal_features, 0, cached_a1)
    artist_dir = rcce / "a1_plus_b" if rcce is not None else args.artist_root / f"formal_s{args.seed}"
    artist_result_path = artist_dir / "ARTIST_B_RESULT.json"
    artist_checkpoint_path = artist_dir / "artist_b_checkpoint.pt"
    artist_predictions_path = artist_dir / "validation_predictions.npz"
    artist_result = json.loads(artist_result_path.read_text())
    artist_checkpoint = torch.load(artist_checkpoint_path, map_location="cpu", weights_only=False)
    if (
        artist_result.get("schema") != ARTIST_SCHEMA
        or artist_result.get("status") != "pass"
        or int(artist_result.get("seed", -1)) != args.seed
        or artist_result.get("checkpoint_sha256") != digest(artist_checkpoint_path)
        or (surface == "validation" and artist_result.get("predictions_sha256") != digest(artist_predictions_path))
        or int(artist_checkpoint.get("seed", artist_checkpoint.get("binding", {}).get("seed", -1))) != args.seed
        or artist_checkpoint.get("binding") != artist_result.get("binding")
        or artist_result["binding"].get("train_feature_sha256") != training_feature_manifest["feature_sha256"]
        or (surface == "validation" and artist_result["binding"].get("validation_feature_sha256") != feature_manifest["feature_sha256"])
    ):
        raise ValueError("artist_b_binding_invalid")
    artist = ProposalResidual(proposal_features.shape[-1], int(artist_result["binding"]["hidden_dim"])).to(args.device)
    artist.load_state_dict(artist_checkpoint["model"], strict=True)
    residual = artist_residual_scores(artist, view["features"], view["mask"], args.device, args.eval_batch_size)
    scores = view["scores"] + float(artist_result["selection"]["alpha"]) * residual
    top10 = stable_topk(scores, view["candidates"])
    if surface == "validation":
        cached_b = dict(np.load(artist_predictions_path, allow_pickle=False))["artist_b_top10"]
        if not np.array_equal(top10, cached_b):
            raise ValueError("artist_b_top10_reconstruction_failed")
    bindings = {
        "a1_result_sha256": digest(a1_result_path),
        "a1_checkpoint_sha256": digest(a1_checkpoint_path),
        "artist_b_result_sha256": digest(artist_result_path),
        "artist_b_checkpoint_sha256": digest(artist_checkpoint_path),
    }
    if surface == "validation":
        bindings.update({
            "a1_predictions_sha256": digest(a1_predictions_path),
            "artist_b_predictions_sha256": digest(artist_predictions_path),
        })
    return view["candidates"].astype(np.int64), scores.astype(np.float32), top10.astype(np.int32), view["positives"].astype(bool), feature_manifest, bindings


def supported_target_values(candidates: np.ndarray, positives: np.ndarray, target_counts: np.ndarray) -> np.ndarray:
    """Pack supported target IDs while retaining the full external denominator."""

    width = int(target_counts.max())
    output = np.full((len(candidates), width), -1, dtype=np.int64)
    for row in range(len(candidates)):
        values = candidates[row][positives[row]]
        output[row, : len(values)] = values
    return output


def surface_context_args(args, feature_manifest, surface):
    """Retain STC's learned coordinates; RCCE's score source is a separate role."""
    selected = argparse.Namespace(**vars(args))
    if surface == "validation" and getattr(args, "validation_base_checkpoint", None) is not None:
        if (digest(args.validation_base_checkpoint) != digest(args.base_checkpoint)
                or digest(args.validation_base_training_manifest) != digest(args.base_training_manifest)):
            raise ValueError("stc_context_encoder_drift")
    if getattr(args, "rcce_dir", None) is not None and surface == "train_cv":
        if (digest(selected.base_checkpoint) != feature_manifest["base_checkpoint_sha256"]
                or digest(selected.base_training_manifest) != feature_manifest["base_manifest_sha256"]):
            raise ValueError("surface_encoder_binding_mismatch")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--features-root", type=Path, default=Path("runs/music4all_artist_b/features"))
    parser.add_argument("--training-features-root", type=Path,
                        help="Canonical feature root used to train frozen A1/B heads when inference uses another Train surface.")
    parser.add_argument("--a1-root", type=Path, default=Path("runs/music4all_candidate_support_a1"))
    parser.add_argument("--artist-root", type=Path, default=Path("runs/music4all_artist_b"))
    parser.add_argument("--rcce-dir", type=Path)
    parser.add_argument("--validation-base-checkpoint", type=Path)
    parser.add_argument("--validation-base-training-manifest", type=Path)
    parser.add_argument("--export-ranking", action="store_true")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("runs/music4all_candidate_support_a1/crossfit_sasrec_s2024/final_model.pt"))
    parser.add_argument("--base-training-manifest", type=Path, default=Path("runs/music4all_candidate_support_a1/crossfit_sasrec_s2024/training_manifest.json"))
    parser.add_argument("--oof-decision", type=Path, default=Path("refine-logs/music4all_onion/component_c2/OOF_CONVERSION_DECISION.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=.25)
    parser.add_argument("--beta-grid", type=float, nargs="+", default=(0.0, .25, .5, 1.0, 2.0))
    parser.add_argument("--max-context", type=int, default=129)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if bool(args.validation_base_checkpoint) != bool(args.validation_base_training_manifest):
        raise ValueError("validation_encoder_requires_checkpoint_and_manifest")
    torch.set_num_threads(args.threads)
    if args.output.exists():
        raise FileExistsError("component_c2_validation_output_exists")
    decision = json.loads(args.oof_decision.read_text())
    if decision.get("decision") != "OOF_CONVERSION_PASS_VALIDATION_UNLOCKED" or decision.get("validation_accessed") is not False or decision.get("test_accessed") is not False:
        raise ValueError("oof_decision_not_unlocked")
    set_seed(args.seed)
    train, train_manifest = load_artist_features(args.features_root / "train_cv_features.npz", args.features_root / "train_cv_features_manifest.json", "train_cv")
    train_candidates, train_scores, train_top10, train_positives, _, train_bindings = frozen_predecessor(args, train, "train_cv")
    train_targets = supported_target_values(train_candidates, train_positives, train["target_counts"])
    data = LFMTimeConcatData(args.data_dir, max_context=args.max_context, horizon=args.horizon)
    train_context = surface_context_args(args, train_manifest, "train_cv")
    train_states, item_array, base_manifest = context_states(train_context, data, train["row_ids"])
    embeddings = torch.as_tensor(item_array, device=args.device)
    fit_mask = user_partition(train["users"])
    fit_rows = np.flatnonzero(fit_mask & train_positives.any(axis=1))
    calibration_rows = np.flatnonzero(~fit_mask)
    context_source = np.arange(len(train["row_ids"]), dtype=np.int64)
    if args.variant == "shuffled_context":
        context_source[fit_rows] = user_disjoint_derangement(fit_rows, train["users"], args.seed + 10000, "c2-val-fit")
        context_source[calibration_rows] = user_disjoint_derangement(calibration_rows, train["users"], args.seed + 20000, "c2-val-cal")
    model = ConversionRanker(item_array.shape[1], args.hidden_dim, contextual=args.variant != "pointwise").to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best = None
    records = []
    updates = 0
    args.output.mkdir(parents=True)
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        order = deterministic_shuffle(fit_rows, args.seed + epoch, "c2-validation-fit")
        losses = []
        model.train()
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            source = context_source[index]
            c = torch.as_tensor(train_candidates[index], dtype=torch.long, device=args.device)
            s = torch.as_tensor(train_scores[index], device=args.device)
            h = torch.as_tensor(train_states[source], device=args.device)
            g = torch.as_tensor(train["gaps"][source], device=args.device)
            residual = model(token_features(c, s, h, g, embeddings))
            loss = conversion_loss(
                s + residual,
                torch.as_tensor(train_positives[index], device=args.device),
                torch.as_tensor(train["target_counts"][index], dtype=torch.long, device=args.device),
                args.margin,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite_component_c2_validation_loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1
            if args.stop_after and updates >= args.stop_after:
                atomic_json(args.output / "SMOKE.json", {"schema": SCHEMA, "status": "engineering_pass", "variant": args.variant, "seed": args.seed, "updates": updates, "validation_accessed": False, "test_accessed": False})
                return
        source = context_source[calibration_rows]
        residual = predict_residual(model, train_candidates[calibration_rows], train_scores[calibration_rows], train_states[source], train["gaps"][source], embeddings, args.device, args.eval_batch_size)
        epoch_best = None
        for beta in args.beta_grid:
            prediction = stable_topk(train_scores[calibration_rows] + float(beta) * residual, train_candidates[calibration_rows])
            recall, ndcg = row_metrics(prediction, train_targets[calibration_rows], train["target_counts"][calibration_rows])
            score = selection_score(recall, ndcg, train["users"][calibration_rows], train["bucket_ids"][calibration_rows], np.ones(len(calibration_rows), dtype=bool))
            candidate = {"epoch": epoch, "beta": float(beta), "selection_score": score}
            if epoch_best is None or (score, -float(beta)) > (epoch_best["selection_score"], -epoch_best["beta"]):
                epoch_best = candidate
        assert epoch_best is not None
        records.append({"epoch": epoch, "mean_loss": float(np.mean(losses)), **epoch_best})
        if best is None or (epoch_best["selection_score"], -epoch_best["beta"], -epoch) > (best["selection_score"], -best["beta"], -best["epoch"]):
            best = {**epoch_best, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
        print(json.dumps({"event": "component_c2_validation_epoch", "variant": args.variant, "seed": args.seed, **records[-1]}), flush=True)
    assert best is not None
    model.load_state_dict(best.pop("model"), strict=True)
    checkpoint = args.output / "component_c2_checkpoint.pt"
    atomic_torch(checkpoint, {"schema": SCHEMA, "variant": args.variant, "seed": args.seed, "selection": best, "model": model.state_dict()})
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("schema") != SCHEMA or saved.get("variant") != args.variant or int(saved.get("seed", -1)) != args.seed or saved.get("selection") != best:
        raise ValueError("component_c2_checkpoint_binding_invalid")
    fresh = ConversionRanker(item_array.shape[1], args.hidden_dim, contextual=args.variant != "pointwise").to(args.device)
    fresh.load_state_dict(saved["model"], strict=True)
    fresh.eval()
    model = fresh

    # Validation is opened exactly once, after all model and calibration choices
    # have been frozen and verified from a fresh checkpoint reload.
    validation, validation_manifest = load_artist_features(args.features_root / "validation_features.npz", args.features_root / "validation_features_manifest.json", "validation")
    validation_candidates, validation_scores, validation_top10, validation_positives, _, validation_bindings = frozen_predecessor(args, validation, "validation")
    if any(validation_bindings.get(key) != value for key, value in train_bindings.items()):
        raise ValueError("predecessor_binding_drift")
    validation_targets = supported_target_values(validation_candidates, validation_positives, validation["target_counts"])
    validation_context = surface_context_args(args, validation_manifest, "validation")
    validation_states, validation_item_array, _ = context_states(validation_context, data, validation["row_ids"])
    if item_array.shape != validation_item_array.shape:
        raise ValueError("item_catalog_shape_drift")
    if not np.array_equal(item_array, validation_item_array):
        raise ValueError("item_embedding_drift")
    validation_embeddings = torch.as_tensor(validation_item_array, device=args.device)
    validation_context_source = np.arange(len(validation["row_ids"]), dtype=np.int64)
    if args.variant == "shuffled_context":
        validation_context_source = user_disjoint_derangement(np.arange(len(validation["row_ids"])), validation["users"], args.seed + 30000, "c2-val-eval")
    source = validation_context_source
    residual = predict_residual(model, validation_candidates, validation_scores, validation_states[source], validation["gaps"][source], validation_embeddings, args.device, args.eval_batch_size)
    predictions = stable_topk(validation_scores + float(best["beta"]) * residual, validation_candidates)
    base_recall, base_ndcg = row_metrics(validation_top10, validation_targets, validation["target_counts"])
    method_recall, method_ndcg = row_metrics(predictions, validation_targets, validation["target_counts"])
    metrics = {}
    for scope_index, scope in enumerate(SCOPES):
        selected = np.ones(len(predictions), dtype=bool) if scope == "overall" else validation["bucket_ids"] == scope_index - 1
        metrics[scope] = {
            "rows": int(selected.sum()),
            "users": int(len(np.unique(validation["users"][selected]))),
            "base_recall_at_10": user_macro(base_recall, validation["users"], selected),
            "method_recall_at_10": user_macro(method_recall, validation["users"], selected),
            "recall_paired": paired_bootstrap(method_recall - base_recall, validation["users"], selected, args.seed + scope_index),
            "base_ndcg_at_10": user_macro(base_ndcg, validation["users"], selected),
            "method_ndcg_at_10": user_macro(method_ndcg, validation["users"], selected),
            "ndcg_paired": paired_bootstrap(method_ndcg - base_ndcg, validation["users"], selected, args.seed + 100 + scope_index),
        }
    artifact = args.output / "validation_predictions.npz"
    atomic_npz(artifact, row_ids=validation["row_ids"], users=validation["users"], bucket_ids=validation["bucket_ids"], base_top10=validation_top10, method_top10=predictions, base_recall=base_recall, method_recall=method_recall, base_ndcg=base_ndcg, method_ndcg=method_ndcg)
    ranking_hashes = {}
    if args.export_ranking:
        train_residual = predict_residual(model, train_candidates, train_scores, train_states[context_source], train["gaps"][context_source], embeddings, args.device, args.eval_batch_size)
        for name, raw, candidates, scores, targets in (
            ("train_cv", train, train_candidates, train_scores + float(best["beta"]) * train_residual, train_targets),
            ("validation", validation, validation_candidates, validation_scores + float(best["beta"]) * residual, validation_targets),
        ):
            path = args.output / f"{name}_ranking.npz"
            atomic_npz(path, row_ids=raw["row_ids"], users=raw["users"], gaps=raw["gaps"], bucket_ids=raw["bucket_ids"], candidates=candidates, scores=scores, targets=targets, target_counts=raw["target_counts"], top10=stable_topk(scores, candidates))
            ranking_hashes[name] = digest(path)
    result = {
        "schema": SCHEMA,
        "status": "pass",
        "variant": args.variant,
        "seed": args.seed,
        "selection": best,
        "metrics": metrics,
        "benefit_harm": {
            "recall_better_rows": int((method_recall > base_recall).sum()),
            "recall_worse_rows": int((method_recall < base_recall).sum()),
            "rescued_zero_to_hit_rows": int(((method_recall > 0) & (base_recall == 0)).sum()),
            "displaced_hit_to_zero_rows": int(((method_recall == 0) & (base_recall > 0)).sum()),
        },
        "binding": {
            "train_feature_sha256": train_manifest["feature_sha256"],
            "validation_feature_sha256": validation_manifest["feature_sha256"],
            **validation_bindings,
            "oof_decision_sha256": digest(args.oof_decision),
            "base_checkpoint_sha256": digest(args.base_checkpoint),
            "base_manifest_sha256": digest(args.base_training_manifest),
            "validation_base_checkpoint_sha256": digest(validation_context.base_checkpoint),
            "validation_base_manifest_sha256": digest(validation_context.base_training_manifest),
            "base_architecture": base_manifest["binding"]["architecture_contract"],
            "producer_sha256": digest(Path(__file__)),
            "validation_accessed_once_after_freeze": True,
            "test_accessed": False,
        },
        "checkpoint_sha256": digest(checkpoint),
        "predictions_sha256": digest(artifact),
        "checkpoint_strict_reload": True,
        "epoch_records": records,
        "ranking_sha256": ranking_hashes,
        "seconds": time.monotonic() - started,
        "test_accessed": False,
    }
    atomic_json(args.output / "VALIDATION_RESULT.json", result)
    print(json.dumps({"event": "component_c2_validation_complete", "variant": args.variant, "seed": args.seed, "overall": metrics["overall"], "long": metrics[">365d"], "benefit_harm": result["benefit_harm"]}), flush=True)


if __name__ == "__main__":
    main()
