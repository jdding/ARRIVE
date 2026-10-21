#!/usr/bin/env python3
"""Joint RCCE--RGSC training on a fixed Train-legal proposal universe.

The same runner is used for SASRec and SSD4Rec.  RCCE learns proposal values
over the 200 behavioral candidates plus at most 20 return-artist candidates;
RGSC converts the selected 200-item support into the final Top-10.  Both heads
are optimized in one loop and stored in one checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import nn

from run_arrive_unified_content_conversion import (
    SCOPES,
    UnifiedContentRanker,
    context_items,
    metric_table,
    multimodal_features_dispatch,
    SparseCategorical,
)
from run_music4all_artist_component_b import load_artist_features, proposal_numeric
from run_music4all_candidate_support_a1 import (
    atomic_json,
    atomic_npz,
    atomic_torch,
    base_method,
    base_topk,
    digest,
    encode_base_states,
    load_base_data,
    load_standard_base,
    numeric_features,
    set_seed,
    stable_topk,
    target_lookup,
    user_partition,
)
from run_music4all_component_c_oof import context_states
from run_music4all_component_c2_conversion_oof import (
    conversion_loss,
    deterministic_shuffle,
    row_metrics,
    selection_score,
    user_macro,
)
from run_music4all_component_c2_validation import supported_target_values
from run_lfm_timeconcat_control import early_stop_reached


SCHEMA = "arrive.joint_rcce_rgsc.v4"
SUPPORTED_BASE_METHODS = {"sasrec", "ssd4rec", "muse", "tisasrec"}
COMPONENT_VARIANTS = ("full", "rcce_only", "rgsc_only")


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@torch.no_grad()
def exact_backbone_universe(args, arrays, feature_manifest, surface):
    """Cache the carrier's exact C0 and states, never the mixed feature prefix.

    Retrieval and scoring reuse the frozen feature builder's carrier helpers.
    Labels are read only after retrieval and cannot enter the candidate pool.
    """
    data = load_base_data(args, args.base_training_manifest)
    model, manifest = load_standard_base(
        args, data, checkpoint_path=args.base_checkpoint,
        manifest_path=args.base_training_manifest,
    )
    if base_method(manifest) not in {"sasrec", "ssd4rec"}:
        raise ValueError("component_ablation_backbone_out_of_scope")
    row_ids = arrays["row_ids"].astype(np.int64)
    binding = {
        "schema": "arrive.exact_backbone_c0.v1", "surface": surface,
        "feature_sha256": feature_manifest["feature_sha256"],
        "checkpoint_sha256": digest(args.base_checkpoint),
        "base_manifest_sha256": digest(args.base_training_manifest),
        "row_ids_sha256": array_digest(row_ids),
        "data_sha256": {name: digest(args.data_dir / name) for name in
                        ("events.npy", "sessions.npy", "train_frequencies.npy")},
        "max_context": args.max_context, "horizon": args.horizon,
        "eval_batch_size": args.eval_batch_size,
        "retrieval": "base_topk_exact_catalog_200",
        "helper_sha256": digest(Path(__file__).with_name("run_music4all_candidate_support_a1.py")),
        "retrieval_source_sha256": {
            name: digest(Path(__file__).with_name(name)) for name in
            ("lfm_timeconcat_control.py", "run_arrive_main_table_baseline.py")
        },
    }
    if getattr(args, "event_times", None) is not None and args.event_times.is_file():
        binding["event_times_sha256"] = digest(args.event_times)
    if base_method(manifest) == "ssd4rec":
        binding["ssd4rec_source_sha256"] = digest(args.ssd4rec_source)
    if surface != "train_cv":
        binding["metric_labels_sha256"] = digest(args.data_dir / "metric_labels.npz")
    cache_key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    cache_dir = args.candidate_cache_dir or args.output.parent / "_exact_c0_cache"
    cache_path = cache_dir / f"{surface}_{cache_key}.npz"
    manifest_path = cache_path.with_suffix(".json")
    if cache_path.exists() and manifest_path.exists():
        cached_manifest = json.loads(manifest_path.read_text())
        if cached_manifest.get("binding") != binding or cached_manifest.get("sha256") != digest(cache_path):
            raise ValueError("exact_c0_cache_binding_mismatch")
        with np.load(cache_path, allow_pickle=False) as cached:
            candidates, raw, states = (cached[key].copy() for key in ("candidates", "raw_scores", "states"))
            if not np.array_equal(cached["row_ids"], row_ids):
                raise ValueError("exact_c0_cache_row_mismatch")
    else:
        candidates = np.empty((len(row_ids), 200), dtype=np.int32)
        raw = np.empty((len(row_ids), 200), dtype=np.float32)
        state_batches = []
        for start in range(0, len(row_ids), args.eval_batch_size):
            stop = min(start + args.eval_batch_size, len(row_ids))
            state, serving = encode_base_states(args, data, model, row_ids[start:stop])
            selected = base_topk(model, state, serving, k=200)
            candidates[start:stop] = selected.cpu().numpy()
            raw[start:stop] = torch.einsum("bd,bkd->bk", state, model.item_embedding(selected)).cpu().numpy()
            state_batches.append(state.cpu().numpy())
        states = np.concatenate(state_batches).astype(np.float32)
        atomic_npz(cache_path, row_ids=row_ids, candidates=candidates, raw_scores=raw, states=states)
        atomic_json(manifest_path, {"binding": binding, "sha256": digest(cache_path)})
    if (candidates.shape != (len(row_ids), 200) or raw.shape != candidates.shape
            or len(states) != len(row_ids) or not np.isfinite(raw).all()
            or not np.isfinite(states).all() or np.any(candidates < 3)
            or np.any(np.diff(np.sort(candidates, axis=1), axis=1) == 0)):
        raise ValueError("exact_c0_candidate_validity_invalid")
    if not np.array_equal(candidates[:, :10], arrays["candidates"][:, :10]):
        raise ValueError("exact_c0_protected_top10_parity_failed")
    labels = None if surface == "train_cv" else target_lookup(args.data_dir)
    positives = np.zeros(candidates.shape, dtype=bool)
    for at, row in enumerate(row_ids):
        truth = set(data.known_suffix(int(row)) if labels is None else labels[int(row)])
        if len(truth) != int(arrays["target_counts"][at]):
            raise ValueError("exact_c0_target_count_mismatch")
        positives[at] = np.isin(candidates[at], list(truth))
    universe = {
        "candidates": candidates.astype(np.int64), "raw_scores": raw,
        "valid": np.ones(candidates.shape, dtype=bool), "positives": positives,
        "proposal_features": np.empty((*candidates.shape, 0), dtype=np.float32),
    }
    identity = {"kind": "pure_backbone_exact_top200", "cache_sha256": digest(cache_path),
                "binding": binding, "candidate_ids_sha256": array_digest(candidates),
                "row_ids_sha256": array_digest(row_ids)}
    items = model.item_embedding.weight.detach().cpu().numpy().astype(np.float32)
    return universe, states, items, manifest, identity


def validate_feature_backbone_binding(
    manifest: dict, checkpoint: Path, training_manifest: Path, method: str, surface: str
) -> None:
    """Bind a feature surface to its exact carrier; tolerate legacy missing method tags.

    Historical SASRec features predate the ``base_method`` field.  Their exact
    checkpoint and training-manifest digests remain the authoritative identity.
    Newer manifests may additionally declare ``base_method`` and must agree.
    """

    if (
        manifest.get("base_checkpoint_sha256") != digest(checkpoint)
        or manifest.get("base_manifest_sha256") != digest(training_manifest)
    ):
        raise ValueError(f"joint_{surface}_feature_backbone_binding_mismatch")
    declared = manifest.get("base_method")
    if declared is not None and declared != method:
        raise ValueError(f"joint_{surface}_feature_method_mismatch")


def resolved_architecture_contract(manifest: dict, method: str) -> dict | str:
    """Return an explicit serving contract for legacy and current backbones."""

    binding = manifest.get("binding", {})
    declared = binding.get("architecture_contract")
    if declared is not None:
        return declared
    configuration = binding.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("joint_architecture_contract_missing")
    return {"method": method, "configuration": configuration}


def validate_backbone_manifests(args: argparse.Namespace) -> tuple[dict, dict, str]:
    train = json.loads(args.base_training_manifest.read_text())
    serving = json.loads(args.serving_base_training_manifest.read_text())
    for label, manifest in (("train", train), ("serving", serving)):
        configuration = manifest.get("binding", {}).get("configuration", {})
        if (
            manifest.get("status") != "pass"
            or manifest.get("test_accessed") is not False
            or manifest.get("binding", {}).get("test_accessed") is not False
            or int(configuration.get("seed", -1)) != args.seed
        ):
            raise ValueError(f"joint_{label}_backbone_manifest_invalid")
    if train.get("validation_accessed") is not False:
        raise ValueError("joint_crossfit_validation_boundary_invalid")
    train_method = base_method(train)
    serving_method = base_method(serving)
    if train_method != serving_method or train_method not in SUPPORTED_BASE_METHODS:
        raise ValueError("joint_backbone_method_mismatch")
    if train.get("final_model_sha256") != digest(args.base_checkpoint):
        raise ValueError("joint_crossfit_checkpoint_hash_mismatch")
    if serving.get("best_model_sha256") != digest(args.serving_base_checkpoint):
        raise ValueError("joint_serving_checkpoint_hash_mismatch")
    return train, serving, serving_method


def joint_universe(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the exact 200+20 real-cue proposal universe and fixed features."""

    original_features, _ = numeric_features(arrays, "full")
    original_extra = np.zeros((*original_features.shape[:2], 4), dtype=np.float32)
    original_extra[..., 0] = 1.0
    proposal_features, proposal_valid = proposal_numeric(arrays, 0)
    candidates = np.concatenate(
        (arrays["candidates"].astype(np.int64), arrays["proposal_candidates"][:, 0].astype(np.int64)),
        axis=1,
    )
    raw_scores = np.concatenate(
        (arrays["base_scores"].astype(np.float32), arrays["proposal_base_scores"][:, 0].astype(np.float32)),
        axis=1,
    )
    valid = np.concatenate(
        (np.ones(arrays["candidates"].shape, dtype=bool), proposal_valid.astype(bool)), axis=1
    )
    positives = np.concatenate(
        (arrays["target_mask"].astype(bool), arrays["proposal_target_mask"][:, 0].astype(bool)), axis=1
    )
    features = np.concatenate(
        (np.concatenate((original_features, original_extra), axis=-1), proposal_features.astype(np.float32)),
        axis=1,
    )
    if candidates.shape[1] != 220 or features.shape[1:] != (220, 12):
        raise ValueError("joint_universe_shape_invalid")
    if not np.all(arrays["base_ranks"][:, :10] == np.arange(1, 11, dtype=np.uint16)):
        raise ValueError("protected_backbone_top10_invalid")
    if np.any(candidates[:, :200] < 3) or np.any(positives & ~valid):
        raise ValueError("joint_universe_validity_invalid")
    return {
        "candidates": candidates,
        "raw_scores": raw_scores,
        "valid": valid,
        "positives": positives,
        "proposal_features": features,
    }


class JointARRIVE(nn.Module):
    def __init__(self, proposal_width: int, behavior_width: int, hidden: int, component_variant: str = "full", interaction_mode: str = "setwise") -> None:
        super().__init__()
        if component_variant not in COMPONENT_VARIANTS:
            raise ValueError("component_variant_invalid")
        self.component_variant = component_variant
        self.proposal = None if component_variant == "rgsc_only" else nn.Sequential(
            nn.Linear(proposal_width, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.conversion = None if component_variant == "rcce_only" else UnifiedContentRanker(behavior_width, hidden, use_content=True, interaction_mode=interaction_mode)
        if self.proposal is not None:
            nn.init.zeros_(self.proposal[-1].weight)
            nn.init.zeros_(self.proposal[-1].bias)

    def proposal_scores(self, raw: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return raw if self.proposal is None else raw + self.proposal(features).squeeze(-1)


def lexicographic_order(
    scores: torch.Tensor,
    valid: torch.Tensor,
    candidates: torch.Tensor,
    exclude_prefix: int = 0,
) -> torch.Tensor:
    """Score-descending/item-id-ascending order, matching the NumPy evaluator."""

    masked = scores.masked_fill(~valid, float("-inf")).clone()
    if exclude_prefix:
        masked[:, :exclude_prefix] = float("-inf")
    item_order = torch.argsort(candidates, dim=1, stable=True)
    ordered_scores = torch.gather(masked, 1, item_order)
    score_order = torch.argsort(ordered_scores, dim=1, descending=True, stable=True)
    return torch.gather(item_order, 1, score_order)


def deterministic_landmark_ids(frequencies: np.ndarray, count: int) -> np.ndarray:
    """Select shared Train-only item landmarks by frequency, then item id."""

    if count <= 0 or count > len(frequencies) - 3:
        raise ValueError("joint_landmark_count_invalid")
    item_ids = np.arange(len(frequencies), dtype=np.int64)
    eligible = item_ids >= 3
    order = np.lexsort((item_ids[eligible], -np.asarray(frequencies)[eligible]))
    landmarks = item_ids[eligible][order[:count]]
    if len(np.unique(landmarks)) != count:
        raise ValueError("joint_landmark_identity_invalid")
    return landmarks


def select_pool(
    scores: torch.Tensor,
    valid: torch.Tensor,
    candidates: torch.Tensor,
    budget: int = 200,
) -> torch.Tensor:
    """Select Top-B while structurally retaining the backbone Top-10."""

    if scores.shape[1] < budget or budget < 10:
        raise ValueError("joint_pool_budget_invalid")
    tail = lexicographic_order(scores, valid, candidates, exclude_prefix=10)[:, : budget - 10]
    protected = torch.arange(10, device=scores.device).expand(scores.shape[0], -1)
    selected = torch.cat((protected, tail), dim=1)
    if not torch.gather(valid, 1, selected).all():
        raise ValueError("joint_pool_selected_invalid")
    return selected


def gather_tokens(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    index = selected
    for _ in range(values.ndim - 2):
        index = index.unsqueeze(-1)
    return torch.gather(values, 1, index.expand(*selected.shape, *values.shape[2:]))


def support_loss(
    scores: torch.Tensor,
    positives: torch.Tensor,
    valid: torch.Tensor,
    target_counts: torch.Tensor,
) -> torch.Tensor:
    active = positives.any(dim=1)
    if not active.any():
        raise ValueError("batch_has_no_universe_positive")
    score = scores[active].masked_fill(~valid[active], float("-inf"))
    positive = positives[active]
    denominator = target_counts[active].to(score.dtype).clamp_min(1.0)
    log_probability = torch.log_softmax(score, dim=1)
    positive_log_probability = torch.where(positive, log_probability, torch.zeros_like(log_probability))
    return (-(positive_log_probability.sum(dim=1)) / denominator).mean()


def ordinary_multipositive_loss(
    scores: torch.Tensor, positives: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
    """Mean cross-entropy against a uniform distribution over reachable positives.

    Rows without a reachable positive provide no listwise target. They remain
    in evaluation and contribute a differentiable zero when a batch has none.
    """
    positives = positives & valid
    if not positives.any():
        return scores.masked_fill(~valid, 0.0).sum() * 0.0
    return support_loss(scores, positives, valid, positives.sum(dim=1))


def support_boundary_loss(
    scores: torch.Tensor,
    positives: torch.Tensor,
    valid: torch.Tensor,
    target_counts: torch.Tensor,
    margin: float,
    protected: int = 10,
    budget: int = 200,
) -> torch.Tensor:
    """Quota-aware surrogate for positive containment in the served Top-B pool."""

    tail_slots = budget - protected
    tail_scores = scores[:, protected:]
    tail_positive = positives[:, protected:] & valid[:, protected:]
    tail_valid = valid[:, protected:]
    active_rows = tail_positive.any(dim=1)
    per_row = scores.masked_fill(~valid, 0.0).sum(dim=1) * 0.0
    if not active_rows.any():
        return per_row.mean()
    score = tail_scores[active_rows]
    positive = tail_positive[active_rows]
    available = tail_valid[active_rows]
    denominator = target_counts[active_rows].to(score.dtype).clamp_min(1.0)
    positive_scores = score.masked_fill(~positive, float("-inf")).sort(
        dim=1, descending=True
    ).values[:, :tail_slots]
    negative_scores = score.masked_fill(~(available & ~positive), float("-inf")).topk(
        tail_slots, dim=1
    ).values
    supported = positive.sum(dim=1).clamp(max=tail_slots)
    quota = torch.zeros_like(denominator)
    for rank in range(tail_slots):
        active = supported > rank
        threshold = negative_scores[:, tail_slots - 1 - rank]
        term = torch.nn.functional.softplus(
            margin + threshold - positive_scores[:, rank]
        ) / denominator
        quota = quota + torch.where(active, term, torch.zeros_like(term))
    active_indices = torch.nonzero(active_rows, as_tuple=False).squeeze(1)
    return per_row.index_copy(0, active_indices, quota).mean()


def deranged_observed(observed: np.ndarray, seed: int, label: str) -> np.ndarray:
    """Deterministically permute the observed return track across rows.

    Shuffled-grounding control: every query keeps its own history, gap,
    candidate pool, and targets, but its grounding signal is conditioned
    on a different row's return track. Fixed points are not enforced: a
    seeded permutation on n rows leaves ~1 self-mapped row in expectation,
    a negligible grounding residue.
    """
    keys = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{label}:{seed}:{int(i)}".encode()).digest()[:8],
                "big",
            )
            for i in range(len(observed))
        ],
        dtype=np.uint64,
    )
    order = np.argsort(keys, kind="stable")
    shuffled = observed[order]
    if len(shuffled) > 1 and (shuffled == observed).all():
        raise ValueError("shuffled_grounding_permutation_is_identity")
    return shuffled


def alignment_loss(proposal: torch.Tensor, final: torch.Tensor, temperature: float) -> torch.Tensor:
    target = torch.softmax(final.detach() / temperature, dim=1)
    return torch.nn.functional.kl_div(
        torch.log_softmax(proposal / temperature, dim=1), target, reduction="batchmean"
    ) * (temperature * temperature)


def training_objective_scores(
    selected_proposal: torch.Tensor,
    residual: torch.Tensor,
    gradient_mode: str,
) -> torch.Tensor:
    """Keep proposal support and Top-K conversion gradients role-consistent."""

    if gradient_mode == "coupled":
        return selected_proposal + residual
    if gradient_mode == "role_separated":
        return selected_proposal.detach() + residual
    raise ValueError("joint_gradient_mode_invalid")


def project_conflicting_gradient(
    support_gradients: list[torch.Tensor | None],
    conversion_gradients: list[torch.Tensor | None],
    epsilon: float = 1e-12,
) -> tuple[list[torch.Tensor], dict[str, float | bool]]:
    """Remove only the conversion component that conflicts with support descent."""

    paired = []
    for support, conversion in zip(support_gradients, conversion_gradients):
        if support is None and conversion is None:
            raise ValueError("joint_parameter_has_no_gradient")
        reference = support if support is not None else conversion
        paired.append(
            (
                torch.zeros_like(reference) if support is None else support,
                torch.zeros_like(reference) if conversion is None else conversion,
            )
        )
    support_norm_sq = sum((support.double().square().sum() for support, _ in paired))
    dot = sum(((support.double() * conversion.double()).sum() for support, conversion in paired))
    support_norm_value = float(support_norm_sq.detach())
    dot_value = float(dot.detach())
    if support_norm_value <= epsilon:
        return [torch.zeros_like(support) for support, _ in paired], {
            "support_norm_sq": support_norm_value,
            "gradient_dot": dot_value,
            "post_projection_dot": 0.0,
            "conflict": False,
            "proposal_frozen": True,
        }
    coefficient = torch.minimum(dot, torch.zeros_like(dot)) / support_norm_sq
    # Project in float64: the float32 subtraction of a tiny conflict
    # component otherwise leaves a reassociation residual that can exceed
    # the projection tolerance on large-magnitude gradients.
    projected = [
        (conversion.double() - coefficient * support.double()).to(conversion.dtype)
        for support, conversion in paired
    ]
    post_dot = sum(
        ((support.double() * conversion.double()).sum() for (support, _), conversion in zip(paired, projected))
    )
    projected_norm_sq = sum(
        conversion.double().square().sum() for conversion in projected
    )
    # The mathematical projection is formed in float64, but the returned
    # gradients must be cast back to the model dtype.  Bound the admissible
    # negative dot product by the model-dtype roundoff scale; this changes no
    # gradient and only prevents a false failure from the final cast.
    roundoff_scale = math.sqrt(
        max(0.0, support_norm_value * float(projected_norm_sq.detach()))
    )
    tolerance = max(
        1e-8 * max(1.0, abs(dot_value)),
        8.0 * torch.finfo(projected[0].dtype).eps * max(1.0, roundoff_scale),
    )
    if float(post_dot.detach()) < -tolerance:
        raise FloatingPointError(
            "joint_gradient_projection_failed:"
            f"support_norm_sq={support_norm_value}:"
            f"projected_norm_sq={float(projected_norm_sq.detach())}:"
            f"gradient_dot={dot_value}:"
            f"post_projection_dot={float(post_dot.detach())}:"
            f"tolerance={tolerance}"
        )
    return projected, {
        "support_norm_sq": support_norm_value,
        "gradient_dot": dot_value,
        "post_projection_dot": float(post_dot.detach()),
        "conflict": dot_value < 0.0,
        "proposal_frozen": False,
    }


@torch.no_grad()
def support_safe_proposal_step(
    model: JointARRIVE,
    proposal_parameters: list[nn.Parameter],
    batch,
    target_counts: torch.Tensor,
    margin: float,
    learning_rate: float,
    before_loss: float,
    max_backtracks: int = 10,
) -> dict[str, float | int | bool]:
    """Apply the largest tested proposal step that does not raise minibatch support loss."""

    originals = [parameter.detach().clone() for parameter in proposal_parameters]
    gradients = [parameter.grad.detach().clone() for parameter in proposal_parameters]
    accepted_scale = 0.0
    after_loss = before_loss
    for attempt in range(max_backtracks + 1):
        scale = 0.5**attempt
        for parameter, original, gradient in zip(proposal_parameters, originals, gradients):
            parameter.copy_(original - learning_rate * scale * gradient)
        candidates, raw, valid, positives, proposal_features = batch[:5]
        candidate_scores = model.proposal_scores(raw, proposal_features).masked_fill(
            ~valid, float("-inf")
        )
        candidate_loss = float(
            support_boundary_loss(candidate_scores, positives, valid, target_counts, margin)
        )
        tolerance = 1e-7 * max(1.0, abs(before_loss))
        if np.isfinite(candidate_loss) and candidate_loss <= before_loss + tolerance:
            accepted_scale = scale
            after_loss = candidate_loss
            return {
                "accepted": True,
                "backtracks": attempt,
                "step_scale": accepted_scale,
                "support_loss_before": before_loss,
                "support_loss_after": after_loss,
            }
    for parameter, original in zip(proposal_parameters, originals):
        parameter.copy_(original)
    return {
        "accepted": False,
        "backtracks": max_backtracks + 1,
        "step_scale": 0.0,
        "support_loss_before": before_loss,
        "support_loss_after": before_loss,
    }


def tensor_batch(
    rows: np.ndarray,
    universe: dict[str, np.ndarray],
    states: np.ndarray,
    gaps: np.ndarray,
    observed: np.ndarray,
    anchors: np.ndarray,
    anchor_mask: np.ndarray,
    item_embeddings: torch.Tensor,
    audio: torch.Tensor,
    genre: torch.Tensor,
    landmark_ids: torch.Tensor,
    objective_version: str,
    device: str,
    *,
    serving_cache=None,
    cue_ranking_mode: str = "observed",
):
    candidates_np = universe["candidates"][rows]
    valid = torch.as_tensor(universe["valid"][rows], device=device)
    safe_candidates = np.where(universe["valid"][rows], candidates_np, 0)
    candidates = torch.as_tensor(safe_candidates, dtype=torch.long, device=device)
    raw = torch.as_tensor(universe["raw_scores"][rows], device=device)
    proposal_features = torch.as_tensor(universe["proposal_features"][rows], device=device)
    positives = torch.as_tensor(universe["positives"][rows], device=device)
    state = torch.as_tensor(states[rows], device=device)
    gap = torch.as_tensor(gaps[rows], device=device)
    item_unit = (torch.nn.functional.normalize(item_embeddings[candidates], dim=-1)
                 if serving_cache is None else serving_cache.items(candidates))
    state_unit = torch.nn.functional.normalize(state, dim=-1).unsqueeze(1)
    interaction = item_unit * state_unit
    cosine = interaction.sum(dim=-1, keepdim=True)
    gap_feature = (torch.log1p(gap) / 8.0).view(-1, 1, 1).expand(-1, raw.shape[1], 1)
    if objective_version == "scalar_v1":
        behavior_static = torch.cat((item_unit, interaction, cosine), dim=-1)
    elif objective_version == "support_safe_v2":
        # Historical corrected path retained for exact diagnostic reproduction.
        behavior_static = cosine
    elif objective_version.startswith("support_safe_v3") or objective_version == "support_safe_v4":
        # Landmark-kernel coordinates retain multi-dimensional collaborative
        # information while remaining invariant to an orthogonal change of the
        # independently trained Train-CV/serving embedding bases.
        landmarks = (torch.nn.functional.normalize(item_embeddings[landmark_ids], dim=-1)
                     if serving_cache is None else serving_cache.landmarks())
        item_landmark = torch.matmul(item_unit, landmarks.T)
        state_landmark = torch.matmul(state_unit, landmarks.T).expand_as(item_landmark)
        landmark_interaction = item_landmark * state_landmark
        observed_ids = torch.as_tensor(observed[rows], dtype=torch.long, device=device)
        observed_unit = (torch.nn.functional.normalize(item_embeddings[observed_ids], dim=-1)
                         if serving_cache is None else serving_cache.items(observed_ids))
        observed_cosine = (item_unit * observed_unit.unsqueeze(1)).sum(
            dim=-1, keepdim=True
        )
        anchor_ids = torch.as_tensor(anchors[rows], dtype=torch.long, device=device)
        anchor_unit = (torch.nn.functional.normalize(item_embeddings[anchor_ids], dim=-1)
                       if serving_cache is None else serving_cache.items(anchor_ids))
        anchor_cosine = torch.einsum("bcd,bad->bca", item_unit, anchor_unit)
        anchor_cosine = anchor_cosine * torch.as_tensor(
            anchor_mask[rows], device=device
        ).unsqueeze(1)
        behavior_static = torch.cat(
            (
                cosine,
                item_landmark,
                state_landmark,
                landmark_interaction,
                observed_cosine,
                anchor_cosine,
            ),
            dim=-1,
        )
    else:
        raise ValueError("joint_objective_version_invalid")
    content_args = (
        candidates,
        torch.as_tensor(observed[rows], dtype=torch.long, device=device),
        torch.as_tensor(anchors[rows], dtype=torch.long, device=device),
        torch.as_tensor(anchor_mask[rows], device=device),
        audio,
        genre,
    )
    content = (multimodal_features_dispatch(*content_args) if serving_cache is None else
               serving_cache.content(content_args, safe_candidates, observed[rows],
                                     anchors[rows], anchor_mask[rows]))
    if cue_ranking_mode == "masked":
        if objective_version != "support_safe_v4" or content.shape[-1] != 4:
            raise ValueError("cue_mask_requires_v4_four_content_channels")
        # Remove only the extra candidate-to-first-track coordinates.  The
        # frozen backbone/proposal still receive the observed first event.
        behavior_static = behavior_static.clone()
        behavior_static[..., 1 + 3 * landmark_ids.numel()] = 0.0
        content = content.clone()
        content[..., (0, 2)] = 0.0
    elif cue_ranking_mode != "observed":
        raise ValueError("cue_ranking_mode_invalid")
    return candidates, raw, valid, positives, proposal_features, behavior_static, gap_feature, content


def score_conditioned_behavior(
    static: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
    gap_feature: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Construct RGSC rank/value features from the score law that serves the pool."""

    count = valid.sum(dim=1, keepdim=True).clamp_min(1)
    score_mean = scores.masked_fill(~valid, 0.0).sum(dim=1, keepdim=True) / count
    centered = (scores - score_mean).masked_fill(~valid, 0.0)
    score_std = torch.sqrt((centered.square().sum(dim=1, keepdim=True) / count).clamp_min(1e-6))
    score_z = (centered / score_std).unsqueeze(-1)
    masked = scores.masked_fill(~valid, float("-inf"))
    order = lexicographic_order(masked, valid, candidates)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1, order, torch.arange(1, scores.shape[1] + 1, device=scores.device).expand_as(order)
    )
    reciprocal_rank = (1.0 / torch.log2(ranks.to(scores.dtype) + 1.0)).unsqueeze(-1)
    tenth = torch.gather(scores, 1, order[:, :10])[:, -1:]
    boundary = (scores - tenth).masked_fill(~valid, 0.0).unsqueeze(-1)
    return torch.cat(
        (static, score_z, reciprocal_rank, boundary, gap_feature), dim=-1
    )


def prelearning_candidate_features(batch) -> torch.Tensor:
    """Exact current pre-learning inputs for a same-information ranker.

    This shares the v4 feature builders without giving a control the learned
    proposal/residual scores or target labels. Invalid proposal slots are
    zeroed and must still be excluded by the exported validity mask.
    """

    candidates, raw, valid, _, proposal, static, gap, content = batch
    behavior = score_conditioned_behavior(static, raw, valid, gap, candidates)
    features = torch.cat((raw.unsqueeze(-1), proposal, behavior, content), dim=-1)
    features = features.masked_fill(~valid.unsqueeze(-1), 0.0)
    if not torch.isfinite(features).all():
        raise FloatingPointError("same_information_nonfinite_features")
    return features


def forward_batch(
    model: JointARRIVE,
    batch,
    beta: float = 1.0,
    objective_version: str = "scalar_v1",
):
    candidates, raw, valid, positives, proposal_features, behavior_static, gap_feature, content = batch
    proposal = model.proposal_scores(raw, proposal_features).masked_fill(~valid, float("-inf"))
    if model.component_variant == "rgsc_only":
        if candidates.shape[1] != 200 or not valid.all():
            raise ValueError("rgsc_only_requires_exact_200_valid_candidates")
        selected = torch.arange(200, device=candidates.device).expand(len(candidates), -1)
    else:
        selected = select_pool(proposal, valid, candidates)
    selected_candidates = gather_tokens(candidates, selected)
    selected_proposal = gather_tokens(proposal, selected)
    selected_positive = gather_tokens(positives, selected)
    if model.component_variant == "rcce_only":
        residual = torch.zeros_like(selected_proposal)
        return proposal, selected, selected_candidates, selected_proposal, selected_positive, residual, selected_proposal
    # Support and conversion have different score roles.  The learned proposal
    # serves candidate containment.  V4 conditions RGSC on the frozen carrier
    # coordinate so its rank/value features do not move whenever the proposal
    # changes; the final score still jointly trains through ``proposal`` below.
    behavior_scores = raw if objective_version == "support_safe_v4" else (
        proposal if objective_version.startswith("support_safe_") else raw
    )
    behavior = score_conditioned_behavior(
        behavior_static,
        behavior_scores,
        valid,
        gap_feature,
        candidates,
    )
    selected_behavior = gather_tokens(behavior, selected)
    selected_content = gather_tokens(content, selected)
    residual = model.conversion(selected_behavior, selected_content)
    final = selected_proposal + beta * residual
    return proposal, selected, selected_candidates, selected_proposal, selected_positive, residual, final


@torch.no_grad()
def predict(
    model: JointARRIVE,
    universe: dict[str, np.ndarray],
    states: np.ndarray,
    gaps: np.ndarray,
    observed: np.ndarray,
    anchors: np.ndarray,
    anchor_mask: np.ndarray,
    item_embeddings: torch.Tensor,
    audio: torch.Tensor,
    genre: torch.Tensor,
    landmark_ids: torch.Tensor,
    device: str,
    batch_size: int,
    beta: float,
    objective_version: str,
    cue_ranking_mode: str = "observed",
):
    n = len(universe["candidates"])
    pool_candidates = np.empty((n, 200), dtype=np.int32)
    pool_scores = np.empty((n, 200), dtype=np.float32)
    final_scores = np.empty((n, 200), dtype=np.float32)
    model.eval()
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        rows = np.arange(start, stop, dtype=np.int64)
        result = forward_batch(
            model,
            tensor_batch(
                rows, universe, states, gaps, observed, anchors, anchor_mask,
                item_embeddings, audio, genre, landmark_ids, objective_version, device,
                cue_ranking_mode=cue_ranking_mode,
            ),
            beta,
            objective_version,
        )
        pool_candidates[start:stop] = result[2].cpu().numpy()
        pool_scores[start:stop] = result[3].cpu().numpy()
        final_scores[start:stop] = result[6].cpu().numpy()
    return pool_candidates, pool_scores, final_scores


def load_surface(args: argparse.Namespace, surface: str):
    return load_artist_features(
        args.features_root / f"{surface}_features.npz",
        args.features_root / f"{surface}_features_manifest.json",
        surface,
    )


@torch.no_grad()
def evaluate_source_counterfactual(args: argparse.Namespace) -> None:
    """Frozen ordinary head on an equal-budget pool built without X retrieval."""

    if args.output.exists():
        raise FileExistsError("source_counterfactual_output_exists")
    a1_manifest_path = args.features_root.parent / "a1/validation_features_manifest.json"
    a1_manifest = json.loads(a1_manifest_path.read_text())
    validation, feature_manifest = load_surface(args, "validation")
    if (
        a1_manifest.get("source_intervention") != "exclude_first_track_transition"
        or feature_manifest.get("a1_feature_sha256") != a1_manifest.get("feature_sha256")
    ):
        raise ValueError("source_counterfactual_feature_binding_invalid")
    _, _, method = validate_backbone_manifests(args)
    validate_feature_backbone_binding(
        feature_manifest, args.serving_base_checkpoint,
        args.serving_base_training_manifest, method, "validation",
    )
    parent_path = args.source_counterfactual_parent_dir / "joint_arrive_checkpoint.pt"
    parent_manifest_path = args.source_counterfactual_parent_dir / "training_manifest.json"
    parent_predictions_path = args.source_counterfactual_parent_dir / "validation_predictions.npz"
    parent_manifest = json.loads(parent_manifest_path.read_text())
    checkpoint = torch.load(parent_path, map_location="cpu", weights_only=False)
    if (
        parent_manifest.get("status") != "pass"
        or parent_manifest.get("test_accessed") is not False
        or parent_manifest.get("checkpoint_sha256") != digest(parent_path)
        or parent_manifest.get("predictions_sha256") != digest(parent_predictions_path)
        or parent_manifest.get("configuration", {}).get("training_rule") != "ordinary_multipositive"
        or checkpoint.get("seed") != args.seed or checkpoint.get("backbone") != method
        or checkpoint.get("component_variant") != "full"
        or checkpoint.get("objective_version") != "support_safe_v4"
        or parent_manifest.get("binding", {}).get("serving_base_checkpoint_sha256") != digest(args.serving_base_checkpoint)
        or parent_manifest.get("binding", {}).get("audio_sha256") != digest(args.content_dir / args.audio_file)
        or parent_manifest.get("binding", {}).get("genre_sha256") != digest(args.content_dir / args.categorical_file)
    ):
        raise ValueError("source_counterfactual_parent_invalid")
    with np.load(parent_predictions_path, allow_pickle=False) as parent_predictions:
        if not np.array_equal(validation["row_ids"], parent_predictions["row_ids"]):
            raise ValueError("source_counterfactual_parent_rows_mismatch")
        parent_pool = parent_predictions["candidate_top200"].copy()
        parent_top10 = parent_predictions["arrive_top10"].copy()
        parent_base_top10 = parent_predictions["backbone_top10"].copy()
    universe = joint_universe(validation)
    if np.any(np.diff(np.sort(universe["candidates"], axis=1), axis=1) == 0):
        # Invalid proposal slots may repeat -1; valid candidates must be unique.
        for row in range(len(validation["row_ids"])):
            selected = universe["candidates"][row][universe["valid"][row]]
            if len(selected) != len(np.unique(selected)):
                raise ValueError("source_counterfactual_duplicate_valid_candidate")
    serving_args = argparse.Namespace(**vars(args))
    serving_args.base_checkpoint = args.serving_base_checkpoint
    serving_args.base_training_manifest = args.serving_base_training_manifest
    data = load_base_data(serving_args, args.serving_base_training_manifest)
    states, items, _ = context_states(serving_args, data, validation["row_ids"])
    observed, anchors, anchor_mask = context_items(data, validation["row_ids"])
    model = JointARRIVE(
        checkpoint["proposal_width"], checkpoint["behavior_width"],
        checkpoint["hidden_dim"], "full", checkpoint.get("interaction_mode", "setwise"),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    audio = torch.as_tensor(np.asarray(np.load(args.content_dir / args.audio_file, mmap_mode="r")), device=args.device)
    genre = torch.as_tensor(
        sparse.load_npz(args.content_dir / args.categorical_file).toarray().astype(np.float32),
        device=args.device,
    )
    landmark_ids = deterministic_landmark_ids(
        data.frequencies, int(checkpoint["configuration"]["landmark_count"])
    )
    if hashlib.sha256(landmark_ids.astype("<i8", copy=False).tobytes()).hexdigest() != checkpoint["configuration"]["landmark_ids_sha256"]:
        raise ValueError("source_counterfactual_landmark_binding_mismatch")
    landmarks = torch.as_tensor(landmark_ids, dtype=torch.long, device=args.device)
    pool, proposal_scores, final_scores = predict(
        model, universe, states, validation["gaps"], observed, anchors,
        anchor_mask, torch.as_tensor(items, device=args.device), audio, genre,
        landmarks, args.device, args.eval_batch_size,
        float(checkpoint["selection"].get("beta", 0.0)), "support_safe_v4",
    )
    base_top10 = stable_topk(validation["base_scores"], validation["candidates"])
    if not np.array_equal(base_top10, parent_base_top10):
        raise ValueError("source_counterfactual_backbone_top10_drift")
    predicted = stable_topk(final_scores, pool)
    targets = supported_target_values(
        universe["candidates"], universe["positives"], validation["target_counts"]
    )
    metrics, _, _, recall, ndcg = metric_table(
        predicted, base_top10, targets, validation, 72000 + args.seed
    )
    pool_recall, _ = row_metrics(pool, targets, validation["target_counts"])
    args.output.mkdir(parents=True)
    predictions_path = args.output / "validation_predictions.npz"
    atomic_npz(
        predictions_path, row_ids=validation["row_ids"], users=validation["users"],
        bucket_ids=validation["bucket_ids"], targets=targets,
        target_counts=validation["target_counts"], candidate_top200=pool,
        arrive_top10=predicted, arrive_recall=recall, arrive_ndcg=ndcg,
        pool_recall=pool_recall, parent_candidate_top200=parent_pool,
        parent_top10=parent_top10,
    )
    atomic_json(args.output / "VALIDATION_RESULT.json", {
        "schema": "arrive.first_cue_source_counterfactual.v1", "status": "pass",
        "seed": args.seed, "metrics": metrics, "test_accessed": False,
        "rows": len(validation["row_ids"]),
        "selected_pool_changed_rows": int(np.any(pool != parent_pool, axis=1).sum()),
        "top10_changed_rows": int(np.any(predicted != parent_top10, axis=1).sum()),
        "binding": {
            "parent_checkpoint_sha256": digest(parent_path),
            "parent_manifest_sha256": digest(parent_manifest_path),
            "parent_predictions_sha256": digest(parent_predictions_path),
            "no_x_a1_manifest_sha256": digest(a1_manifest_path),
            "no_x_artist_feature_sha256": feature_manifest["feature_sha256"],
            "producer_sha256": digest(Path(__file__)),
            "test_accessed": False,
        },
        "predictions_sha256": digest(predictions_path),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(2024, 2025, 2026))
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-training-manifest", type=Path, required=True)
    parser.add_argument("--serving-base-checkpoint", type=Path, required=True)
    parser.add_argument("--serving-base-training-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component-variant", choices=COMPONENT_VARIANTS, default="full")
    parser.add_argument("--candidate-cache-dir", type=Path)
    parser.add_argument("--frozen-proposal-dir", type=Path, help="Attribution only: fixed learned selector; refit the ranking head.")
    parser.add_argument("--source-counterfactual-parent-dir", type=Path,
                        help="Evaluate the frozen ordinary head on no-X Validation features, without fitting.")
    parser.add_argument("--cue-ranking-mode", choices=("observed", "masked"), default="observed",
                        help="Fixed-selector ordinary refit: remove candidate-to-first-track coordinates during fit and inference.")
    parser.add_argument("--interaction-mode", choices=("setwise", "pointwise"), default="setwise")
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--content-dir", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1"))
    parser.add_argument("--audio-file", default="musicnn_normalized.npy")
    parser.add_argument("--categorical-file", default="genre_normalized.npz")
    parser.add_argument("--categorical-sparse", action="store_true")
    parser.add_argument("--event-times", type=Path, default=Path("data/music4all_onion/baseline_features_v1/event_times.npy"))
    parser.add_argument("--ssd4rec-source", type=Path, default=Path("vendor/ssd4rec_official/ssd4rec.py"))
    parser.add_argument(
        "--muse-counts",
        type=Path,
        default=Path("data/music4all_onion/main_table_repair_v1/muse_transition_counts.npz"),
    )
    parser.add_argument(
        "--train-muse-counts",
        type=Path,
        help="Train-CV-excluded MUSE transitions bound to the cross-fit carrier.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=.25)
    parser.add_argument("--lambda-topk", type=float, default=1.0)
    parser.add_argument("--mu-align", type=float, default=.1)
    parser.add_argument(
        "--objective-version",
        choices=("scalar_v1", "support_safe_v2", "support_safe_v3", "support_safe_v3r1", "support_safe_v4"),
        required=True,
    )
    parser.add_argument(
        "--gradient-mode", choices=("coupled", "role_separated"), default="coupled"
    )
    parser.add_argument("--training-rule", choices=("arrive", "ordinary_multipositive"),
                        default="arrive", help="Matched two-head training control; unchanged v4 forward.")
    parser.add_argument(
        "--grounding-mode",
        choices=("observed", "shuffled"),
        default="observed",
        help="shuffled permutes the observed return track across rows with a "
        "seeded derangement (all other signals untouched); control only",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta-grid", type=float, nargs="+", default=(0.0, .25, .5, 1.0, 2.0))
    parser.add_argument("--max-context", type=int, default=129)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--landmark-count", type=int, default=32)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    parser.add_argument("--resume-if-present", action="store_true")
    parser.add_argument("--return-query", action="store_true", help="Internal query-state quality/cost pilot; no Test.")
    parser.add_argument("--return-query-eval-only", action="store_true", help="Export frozen query checkpoint without retraining")
    args = parser.parse_args()
    ordinary_training = args.training_rule == "ordinary_multipositive"
    if args.source_counterfactual_parent_dir is not None:
        if (not ordinary_training or args.frozen_proposal_dir is not None
                or args.cue_ranking_mode != "observed" or args.objective_version != "support_safe_v4"):
            raise ValueError("source_counterfactual_requires_frozen_ordinary_parent")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("cuda_unavailable")
        torch.set_num_threads(args.threads)
        evaluate_source_counterfactual(args)
        return
    if ordinary_training and (
        args.objective_version != "support_safe_v4" or args.component_variant != "full"
        or args.gradient_mode != "coupled" or args.grounding_mode != "observed"
        or args.interaction_mode != "setwise"
        or args.return_query
    ):
        raise ValueError("ordinary_control_requires_full_observed_coupled_v4")
    if args.cue_ranking_mode != "observed" and not (
        ordinary_training and args.frozen_proposal_dir is not None
    ):
        raise ValueError("cue_mask_requires_ordinary_fixed_selector")
    if args.return_query:
        from arrive_return_query import run
        run(args)
        return
    if args.interaction_mode == "pointwise" and args.frozen_proposal_dir is None:
        raise ValueError("pointwise_requires_frozen_proposal_attribution")
    if args.frozen_proposal_dir is not None and (
        args.component_variant != "full" or args.objective_version != "support_safe_v4"
        or args.gradient_mode != "coupled" or args.grounding_mode != "observed"
    ):
        raise ValueError("mechanism_requires_full_observed_v4")
    if args.component_variant != "full" and (
        args.objective_version != "support_safe_v4" or args.gradient_mode != "coupled"
        or args.grounding_mode != "observed"
    ):
        raise ValueError("component_ablation_requires_observed_coupled_support_safe_v4")
    if args.component_variant == "rcce_only":
        args.beta_grid = ()
    if args.output.exists() and not args.resume_if_present:
        raise FileExistsError("joint_arrive_output_exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    started = time.monotonic()
    configuration = {
        "seed": args.seed,
        "epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "margin": args.margin,
        "lambda_topk": args.lambda_topk,
        "mu_align": args.mu_align,
        "gradient_mode": args.gradient_mode,
        "objective_version": args.objective_version,
        "grounding_mode": args.grounding_mode,
        "audio_file": args.audio_file,
        "categorical_file": args.categorical_file,
        "categorical_sparse": args.categorical_sparse,
        "temperature": args.temperature,
        "beta_grid": list(args.beta_grid),
        "max_context": args.max_context,
        "horizon": args.horizon,
        "landmark_count": args.landmark_count,
        "proposal_gradient_clip": 5.0,
        "conversion_gradient_clip": 5.0,
        "proposal_optimizer": "backtracked_plain_sgd" if args.objective_version.startswith("support_safe_") else "adamw",
        "conversion_optimizer": "adamw",
    }
    # Keep the legacy full configuration hash/resume binding unchanged.
    if ordinary_training:
        configuration.update(training_rule=args.training_rule, proposal_optimizer="adamw",
                             selection_loss="uniform_reachable_positive_cross_entropy",
                             ranking_loss="uniform_reachable_positive_cross_entropy",
                             gradient_update="ordinary_joint_adamw_global_clip5",
                             alignment_weight_effective=0.0)
    if args.frozen_proposal_dir is not None and ordinary_training:
        configuration.update(
            proposal_optimizer="frozen_parent",
            conversion_optimizer="adamw",
            gradient_update="ordinary_fixed_selector_adamw_clip5",
            cue_ranking_mode=args.cue_ranking_mode,
        )
    if args.component_variant != "full":
        configuration.update(component_variant=args.component_variant,
                             proposal_optimizer="none" if args.component_variant == "rgsc_only" else "backtracked_plain_sgd",
                             conversion_optimizer="none" if args.component_variant == "rcce_only" else "adamw")
    configuration_sha256 = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    train_base_manifest, serving_manifest, method = validate_backbone_manifests(args)
    train, train_manifest = load_surface(args, "train_cv")
    original_train_universe = joint_universe(train)
    train_universe = original_train_universe
    validate_feature_backbone_binding(
        train_manifest, args.base_checkpoint, args.base_training_manifest, method, "train"
    )

    serving_args = argparse.Namespace(**vars(args))
    serving_args.base_checkpoint = args.serving_base_checkpoint
    serving_args.base_training_manifest = args.serving_base_training_manifest
    data = load_base_data(serving_args, args.serving_base_training_manifest)
    train_context_args = argparse.Namespace(**vars(args))
    if args.objective_version.startswith("support_safe_"):
        train_context_args.base_checkpoint = args.base_checkpoint
        train_context_args.base_training_manifest = args.base_training_manifest
        if method == "muse":
            if args.train_muse_counts is None:
                raise ValueError("train_muse_counts_required")
            train_context_args.muse_counts = args.train_muse_counts
    else:
        # Preserve the historical scalar-v1 path for exact diagnostic reproduction.
        train_context_args.base_checkpoint = args.serving_base_checkpoint
        train_context_args.base_training_manifest = args.serving_base_training_manifest
    train_candidate_identity = {"kind": "rcce_four_source_universe"}
    if args.component_variant == "rgsc_only":
        train_universe, train_states, item_array, base_manifest, train_candidate_identity = exact_backbone_universe(
            train_context_args, train, train_manifest, "train_cv"
        )
    else:
        train_states, item_array, base_manifest = context_states(
            train_context_args, data, train["row_ids"]
        )
    train_observed, train_anchors, train_anchor_mask = context_items(data, train["row_ids"])
    if args.grounding_mode == "shuffled":
        train_observed = deranged_observed(
            train_observed, args.seed + 917, "arrive-joint-shuffled-grounding-train"
        )
    item_embeddings = torch.as_tensor(item_array, device=args.device)
    landmark_ids_np = deterministic_landmark_ids(data.frequencies, args.landmark_count)
    landmark_ids = torch.as_tensor(landmark_ids_np, dtype=torch.long, device=args.device)
    configuration["landmark_ids_sha256"] = hashlib.sha256(
        landmark_ids_np.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    configuration_sha256 = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    audio = torch.as_tensor(
        np.asarray(np.load(args.content_dir / args.audio_file, mmap_mode="r")),
        device=args.device,
    )
    if args.categorical_sparse:
        cat = sparse.load_npz(args.content_dir / args.categorical_file).tocsr().astype(np.float32)
        genre = SparseCategorical(
            torch.as_tensor(np.asarray(cat.indptr, dtype=np.int64), device=args.device),
            torch.as_tensor(np.asarray(cat.indices, dtype=np.int64), device=args.device),
            torch.as_tensor(np.asarray(cat.data, dtype=np.float32), device=args.device),
            (int(cat.shape[0]), int(cat.shape[1])),
        )
    else:
        genre = torch.as_tensor(
            sparse.load_npz(args.content_dir / args.categorical_file).toarray().astype(np.float32),
            device=args.device,
        )

    probe = tensor_batch(
        np.asarray([0]), train_universe, train_states, train["gaps"], train_observed,
        train_anchors, train_anchor_mask, item_embeddings, audio, genre,
        landmark_ids, args.objective_version, args.device,
        cue_ranking_mode=args.cue_ranking_mode,
    )
    behavior_width = int(probe[5].shape[-1] + 4)
    model = JointARRIVE(probe[4].shape[-1], behavior_width, args.hidden_dim, args.component_variant, args.interaction_mode).to(args.device)
    frozen_proposal_state = None
    if args.frozen_proposal_dir is not None:
        parent_path = args.frozen_proposal_dir / "joint_arrive_checkpoint.pt"
        parent_manifest_path = args.frozen_proposal_dir / "training_manifest.json"
        parent_manifest = json.loads(parent_manifest_path.read_text())
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        if (parent_manifest.get("status") != "pass" or parent_manifest.get("test_accessed") is not False
                or parent_manifest.get("checkpoint_sha256") != digest(parent_path)
                or parent.get("seed") != args.seed or parent.get("backbone") != method
                or parent.get("objective_version") != args.objective_version
                or parent.get("component_variant", "full") != "full"):
            raise ValueError("frozen_proposal_parent_invalid")
        if ordinary_training and parent_manifest.get("configuration", {}).get("training_rule") != "ordinary_multipositive":
            raise ValueError("frozen_proposal_parent_not_ordinary")
        for key, value in {
            "train_feature_sha256": train_manifest["feature_sha256"],
            "train_base_checkpoint_sha256": digest(args.base_checkpoint),
            "serving_base_checkpoint_sha256": digest(args.serving_base_checkpoint),
            "audio_sha256": digest(args.content_dir / args.audio_file),
            "genre_sha256": digest(args.content_dir / args.categorical_file),
        }.items():
            if parent_manifest["binding"].get(key) != value:
                raise ValueError(f"frozen_proposal_parent_binding:{key}")
        frozen_proposal_state = {k.removeprefix("proposal."): v for k, v in parent["model"].items() if k.startswith("proposal.")}
        model.proposal.load_state_dict(frozen_proposal_state, strict=True)
        model.proposal.requires_grad_(False)
        configuration.update(frozen_proposal_checkpoint_sha256=digest(parent_path),
                             frozen_proposal_manifest_sha256=digest(parent_manifest_path),
                             interaction_mode=args.interaction_mode,
                             conversion_parameters=sum(p.numel() for p in model.conversion.parameters()))
        configuration_sha256 = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if args.frozen_proposal_dir is not None and ordinary_training:
        conversion_optimizer = torch.optim.AdamW(
            model.conversion.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        optimizer = proposal_optimizer = None
    elif args.objective_version.startswith("support_safe_") and not ordinary_training:
        conversion_optimizer = None if model.conversion is None else torch.optim.AdamW(
            model.conversion.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        optimizer = None
        proposal_optimizer = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        proposal_optimizer = conversion_optimizer = None
    fit_mask = user_partition(train["users"])
    # Reuse the full run's fit rows and deterministic order, including rows
    # whose targets are unreachable in C0. Evaluation always retains all rows.
    fit_rows = np.flatnonzero(fit_mask & original_train_universe["positives"].any(axis=1))
    calibration_rows = np.flatnonzero(~fit_mask)
    records = []
    best = None
    updates = 0
    stop_reason = "max_epochs"
    completed_epoch = 0
    args.output.mkdir(parents=True, exist_ok=args.resume_if_present)
    state_path = args.output / "last_state.pt"
    resume_binding = {
        "configuration_sha256": configuration_sha256,
        "backbone": method,
        "train_feature_sha256": train_manifest["feature_sha256"],
        "train_base_checkpoint_sha256": digest(args.base_checkpoint),
        "train_base_manifest_sha256": digest(args.base_training_manifest),
        "serving_base_checkpoint_sha256": digest(args.serving_base_checkpoint),
        "serving_base_manifest_sha256": digest(args.serving_base_training_manifest),
        "audio_sha256": digest(args.content_dir / args.audio_file),
        "categorical_sha256": digest(args.content_dir / args.categorical_file),
    }
    if args.component_variant != "full" or args.frozen_proposal_dir is not None or ordinary_training:
        resume_binding.update(component_variant=args.component_variant,
                              producer_sha256=digest(Path(__file__)),
                              train_candidate_identity=train_candidate_identity)
    if args.frozen_proposal_dir is not None:
        resume_binding["ranker_source_sha256"] = digest(Path(__file__).with_name("run_arrive_unified_content_conversion.py"))

    def save_resume_state(epoch: int) -> None:
        optimizer_state = (
            conversion_optimizer.state_dict()
            if conversion_optimizer is not None
            else optimizer.state_dict() if optimizer is not None else None
        )
        atomic_torch(
            state_path,
            {
                "schema": SCHEMA,
                "resume_binding": resume_binding,
                "target_epochs": args.epochs,
                "epoch": epoch,
                "updates": updates,
                "records": records,
                "best": best,
                "model": model.state_dict(),
                "optimizer": optimizer_state,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        )

    if args.resume_if_present and state_path.exists():
        resume = torch.load(state_path, map_location="cpu", weights_only=False)
        if (
            resume.get("schema") != SCHEMA
            or resume.get("resume_binding") != resume_binding
            or int(resume.get("target_epochs", -1)) != args.epochs
        ):
            raise ValueError("joint_arrive_resume_binding_mismatch")
        model.load_state_dict(resume["model"], strict=True)
        if conversion_optimizer is not None:
            conversion_optimizer.load_state_dict(resume["optimizer"])
        elif optimizer is not None:
            optimizer.load_state_dict(resume["optimizer"])
        elif resume["optimizer"] is not None:
            raise ValueError("rcce_only_unexpected_optimizer_state")
        completed_epoch = int(resume["epoch"])
        updates = int(resume["updates"])
        records = list(resume["records"])
        best = resume["best"]
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"])
        if torch.cuda.is_available() and resume.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
        print(
            json.dumps(
                {"event": "joint_arrive_resumed", "epoch": completed_epoch, "updates": updates}
            ),
            flush=True,
        )
    else:
        save_resume_state(0)

    # A run can finish optimization and fail only while serializing an empty
    # dataset bucket.  Preserve the already reached stopping point on resume;
    # do not spend an extra epoch merely to regenerate the final artifacts.
    if completed_epoch and early_stop_reached(
        records,
        patience=args.patience,
        min_epochs=args.min_epochs,
        min_delta=args.min_delta,
    ):
        stop_reason = "selection_patience"
        completed_epoch = args.epochs
        print(
            json.dumps(
                {
                    "event": "joint_arrive_recovered_after_selection_stop",
                    "completed_epochs": len(records),
                    "best_epoch": best["epoch"],
                }
            ),
            flush=True,
        )

    for epoch in range(completed_epoch + 1, args.epochs + 1):
        order = deterministic_shuffle(fit_rows, args.seed + epoch, "arrive-joint-fit")
        losses = []
        model.train()
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            batch = tensor_batch(
                rows, train_universe, train_states, train["gaps"], train_observed,
                train_anchors, train_anchor_mask, item_embeddings, audio, genre,
                landmark_ids, args.objective_version, args.device,
                cue_ranking_mode=args.cue_ranking_mode,
            )
            proposal, _, _, selected_proposal, selected_positive, residual, final = forward_batch(
                model, batch, objective_version=args.objective_version
            )
            targets = torch.as_tensor(
                np.asarray(train["target_counts"][rows], dtype=np.int64), device=args.device
            )
            l_support = (
                final.sum() * 0.0 if ordinary_training and args.frozen_proposal_dir is not None else
                ordinary_multipositive_loss(proposal, batch[3], batch[2]) if ordinary_training else
                final.sum() * 0.0 if args.component_variant == "rgsc_only" else
                support_boundary_loss(proposal, batch[3], batch[2], targets, args.margin)
                if args.objective_version.startswith("support_safe_")
                else support_loss(proposal, batch[3], batch[2], targets)
            )
            objective_scores = training_objective_scores(
                selected_proposal, residual, args.gradient_mode
            )
            l_topk = (
                ordinary_multipositive_loss(objective_scores, selected_positive,
                                            torch.ones_like(selected_positive)) if ordinary_training else
                conversion_loss(objective_scores, selected_positive, targets, args.margin)
                if selected_positive.any()
                else final.sum() * 0.0
            )
            l_align = (
                alignment_loss(selected_proposal, final, args.temperature)
                if args.gradient_mode == "coupled" and args.objective_version == "scalar_v1"
                else final.sum() * 0.0
            )
            loss = l_support + args.lambda_topk * l_topk + args.mu_align * l_align
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite_joint_arrive_loss")
            gradient_audit = None
            if args.component_variant == "rgsc_only" or args.frozen_proposal_dir is not None:
                conversion_optimizer.zero_grad(set_to_none=True)
                (args.lambda_topk * l_topk).backward()
                torch.nn.utils.clip_grad_norm_(model.conversion.parameters(), 5.0, error_if_nonfinite=True)
                conversion_optimizer.step()
            elif args.objective_version.startswith("support_safe_") and not ordinary_training:
                proposal_parameters = list(model.proposal.parameters())
                conversion_parameters = [] if model.conversion is None else list(model.conversion.parameters())
                support_gradients = torch.autograd.grad(
                    l_support, proposal_parameters, retain_graph=True, allow_unused=True
                )
                conversion_proposal_gradients = torch.autograd.grad(
                    l_topk, proposal_parameters, retain_graph=True, allow_unused=True
                )
                conversion_head_gradients = torch.autograd.grad(
                    l_topk, conversion_parameters, allow_unused=True
                ) if conversion_parameters else []
                projected, gradient_audit = project_conflicting_gradient(
                    list(support_gradients), list(conversion_proposal_gradients)
                )
                if conversion_optimizer is not None:
                    conversion_optimizer.zero_grad(set_to_none=True)
                if not gradient_audit["proposal_frozen"]:
                    for parameter, support_gradient, conversion_gradient in zip(
                        proposal_parameters, support_gradients, projected
                    ):
                        parameter.grad = (
                            torch.zeros_like(parameter)
                            if support_gradient is None
                            else support_gradient.detach().clone()
                        ) + args.lambda_topk * conversion_gradient.detach()
                    torch.nn.utils.clip_grad_norm_(
                        proposal_parameters, 5.0, error_if_nonfinite=True
                    )
                    step_audit = support_safe_proposal_step(
                        model,
                        proposal_parameters,
                        batch,
                        targets,
                        args.margin,
                        args.learning_rate,
                        float(l_support.detach()),
                    )
                else:
                    step_audit = {
                        "accepted": False,
                        "backtracks": 0,
                        "step_scale": 0.0,
                        "support_loss_before": float(l_support.detach()),
                        "support_loss_after": float(l_support.detach()),
                    }
                for parameter, gradient in zip(conversion_parameters, conversion_head_gradients):
                    parameter.grad = (
                        torch.zeros_like(parameter) if gradient is None else args.lambda_topk * gradient.detach()
                    )
                torch.nn.utils.clip_grad_norm_(
                    conversion_parameters, 5.0, error_if_nonfinite=True
                )
                if conversion_optimizer is not None:
                    conversion_optimizer.step()
                gradient_audit.update(step_audit)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
            updates += 1
            losses.append((
                float(loss.detach()), float(l_support.detach()), float(l_topk.detach()),
                float(l_align.detach()),
                float(gradient_audit["gradient_dot"]) if gradient_audit else 0.0,
                float(gradient_audit["post_projection_dot"]) if gradient_audit else 0.0,
                float(bool(gradient_audit["conflict"])) if gradient_audit else 0.0,
                float(bool(gradient_audit["proposal_frozen"])) if gradient_audit else 0.0,
                float(gradient_audit["step_scale"]) if gradient_audit else 0.0,
                float(gradient_audit["backtracks"]) if gradient_audit else 0.0,
                float(not bool(gradient_audit["accepted"])) if gradient_audit else 0.0,
            ))
            if args.stop_after and updates >= args.stop_after:
                atomic_json(args.output / "SMOKE.json", {
                    "schema": SCHEMA, "status": "engineering_pass", "seed": args.seed,
                    "component_variant": args.component_variant,
                    "grounding_mode": args.grounding_mode,
                    "audio_file": args.audio_file,
                    "categorical_file": args.categorical_file,
                    "categorical_sparse": args.categorical_sparse,
                    "updates": updates, "validation_accessed": False, "test_accessed": False,
                })
                return

        pool_candidates, pool_scores, residual_scores = predict(
            model,
            {key: value[calibration_rows] for key, value in train_universe.items()},
            train_states[calibration_rows], train["gaps"][calibration_rows],
            train_observed[calibration_rows], train_anchors[calibration_rows],
            train_anchor_mask[calibration_rows], item_embeddings, audio, genre,
            landmark_ids, args.device, args.eval_batch_size, 1.0, args.objective_version,
            args.cue_ranking_mode,
        )
        targets = supported_target_values(
            train_universe["candidates"][calibration_rows],
            train_universe["positives"][calibration_rows],
            train["target_counts"][calibration_rows],
        )
        epoch_best = None
        for beta in ((None,) if args.component_variant == "rcce_only" else args.beta_grid):
            prediction = stable_topk(pool_scores if beta is None else pool_scores + float(beta) * (residual_scores - pool_scores), pool_candidates)
            recall, ndcg = row_metrics(prediction, targets, train["target_counts"][calibration_rows])
            score = selection_score(
                recall, ndcg, train["users"][calibration_rows], train["bucket_ids"][calibration_rows],
                np.ones(len(calibration_rows), dtype=bool),
            )
            candidate = {"epoch": epoch, "selection_score": float(score)}
            if beta is not None:
                candidate["beta"] = float(beta)
            if epoch_best is None or (score, -float(beta or 0.0)) > (epoch_best["selection_score"], -epoch_best.get("beta", 0.0)):
                epoch_best = candidate
        mean = np.asarray(losses).mean(axis=0)
        records.append({
            **epoch_best,
            "mean_loss": float(mean[0]), "mean_support_loss": float(mean[1]),
            "mean_topk_loss": float(mean[2]), "mean_align_loss": float(mean[3]),
            "mean_gradient_dot": float(mean[4]), "mean_post_projection_dot": float(mean[5]),
            "gradient_conflict_rate": float(mean[6]), "proposal_frozen_rate": float(mean[7]),
            "mean_proposal_step_scale": float(mean[8]), "mean_backtracks": float(mean[9]),
            "proposal_reject_rate": float(mean[10]),
        })
        if best is None or (epoch_best["selection_score"], -epoch_best.get("beta", 0.0), -epoch) > (best["selection_score"], -best.get("beta", 0.0), -best["epoch"]):
            best = {**epoch_best, "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
        save_resume_state(epoch)
        if frozen_proposal_state is not None:
            assert all(torch.equal(value.detach().cpu(), frozen_proposal_state[key]) for key, value in model.proposal.state_dict().items()), "frozen_proposal_changed"
        print(json.dumps({"event": "joint_arrive_epoch", "seed": args.seed, **records[-1]}), flush=True)
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            print(
                json.dumps({"event": "joint_arrive_interrupted_for_resume_smoke", "epoch": epoch}),
                flush=True,
            )
            return
        # Convergence policy mirrors the repaired public baselines exactly: the
        # shared early_stop_reached helper from run_lfm_timeconcat_control,
        # applied to the same per-epoch calibration selection scores recorded
        # above.  Checkpoint selection keeps the exact best epoch regardless of
        # when the patience stop fires.
        if early_stop_reached(
            records,
            patience=args.patience,
            min_epochs=args.min_epochs,
            min_delta=args.min_delta,
        ):
            stop_reason = "selection_patience"
            print(json.dumps({
                "event": "joint_arrive_early_stop", "seed": args.seed,
                "completed_epochs": len(records), "best_epoch": best["epoch"],
            }), flush=True)
            break

    state = best.pop("model")
    model.load_state_dict(state, strict=True)
    checkpoint = args.output / "joint_arrive_checkpoint.pt"
    checkpoint_payload = {
        "schema": SCHEMA, "seed": args.seed, "backbone": method, "selection": best,
        "proposal_width": int(probe[4].shape[-1]), "behavior_width": behavior_width,
        "hidden_dim": args.hidden_dim, "model": model.state_dict(),
        "objective_version": args.objective_version,
        "component_variant": args.component_variant,
        "interaction_mode": args.interaction_mode,
        "grounding_mode": args.grounding_mode,
        "cue_ranking_mode": args.cue_ranking_mode,
        "categorical_sparse": args.categorical_sparse,
        "gradient_mode": args.gradient_mode,
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
    }
    atomic_torch(checkpoint, checkpoint_payload)
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fresh = JointARRIVE(reloaded["proposal_width"], reloaded["behavior_width"], reloaded["hidden_dim"], reloaded.get("component_variant", "full"), reloaded.get("interaction_mode", "setwise")).to(args.device)
    fresh.load_state_dict(reloaded["model"], strict=True)
    model = fresh

    # Validation is opened exactly once, after epoch/beta selection and a
    # strict checkpoint reload have frozen the joint model.
    validation, validation_manifest = load_surface(args, "validation")
    validate_feature_backbone_binding(
        validation_manifest,
        args.serving_base_checkpoint,
        args.serving_base_training_manifest,
        method,
        "validation",
    )
    validation_candidate_identity = {"kind": "rcce_four_source_universe"}
    if args.component_variant == "rgsc_only":
        validation_universe, validation_states, validation_items, _, validation_candidate_identity = exact_backbone_universe(
            serving_args, validation, validation_manifest, "validation"
        )
    else:
        validation_universe = joint_universe(validation)
        validation_states, validation_items, _ = context_states(
            serving_args, data, validation["row_ids"]
        )
    if item_array.shape != validation_items.shape:
        raise ValueError("joint_item_embedding_catalog_shape_drift")
    validation_item_embeddings = torch.as_tensor(validation_items, device=args.device)
    validation_observed, validation_anchors, validation_anchor_mask = context_items(
        data, validation["row_ids"]
    )
    if args.grounding_mode == "shuffled":
        validation_observed = deranged_observed(
            validation_observed, args.seed + 917, "arrive-joint-shuffled-grounding-val"
        )
    pool_candidates, proposal_scores, final_scores = predict(
        model, validation_universe, validation_states, validation["gaps"], validation_observed,
        validation_anchors, validation_anchor_mask, validation_item_embeddings, audio, genre,
        landmark_ids, args.device, args.eval_batch_size, float(best.get("beta", 0.0)), args.objective_version,
        args.cue_ranking_mode,
    )
    backbone_top10 = stable_topk(validation["base_scores"], validation["candidates"])
    rcce_top10 = stable_topk(proposal_scores, pool_candidates)
    method_top10 = stable_topk(final_scores, pool_candidates)
    targets = supported_target_values(
        validation_universe["candidates"], validation_universe["positives"], validation["target_counts"]
    )
    metrics, base_recall, base_ndcg, method_recall, method_ndcg = metric_table(
        method_top10, backbone_top10, targets, validation, 72000 + args.seed
    )
    rcce_recall, rcce_ndcg = row_metrics(rcce_top10, targets, validation["target_counts"])
    pool_recall, _ = row_metrics(pool_candidates, targets, validation["target_counts"])
    predictions_path = args.output / "validation_predictions.npz"
    atomic_npz(
        predictions_path,
        row_ids=validation["row_ids"], users=validation["users"], bucket_ids=validation["bucket_ids"],
        targets=targets, target_counts=validation["target_counts"], backbone_top10=backbone_top10,
        rcce_top10=rcce_top10, arrive_top10=method_top10, backbone_recall=base_recall,
        backbone_ndcg=base_ndcg, rcce_recall=rcce_recall, rcce_ndcg=rcce_ndcg,
        pool_recall=pool_recall,
        candidate_top200=pool_candidates,
        arrive_recall=method_recall, arrive_ndcg=method_ndcg,
    )
    result = {
        "schema": SCHEMA, "status": "pass", "seed": args.seed, "backbone": method,
        "method": "joint RCCE proposal support plus RGSC set conversion in one optimization loop",
        "grounding_mode": args.grounding_mode,
        "gradient_mode": args.gradient_mode,
        "objective_version": args.objective_version,
        "component_variant": args.component_variant,
        "candidate_top200_sha256": array_digest(pool_candidates),
        "row_ids_sha256": array_digest(validation["row_ids"]),
        "candidate_identity": {"train": train_candidate_identity, "validation": validation_candidate_identity},
        "fit_row_ids_sha256": array_digest(train["row_ids"][fit_rows]),
        "selection_row_ids_sha256": array_digest(train["row_ids"][calibration_rows]),
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "selection": best,
        "training": {
            "epochs": args.epochs,
            "completed_epochs": len(records),
            "stop_reason": stop_reason,
            "updates": updates,
            "records": records,
        },
        "metrics": metrics,
        "ablation_rows": {
            "backbone": {"overall_recall_at_10": metrics["overall"]["base_recall_at_10"], "overall_ndcg_at_10": metrics["overall"]["base_ndcg_at_10"]},
            "backbone_plus_rcce": {
                "overall_recall_at_10": user_macro(rcce_recall, validation["users"], np.ones(len(validation["users"]), dtype=bool)),
                "overall_ndcg_at_10": user_macro(rcce_ndcg, validation["users"], np.ones(len(validation["users"]), dtype=bool)),
                "overall_recall_at_200": user_macro(pool_recall, validation["users"], np.ones(len(validation["users"]), dtype=bool)),
            },
            "backbone_plus_rcce_plus_rgsc": {"overall_recall_at_10": metrics["overall"]["method_recall_at_10"], "overall_ndcg_at_10": metrics["overall"]["method_ndcg_at_10"]},
        },
        "benefit_harm": {
            "rescued_zero_to_hit_rows": int(((method_recall > 0) & (base_recall == 0)).sum()),
            "displaced_hit_to_zero_rows": int(((method_recall == 0) & (base_recall > 0)).sum()),
        },
        "binding": {
            "train_feature_sha256": train_manifest["feature_sha256"],
            "validation_feature_sha256": validation_manifest["feature_sha256"],
            "train_base_checkpoint_sha256": digest(args.base_checkpoint),
            "train_base_manifest_sha256": digest(args.base_training_manifest),
            "serving_base_checkpoint_sha256": digest(args.serving_base_checkpoint),
            "serving_base_manifest_sha256": digest(args.serving_base_training_manifest),
            "architecture_contract": resolved_architecture_contract(base_manifest, method),
            "audio_sha256": digest(args.content_dir / args.audio_file),
            "genre_sha256": digest(args.content_dir / args.categorical_file),
            "producer_sha256": digest(Path(__file__)),
            "validation_accessed_after_freeze": True,
            "test_accessed": False,
        },
        "checkpoint_sha256": digest(checkpoint), "predictions_sha256": digest(predictions_path),
        "checkpoint_strict_reload": True, "seconds": time.monotonic() - started,
    }
    if ordinary_training:
        result["method"] = "ordinary_joint_two_head"
    if args.component_variant != "full":
        result["method"] = args.component_variant
        result["ablation_rows"] = {
            "backbone": result["ablation_rows"]["backbone"],
            args.component_variant: {
                "overall_recall_at_10": metrics["overall"]["method_recall_at_10"],
                "overall_ndcg_at_10": metrics["overall"]["method_ndcg_at_10"],
                "overall_recall_at_200": user_macro(pool_recall, validation["users"], np.ones(len(validation["users"]), dtype=bool)),
            },
        }
    if args.frozen_proposal_dir is not None:
        result["method"] = f"fixed RCCE pool; refit {args.interaction_mode} ranking head"
        result["mechanism_attribution"] = {
            "frozen_proposal_checkpoint_sha256": configuration["frozen_proposal_checkpoint_sha256"],
            "interaction_mode": args.interaction_mode,
            "cue_ranking_mode": args.cue_ranking_mode,
            "proposal_updated": False,
            "conversion_parameters": configuration["conversion_parameters"],
            "evidence_role": "supplementary_validation_not_independent_confirmation",
        }
    atomic_json(args.output / "VALIDATION_RESULT.json", result)
    atomic_json(args.output / "training_manifest.json", {
        "schema": SCHEMA, "status": "pass", "seed": args.seed, "backbone": method,
        "objective_version": args.objective_version,
        "component_variant": args.component_variant,
        "candidate_identity": result["candidate_identity"],
        "gradient_mode": args.gradient_mode,
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "completed_epochs": len(records),
        "stop_reason": stop_reason,
        "completed_updates": updates,
        "selection": best, "checkpoint_sha256": result["checkpoint_sha256"],
        "predictions_sha256": result["predictions_sha256"], "binding": result["binding"],
        "validation_accessed_once_after_freeze": True, "test_accessed": False,
    })
    print(json.dumps({"event": "joint_arrive_complete", "seed": args.seed, "backbone": method, "overall": metrics["overall"]}), flush=True)


if __name__ == "__main__":
    main()
