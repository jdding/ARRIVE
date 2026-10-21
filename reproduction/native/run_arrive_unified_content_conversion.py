#!/usr/bin/env python3
"""Unified RCCE candidate conversion with return-grounded multimodal evidence.

The runner replaces the old sequential STC -> LightGBM MIG heads with one
set-aware neural scorer trained against the multi-positive Top-10 objective.
RCCE candidates and the seed-owned SASRec encoder remain frozen inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import nn

from lfm_timeconcat_control import LFMTimeConcatData
from run_music4all_artist_component_b import load_artist_features
from run_music4all_candidate_support_a1 import (
    atomic_json,
    atomic_npz,
    atomic_torch,
    base_method,
    digest,
    load_base_data,
    recent_anchors,
    set_seed,
    stable_topk,
    user_partition,
)
from run_music4all_component_c_oof import context_states
from run_music4all_component_c2_conversion_oof import (
    conversion_loss,
    deterministic_shuffle,
    paired_bootstrap,
    row_metrics,
    selection_score,
    token_features,
    user_disjoint_derangement,
    user_macro,
)
from run_music4all_component_c2_validation import frozen_predecessor, supported_target_values


SCHEMA = "arrive.unified_content_conversion.v1"
SCOPES = ("overall", "30-90d", "90-180d", "180-365d", ">365d")


def context_items(data: LFMTimeConcatData, row_ids: np.ndarray, width: int = 5):
    observed = np.empty(len(row_ids), dtype=np.int64)
    anchors = np.full((len(row_ids), width), 0, dtype=np.int64)
    mask = np.zeros((len(row_ids), width), dtype=np.float32)
    for at, row_id in enumerate(row_ids):
        session = data.sessions[int(row_id)]
        observed[at] = int(data.events[int(session[2])])
        values = recent_anchors(data.events, session, width)
        anchors[at, : len(values)] = values
        mask[at, : len(values)] = 1.0
    return observed, anchors, mask


def _audio_terms(
    candidates: torch.Tensor,
    observed: torch.Tensor,
    anchors: torch.Tensor,
    anchor_mask: torch.Tensor,
    audio: torch.Tensor,
) -> torch.Tensor:
    candidate_audio = audio[candidates]
    observed_audio = audio[observed]
    weights = anchor_mask.unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    history_audio = (audio[anchors] * weights).sum(dim=1) / denominator
    return torch.stack(
        (
            (candidate_audio * observed_audio.unsqueeze(1)).sum(dim=-1),
            (candidate_audio * history_audio.unsqueeze(1)).sum(dim=-1),
        ),
        dim=-1,
    )


def _dense_categorical_terms(
    candidates: torch.Tensor,
    observed: torch.Tensor,
    anchors: torch.Tensor,
    anchor_mask: torch.Tensor,
    categorical: torch.Tensor,
) -> torch.Tensor:
    candidate_genre = categorical[candidates]
    observed_genre = categorical[observed]
    weights = anchor_mask.unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    history_genre = (categorical[anchors] * weights).sum(dim=1) / denominator
    return torch.stack(
        (
            (candidate_genre * observed_genre.unsqueeze(1)).sum(dim=-1),
            (candidate_genre * history_genre.unsqueeze(1)).sum(dim=-1),
        ),
        dim=-1,
    )


class SparseCategorical:
    """crow/col/value triple of an L2-normalized multi-hot matrix.

    A minimal sparse representation that avoids torch's beta sparse CSR
    ops entirely: row gathers are plain vectorized indexing over the
    flattened storage, so the same code runs on CPU and CUDA on any
    torch build. Rows are indexed by model_id; reserved rows 0/1/2 and
    items without content hold no entries (zero semantics).
    """

    __slots__ = ("crow", "cols", "values", "shape")

    def __init__(self, crow: torch.Tensor, cols: torch.Tensor,
                 values: torch.Tensor, shape: tuple[int, int]) -> None:
        if crow.dim() != 1 or cols.dim() != 1 or values.dim() != 1:
            raise ValueError("sparse_categorical_dims_invalid")
        if crow.numel() != shape[0] + 1:
            raise ValueError("sparse_categorical_crow_length_mismatch")
        if cols.numel() != values.numel():
            raise ValueError("sparse_categorical_col_value_length_mismatch")
        if int(crow[0]) != 0 or int(crow[-1]) != cols.numel():
            raise ValueError("sparse_categorical_crow_endpoints_invalid")
        if not bool((crow[1:] >= crow[:-1]).all()):
            raise ValueError("sparse_categorical_crow_not_monotone")
        if cols.numel() and (int(cols.min()) < 0 or int(cols.max()) >= shape[1]):
            raise ValueError("sparse_categorical_col_index_out_of_range")
        if not (crow.device == cols.device == values.device):
            raise ValueError("sparse_categorical_device_mismatch")
        self.crow = crow
        self.cols = cols
        self.values = values
        self.shape = shape


def _gather_rows(cat: SparseCategorical, rows: torch.Tensor):
    """Gather the (col, value, local_row) triplets of the given rows."""
    rows = rows.long().reshape(-1)
    starts = cat.crow[rows]
    lengths = cat.crow[rows + 1] - starts
    total = int(lengths.sum())
    device = cat.cols.device
    if total == 0:
        empty = cat.cols.new_empty(0, dtype=torch.long)
        return empty, cat.values.new_empty(0), empty
    within = (
        torch.arange(total, device=device)
        - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        + torch.repeat_interleave(starts, lengths)
    )
    cols = cat.cols[within]
    vals = cat.values[within]
    row_of = torch.repeat_interleave(
        torch.arange(rows.numel(), device=device), lengths
    )
    return cols, vals, row_of


def _sparse_categorical_terms(
    candidates: torch.Tensor,
    observed: torch.Tensor,
    anchors: torch.Tensor,
    anchor_mask: torch.Tensor,
    categorical: SparseCategorical,
) -> torch.Tensor:
    """The two categorical inner-product terms with the matrix left sparse.

    Per batch, every touched column is remapped onto a compact local index
    and only the observed/history vectors are densified over that union.
    Restricting both operands of an inner product to the union of their
    supports preserves the values exactly: absent columns contribute zero
    in the dense computation too. Rows with no support (items without
    content) produce zero terms, matching the dense zero-row semantics.
    """
    batch, width = candidates.shape
    device = categorical.values.device
    cand_cols, cand_vals, cand_row_of = _gather_rows(
        categorical, candidates.reshape(-1)
    )
    obs_cols, obs_vals, obs_row_of = _gather_rows(categorical, observed)
    anch_cols, anch_vals, anch_row_of = _gather_rows(categorical, anchors.reshape(-1))
    union = torch.unique(torch.cat((cand_cols, obs_cols, anch_cols)))
    local = torch.searchsorted(
        union, torch.cat((cand_cols, obs_cols, anch_cols))
    )
    local_cand = local[: cand_cols.numel()]
    local_obs = local[cand_cols.numel(): cand_cols.numel() + obs_cols.numel()]
    local_anch = local[cand_cols.numel() + obs_cols.numel():]

    obs_block = torch.zeros(
        (batch, union.numel()), device=device, dtype=categorical.values.dtype
    )
    obs_block.index_put_((obs_row_of, local_obs), obs_vals, accumulate=True)
    anch_block = torch.zeros(
        (batch, union.numel()), device=device, dtype=categorical.values.dtype
    )
    # anch_row_of indexes the flattened (batch, anchor) slots; the block row
    # is the batch index slot // A, and the weight belongs to the slot.
    anch_batch = anch_row_of // anchor_mask.shape[1]
    anch_block.index_put_(
        (anch_batch, local_anch),
        anch_vals * anchor_mask.reshape(-1)[anch_row_of],
        accumulate=True,
    )
    denominator = anchor_mask.sum(dim=1).clamp_min(1.0)
    hist_block = anch_block / denominator.unsqueeze(-1)

    b_of = (cand_row_of // width).long()
    t_obs = torch.zeros(
        batch * width, device=device, dtype=categorical.values.dtype
    )
    t_obs.index_add_(0, cand_row_of, cand_vals * obs_block[b_of, local_cand])
    t_hist = torch.zeros(
        batch * width, device=device, dtype=categorical.values.dtype
    )
    t_hist.index_add_(0, cand_row_of, cand_vals * hist_block[b_of, local_cand])
    return torch.stack(
        (t_obs.reshape(batch, width), t_hist.reshape(batch, width)), dim=-1
    )


def multimodal_features(
    candidates: torch.Tensor,
    observed: torch.Tensor,
    anchors: torch.Tensor,
    anchor_mask: torch.Tensor,
    audio: torch.Tensor,
    genre: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            _audio_terms(candidates, observed, anchors, anchor_mask, audio),
            _dense_categorical_terms(candidates, observed, anchors, anchor_mask, genre),
        ),
        dim=-1,
    )


def multimodal_features_dispatch(
    candidates: torch.Tensor,
    observed: torch.Tensor,
    anchors: torch.Tensor,
    anchor_mask: torch.Tensor,
    audio: torch.Tensor,
    categorical: torch.Tensor,
) -> torch.Tensor:
    """Grounding terms for a dense or sparse-csr categorical channel.

    Both paths return the same (batch, width, 4) layout: audio observed /
    audio history / categorical observed / categorical history. The dense
    branch is the original Onion computation; the sparse-csr branch exists
    because densifying the Yambda artist matrix (vocab x 177,723) is
    infeasible. Values agree with the dense path within one float32 ULP
    (reassociation only); verify with
    tools/verify_sparse_grounding_equivalence.py.
    """
    if isinstance(categorical, SparseCategorical):
        return torch.cat(
            (
                _audio_terms(candidates, observed, anchors, anchor_mask, audio),
                _sparse_categorical_terms(
                    candidates, observed, anchors, anchor_mask, categorical
                ),
            ),
            dim=-1,
        )
    if isinstance(categorical, torch.Tensor) and categorical.layout == torch.strided:
        return multimodal_features(
            candidates, observed, anchors, anchor_mask, audio, categorical
        )
    raise ValueError("categorical_layout_unsupported")


class UnifiedContentRanker(nn.Module):
    def __init__(self, behavior_width: int, hidden: int, use_content: bool, interaction_mode: str = "setwise") -> None:
        super().__init__()
        self.use_content = use_content
        width = behavior_width + (4 if use_content else 0)
        self.input = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.attention = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        self.feed_norm = nn.LayerNorm(hidden)
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        if interaction_mode not in ("setwise", "pointwise"):
            raise ValueError("interaction_mode_invalid")
        self.interaction_mode = interaction_mode
        if interaction_mode == "pointwise":
            # Keep common-layer initialization and the caller's RNG identical.
            # This token-wise MLP has only hidden fewer parameters than MHA.
            with torch.random.fork_rng(devices=[]):
                self.pointwise = nn.Sequential(
                    nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden)
                )
            self.attention = None

    def forward(self, behavior: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        features = torch.cat((behavior, content), dim=-1) if self.use_content else behavior
        hidden = self.input(features)
        if self.interaction_mode == "pointwise":
            attended = self.pointwise(hidden)
        else:
            attended, _ = self.attention(hidden, hidden, hidden, need_weights=False)
        hidden = self.norm(hidden + attended)
        hidden = self.feed_norm(hidden + self.feed_forward(hidden))
        return self.output(hidden).squeeze(-1)


def batch_inputs(
    rows: np.ndarray,
    source: np.ndarray,
    candidates: np.ndarray,
    scores: np.ndarray,
    states: np.ndarray,
    gaps: np.ndarray,
    observed: np.ndarray,
    anchors: np.ndarray,
    anchor_mask: np.ndarray,
    item_embeddings: torch.Tensor,
    audio: torch.Tensor,
    genre: torch.Tensor,
    device: str,
    use_content: bool,
):
    source_rows = source[rows]
    c = torch.as_tensor(candidates[rows], dtype=torch.long, device=device)
    s = torch.as_tensor(scores[rows], device=device)
    h = torch.as_tensor(states[rows], device=device)
    g = torch.as_tensor(gaps[rows], device=device)
    behavior = token_features(c, s, h, g, item_embeddings)
    if use_content:
        o = torch.as_tensor(observed[source_rows], dtype=torch.long, device=device)
        a = torch.as_tensor(anchors[source_rows], dtype=torch.long, device=device)
        m = torch.as_tensor(anchor_mask[source_rows], device=device)
        content = multimodal_features(c, o, a, m, audio, genre)
    else:
        content = behavior.new_empty((*behavior.shape[:2], 0))
    return s, behavior, content


@torch.no_grad()
def predict_residual(
    model: nn.Module,
    candidates: np.ndarray,
    scores: np.ndarray,
    states: np.ndarray,
    gaps: np.ndarray,
    observed: np.ndarray,
    anchors: np.ndarray,
    anchor_mask: np.ndarray,
    source: np.ndarray,
    item_embeddings: torch.Tensor,
    audio: torch.Tensor,
    genre: torch.Tensor,
    device: str,
    batch_size: int,
    use_content: bool,
) -> np.ndarray:
    output = np.empty(scores.shape, dtype=np.float32)
    model.eval()
    for start in range(0, len(scores), batch_size):
        stop = min(start + batch_size, len(scores))
        rows = np.arange(start, stop, dtype=np.int64)
        _, behavior, content = batch_inputs(
            rows, source, candidates, scores, states, gaps, observed, anchors,
            anchor_mask, item_embeddings, audio, genre, device, use_content,
        )
        output[start:stop] = model(behavior, content).cpu().numpy()
    return output


def metric_table(predictions, base_top10, targets, raw, seed):
    base_recall, base_ndcg = row_metrics(base_top10, targets, raw["target_counts"])
    method_recall, method_ndcg = row_metrics(predictions, targets, raw["target_counts"])
    metrics = {}
    for scope_index, scope in enumerate(SCOPES):
        selected = np.ones(len(predictions), dtype=bool) if scope == "overall" else raw["bucket_ids"] == scope_index - 1
        metrics[scope] = {
            "rows": int(selected.sum()),
            "users": int(len(np.unique(raw["users"][selected]))),
            "base_recall_at_10": user_macro(base_recall, raw["users"], selected),
            "method_recall_at_10": user_macro(method_recall, raw["users"], selected),
            "recall_paired": paired_bootstrap(method_recall - base_recall, raw["users"], selected, seed + scope_index),
            "base_ndcg_at_10": user_macro(base_ndcg, raw["users"], selected),
            "method_ndcg_at_10": user_macro(method_ndcg, raw["users"], selected),
            "ndcg_paired": paired_bootstrap(method_ndcg - base_ndcg, raw["users"], selected, seed + 100 + scope_index),
        }
    return metrics, base_recall, base_ndcg, method_recall, method_ndcg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(2024, 2025, 2026))
    parser.add_argument("--variant", choices=("behavior_setwise", "multimodal_setwise"), required=True)
    parser.add_argument("--rcce-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-training-manifest", type=Path, required=True)
    parser.add_argument("--validation-base-checkpoint", type=Path)
    parser.add_argument("--validation-base-training-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/music4all_onion/prepared_v1"))
    parser.add_argument("--content-dir", type=Path, default=Path("data/music4all_onion/component_b2_attainability_v1"))
    parser.add_argument("--event-times", type=Path, default=Path("data/music4all_onion/baseline_features_v1/event_times.npy"))
    parser.add_argument("--ssd4rec-source", type=Path, default=Path("vendor/ssd4rec_official/ssd4rec.py"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=.25)
    parser.add_argument("--beta-grid", type=float, nargs="+", default=(0.0, .25, .5, 1.0, 2.0))
    parser.add_argument("--max-context", type=int, default=129)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if (args.validation_base_checkpoint is None) != (args.validation_base_training_manifest is None):
        raise ValueError("validation_base_binding_incomplete")
    validation_base_checkpoint = args.validation_base_checkpoint or args.base_checkpoint
    validation_base_manifest = args.validation_base_training_manifest or args.base_training_manifest
    if args.output.exists():
        raise FileExistsError("unified_content_output_exists")
    args.features_root = args.rcce_dir / "artist_features"
    args.training_features_root = args.features_root
    args.a1_root = Path("unused")
    args.artist_root = Path("unused")
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    started = time.monotonic()

    train, train_manifest = load_artist_features(
        args.features_root / "train_cv_features.npz",
        args.features_root / "train_cv_features_manifest.json", "train_cv",
    )
    validation_base_metadata = json.loads(validation_base_manifest.read_text())
    serving_method = base_method(validation_base_metadata)
    if int(validation_base_metadata["binding"]["configuration"].get("seed", -1)) != args.seed:
        raise ValueError("serving_base_seed_mismatch")
    if train_manifest.get("base_method") not in (None, serving_method):
        raise ValueError("train_candidate_serving_base_method_mismatch")
    train_candidates, train_scores, train_top10, train_positives, _, train_bindings = frozen_predecessor(args, train, "train_cv")
    train_targets = supported_target_values(train_candidates, train_positives, train["target_counts"])
    # Candidate scores are cross-fit on Train, but coordinate-valued RGSC
    # features must come from one frozen serving encoder on both Train and
    # Validation.  Otherwise independently trained encoders create unrelated
    # item/state coordinates and the deployed system needs two backbones.
    serving_args = argparse.Namespace(**vars(args))
    serving_args.base_checkpoint = validation_base_checkpoint
    serving_args.base_training_manifest = validation_base_manifest
    data = load_base_data(serving_args, validation_base_manifest)
    train_states, item_array, base_manifest = context_states(serving_args, data, train["row_ids"])
    observed, anchors, anchor_mask = context_items(data, train["row_ids"])
    item_embeddings = torch.as_tensor(item_array, device=args.device)
    audio = torch.as_tensor(np.asarray(np.load(args.content_dir / "musicnn_normalized.npy", mmap_mode="r")), device=args.device)
    genre = torch.as_tensor(sparse.load_npz(args.content_dir / "genre_normalized.npz").toarray().astype(np.float32), device=args.device)

    probe_rows = np.asarray([0], dtype=np.int64)
    identity = np.arange(len(train["row_ids"]), dtype=np.int64)
    _, probe_behavior, _ = batch_inputs(
        probe_rows, identity, train_candidates, train_scores, train_states, train["gaps"], observed,
        anchors, anchor_mask, item_embeddings, audio, genre, args.device,
        args.variant == "multimodal_setwise",
    )
    model = UnifiedContentRanker(
        probe_behavior.shape[-1], args.hidden_dim, args.variant == "multimodal_setwise"
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    fit_mask = user_partition(train["users"])
    fit_rows = np.flatnonzero(fit_mask & train_positives.any(axis=1))
    calibration_rows = np.flatnonzero(~fit_mask)
    best = None
    records = []
    updates = 0
    args.output.mkdir(parents=True)
    for epoch in range(1, args.epochs + 1):
        order = deterministic_shuffle(fit_rows, args.seed + epoch, "arrive-unified-fit")
        losses = []
        model.train()
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            s, behavior, content = batch_inputs(
                rows, identity, train_candidates, train_scores, train_states, train["gaps"], observed,
                anchors, anchor_mask, item_embeddings, audio, genre, args.device,
                args.variant == "multimodal_setwise",
            )
            residual = model(behavior, content)
            loss = conversion_loss(
                s + residual,
                torch.as_tensor(train_positives[rows], device=args.device),
                torch.as_tensor(train["target_counts"][rows], dtype=torch.long, device=args.device),
                args.margin,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite_unified_content_loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1
            if args.stop_after and updates >= args.stop_after:
                atomic_json(args.output / "SMOKE.json", {
                    "schema": SCHEMA, "status": "engineering_pass", "seed": args.seed,
                    "variant": args.variant, "updates": updates, "validation_accessed": False,
                    "test_accessed": False,
                })
                return
        residual = predict_residual(
            model, train_candidates[calibration_rows], train_scores[calibration_rows],
            train_states[calibration_rows], train["gaps"][calibration_rows], observed[calibration_rows],
            anchors[calibration_rows], anchor_mask[calibration_rows], np.arange(len(calibration_rows)),
            item_embeddings, audio, genre, args.device, args.eval_batch_size,
            args.variant == "multimodal_setwise",
        )
        epoch_best = None
        for beta in args.beta_grid:
            prediction = stable_topk(
                train_scores[calibration_rows] + float(beta) * residual,
                train_candidates[calibration_rows],
            )
            recall, ndcg = row_metrics(prediction, train_targets[calibration_rows], train["target_counts"][calibration_rows])
            score = selection_score(
                recall, ndcg, train["users"][calibration_rows], train["bucket_ids"][calibration_rows],
                np.ones(len(calibration_rows), dtype=bool),
            )
            candidate = {"epoch": epoch, "beta": float(beta), "selection_score": score}
            if epoch_best is None or (score, -float(beta)) > (epoch_best["selection_score"], -epoch_best["beta"]):
                epoch_best = candidate
        assert epoch_best is not None
        records.append({"epoch": epoch, "mean_loss": float(np.mean(losses)), **epoch_best})
        if best is None or (epoch_best["selection_score"], -epoch_best["beta"], -epoch) > (best["selection_score"], -best["beta"], -best["epoch"]):
            best = {**epoch_best, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
        print(json.dumps({"event": "unified_content_epoch", "seed": args.seed, "variant": args.variant, **records[-1]}), flush=True)

    assert best is not None
    state = best.pop("model")
    model.load_state_dict(state, strict=True)
    checkpoint = args.output / "unified_content_checkpoint.pt"
    atomic_torch(checkpoint, {
        "schema": SCHEMA, "seed": args.seed, "variant": args.variant,
        "selection": best, "behavior_width": int(probe_behavior.shape[-1]),
        "hidden_dim": args.hidden_dim, "model": model.state_dict(),
    })
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fresh = UnifiedContentRanker(saved["behavior_width"], saved["hidden_dim"], args.variant == "multimodal_setwise").to(args.device)
    fresh.load_state_dict(saved["model"], strict=True)
    model = fresh

    validation, validation_manifest = load_artist_features(
        args.features_root / "validation_features.npz",
        args.features_root / "validation_features_manifest.json", "validation",
    )
    validation_candidates, validation_scores, validation_top10, validation_positives, _, validation_bindings = frozen_predecessor(args, validation, "validation")
    validation_targets = supported_target_values(
        validation_candidates, validation_positives, validation["target_counts"]
    )
    if any(validation_bindings.get(key) != value for key, value in train_bindings.items()):
        raise ValueError("unified_predecessor_binding_drift")
    validation_data = data
    validation_states, validation_items, validation_base = context_states(
        serving_args, validation_data, validation["row_ids"]
    )
    if not np.array_equal(item_array, validation_items):
        raise ValueError("unified_serving_item_embedding_drift")
    validation_item_embeddings = torch.as_tensor(validation_items, device=args.device)
    validation_observed, validation_anchors, validation_anchor_mask = context_items(
        validation_data, validation["row_ids"]
    )
    validation_identity = np.arange(len(validation["row_ids"]), dtype=np.int64)
    residual = predict_residual(
        model, validation_candidates, validation_scores, validation_states, validation["gaps"],
        validation_observed, validation_anchors, validation_anchor_mask, validation_identity,
        validation_item_embeddings, audio, genre, args.device, args.eval_batch_size,
        args.variant == "multimodal_setwise",
    )
    predictions = stable_topk(validation_scores + float(best["beta"]) * residual, validation_candidates)
    metrics, base_r, base_n, method_r, method_n = metric_table(
        predictions, validation_top10, validation_targets, validation, 52000 + args.seed
    )

    shuffled_predictions = validation_top10.copy()
    shuffled_metrics = None
    if args.variant == "multimodal_setwise":
        shuffled_source = user_disjoint_derangement(
            validation_identity, validation["users"], args.seed + 60000, "arrive-unified-content-validation"
        )
        shuffled_residual = predict_residual(
            model, validation_candidates, validation_scores, validation_states, validation["gaps"],
            validation_observed, validation_anchors, validation_anchor_mask, shuffled_source,
            validation_item_embeddings, audio, genre, args.device, args.eval_batch_size, True,
        )
        shuffled_predictions = stable_topk(
            validation_scores + float(best["beta"]) * shuffled_residual, validation_candidates
        )
        shuffled_metrics, _, _, _, _ = metric_table(
            shuffled_predictions, validation_top10, validation_targets, validation, 62000 + args.seed
        )

    predictions_path = args.output / "validation_predictions.npz"
    atomic_npz(
        predictions_path, row_ids=validation["row_ids"], users=validation["users"],
        bucket_ids=validation["bucket_ids"], targets=validation_targets,
        target_counts=validation["target_counts"], rcce_top10=validation_top10,
        method_top10=predictions, shuffled_content_top10=shuffled_predictions,
        base_recall=base_r, base_ndcg=base_n, method_recall=method_r, method_ndcg=method_n,
    )
    result = {
        "schema": SCHEMA, "status": "pass", "seed": args.seed, "variant": args.variant,
        "method": "one set-aware neural scorer over frozen RCCE candidates with optional return-grounded audio/genre evidence",
        "selection": best, "training": {"epochs": args.epochs, "records": records, "updates": updates},
        "metrics": metrics, "shuffled_content_metrics": shuffled_metrics,
        "benefit_harm": {
            "recall_better_rows": int((method_r > base_r).sum()),
            "recall_worse_rows": int((method_r < base_r).sum()),
            "rescued_zero_to_hit_rows": int(((method_r > 0) & (base_r == 0)).sum()),
            "displaced_hit_to_zero_rows": int(((method_r == 0) & (base_r > 0)).sum()),
        },
        "binding": {
            "train_feature_sha256": train_manifest["feature_sha256"],
            "validation_feature_sha256": validation_manifest["feature_sha256"],
            **validation_bindings,
            "train_rcce_base_checkpoint_sha256": digest(args.base_checkpoint),
            "train_rcce_base_manifest_sha256": digest(args.base_training_manifest),
            "rgsc_serving_encoder_checkpoint_sha256": digest(validation_base_checkpoint),
            "rgsc_serving_encoder_manifest_sha256": digest(validation_base_manifest),
            "rgsc_serving_encoder_architecture": base_manifest["binding"]["architecture_contract"],
            "rgsc_serving_encoder_shared_across_surfaces": True,
            "base_method": serving_method,
            "audio_sha256": digest(args.content_dir / "musicnn_normalized.npy"),
            "genre_sha256": digest(args.content_dir / "genre_normalized.npz"),
            "producer_sha256": digest(Path(__file__)),
            "validation_accessed_after_freeze": True,
            "test_accessed": False,
        },
        "checkpoint_sha256": digest(checkpoint),
        "predictions_sha256": digest(predictions_path),
        "checkpoint_strict_reload": True,
        "seconds": time.monotonic() - started,
    }
    atomic_json(args.output / "VALIDATION_RESULT.json", result)
    print(json.dumps({"event": "unified_content_complete", "seed": args.seed, "variant": args.variant, "metrics": metrics, "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
