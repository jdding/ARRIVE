"""DeltaGate operators adapted to ARRIVE's observed-prefix continuation contract.

Gating-mechanism source: vendor/deltagate_corrected/src/deltagate.py
(``DeltaGateSASRec``, RecSys 2026 camera-ready 266). This is a task
adaptation, not a reproduction of its zero-observation reactivation
experiment; see the fidelity note in this module docstring.

Fidelity note (task adaptation boundary):

- Retained from DeltaGate: the output-layer dimensional gate — a two-layer
  MLP over ``[log1p(gap), personalized_state]`` with a sigmoid, mixing the
  personalized state with a learned global prior per dimension,
  ``h = g * h_personalized + (1 - g) * global_prior`` — and the SASRec-style
  causal transformer that produces the personalized state. The gap enters
  only the gate; item/position embeddings feed the encoder.
- Changed for the ARRIVE estimand: the input sequence is the observed
  prefix ``H^- ++ [x_0]`` (the return track is observed input, not a
  prediction target), the gate is conditioned on the same hiatus gap, and
  the head is the shared tied-item multi-positive set objective of the
  main-table contract rather than DeltaGate's next-item softmax. The gate
  is applied to the state at the observed return position
  (``initial_positions``), i.e. after the return track has been encoded.
- Not used: DeltaGate's original evaluation protocol (predict the first
  post-gap item from history alone); those numbers are not comparable and
  are not reused.
"""

from __future__ import annotations

import torch
from torch import nn

from arrive_main_table_models import MainTableModel
from lfm_timeconcat_control import PAD

UPSTREAM_FILE = "vendor/deltagate_corrected/src/deltagate.py"
UPSTREAM_MODEL = "DeltaGateSASRec"


class DeltaGateRepairedAdapter(MainTableModel):
    """SASRec encoder + DeltaGate gap-conditioned dimensional gating.

    Mirrors ``arrive_main_table_models.CausalTransformer`` for the encoder
    (right-padded sequences, learned positions, pre-norm transformer, GELU)
    and ``vendor/deltagate_corrected/src/deltagate.py`` for the gate and
    prior mix. The tied item-embedding head is inherited from
    ``MainTableModel``.
    """

    def __init__(
        self,
        vocab_size: int,
        noise_probabilities,
        *,
        embed_dim: int,
        max_sequence: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        **_,
    ) -> None:
        super().__init__(vocab_size, noise_probabilities, embed_dim)
        self.max_sequence = int(max_sequence)
        self.position_embedding = nn.Embedding(max_sequence, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # DeltaGate gate: [log1p(gap), h] -> per-dimension retention sigmoid.
        self.gate_mlp = nn.Sequential(
            nn.Linear(1 + embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.global_prior = nn.Parameter(torch.zeros(embed_dim))

    def encode_personalized(self, sequence: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
        hidden = self.dropout(self.item_vectors(sequence) + self.position_embedding(positions))
        hidden = hidden.masked_fill(sequence.eq(PAD).unsqueeze(-1), 0.0)
        causal = torch.triu(
            torch.full((sequence.shape[1], sequence.shape[1]), float("-inf"), device=sequence.device),
            diagonal=1,
        )
        return self.norm(self.encoder(hidden, mask=causal)).masked_fill(
            sequence.eq(PAD).unsqueeze(-1), 0.0
        )

    def compute_gate(self, state: torch.Tensor, gap_days: torch.Tensor) -> torch.Tensor:
        if gap_days.shape != (state.shape[0],):
            raise ValueError("gap_days must be one value per row")
        if bool(torch.isfinite(gap_days).all()) is False or bool((gap_days >= 0).all()) is False:
            raise ValueError("gap_days must be finite and non-negative")
        gate_input = torch.cat([torch.log1p(gap_days).unsqueeze(-1), state], dim=-1)
        return self.gate_mlp(gate_input)

    def encode_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        sequence = batch["sequence"]
        hidden = self.encode_personalized(sequence)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        positions = batch["initial_positions"]
        if bool((sequence[rows, positions] == PAD).any()):
            raise ValueError("initial position must land on an observed token")
        state = hidden[rows, positions]
        gate = self.compute_gate(state, batch["gap_days"])
        hidden = hidden.clone()
        hidden[rows, positions] = gate * state + (1.0 - gate) * self.global_prior
        return hidden

    def auxiliary_loss(self, batch: dict[str, torch.Tensor], hidden: torch.Tensor) -> torch.Tensor:
        del batch, hidden
        # Vendor fidelity: the global prior is a completely free parameter
        # (no L2 pull), so the fallback branch keeps DeltaGate's original
        # learning dynamics under the shared optimizer.
        return self.global_prior.sum() * 0.0


# --- warm-start support (revisions warmhead20260917 / warmfull20260917) ---

_BACKBONE_KEY_MAP = {"transformer.": "encoder.", "layer_norm.": "norm."}


def load_warm_backbone(model: DeltaGateRepairedAdapter, state_dict, mode: str) -> None:
    """Load a Music4AllSASRec checkpoint into the adapter and apply warm mode.

    ``mode='head_only'`` freezes the loaded backbone (embeddings, encoder,
    norm) and trains only the gate MLP and global prior; ``mode='full'``
    fine-tunes everything. In both modes the gate output bias is shifted by
    +2.5 so training starts near the personalized (gate ~ .92) solution —
    the regime where DeltaGate's function class contains SASRec.

    The baseline encoder runs over a 192-wide context+horizon window while
    this adapter encodes the 129-wide observed prefix; both are left-aligned
    absolute positions, so the first ``max_sequence`` rows of the backbone's
    position table are loaded directly (shape-checked, fail-closed).
    """

    if mode not in ("head_only", "full"):
        raise ValueError(f"invalid warm mode: {mode}")
    vocab, embed = model.item_embedding.weight.shape
    width = int(model.position_embedding.weight.shape[0])
    mapped = {}
    for key, value in state_dict.items():
        if key.startswith("gate_mlp.") or key == "global_prior":
            raise ValueError("backbone state must not contain gate parameters")
        target = key
        for source_prefix, adapter_prefix in _BACKBONE_KEY_MAP.items():
            if key.startswith(source_prefix):
                target = adapter_prefix + key[len(source_prefix):]
                break
        if key == "item_embedding.weight":
            if tuple(value.shape) != (vocab, embed):
                raise ValueError(f"backbone vocab mismatch: {tuple(value.shape)} vs {(vocab, embed)}")
        if key == "position_embedding.weight":
            if value.shape[0] < width or value.shape[1] != embed:
                raise ValueError(f"backbone position table too small: {tuple(value.shape)}")
            value = value[:width].clone()
        mapped[target] = value
    expected = {
        name for name, _ in model.named_parameters()
        if not name.startswith(("gate_mlp.", "global_prior"))
    }
    if set(mapped) != expected:
        missing = sorted(expected - set(mapped))
        extra = sorted(set(mapped) - expected)
        raise ValueError(f"backbone key mismatch: missing={missing[:4]} extra={extra[:4]}")
    result = model.load_state_dict(mapped, strict=False)
    bad_missing = [k for k in result.missing_keys if not k.startswith(("gate_mlp.", "global_prior"))]
    if result.unexpected_keys or bad_missing:
        raise ValueError(f"backbone load mismatch: unexpected={result.unexpected_keys} bad_missing={bad_missing}")
    if not getattr(model, "_warm_gate_initialized", False):
        with torch.no_grad():
            # Uniform gate at load: zero the output weights so gate = sigma(2.5)
            # exactly on every dimension — the loaded model's ranking is then
            # EXACTLY the backbone SASRec's (positive scalar times the state;
            # global_prior is zero-initialized). Gradients still reach the gate
            # through the (nonzero) hidden input.
            model.gate_mlp[2].weight.zero_()
            model.gate_mlp[2].bias.fill_(2.5)
            model._warm_gate_initialized = True
    if mode == "head_only":
        frozen = 0
        for name, parameter in model.named_parameters():
            if not name.startswith(("gate_mlp.", "global_prior")):
                parameter.requires_grad_(False)
                frozen += 1
        if frozen == 0:
            raise ValueError("head_only froze nothing")
