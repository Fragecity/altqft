from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn


class DeepSetPeriodPredictor(nn.Module):
    def __init__(self, nqubit: int, num_periods: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if num_periods < 1:
            raise ValueError("num_periods must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.num_periods = num_periods
        self.bit_width = compact_label_bit_width(num_periods)
        feature_dim = 16 * nqubit
        self.phi = nn.Sequential(
            nn.Linear(nqubit, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_periods),
        )

    def forward(self, bit_matrices: Tensor) -> Tensor:
        features_input = (
            bit_matrices
            if bit_matrices.dtype == torch.float32
            else bit_matrices.to(dtype=torch.float32)
        )
        features = self.phi(features_input)
        pooled = features.mean(dim=1)
        return cast(Tensor, self.head(pooled))

    def predict_topk_periods(
        self,
        bit_matrices: Tensor,
        candidate_periods: Sequence[int],
        k: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if len(candidate_periods) != self.num_periods:
            raise ValueError("candidate_periods length does not match model output space")
        logits = self(bit_matrices)
        return decode_topk_periods_from_class_logits(logits, candidate_periods, k)


def compact_label_bit_width(num_periods: int) -> int:
    if num_periods < 1:
        raise ValueError("num_periods must be positive")
    return max(1, (num_periods - 1).bit_length())


def class_indices_to_bits(indices: Tensor, bit_width: int) -> Tensor:
    if bit_width < 1:
        raise ValueError("bit_width must be positive")
    shifts = torch.arange(bit_width - 1, -1, -1, device=indices.device, dtype=torch.long)
    return ((indices.to(torch.long).unsqueeze(-1) >> shifts) & 1).to(torch.long)


def period_bit_loss(bit_logits: Tensor, labels: Tensor) -> Tensor:
    target_bits = class_indices_to_bits(labels, bit_logits.shape[1]).reshape(-1)
    flattened_logits = bit_logits.reshape(-1, 2)
    return cast(Tensor, nn.functional.cross_entropy(flattened_logits, target_bits))


def period_class_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    label_smoothing: float = 0.0,
) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("class logits must have shape (batch, num_classes)")
    return cast(
        Tensor,
        nn.functional.cross_entropy(logits, labels, label_smoothing=label_smoothing),
    )


def decode_topk_class_indices(
    bit_logits: Tensor,
    k: int,
    *,
    num_classes: int,
) -> tuple[Tensor, Tensor]:
    if k < 1:
        raise ValueError("k must be positive")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if bit_logits.ndim != 3 or bit_logits.shape[2] != 2:
        raise ValueError("bit_logits must have shape (batch, bit_width, 2)")

    batch_size, bit_width, _ = bit_logits.shape
    log_probs = bit_logits.log_softmax(dim=-1)
    beam_width = min(k, num_classes)
    bit_values = torch.arange(2, device=bit_logits.device, dtype=torch.long).view(
        1,
        1,
        2,
    )
    beam_indices = torch.zeros((batch_size, 1), dtype=torch.long, device=bit_logits.device)
    beam_scores = torch.zeros(
        (batch_size, 1),
        dtype=bit_logits.dtype,
        device=bit_logits.device,
    )

    for bit_index in range(bit_width):
        expanded_indices = (beam_indices.unsqueeze(-1) << 1) | bit_values
        expanded_scores = beam_scores.unsqueeze(-1) + log_probs[
            :,
            bit_index,
            :,
        ].unsqueeze(1)
        remaining_bits = bit_width - bit_index - 1
        valid_prefixes = (expanded_indices << remaining_bits) < num_classes
        expanded_scores = expanded_scores.masked_fill(~valid_prefixes, float("-inf"))

        flat_indices = expanded_indices.reshape(batch_size, -1)
        flat_scores = expanded_scores.reshape(batch_size, -1)
        current_width = min(beam_width, flat_scores.shape[1])
        top_scores, top_positions = flat_scores.topk(current_width, dim=1)
        beam_indices = flat_indices.gather(1, top_positions)
        beam_scores = top_scores

    return beam_indices, beam_scores


def decode_topk_periods(
    bit_logits: Tensor,
    candidate_periods: Sequence[int],
    k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if not candidate_periods:
        raise ValueError("candidate_periods must not be empty")

    top_indices, top_scores = decode_topk_class_indices(
        bit_logits,
        k,
        num_classes=len(candidate_periods),
    )
    candidate_tensor = torch.tensor(
        candidate_periods,
        dtype=torch.long,
        device=bit_logits.device,
    )
    top_periods = candidate_tensor[top_indices]
    top_bits = class_indices_to_bits(top_indices, bit_logits.shape[1])
    return top_periods, top_bits, top_scores


def decode_topk_periods_from_class_logits(
    logits: Tensor,
    candidate_periods: Sequence[int],
    k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if logits.ndim != 2:
        raise ValueError("class logits must have shape (batch, num_classes)")
    if not candidate_periods:
        raise ValueError("candidate_periods must not be empty")
    if logits.shape[1] != len(candidate_periods):
        raise ValueError("candidate_periods length does not match class logits width")

    beam_width = min(k, len(candidate_periods))
    top_scores, top_indices = logits.log_softmax(dim=1).topk(beam_width, dim=1)
    candidate_tensor = torch.tensor(
        candidate_periods,
        dtype=torch.long,
        device=logits.device,
    )
    top_periods = candidate_tensor[top_indices]
    top_bits = class_indices_to_bits(
        top_indices,
        compact_label_bit_width(len(candidate_periods)),
    )
    return top_periods, top_bits, top_scores
