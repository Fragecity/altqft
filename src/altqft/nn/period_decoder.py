from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

TOKEN_BITS = 4
TOKEN_CLASSES = 1 << TOKEN_BITS
BOS_TOKEN = TOKEN_CLASSES
DEFAULT_BEAM_WIDTH = 4
DECODER_TYPE = "nibble"


class ResidualMLPBlock(nn.Module):
    def __init__(self, feature_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, inputs + self.net(inputs))


class DeepSetPeriodPredictor(nn.Module):
    def __init__(
        self,
        nqubit: int,
        num_periods: int | None = None,
        *,
        period_min: int = 2,
        period_max: int | None = None,
        dropout: float = 0.0,
        architecture: str = "weighted",
        beam_width: int = DEFAULT_BEAM_WIDTH,
    ) -> None:
        super().__init__()
        if nqubit < 1:
            raise ValueError("nqubit must be positive")
        if num_periods is not None and num_periods < 1:
            raise ValueError("num_periods must be positive")
        if period_min < 0:
            raise ValueError("period_min must be non-negative")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if architecture not in {"legacy", "weighted"}:
            raise ValueError("architecture must be 'legacy' or 'weighted'")
        if beam_width < 1:
            raise ValueError("beam_width must be positive")

        resolved_period_max = (
            period_min + num_periods - 1
            if period_max is None and num_periods is not None
            else (1 << nqubit) - 1
            if period_max is None
            else period_max
        )
        if resolved_period_max < period_min:
            raise ValueError("period_max must not be smaller than period_min")
        if resolved_period_max >= 1 << nqubit:
            raise ValueError("period_max must fit in nqubit bits")
        resolved_num_periods = resolved_period_max - period_min + 1
        if num_periods is not None and num_periods != resolved_num_periods:
            raise ValueError("num_periods does not match period_min/period_max")

        self.nqubit = nqubit
        self.num_periods = resolved_num_periods
        self.period_min = period_min
        self.period_max = resolved_period_max
        self.architecture = architecture
        self.beam_width = beam_width
        self.bit_width = max(1, resolved_period_max.bit_length())
        self.token_count = (self.bit_width + TOKEN_BITS - 1) // TOKEN_BITS
        feature_dim = 16 * nqubit
        self.feature_dim = feature_dim

        if architecture == "legacy":
            self.phi = nn.Sequential(
                nn.Linear(nqubit, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
            )
        else:
            bit_embedding_dim = max(8, nqubit)
            self.bit_value_embedding = nn.Embedding(2, bit_embedding_dim)
            self.bit_position_embedding = nn.Embedding(nqubit, bit_embedding_dim)
            self.bit_input_norm = nn.LayerNorm(nqubit * bit_embedding_dim)
            self.phi_input = nn.Sequential(
                nn.Linear(nqubit * bit_embedding_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
            )
            self.phi_blocks = nn.ModuleList(
                ResidualMLPBlock(feature_dim, dropout=dropout) for _ in range(3)
            )

        self.head_norm = nn.LayerNorm(feature_dim)
        self.head_blocks = nn.ModuleList(
            ResidualMLPBlock(feature_dim, dropout=dropout) for _ in range(2)
        )
        self.decoder_init = nn.Linear(feature_dim, feature_dim)
        self.token_embedding = nn.Embedding(TOKEN_CLASSES + 1, feature_dim)
        self.decoder_cell = nn.GRUCell(feature_dim, feature_dim)
        self.token_classifier = nn.Linear(feature_dim, TOKEN_CLASSES)

    def _legacy_features(self, bit_matrices: Tensor) -> Tensor:
        features_input = (
            bit_matrices
            if bit_matrices.dtype == torch.float32
            else bit_matrices.to(dtype=torch.float32)
        )
        return cast(Tensor, self.phi(features_input))

    def _weighted_features(self, bit_matrices: Tensor) -> Tensor:
        if bit_matrices.ndim != 3:
            raise ValueError(
                "bit_matrices must have shape (batch, sample_count, nqubit)"
            )
        if bit_matrices.shape[2] != self.nqubit:
            raise ValueError("bit_matrices last dimension must match nqubit")

        bits = bit_matrices.to(dtype=torch.long).clamp(0, 1)
        positions = torch.arange(self.nqubit, device=bits.device, dtype=torch.long)
        position_embedding = self.bit_position_embedding(positions).view(
            1,
            1,
            self.nqubit,
            -1,
        )
        embedded = self.bit_value_embedding(bits) + position_embedding
        features = embedded.reshape(*embedded.shape[:2], -1)
        features = self.phi_input(self.bit_input_norm(features))
        for block in self.phi_blocks:
            features = block(features)
        return cast(Tensor, features)

    def _pool_features(self, features: Tensor, sample_weights: Tensor | None) -> Tensor:
        if sample_weights is None:
            return cast(Tensor, features.mean(dim=1))
        if sample_weights.ndim != 2:
            raise ValueError("sample_weights must have shape (batch, sample_count)")
        if sample_weights.shape != features.shape[:2]:
            raise ValueError(
                "sample_weights shape must match bit_matrices batch/sample dimensions"
            )

        weights = sample_weights.to(device=features.device, dtype=features.dtype)
        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(features.dtype).eps,
        )
        normalized = weights / weight_sum
        return cast(Tensor, (features * normalized.unsqueeze(-1)).sum(dim=1))

    def pooled_features(
        self,
        bit_matrices: Tensor,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        features = (
            self._legacy_features(bit_matrices)
            if self.architecture == "legacy"
            else self._weighted_features(bit_matrices)
        )
        pooled = self.head_norm(self._pool_features(features, sample_weights))
        for block in self.head_blocks:
            pooled = block(pooled)
        return cast(Tensor, pooled)

    def _initial_decoder_state(self, pooled: Tensor) -> Tensor:
        return cast(Tensor, torch.tanh(self.decoder_init(pooled)))

    def decode_teacher_forced(self, pooled: Tensor, periods: Tensor) -> Tensor:
        tokens = periods_to_tokens(periods, self.token_count)
        hidden = self._initial_decoder_state(pooled)
        previous = torch.full(
            (pooled.shape[0],),
            BOS_TOKEN,
            dtype=torch.long,
            device=pooled.device,
        )
        logits: list[Tensor] = []
        for token_index in range(self.token_count):
            hidden = self.decoder_cell(self.token_embedding(previous), hidden)
            logits.append(self.token_classifier(hidden))
            previous = tokens[:, token_index]
        return torch.stack(logits, dim=1)

    def _valid_next_tokens(self, prefixes: Tensor, token_index: int) -> Tensor:
        token_values = torch.arange(
            TOKEN_CLASSES,
            dtype=torch.long,
            device=prefixes.device,
        )
        next_prefixes = prefixes.unsqueeze(-1) * TOKEN_CLASSES + token_values
        remaining_tokens = self.token_count - token_index - 1
        suffix_values = TOKEN_CLASSES**remaining_tokens
        lower = next_prefixes * suffix_values
        upper = lower + suffix_values - 1
        return cast(Tensor, (upper >= self.period_min) & (lower <= self.period_max))

    def _greedy_logits(self, pooled: Tensor) -> Tensor:
        hidden = self._initial_decoder_state(pooled)
        previous = torch.full(
            (pooled.shape[0],),
            BOS_TOKEN,
            dtype=torch.long,
            device=pooled.device,
        )
        prefixes = torch.zeros_like(previous)
        logits: list[Tensor] = []
        for token_index in range(self.token_count):
            hidden = self.decoder_cell(self.token_embedding(previous), hidden)
            step_logits = self.token_classifier(hidden)
            valid = self._valid_next_tokens(prefixes, token_index)
            constrained = step_logits.masked_fill(~valid, float("-inf"))
            logits.append(constrained)
            previous = constrained.argmax(dim=-1)
            prefixes = prefixes * TOKEN_CLASSES + previous
        return torch.stack(logits, dim=1)

    def decode_topk_from_pooled_features(
        self,
        pooled: Tensor,
        k: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not 1 <= k <= self.beam_width:
            raise ValueError(
                f"k must be between 1 and decoder beam width {self.beam_width}"
            )

        batch_size = pooled.shape[0]
        retained_width = min(self.beam_width, self.num_periods)
        returned_width = min(k, retained_width)
        hidden = self._initial_decoder_state(pooled).unsqueeze(1)
        previous = torch.full(
            (batch_size, 1),
            BOS_TOKEN,
            dtype=torch.long,
            device=pooled.device,
        )
        prefixes = torch.zeros(
            (batch_size, 1),
            dtype=torch.long,
            device=pooled.device,
        )
        scores = torch.zeros(
            (batch_size, 1),
            dtype=pooled.dtype,
            device=pooled.device,
        )

        for token_index in range(self.token_count):
            beam_count = prefixes.shape[1]
            flat_hidden = hidden.reshape(batch_size * beam_count, -1)
            flat_previous = previous.reshape(-1)
            updated_hidden = self.decoder_cell(
                self.token_embedding(flat_previous),
                flat_hidden,
            ).reshape(batch_size, beam_count, -1)
            logits = self.token_classifier(updated_hidden)
            valid = self._valid_next_tokens(prefixes, token_index)
            log_probs = logits.masked_fill(~valid, float("-inf")).log_softmax(dim=-1)
            log_probs = torch.nan_to_num(log_probs, nan=float("-inf"))
            expanded_scores = scores.unsqueeze(-1) + log_probs
            expanded_prefixes = (
                prefixes.unsqueeze(-1) * TOKEN_CLASSES
                + torch.arange(TOKEN_CLASSES, device=pooled.device).view(1, 1, -1)
            )

            flat_scores = expanded_scores.reshape(batch_size, -1)
            flat_prefixes = expanded_prefixes.expand(batch_size, beam_count, -1).reshape(
                batch_size,
                -1,
            )
            current_width = min(retained_width, flat_scores.shape[1])
            scores, positions = flat_scores.topk(current_width, dim=1)
            prefixes = flat_prefixes.gather(1, positions)
            parent_indices = positions // TOKEN_CLASSES
            previous = positions % TOKEN_CLASSES
            hidden = updated_hidden.gather(
                1,
                parent_indices.unsqueeze(-1).expand(-1, -1, self.feature_dim),
            )

        periods = prefixes[:, :returned_width]
        period_bits = integers_to_bits(periods, self.bit_width)
        return periods, period_bits, scores[:, :returned_width]

    def forward(
        self,
        bit_matrices: Tensor,
        sample_weights: Tensor | None = None,
        *,
        target_periods: Tensor | None = None,
    ) -> Tensor:
        pooled = self.pooled_features(bit_matrices, sample_weights)
        if target_periods is not None:
            return self.decode_teacher_forced(pooled, target_periods)
        return self._greedy_logits(pooled)

    def predict_topk_periods(
        self,
        bit_matrices: Tensor,
        candidate_periods: Sequence[int],
        k: int,
        *,
        sample_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self._validate_candidate_periods(candidate_periods)
        pooled = self.pooled_features(bit_matrices, sample_weights)
        return self.decode_topk_from_pooled_features(pooled, k)

    def _validate_candidate_periods(self, candidate_periods: Sequence[int]) -> None:
        if len(candidate_periods) != self.num_periods:
            raise ValueError("candidate_periods length does not match model period range")
        if not candidate_periods:
            raise ValueError("candidate_periods must not be empty")
        if (
            int(candidate_periods[0]) != self.period_min
            or int(candidate_periods[-1]) != self.period_max
        ):
            raise ValueError("candidate_periods bounds do not match model period range")


def predictor_from_checkpoint(
    payload: Mapping[str, Any],
    *,
    nqubit: int,
) -> tuple[DeepSetPeriodPredictor, tuple[int, ...]]:
    candidate_periods = payload.get("candidate_periods")
    state_dict = payload.get("state_dict")
    if not isinstance(candidate_periods, list) or not all(
        isinstance(value, int) for value in candidate_periods
    ):
        raise ValueError("checkpoint is missing integer candidate_periods")
    if not candidate_periods:
        raise ValueError("checkpoint candidate_periods must not be empty")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint is missing state_dict")

    if payload.get("decoder_type") != DECODER_TYPE:
        raise ValueError(f"checkpoint decoder_type must be '{DECODER_TYPE}'")
    if payload.get("token_bits") != TOKEN_BITS:
        raise ValueError(f"checkpoint token_bits must be {TOKEN_BITS}")
    architecture = payload.get("model_architecture")
    if architecture not in {"legacy", "weighted"}:
        raise ValueError("checkpoint model_architecture must be 'legacy' or 'weighted'")

    model = DeepSetPeriodPredictor(
        nqubit,
        len(candidate_periods),
        period_min=int(payload.get("period_min", candidate_periods[0])),
        period_max=int(payload.get("period_max", candidate_periods[-1])),
        architecture=cast(str, architecture),
        beam_width=int(payload.get("beam_width", DEFAULT_BEAM_WIDTH)),
    )
    model.load_state_dict(cast(dict[str, Tensor], state_dict))
    return model, tuple(int(value) for value in candidate_periods)


def periods_to_tokens(periods: Tensor, token_count: int) -> Tensor:
    if token_count < 1:
        raise ValueError("token_count must be positive")
    shifts = torch.arange(
        token_count - 1,
        -1,
        -1,
        device=periods.device,
        dtype=torch.long,
    ) * TOKEN_BITS
    return ((periods.to(torch.long).unsqueeze(-1) >> shifts) & (TOKEN_CLASSES - 1)).to(
        torch.long
    )


def integers_to_bits(values: Tensor, bit_width: int) -> Tensor:
    if bit_width < 1:
        raise ValueError("bit_width must be positive")
    shifts = torch.arange(
        bit_width - 1,
        -1,
        -1,
        device=values.device,
        dtype=torch.long,
    )
    return ((values.to(torch.long).unsqueeze(-1) >> shifts) & 1).to(torch.long)


def period_token_loss(
    token_logits: Tensor,
    periods: Tensor,
    *,
    label_smoothing: float = 0.0,
) -> Tensor:
    if token_logits.ndim != 3 or token_logits.shape[2] != TOKEN_CLASSES:
        raise ValueError("token_logits must have shape (batch, token_count, 16)")
    target_tokens = periods_to_tokens(periods, token_logits.shape[1]).reshape(-1)
    return cast(
        Tensor,
        nn.functional.cross_entropy(
            token_logits.reshape(-1, TOKEN_CLASSES),
            target_tokens,
            label_smoothing=label_smoothing,
        ),
    )
