from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LayerSpec:
    controls: tuple[int, ...]
    targets: tuple[int, ...]


def layer_members(hlayout: Sequence[int], layer: int) -> tuple[int, ...]:
    return tuple(index for index, value in enumerate(hlayout) if value == layer)


def final_layer(hlayout: Sequence[int]) -> tuple[int, ...]:
    return layer_members(hlayout, max(hlayout, default=-1))


def iter_active_layers(hlayout: Sequence[int]) -> Iterator[LayerSpec]:
    remaining = set(range(len(hlayout)))

    for layer in range(max(hlayout, default=-1)):
        controls = layer_members(hlayout, layer)
        remaining.difference_update(controls)
        yield LayerSpec(controls=controls, targets=tuple(sorted(remaining)))


def count_required_phases(hlayout: Sequence[int]) -> int:
    return sum(
        len(layer.controls) * len(layer.targets)
        for layer in iter_active_layers(hlayout)
    )


def alternating_layout(nqubit: int) -> list[int]:
    return [index % 2 for index in range(nqubit)]
