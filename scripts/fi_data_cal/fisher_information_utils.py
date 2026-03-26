from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from qiskit import QuantumCircuit

from altqft.circuits.ph_generators import (
    ph_1,
    ph_1_random,
    ph_random,
    ph_random_phase,
    qft,
)
from altqft.nn.process_qc import min_fi

CircuitFactory = Callable[[], QuantumCircuit]


@dataclass(frozen=True)
class FiExperimentConfig:
    circuit_type: str
    nqubit: int
    repeat: int = 1
    nlayer: int | None = None


@dataclass(frozen=True)
class FiResult:
    circuit_type: str
    nqubit: int
    fi_value: float
    nlayer: int | None = None


def _require_nlayer(circuit_type: str, nlayer: int | None) -> int:
    if nlayer is None:
        raise ValueError(f"{circuit_type} requires nlayer")
    return nlayer


def build_circuit(
    circuit_type: str,
    nqubit: int,
    nlayer: int | None = None,
) -> QuantumCircuit:
    circuit_key = circuit_type.lower()
    builders: dict[str, CircuitFactory] = {
        "qft": lambda: qft(nqubit),
        "ph1": lambda: ph_1(nqubit),
        "ph_random": lambda: ph_random(nqubit, _require_nlayer("ph_random", nlayer)),
        "ph_1_random": lambda: ph_1_random(nqubit),
        "ph_random_phase": lambda: ph_random_phase(
            nqubit,
            _require_nlayer("ph_random_phase", nlayer),
        ),
    }

    if circuit_key not in builders:
        raise ValueError(f"unsupported circuit type: {circuit_type}")
    return builders[circuit_key]()


def default_period_range(nqubit: int) -> range:
    upper_bound = min(
        max(int(2 ** (nqubit / 4)), nqubit**2),
        max(nqubit, int(nqubit**2 / 2)),
    )
    return range(nqubit, upper_bound + 1)


def calculate_fi_results(config: FiExperimentConfig) -> list[FiResult]:
    period_range = default_period_range(config.nqubit)
    return [
        FiResult(
            circuit_type=config.circuit_type,
            nqubit=config.nqubit,
            fi_value=min_fi(
                build_circuit(config.circuit_type, config.nqubit, config.nlayer),
                period_range=period_range,
            ),
            nlayer=config.nlayer,
        )
        for _ in range(config.repeat)
    ]


def extend_results(target: list[FiResult], chunks: Iterable[Iterable[FiResult]]) -> None:
    for chunk in chunks:
        target.extend(chunk)
