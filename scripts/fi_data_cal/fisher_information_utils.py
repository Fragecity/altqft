from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from qiskit import QuantumCircuit

from altqft.circuits.ph_generators import (
    ph_1,
    ph_1_random,
    ph_random,
    ph_random_phase,
    qft,
)
from altqft.nn.process_qc import min_fi


@dataclass(frozen=True)
class FiExperimentConfig:
    circuit_type: str
    nqubit: int
    repeat: int = 1
    nlayer: Optional[int] = None


@dataclass(frozen=True)
class FiResult:
    circuit_type: str
    nqubit: int
    fi_value: float
    nlayer: Optional[int] = None


def _require_nlayer(circuit_type: str, nlayer: Optional[int]) -> int:
    if nlayer is None:
        raise ValueError(f"{circuit_type} 电路需要提供 nlayer。")
    return nlayer


def build_circuit(
    circuit_type: str,
    nqubit: int,
    nlayer: Optional[int] = None,
) -> QuantumCircuit:
    circuit_key = circuit_type.lower()
    circuit_builders: dict[str, Callable[[], QuantumCircuit]] = {
        "qft": lambda: qft(nqubit),
        "ph1": lambda: ph_1(nqubit),
        "ph_random": lambda: ph_random(nqubit, _require_nlayer("ph_random", nlayer)),
        "ph_1_random": lambda: ph_1_random(nqubit),
        "ph_random_phase": lambda: ph_random_phase(
            nqubit,
            _require_nlayer("ph_random_phase", nlayer),
        ),
    }

    try:
        return circuit_builders[circuit_key]()
    except KeyError as exc:
        raise ValueError(f"暂不支持的电路类型: {circuit_type}") from exc


def default_period_range(nqubit: int) -> range:
    upper_bound = min(
        max(int(2 ** (nqubit / 4)), nqubit**2),
        max(nqubit, int(nqubit**2 / 2)),
    )
    return range(nqubit, upper_bound + 1)


def calculate_fi_results(config: FiExperimentConfig) -> list[FiResult]:
    results: list[FiResult] = []
    nqubit = config.nqubit
    period_range = default_period_range(nqubit)

    for _ in range(config.repeat):
        # 移入循环内：每次迭代都生成一个新随机参数的电路
        circuit = build_circuit(config.circuit_type, nqubit, config.nlayer)
        
        fi_value = min_fi(circuit, period_range=period_range)
        results.append(
            FiResult(
                circuit_type=config.circuit_type,
                nqubit=nqubit,
                fi_value=fi_value,
                nlayer=config.nlayer,
            )
        )
    return results


def extend_results(target: list[FiResult], chunks: Iterable[Iterable[FiResult]]) -> None:
    for chunk in chunks:
        target.extend(chunk)
