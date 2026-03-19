from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Optional

from qiskit import QuantumCircuit

from altqft.circuits.ph import ph_phase, qft
from altqft.nn.process_qc import min_fi


@dataclass(frozen=True)
class FiExperimentConfig:
    circuit_type: str
    nqubit_values: list[int]
    repeat: int = 1
    hlayout: Optional[list[int]] = None


@dataclass(frozen=True)
class FiResult:
    circuit_type: str
    nqubit: int
    fi_value: float
    hlayout: Optional[list[int]] = None


def build_circuit(
    circuit_type: str,
    nqubit: int,
    hlayout: Optional[list[int]] = None,
) -> QuantumCircuit:
    circuit_key = circuit_type.lower()
    if circuit_key == "qft":
        return qft(nqubit)
    if circuit_key == "ph":
        if hlayout is None:
            raise ValueError("ph 电路需要提供 hlayout。")
        if len(hlayout) != nqubit:
            raise ValueError("hlayout 的长度必须与 nqubit 一致。")
        return ph_phase(hlayout)
    raise ValueError(f"暂不支持的电路类型: {circuit_type}")


def default_period_range(nqubit: int) -> range:
    upper_bound = min(
        max(int(2 ** (nqubit / 4)), nqubit**2),
        max(nqubit, int(nqubit**2 / 2)),
    )
    return range(nqubit, upper_bound + 1)


def calculate_fi_results(config: FiExperimentConfig) -> list[FiResult]:
    results: list[FiResult] = []
    for nqubit in config.nqubit_values:
        circuit = build_circuit(config.circuit_type, nqubit, config.hlayout)
        period_range = default_period_range(nqubit)
        for _ in range(config.repeat):
            fi_value = min_fi(circuit, period_range=period_range)
            results.append(
                FiResult(
                    circuit_type=config.circuit_type,
                    nqubit=nqubit,
                    fi_value=fi_value,
                    hlayout=config.hlayout,
                )
            )
    return results


def random_hlayout(nqubit: int, rng: random.Random | None = None) -> list[int]:
    random_generator = rng or random
    hlayout = [0]
    current_max = 0

    for _ in range(1, nqubit):
        layer = random_generator.randint(0, current_max + 1)
        hlayout.append(layer)
        current_max = max(current_max, layer)

    return hlayout


def extend_results(target: list[FiResult], chunks: Iterable[Iterable[FiResult]]) -> None:
    for chunk in chunks:
        target.extend(chunk)
