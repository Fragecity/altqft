from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from qiskit import QuantumCircuit

# 这里需要确保 altqft.circuits.ph 已经包含了你新写的 ph_1 和 ph_random 函数
from altqft.circuits.ph_generators import qft, ph_1, ph_random
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


def build_circuit(
    circuit_type: str,
    nqubit: int,
    nlayer: Optional[int] = None,
) -> QuantumCircuit:
    circuit_key = circuit_type.lower()
    
    if circuit_key == "qft":
        return qft(nqubit)
        
    if circuit_key == "ph1":
        return ph_1(nqubit)
        
    if circuit_key == "ph_random":
        if nlayer is None:
            raise ValueError("ph_random 电路需要提供 nlayer。")
        return ph_random(nqubit, nlayer)
        
    raise ValueError(f"暂不支持的电路类型: {circuit_type}")


def default_period_range(nqubit: int) -> range:
    upper_bound = min(
        max(int(2 ** (nqubit / 4)), nqubit**2),
        max(nqubit, int(nqubit**2 / 2)),
    )
    return range(nqubit, upper_bound + 1)


def calculate_fi_results(config: FiExperimentConfig) -> list[FiResult]:
    """跑单个Config：取消外层循环，直接用 config.nqubit"""
    results: list[FiResult] = []
    
    nqubit = config.nqubit
    circuit = build_circuit(config.circuit_type, nqubit, config.nlayer)
    period_range = default_period_range(nqubit)
    
    for _ in range(config.repeat):
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