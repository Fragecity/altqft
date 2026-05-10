from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit.synthesis.qft import synth_qft_full

CountDict: TypeAlias = dict[str, int]


def default_counting_qubits(N: int) -> int:
    if N < 2:
        raise ValueError("N must be at least 2")
    return 2 * math.ceil(math.log2(N))


def _work_register_qubits(N: int) -> int:
    if N < 2:
        raise ValueError("N must be at least 2")
    return math.ceil(math.log2(N))


@dataclass(slots=True)
class ShorConfig:
    N: int
    a: int
    counting_qubits: int
    shots: int = 1024
    seed: int | None = 7

    def __post_init__(self) -> None:
        if self.N < 3:
            raise ValueError("N must be at least 3")
        if self.a <= 1 or self.a >= self.N:
            raise ValueError("a must satisfy 1 < a < N")
        if self.counting_qubits < 1:
            raise ValueError("counting_qubits must be positive")
        if self.shots < 1:
            raise ValueError("shots must be positive")


@dataclass(frozen=True, slots=True)
class ShorCandidate:
    bitstring: str
    count: int
    phase: float
    continued_fraction_denominator: int | None
    validated_order: int | None
    status: str


@dataclass(frozen=True, slots=True)
class ShorResult:
    success: bool
    factors: tuple[int, int] | None
    order: int | None
    candidates: list[ShorCandidate]


def _modular_multiplication_matrix(multiplier: int, modulus: int) -> np.ndarray:
    if modulus < 3:
        raise ValueError("modulus must be at least 3")
    if math.gcd(multiplier, modulus) != 1:
        raise ValueError("multiplier must be coprime with modulus")

    dimension = 1 << _work_register_qubits(modulus)
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for basis_state in range(dimension):
        target_state = basis_state
        if basis_state < modulus:
            target_state = (multiplier * basis_state) % modulus
        matrix[target_state, basis_state] = 1.0
    return matrix


def _modular_multiplication_gate(multiplier: int, modulus: int) -> Gate:
    matrix = _modular_multiplication_matrix(multiplier, modulus)
    return UnitaryGate(matrix, label=f"mul_{multiplier}_mod_{modulus}")


def _controlled_modular_multiplication_gate(base: int, exponent: int, modulus: int) -> Gate:
    multiplier = pow(base, 1 << exponent, modulus)
    gate = _modular_multiplication_gate(multiplier, modulus)
    return gate.control(1, label=f"c_mul_{multiplier}_mod_{modulus}")


def _build_modular_exponentiation_circuit(config: ShorConfig) -> QuantumCircuit:
    counting = QuantumRegister(config.counting_qubits, "count")
    work = QuantumRegister(_work_register_qubits(config.N), "work")
    circuit = QuantumCircuit(counting, work)

    circuit.x(work[0])
    circuit.h(counting)
    for exponent, control_qubit in enumerate(counting):
        controlled_gate = _controlled_modular_multiplication_gate(
            config.a,
            exponent,
            config.N,
        )
        circuit.append(controlled_gate, [control_qubit, *work])
    return circuit


def _inverse_qft_circuit(counting_qubits: int) -> QuantumCircuit:
    return synth_qft_full(counting_qubits, do_swaps=True, inverse=True)


def _modular_exponentiation_outputs(config: ShorConfig) -> np.ndarray:
    outputs = np.empty(1 << config.counting_qubits, dtype=np.int64)
    value = 1
    for exponent in range(outputs.size):
        if exponent == 0:
            outputs[exponent] = 1
            continue
        value = (value * config.a) % config.N
        outputs[exponent] = value
    return outputs


def _post_modular_exponentiation_state(config: ShorConfig) -> Statevector:
    work_dimension = 1 << _work_register_qubits(config.N)
    counting_dimension = 1 << config.counting_qubits
    amplitudes = np.zeros((work_dimension, counting_dimension), dtype=np.complex128)
    normalization = 1.0 / math.sqrt(counting_dimension)
    for count_value, work_value in enumerate(_modular_exponentiation_outputs(config)):
        amplitudes[work_value, count_value] = normalization
    return Statevector(amplitudes.reshape(-1), dims=(2,) * (_work_register_qubits(config.N) + config.counting_qubits))


def _reshape_joint_state(config: ShorConfig, state: Statevector) -> np.ndarray:
    work_dimension = 1 << _work_register_qubits(config.N)
    counting_dimension = 1 << config.counting_qubits
    return np.asarray(state.data).reshape(work_dimension, counting_dimension)


def _work_register_probabilities(config: ShorConfig, state: Statevector) -> np.ndarray:
    reshaped = _reshape_joint_state(config, state)
    probabilities = np.sum(np.abs(reshaped) ** 2, axis=1)
    return np.asarray(probabilities / probabilities.sum(), dtype=np.float64)


def _collapsed_counting_state(
    config: ShorConfig,
    state: Statevector,
    work_outcome: int,
) -> Statevector:
    reshaped = _reshape_joint_state(config, state)
    amplitudes = np.asarray(reshaped[work_outcome], dtype=np.complex128)
    probability = float(np.sum(np.abs(amplitudes) ** 2))
    if probability <= 0.0:
        raise ValueError(f"work outcome {work_outcome} has zero probability")
    normalized = amplitudes / math.sqrt(probability)
    return Statevector(normalized, dims=(2,) * config.counting_qubits)


def _phase_measurement_probabilities(
    config: ShorConfig,
    state: Statevector,
    work_outcome: int,
) -> np.ndarray:
    counting_state = _collapsed_counting_state(config, state, work_outcome)
    transformed = counting_state.evolve(_inverse_qft_circuit(config.counting_qubits))
    probabilities = np.asarray(transformed.probabilities(), dtype=np.float64)
    return np.asarray(probabilities / probabilities.sum(), dtype=np.float64)


def _sample_counts_from_probabilities(
    probabilities: np.ndarray,
    *,
    shots: int,
    width: int,
    rng: np.random.Generator,
) -> CountDict:
    if shots < 1:
        return {}
    samples = rng.choice(len(probabilities), size=shots, p=probabilities)
    counts = Counter(int(sample) for sample in samples)
    return {
        format(measurement, f"0{width}b"): count
        for measurement, count in counts.items()
    }


def _sample_order_finding_counts(config: ShorConfig) -> CountDict:
    joint_state = _post_modular_exponentiation_state(config)
    work_probabilities = _work_register_probabilities(config, joint_state)
    work_dimension = len(work_probabilities)
    rng = np.random.default_rng(config.seed)
    work_samples = rng.choice(work_dimension, size=config.shots, p=work_probabilities)
    work_counts = Counter(int(sample) for sample in work_samples)

    phase_counts: Counter[str] = Counter()
    conditional_phase_probabilities = {
        work_outcome: _phase_measurement_probabilities(config, joint_state, work_outcome)
        for work_outcome, probability in enumerate(work_probabilities)
        if probability > 0.0
    }
    for work_outcome, shot_count in work_counts.items():
        sampled_counts = _sample_counts_from_probabilities(
            conditional_phase_probabilities[work_outcome],
            shots=shot_count,
            width=config.counting_qubits,
            rng=rng,
        )
        phase_counts.update(sampled_counts)
    return dict(phase_counts)


def _candidate_denominator(measurement: int, config: ShorConfig) -> int:
    phase_fraction = Fraction(measurement, 1 << config.counting_qubits)
    return phase_fraction.limit_denominator(config.N).denominator


def _recover_order_from_denominator(
    denominator: int | None,
    *,
    a: int,
    N: int,
) -> int | None:
    if denominator is None or denominator < 1:
        return None

    for multiple in range(1, N + 1):
        candidate_order = denominator * multiple
        if pow(a, candidate_order, N) == 1:
            return candidate_order
    return None


def _recover_factors_from_order(
    order: int,
    *,
    a: int,
    N: int,
) -> tuple[tuple[int, int] | None, str]:
    if order % 2 != 0:
        return None, "validated order is odd"

    half_power = pow(a, order // 2, N)
    if half_power in (1, N - 1):
        return None, f"a^(r/2) mod N is trivial ({half_power})"

    factor_minus = math.gcd(half_power - 1, N)
    factor_plus = math.gcd(half_power + 1, N)
    for factor in (factor_minus, factor_plus):
        if 1 < factor < N and N % factor == 0:
            other_factor = N // factor
            if 1 < other_factor < N:
                factors = tuple(sorted((factor, other_factor)))
                return factors, f"success factors={factors}"

    return None, f"gcd recovery was trivial ({factor_minus}, {factor_plus})"


def _analyze_candidate(
    bitstring: str,
    count: int,
    config: ShorConfig,
) -> tuple[ShorCandidate, tuple[int, int] | None, int | None]:
    measurement = int(bitstring, 2)
    phase = measurement / float(1 << config.counting_qubits)
    denominator = _candidate_denominator(measurement, config)
    order = _recover_order_from_denominator(
        denominator,
        a=config.a,
        N=config.N,
    )
    if order is None:
        candidate = ShorCandidate(
            bitstring=bitstring,
            count=count,
            phase=phase,
            continued_fraction_denominator=denominator,
            validated_order=None,
            status=f"no order found for denominator {denominator}",
        )
        return candidate, None, None

    factors, status = _recover_factors_from_order(order, a=config.a, N=config.N)
    candidate = ShorCandidate(
        bitstring=bitstring,
        count=count,
        phase=phase,
        continued_fraction_denominator=denominator,
        validated_order=order,
        status=status,
    )
    return candidate, factors, order


def _sorted_measurement_counts(counts: CountDict) -> list[tuple[str, int]]:
    return sorted(
        counts.items(),
        key=lambda item: (-item[1], -int(item[0], 2)),
    )


def _result_from_counts(config: ShorConfig, counts: CountDict) -> ShorResult:
    candidates: list[ShorCandidate] = []
    for bitstring, count in _sorted_measurement_counts(counts):
        candidate, factors, order = _analyze_candidate(bitstring, count, config)
        candidates.append(candidate)
        if factors is not None:
            return ShorResult(
                success=True,
                factors=factors,
                order=order,
                candidates=candidates,
            )

    return ShorResult(
        success=False,
        factors=None,
        order=None,
        candidates=candidates,
    )


def run_shor(config: ShorConfig) -> ShorResult:
    gcd_value = math.gcd(config.a, config.N)
    if 1 < gcd_value < config.N:
        factors = tuple(sorted((gcd_value, config.N // gcd_value)))
        return ShorResult(
            success=True,
            factors=factors,
            order=None,
            candidates=[],
        )

    counts = _sample_order_finding_counts(config)
    return _result_from_counts(config, counts)
