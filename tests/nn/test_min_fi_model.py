import math

import pytest
import torch
from qiskit.quantum_info import Operator

from altqft.circuits.ph_generators import ph_1_parametrized
from altqft.nn.model import PH1MinFIModel
from altqft.nn.periods import build_default_period_range
from altqft.nn.process_qc import min_fi


def test_ph1_min_fi_model_matches_qiskit_circuit() -> None:
    phases = [0.1, 0.2, 0.3, 0.4]
    model = PH1MinFIModel(nqubit=4, init_phases=phases)

    torch_operator = model.build_unitary().detach().cpu().numpy()
    qiskit_operator = Operator(ph_1_parametrized(4, phases)).data

    assert torch_operator.shape == qiskit_operator.shape
    assert torch.allclose(
        torch.from_numpy(torch_operator),
        torch.from_numpy(qiskit_operator).to(torch.complex64),
        atol=1e-6,
        rtol=1e-6,
    )


def test_ph1_min_fi_model_forward_returns_scalar() -> None:
    model = PH1MinFIModel(nqubit=4, init_phases=[0.1, 0.2, 0.3, 0.4])
    value = model([2, 4])
    assert value.ndim == 0
    assert math.isfinite(float(value.detach().cpu().item()))


def test_ph1_min_fi_model_matches_process_qc_min_fi() -> None:
    phases = [0.1, 0.2, 0.3, 0.4]
    period_range = build_default_period_range(4)
    model = PH1MinFIModel(nqubit=4, init_phases=phases)

    expected = min_fi(ph_1_parametrized(4, phases), period_range, device="cpu")
    actual = float(model(period_range).detach().cpu().item())

    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)
