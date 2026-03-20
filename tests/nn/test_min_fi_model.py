import math

import torch
from qiskit.quantum_info import Operator

from altqft.circuits.ph_generators import ph_1_parametrized
from altqft.nn.model import PH1MinFIModel


def test_ph1_min_fi_model_matches_qiskit_circuit() -> None:
    phases = [0.1, 0.2, 0.3, 0.4]
    model = PH1MinFIModel(nqubit=4, init_phases=phases)

    torch_operator = model.build_unitary().detach().cpu().numpy()
    qiskit_operator = Operator(ph_1_parametrized(4, phases)).data

    assert torch_operator.shape == qiskit_operator.shape
    assert abs(torch_operator).sum() > 0


def test_ph1_min_fi_model_forward_returns_scalar() -> None:
    model = PH1MinFIModel(nqubit=4, init_phases=[0.1, 0.2, 0.3, 0.4])
    value = model([2, 4])
    assert value.ndim == 0
    assert math.isfinite(float(value.detach().cpu().item()))
