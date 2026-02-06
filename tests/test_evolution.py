import numpy as np

from qsim.circuit import Circuit
from qsim.simulator import run_statevector
from qsim.evolution import evolve_statevector


def test_evolution_zero_hamiltonian_no_change():
    c = Circuit(1)
    c.h(0)
    psi = run_statevector(c)

    H = np.zeros((2, 2), dtype=complex)
    psi2 = evolve_statevector(psi, H, t=1.234)

    assert np.allclose(psi2, psi, atol=1e-12)


def test_evolution_t_zero_no_change():
    c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    psi = run_statevector(c)

    # any Hermitian H; t=0 should be identity
    H = np.diag([0.0, 1.0, 2.0, 3.0]).astype(complex)
    psi2 = evolve_statevector(psi, H, t=0.0)

    assert np.allclose(psi2, psi, atol=1e-12)


def test_z_hamiltonian_preserves_probabilities_on_plus():
    # H = Z, evolve |+> should keep probs = (0.5,0.5) for any t
    c = Circuit(1)
    c.h(0)
    psi = run_statevector(c)

    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    psi2 = evolve_statevector(psi, Z, t=0.37)

    p1 = np.abs(psi) ** 2
    p2 = np.abs(psi2) ** 2
    assert np.allclose(p2, p1, atol=1e-12)
    assert np.isclose(np.linalg.norm(psi2), 1.0, atol=1e-12)
