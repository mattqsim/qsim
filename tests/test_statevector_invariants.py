import numpy as np

from qsim.circuit import Circuit
from qsim.simulator import run_statevector


def test_empty_circuit_is_zero_state():
    c = Circuit(3)
    psi = run_statevector(c)
    assert psi.shape == (8,)
    assert np.allclose(psi, np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex))


def test_x_on_qubit0_flips_msb():
    # With your MSB convention, |100> corresponds to index 4 for 3 qubits.
    c = Circuit(3)
    c.x(0)
    psi = run_statevector(c)

    expected = np.zeros(8, dtype=complex)
    expected[4] = 1.0
    assert np.allclose(psi, expected)


def test_h_on_single_qubit_gives_half_half_probs():
    c = Circuit(1)
    c.h(0)
    psi = run_statevector(c)

    probs = np.abs(psi) ** 2
    assert np.allclose(probs.sum(), 1.0)
    assert np.allclose(probs, np.array([0.5, 0.5], dtype=float), atol=1e-12)


def test_bell_state_support_only_00_11():
    c = Circuit(2)
    c.h(0)
    c.cx(0, 1)

    psi = run_statevector(c)

    probs = np.abs(psi) ** 2
    assert np.allclose(probs.sum(), 1.0)

    # Bell = (|00> + |11>)/sqrt(2) => indices 0 and 3 only
    assert np.isclose(probs[0], 0.5, atol=1e-12)
    assert np.isclose(probs[3], 0.5, atol=1e-12)
    assert np.isclose(probs[1], 0.0, atol=1e-12)
    assert np.isclose(probs[2], 0.0, atol=1e-12)


def test_toffoli_deterministic_mapping_110_to_111():
    c = Circuit(3)
    c.x(0)
    c.x(1)          # prepares |110>
    c.tof(0, 1, 2)  # flips qubit 2 iff q0=q1=1 => |111>

    psi = run_statevector(c)

    expected = np.zeros(8, dtype=complex)
    expected[7] = 1.0  # |111> is index 7
    assert np.allclose(psi, expected)


def test_fredkin_deterministic_mapping_101_to_110():
    c = Circuit(3)
    c.x(0)
    c.x(2)              # prepares |101>
    c.fred(0, 1, 2)     # swap (1,2) iff control 0 is 1 => |110>

    psi = run_statevector(c)

    expected = np.zeros(8, dtype=complex)
    expected[6] = 1.0   # |110> is index 6
    assert np.allclose(psi, expected)
