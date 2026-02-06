import numpy as np

from qsim.circuit import Circuit
from qsim.simulator import run_statevector
from qsim.measurement import measure_all, measure_qubit


def test_measure_all_returns_valid_basis_ket_and_probability():
    c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    psi = run_statevector(c)

    rng = np.random.default_rng(123)
    outcome, prob, post = measure_all(psi, num_qubits=2, rng=rng)

    assert isinstance(outcome, int)
    assert 0 <= outcome < 4
    assert 0.0 <= prob <= 1.0

    # post should be a basis ket: exactly one nonzero, norm 1
    assert post.shape == (4,)
    assert np.isclose(np.linalg.norm(post), 1.0, atol=1e-12)
    assert np.count_nonzero(np.abs(post) > 0) == 1
    assert np.isclose(post[outcome], 1.0 + 0.0j, atol=1e-12)


def test_measure_qubit_on_plus_state_is_half_half_and_collapses():
    c = Circuit(1)
    c.h(0)  # |+>
    psi = run_statevector(c)

    rng = np.random.default_rng(7)
    res = measure_qubit(psi, qubit=0, num_qubits=1, rng=rng)

    assert res.outcome in (0, 1)
    assert np.isclose(res.probability, 0.5, atol=1e-12)

    # collapsed state must be |0> or |1>
    if res.outcome == 0:
        assert np.allclose(res.post_state, np.array([1, 0], dtype=complex))
    else:
        assert np.allclose(res.post_state, np.array([0, 1], dtype=complex))


def test_measure_qubit_out_of_range_raises():
    psi = np.array([1, 0], dtype=complex)
    try:
        measure_qubit(psi, qubit=1, num_qubits=1)
        assert False, "Expected ValueError for out-of-range qubit"
    except ValueError:
        pass
