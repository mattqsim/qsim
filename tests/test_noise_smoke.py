import numpy as np

from qsim.circuit import Circuit
from qsim.simulator import run_statevector, run_noisy_statevector
from qsim.noise import NoiseModel


def test_noisy_runner_with_zero_noise_matches_clean_runner():
    c = Circuit(2)
    c.h(0)
    c.cx(0, 1)

    clean = run_statevector(c)
    noisy = run_noisy_statevector(c, model=NoiseModel(), rng=np.random.default_rng(0))

    assert np.allclose(noisy, clean, atol=1e-12)


def test_bit_flip_p1_flips_after_identity_gate():
    # noise is applied after each op on each target in run_noisy_statevector
    c = Circuit(1)
    c.i(0)  # identity op just to trigger the noise application

    model = NoiseModel(bit_flip_p=1.0)
    psi = run_noisy_statevector(c, model=model, rng=np.random.default_rng(123))

    # start |0>, apply I, then X with prob 1 => |1>
    assert np.allclose(psi, np.array([0, 1], dtype=complex), atol=1e-12)


def test_noise_output_is_normalized():
    c = Circuit(3)
    c.h(0)
    c.cx(0, 1)
    c.tof(0, 1, 2)

    model = NoiseModel(depolarizing_p=0.5, phase_flip_p=0.2, bit_flip_p=0.2)
    psi = run_noisy_statevector(c, model=model, rng=np.random.default_rng(5))

    assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-10)
