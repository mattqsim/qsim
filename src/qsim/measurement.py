from __future__ import annotations

from dataclasses import dataclass 
from typing import Optional, Tuple
import numpy as np 

from .state import validate_state, copy_state 

@dataclass(frozen=True) 
class MeasurementResult: 
    outcome: int
    probability: float 
    post_state: np.ndarray 

def _bit_is_one(index: int, qubit: int, num_qubits: int) -> bool: 
    shift = (num_qubits - 1) - qubit
    return ((index >> shift) & 1) == 1


def measure_qubit(
    state: np.ndarray, 
    qubit: int, 
    num_qubits: int,
    *, 
    rng: Optional[np.random.Generator] = None, 
    validate: bool = True, 
) -> MeasurementResult: 

    if validate: 
        validate_state(state, num_qubits) 

    if qubit < 0 or qubit >= num_qubits: 
        raise ValueError("qubit out of range") 

    psi = copy_state(state) 

    dim = 2 ** num_qubits

    p1 = 0.0 
    for i in range(dim): 
        if _bit_is_one(i, qubit, num_qubits): 
            amp = psi[i]
            p1 += (amp.real * amp.real + amp.imag * amp.imag) 

    p0 = 1.0 - p1
    if p0 < 0 and p0 > -1e-12:
        p0 = 0.0
    if p1 < 0 and p1 > -1e-12:
        p1 = 0.0
    if p0 < 0 or p1 < 0: 
        raise ValueError("Probabilities Invalid") 

    if rng is None: 
        rng = np.random.default_rng() 

    r = rng.random()
    outcome = 1 if r < p1 else 0
    prob = p1 if outcome == 1 else p0 

    post = psi
    if outcome == 1: 
        for i in range(dim): 
            if not _bit_is_one(i, qubit, num_qubits):
                post[i] = 0.0 
    else: 
        for i in range(dim):
            if _bit_is_one(i, qubit, num_qubits): 
                post[i] = 0.0

    if prob == 0.0: 
        raise ValueError("Measured an outcome with zero probability") 

    post /= np.sqrt(prob)

    return MeasurementResult(outcome=outcome, probability=float(prob), post_state=post) 


def measure_all(
    state: np.ndarray, 
    num_qubits: int,
    *,
    rng: Optional[np.random.Generator] = None, 
    validate: bool = True, 
) -> Tuple[int, float, np.ndarray]:

    if validate:
        validate_state(state, num_qubits) 

    psi = copy_state(state)
    probs = (psi.real * psi.real + psi.imag * psi.imag)

    total = float(probs.sum())
    if not np.isclose(total, 1.0, atol=1e-10):
        raise ValueError("State is not normalised") 

    if rng is None: 
        rng = np.random.default_rng()

    outcome_index = int(rng.choice(len(probs), p=probs))
    prob = float(probs[outcome_index])

    post = np.zeros_like(psi)
    post[outcome_index] = 1.0 + 0.0j 

    return outcome_index, prob, post 