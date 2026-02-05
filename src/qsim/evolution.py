from __future__ import annotations

import numpy as np 

def unitary_from_hamiltonian(H: np.ndarray, t: float) -> np.ndarray:

 ##  Build U = exp(-i H t) for Hermitian H

    H = np.asarray(H, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be square matrix") 

    w, V = np.linalg.eigh(H)
    phases = np.exp(-1j * w * float(t))
    return V @ np.diag(phases) @ V.conj().T


def evolve_statevector(state: np.ndarray, H: np.ndarray, t: float) -> np.ndarray: 
    U = unitary_from_hamiltonian(H, t)
    return U @ np.asarray(state, dtype=complex)

def evolve_statevector_step( state: np.ndarray, H: np.ndarray, dt: float, steps: int) -> np.ndarray:
    if steps <= 0: 
        raise ValueError("Steps must be positive") 

    U = unitary_from_hamiltonian(H, dt)
    out = np.asarray(state, dtype=complex) 

    for _ in range(int(steps)):
        out = U @ out 

    return out 
 