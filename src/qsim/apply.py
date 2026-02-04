import numpy as np
from .state import validate_state

def _inverse_permutation(perm: list[int]) -> list[int]: 
    inv = [0] * len(perm) 
    for i, p in enumerate(perm): 
        inv[p] = i 
    return inv

def apply_k_qubit_gate(state, U, targets: list[int], num_qubits: int) -> np.ndarray: 
    """
    Apply a k-qubit unitary U to the given statevector on the specified target qubits. 
    Conventions: 
    - Statevector length 2**num_qubits. 
    - qubit indices are 0...num_qubits-1. 
    - qubit 0 is most significant in basis ordering 
    """
    validate_state(state, num_qubits) 

    U = np.asarray(U, dtype=complex) 
    state = np.asarray(state, dtype=complex) 

    if U.ndim != 2 or U.shape[0] != U.shape[1]: 
        raise ValueError(f"U must be a square matrix, got shape {U.shape}") 

    k = len(targets) 
    if k == 0: 
        return state.copy()

    if len(set(targets)) != k: 
        raise ValueError(f"targets must be unique, got {targets}") 

    if any((t < 0 or t >= num_qubits) for t in targets): 
        raise ValueError(f"targets out of range for num_qubits={num_qubits}: {targets}") 

    dim_k = 2 ** k 
    if U.shape != (dim_k, dim_k): 
        raise ValueError(f"U shape must be {(dim_k, dim_k)} for k={k}, got {U.shape}") 

    psi = state.reshape((2,) * num_qubits)

    remaining = [q for q in range(num_qubits) if q not in targets]
    perm = list(targets) + remaining 

    psi_perm = np.transpose(psi, axes=perm)

    psi_mat = psi_perm.reshape(dim_k, -1) 

    psi_mat2 = U @ psi_mat 

    psi_perm2 = psi_mat2.reshape((2,) * num_qubits)
    inv_perm = _inverse_permutation(perm) 
    psi2 = np.transpose(psi_perm2, axes=inv_perm) 

    return psi2.reshape(-1) 