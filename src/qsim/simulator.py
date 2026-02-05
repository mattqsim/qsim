from __future__ import annotations 
from typing     import Optional 

import numpy as np 

from .state   import zero_state, validate_state, copy_state 
from .apply   import apply_k_qubit_gate
from .circuit import Circuit, Operation 


def _validate_operation(op: Operation, num_qubits: int) -> None: 
    targets = tuple(op.targets) 
    k = len(targets) 

    if k == 0: 
        raise ValueError("Operation has no targets") 

    if any(t < 0 or t >= num_qubits for t in targets): 
        raise ValueError("Operation targets out of range") 

    if len(set(targets)) != len(targets): 
        raise ValueError("Operations targets must be unique") 

    expected_dim = 2**k 
    if op.U.shape != (expected_dim, expected_dim): 
        raise ValueError("Gate matrix shape mismatch") 


def run_statevector(
    circuit: Circuit, 
    initial_state: Optional[np.ndarray] = None, 
    *, validate: bool = True,
) -> np.ndarray: 

    num_qubits = circuit.num_qubits
    
    if initial_state is None: 
        state = zero_state(num_qubits)
    else:
        validate_state(initial_state, num_qubits) 
        state = copy_state(initial_state)

    for op in circuit.ops: 
        if validate: 
            _validate_operation(op, num_qubits) 

        state = apply_k_qubit_gate(
            state=state,
            U=op.U,
            targets=op.targets, 
            num_qubits=num_qubits,
        ) 

    if validate: 
        norm = np.linalg.norm(state)
        if not np.isclose(norm, 1.0, atol=1e-10):
            raise ValueError("State Normal Drifted") 

    return state 
    