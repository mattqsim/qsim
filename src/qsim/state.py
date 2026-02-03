import numpy as np

def basis_state(index: int, num_qubits: int) -> np.ndarray:
	dim = 2 ** num_qubits
	if index < 0 or index >= dim: 
		raise ValueError(f"index must be btween 0 and {dim-1}")
	
	state = np.zeros(dim, dtype=complex)
	state[index] = 1.0
	return state

def zero_state(num_qubits: int) -> np.ndarray:
	return basis_state(0, num_qubits)

def normalize(state: np.ndarray) -> np.ndarray:
	norm = np.linalg.norm(state) 
	if norm == 0: 
		raise ValueError("Cannot normalise zero vector!")
	return state / norm 
