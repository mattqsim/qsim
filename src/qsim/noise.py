from __future__  import annotations
from dataclasses import dataclass
from typing      import Optional

import numpy as np

from .apply import apply_k_qubit_gate
from .      import gates


@dataclass(frozen=True) 
class NoiseModel: 
    bit_flip_p:     float = 0.0
    phase_flip_p:   float = 0.0
    depolarizing_p: float = 0.0


def _validate_p(p: float, name: str) -> None: 
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"{name} must be [0,1], got {p}") 


def apply_noise_to_statevector(
    state: np.ndarray,
    *,
    num_qubits: int,
    qubit: int, 
    model: NoiseModel, 
    rng: Optional[np.random.Generator] = None, 
) -> np.ndarray:

    if qubit < 0 or qubit >= num_qubits: 
        raise ValueError(f"qubit out of range: {qubit} for num_qubits={num_qubits}") 

    if rng is None: 
        rng = np.random.default_rng() 

    _validate_p(model.bit_flip_p, "bit_flip_p")
    _validate_p(model.phase_flip_p, "phase_flip_p")
    _validate_p(model.depolarizing_p, "depolarizing_p")

    out = state 


    # Bit-flip
    if model.bit_flip_p > 0.0 and rng.random() < model.bit_flip_p: 
        out = apply_k_qubit_gate(out, gates.X, (qubit,), num_qubits)

    # Phase-flip
    if model.phase_flip_p > 0.0 and rng.random() < model.phase_flip_p: 
        out = apply_k_qubit_gate(out, gates.Z, (qubit,), num_qubits)
        
    # Bit-flip
    if model.depolarizing_p > 0.0 and rng.random() < model.depolarizing_p:
        gate = rng.choice([gates.X, gates.Y, gates.Z])
        out = apply_k_qubit_gate(out, gate, (qubit,), num_qubits)

    return out 