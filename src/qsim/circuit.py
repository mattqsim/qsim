import numpy as np
from dataclasses import dataclass
from typing import Optional 

from . import gates

@dataclass(frozen=True) 
class Operation: 
    """
    Single circuit operation: apply unitary U to the target of given qubits
    no state mutation just discriptive. 
    """

    U: np.ndarray
    targets: tuple[int, ...]
    label: str = "" 


class Circuit: 
    """
    List of operations, No physics executed in the new circuit file.
    Physics will now happen in simulator using apply_k_qubit_gate(). 
    """ 

    def __init__(self, num_qubits: int): 
        if num_qubits <=0: 
            raise ValueError("Must be at least 1 qubit")
        self.num_qubits = int(num_qubits)
        self.ops: list[Operation] = [] 

    def add_gate(self, U: np.ndarray, targets: list[int] | tuple[int, ...], *, label: str = "") -> None:
        U = np.asarray(U, dtype=complex) 

        # Target validation 
        t = tuple(int(q) for q in targets) 
        if len(t) == 0: 
            raise ValueError("targets cannot be empty") 
        if len(set(t)) != len(t): 
            raise ValueError(f"targets must be unique, got {t}") 
        if any((q < 0 or q >= self.num_qubits) for q in t): 
            raise ValueError(f"target out of range for num_qubits={self.num_qubits}: {t}") 

        self.ops.append(Operation(U=U, targets=t, label=label))

    # Gate Wrappers 

    # 1-qubit 
    def i(self, q: int): self.add_gate(gates.I, [q], label="I") 
    def x(self, q: int): self.add_gate(gates.X, [q], label="X")  
    def y(self, q: int): self.add_gate(gates.Y, [q], label="Y")  
    def z(self, q: int): self.add_gate(gates.Z, [q], label="Z") 
    def h(self, q: int): self.add_gate(gates.H, [q], label="H") 
    def s(self, q: int): self.add_gate(gates.S, [q], label="S") 
    def t(self, q: int): self.add_gate(gates.T, [q], label="T") 
    
    def rz(self, q: int, theta: float): self.add_gate(gates.RZ(theta), [q], label=f"RZ({theta})")
    def ry(self, q: int, theta: float): self.add_gate(gates.RY(theta), [q], label=f"RY({theta})")
    def rx(self, q: int, theta: float): self.add_gate(gates.RX(theta), [q], label=f"RX({theta})")

    # 2-qubit 
    def cx(self, control: int, target: int): self.add_gate(gates.CNOT, [control, target], label="CNOT") 
    def cz(self, control: int, target: int): self.add_gate(gates.CZ, [control, target], label="CZ") 
    def swap(self, a: int, b: int): self.add_gate(gates.SWAP, [a, b], label="SWAP")
    def cs(self, control: int, target: int): 
        self.add_gate(gates.CS, [control, target], label="CS") 
    def cp(self, control: int, target: int, theta: float): 
        self.add_gate(gates.CP(theta), [control, target], label=f"CP({theta})") 

    # 3-qubit
    def tof(self, c1: int, c2: int, target: int): self.add_gate(gates.TOF, [c1, c2, target], label="TOF") 
    def fred(self, control: int, a: int, b: int): self.add_gate(gates.FRED, [control, a, b], label="FRED") 