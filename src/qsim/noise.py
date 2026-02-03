import numpy as np
from dataclasses import dataclass
from .circuit import expand_single_qubit_gate


@dataclass
class NoiseModel:
    bit_flip_p: float = 0.0       # X with probability p
    phase_flip_p: float = 0.0     # Z with probability p
    depolarizing_p: float = 0.0   # apply random {X,Y,Z} with prob p


def _validate_p(p: float, name: str):
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"{name} must be in [0,1], got {p}")


def apply_bit_flip(circuit, qubit: int, p: float):
    _validate_p(p, "bit_flip_p")
    if np.random.random() < p:
        from .gates import X
        circuit.apply_gate(X, qubit, label="X_err", noisy=False)


def apply_phase_flip(circuit, qubit: int, p: float):
    _validate_p(p, "phase_flip_p")
    if np.random.random() < p:
        from .gates import Z
        circuit.apply_gate(Z, qubit, label="Z_err", noisy=False)


def apply_depolarizing(circuit, qubit: int, p: float):
    _validate_p(p, "depolarizing_p")
    if np.random.random() < p:
        from .gates import X, Y, Z
        choices = [(X, "X_err"), (Y, "Y_err"), (Z, "Z_err")]
        gate, lab = choices[np.random.randint(0, 3)]

        circuit.apply_gate(gate, qubit, label=lab, noisy=False)


def apply_noise_model(circuit, qubits: list[int], model: NoiseModel):
    if model is None:
        return

    _validate_p(model.bit_flip_p, "bit_flip_p")
    _validate_p(model.phase_flip_p, "phase_flip_p")
    _validate_p(model.depolarizing_p, "depolarizing_p")

    if getattr(circuit, "backend", "statevector") == "density":
        from .gates import X, Y, Z

        for q in qubits:
            rho = circuit.rho

            # Bit-flip channel: (1-p) rho + p X rho X
            if model.bit_flip_p > 0.0:
                p = float(model.bit_flip_p)
                Xq = expand_single_qubit_gate(X, q, circuit.num_qubits)
                rho = (1.0 - p) * rho + p * (Xq @ rho @ Xq.conj().T)

            # Phase-flip channel: (1-p) rho + p Z rho Z
            if model.phase_flip_p > 0.0:
                p = float(model.phase_flip_p)
                Zq = expand_single_qubit_gate(Z, q, circuit.num_qubits)
                rho = (1.0 - p) * rho + p * (Zq @ rho @ Zq.conj().T)

            # Depolarizing channel: (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)
            if model.depolarizing_p > 0.0:
                p = float(model.depolarizing_p)
                Xq = expand_single_qubit_gate(X, q, circuit.num_qubits)
                Yq = expand_single_qubit_gate(Y, q, circuit.num_qubits)
                Zq = expand_single_qubit_gate(Z, q, circuit.num_qubits)

                rho = (1.0 - p) * rho + (p / 3.0) * (
                    (Xq @ rho @ Xq.conj().T) +
                    (Yq @ rho @ Yq.conj().T) +
                    (Zq @ rho @ Zq.conj().T)
                )

            circuit.rho = rho

        return

    # Otherwise (statevector backend): sample errors by applying gates randomly
    for q in qubits:
        if model.bit_flip_p > 0:
            apply_bit_flip(circuit, q, model.bit_flip_p)
        if model.phase_flip_p > 0:
            apply_phase_flip(circuit, q, model.phase_flip_p)
        if model.depolarizing_p > 0:
            apply_depolarizing(circuit, q, model.depolarizing_p)
