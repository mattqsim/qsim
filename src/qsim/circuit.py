import numpy as np

from .gates import I
from .states import zero_state

def expand_single_qubit_gate(gate: np.ndarray, target: int, num_qubits: int) -> np.ndarray:
    if target < 0 or target >= num_qubits:
        raise ValueError(f"target must be in [0, {num_qubits - 1}], got {target}") 

    ops = []
    for qubit in range(num_qubits): 
        if qubit == target:
            ops.append(gate)
        else:
            ops.append(I)

    big_op = ops[0]
    for op in ops[1:]:
        big_op = np.kron(big_op, op)

    return big_op


def unitary_from_hamiltonian(H: np.ndarray, t: float) -> np.ndarray:
    """
    Build U = exp(-i H t) for (assumed) Hermitian H using eigendecomposition.
    No SciPy required.
    """
    H = np.asarray(H, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be a square matrix, got shape {H.shape}")

    # For Hermitian H, eigh is stable and eigenvalues are real (up to tiny numerical noise).
    w, V = np.linalg.eigh(H)
    phases = np.exp(-1j * w * float(t))
    return V @ np.diag(phases) @ V.conj().T


class Circuit:    
    def __init__(self, num_qubits: int, *, backend: str = "statevector"):
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")

        if backend not in ("statevector", "density"): 
            raise ValueError("backend must be 'statevector' or 'density'")

        self.num_qubits = num_qubits
        self.backend = backend
        self.state = zero_state(num_qubits)

        self.rho = None
        if self.backend == "density": 
            ket = self.state.reshape(-1, 1) 
            self.rho = ket @ ket.conj().T
        
        self.history = []
        self.noise = None


    def apply_gate(self, gate, target, label=None, *, noisy: bool = True):

        big_gate = expand_single_qubit_gate(gate, target, self.num_qubits) 

        if self.backend == "statevector": 
            self.state = big_gate @ self.state
        else: 
            self.rho = big_gate @ self.rho @ big_gate.conj().T

        if label is not None:
            self.history.append((label, target))
        else:
            self.history.append(("GATE", target))

        if noisy:
           self._apply_noise([target])
    
        return self.state if self.backend == "statevector" else self.rho 


    def apply_cnot(self, control: int, target: int):
        if control == target:
           raise ValueError("control and target must be different")
        if not (0 <= control < self.num_qubits):
           raise ValueError(f"control must be in [0, {self.num_qubits - 1}], got {control}") 
        if not (0 <= target < self.num_qubits):
           raise ValueError(f"target must be in [0, {self.num_qubits - 1}], got {target}")

        n = self.num_qubits
        dim = 2 ** n

        control_mask = 1 << (n - 1 - control)
        target_mask = 1 << (n - 1 - target)

        if self.backend == "statevector":
            new_state = np.zeros_like(self.state)
            for i in range(dim):
                amp = self.state[i]
                if amp == 0:
                   continue

                if i & control_mask: 

                   j = i ^ target_mask
                   new_state[j] += amp
                else: 
                   new_state[i] += amp

            self.state = new_state
        
        else:
            U = np.zeros((dim, dim), dtype=complex) 
            for i in range(dim): 
                if i & control_mask: 
                    j = i ^ target_mask
                else:
                    j = i 
                U[j, i] = 1.0 
            self.rho = U @ self.rho @ U.conj().T
    
            
        self._apply_noise([control, target])
        self.history.append(("cnot", control, target))

        return self.state if self.backend == "statevector" else self.rho

    
    def measure_all(self) -> str: 
        p = self.probs()
        dim = len(p) 
        outcome_index = np.random.choice(dim, p=p)
        
        self._collapse_to_basis_index(outcome_index)
        bitstring = format(outcome_index, f"0{self.num_qubits}b")
        return bitstring

    
    def run(self, shots: int = 1024) -> dict:
        counts = {}
    
        for _ in range(shots):
            saved_state = self.state.copy()
            saved_rho = None
            if self.backend == "density":
                saved_rho = self.rho.copy()
    
            outcome = self.measure_all()
    
            # restore pre-measurement state
            self.state = saved_state
            if self.backend == "density":
                self.rho = saved_rho
    
            counts[outcome] = counts.get(outcome, 0) + 1
    
        return counts


    def measure_qubit(self, qubit: int) -> str:

        if not (0 <= qubit < self.num_qubits): 
            raise ValueError("qubit error")

        n = self.num_qubits
        dim = 2 ** n

        mask = 1 << (n - 1 - qubit) 

        probs0 = 0.0 
        probs1 = 0.0 
        for i in range(dim): 
            p = float(np.abs(self.state[i]) ** 2) 
            if i & mask: 
                probs1 += p 
            else: 
                probs0 += p 

        total = probs0 + probs1 
        if total == 0.0: 
            raise ValueError("Zero total probability") 

        probs0 /= total 
        probs1 /= total 

        outcome = int(np.random.choice([0, 1], p=[probs0, probs1]))

        new_state = np.zeros_like(self.state) 
        for i in range(dim): 
            bit_is_1 = bool(i & mask) 
            if bit_is_1 == (outcome ==1): 
                new_state[i] = self.state[i]

        norm = np.linalg.norm(new_state)
        if norm == 0.0:
            raise ValueError("Collapse produced zero vector") 
        new_state = new_state / norm

        self.state = new_state
        return "1" if outcome == 1 else "0"


    def measure_qubits(self, qubits: list[int]) -> str:

        for q in qubits:
            if not (0 <= q < self.num_qubits):
                raise ValueError("Invalid") 

        n = self.num_qubits
        dim = 2 ** n 

        masks = [(q, 1 << (n - 1 - q)) for q in qubits]

        probs = {}
        for i in range(dim): 
            amp = self.state[i]
            if amp == 0: 
                continue

            outcome_bits = []
            for _, mask in masks: 
                outcome_bits.append("1" if (i & mask) else "0")

            outcome = "".join(outcome_bits) 
            probs[outcome] = probs.get(outcome, 0.0) + abs(amp) ** 2

        total = sum(probs.values())
        if total == 0: 
            raise ValueError("Total probability is zero") 
        for k in probs: 
            probs[k] /= total

        outcomes = list(probs.keys())
        weights = [probs[o] for o in outcomes]
        measured = np.random.choice(outcomes, p=weights)

        new_state = np.zeros_like(self.state) 
        for i in range(dim): 
            keep = True
            for bit, (_, mask) in zip(measured, masks):
                if ((i & mask) != 0) != (bit == "1"):
                    keep = False 
                    break
            if keep: 
                new_state[i] = self.state[i]

        norm = np.linalg.norm(new_state) 
        if norm == 0: 
            raise ValueError("Collapse produced zero vector") 
        new_state /= norm 

        self.state = new_state
        return measured 


    def apply_swap(self, q1: int, q2: int): 
        if q1 == q2: 
            return self.state 
        if not (0 <= q1 < self.num_qubits) or not (0 <= q2 < self.num_qubits): 
            raise ValueError("q1 and q2 must be valid qubit indices")

        n = self.num_qubits
        dim = 2 ** n
        m1 = 1 << (n - 1 - q1) 
        m2 = 1 << (n - 1 - q2) 

        new_state = self.state.copy()
        for i in range(dim): 
            b1 = 1 if (i & m1) else 0
            b2 = 1 if (i & m2) else 0 
            if b1 != b2:
                j = i ^ (m1 | m2) 
                if i < j: 
                    new_state[i], new_state[j] = new_state[j], new_state[i] 

        self.state = new_state
        self._apply_noise([q1, q2])
        self.history.append(("SWAP", q1, q2))
  
        return self.state 


    def apply_cz(self, q1: int, q2: int): 
        if q1 == q2:
            raise ValueError("q1 and q2 must be different") 
        if not (0 <= q1 < self.num_qubits) or not (0 <= q2 < self.num_qubits):
            raise ValueError("q1 and q2 must be valid qubit indices") 

        n = self.num_qubits 
        dim = 2 ** n 
        m1 = 1 << (n - 1 - q1)
        m2 = 1 << (n - 1 - q2) 

        new_state = self.state.copy() 
        for i in range(dim): 
            if (i & m1) and (i & m2): 
                new_state[i] *= -1 

        self.state = new_state
        self._apply_noise([q1, q2])
        if hasattr(self, "history"):
            self.history.append(("CZ", q1, q2))
        return self.state


    def apply_cphase(self, q1: int, q2: int, theta: float):
        if q1 == q2:
            raise ValueError("q1 and q2 must be different")
        if not (0 <= q1 < self.num_qubits) or not (0 <= q2 < self.num_qubits):
            raise ValueError("q1 and q2 must be valid qubit indices")

        n = self.num_qubits
        dim = 2 ** n
        m1 = 1 << (n - 1 - q1)
        m2 = 1 << (n - 1 - q2)

        phase = np.exp(1j * theta)

        new_state = self.state.copy()
        for i in range(dim):
            if (i & m1) and (i & m2):
                new_state[i] *= phase

        self.state = new_state
        self._apply_noise([q1, q2])
        if hasattr(self, "history"):
            self.history.append(("CPHASE", q1, q2, float(theta)))
        return self.state


    def apply_toffoli(self, c1: int, c2: int, target: int): 
        if len({c1, c2, target}) !=3: 
            raise ValueError("c1, c2, and target must be different qubits") 
        if not (0 <= c1 < self.num_qubits) or not (0 <= c2 < self.num_qubits) or not (0 <= target < self.num_qubits): 
            raise ValueError("c1, c2, target must be valid qubit indices")


        n = self.num_qubits
        dim = 2 ** n 
        m1 = 1 << (n - 1 - c1) 
        m2 = 1 << (n - 1 - c2) 
        mt = 1 << (n - 1 - target) 

        new_state = np.zeros_like(self.state) 

        for i in range(dim): 
            amp = self.state[i]
            if amp == 0:
                continue 

            if (i & m1) and (i & m2): 
                j = i ^ mt 
                new_state[j] += amp 
            else: 
                new_state[i] += amp 

        self.state = new_state
        self._apply_noise([c1, c2, target])
        self.history.append(("TOFFOLI", c1, c2, target))
        return self.state

        """ 
        Adding describe 
        will print state vector as sum of basic kets 

        Args:
          tol: ignore amplitudes with |amp| < tol
          max terms: limit number of printed terms (None for no limit)   
        """

    def describe(self, tol: float = 1e-12, max_terms: int | None = 32): 

        terms = []
        dim = len(self.state) 

        for i in range(dim): 
            amp = self.state[i]
            if abs(amp) < tol: 
                continue
            ket = format(i, f"0{self.num_qubits}b")
            prob = (abs(amp) ** 2) 
            terms.append((i, amp, prob, ket)) 

        terms.sort(key=lambda t: t[2], reverse=True)

        if max_terms is not None: 
            terms = terms[:max_terms]

        print(f"{self.num_qubits}-qubit state |ψ⟩ with {len(terms)} shown term(s):")
        for _, amp, prob, ket in terms:

            a = complex(amp)
            print(f"  {a.real:+.6f}{a.imag:+.6f}j  |{ket}⟩   P={prob:.6f}")


    def __repr__(self):
        return f"Circuit(num_qubits={self.num_qubits}, dim={len(self.state)})"

    def set_noise(self, noise_model):
        self.noise = noise_model

    def probs(self) -> np.ndarray:
        if self.backend == "statevector":
            p = np.abs(self.state) ** 2
            s = p.sum() 
            return p / s if s != 0 else p 

        diag = np.real(np.diag(self.rho))
        s = diag.sum()
        return diag / s if s != 0 else diag 

    def _collapse_to_basis_index(self, outcome_index: int): 
        dim = 2 ** self.num_qubits 

        new_state = np.zeros(dim, dtype=complex)
        new_state[outcome_index] = 1.0 + 0.0j
        self.state = new_state 

        if self.backend == "density": 
            ket = new_state.reshape(-1, 1) 
            self.rho = ket @ ket.conj().T 

    def _apply_noise(self, qubits: list[int]):
        if self.noise is None:
            return
        from .noise import apply_noise_model
        apply_noise_model(self, qubits, self.noise)
        

    def _rho_like(self) -> np.ndarray:
        
        if getattr(self, "backend", "statevector") == "density": 
            return self.rho

        ket = self.state.reshape(-1, 1) 
        return ket @ ket.conj().T 
        

    def purity(self) -> float: 
        
        rho = self._rho_like() 
        return float(np.real(np.trace(rho @ rho))) 


    def coherence_l1_offdiag(self) -> float:

        rho = self._rho_like() 
        off = rho.copy()
        np.fill_diagonal(off, 0.0) 
        return float(np.sum(np.abs(off)))

        
    def expectation_operator(self, A: np.ndarray) -> float:
        """
        Expectation value of a full-system observable A.

        - Statevector backend: <psi|A|psi>
        - Density backend: Tr(rho A)

        Returns a real float (imag part discarded if tiny numerical noise).
        """
        A = np.asarray(A, dtype=complex)

        dim = 2 ** self.num_qubits
        if A.shape != (dim, dim):
            raise ValueError(f"A must have shape ({dim},{dim}), got {A.shape}")

        if self.backend == "statevector":
            v = np.vdot(self.state, A @ self.state)
        else:
            v = np.trace(self.rho @ A)

        return float(np.real(v))


    def evolve(self, H: np.ndarray, t: float):
        """
        Unitary time evolution under Hamiltonian H for time t.

          statevector: |psi> <- U|psi>
          density:     rho <- U rho U†

        where U = exp(-i H t).
        """
        dim = 2 ** self.num_qubits
        H = np.asarray(H, dtype=complex)
        if H.shape != (dim, dim):
            raise ValueError(f"H must have shape ({dim},{dim}), got {H.shape}")

        U = unitary_from_hamiltonian(H, t)

        if self.backend == "statevector":
            self.state = U @ self.state
        else:
            self.rho = U @ self.rho @ U.conj().T

        if hasattr(self, "history"):
            self.history.append(("EVOLVE", float(t)))

        return self.state if self.backend == "statevector" else self.rho


    def evolve_step(self, H: np.ndarray, dt: float, steps: int):
        """
        Stepped evolution: apply U_dt = exp(-i H dt) repeatedly.
        Useful for "small-dt stepping" experiments.
        """
        if steps <= 0:
            raise ValueError("steps must be positive")

        dim = 2 ** self.num_qubits
        H = np.asarray(H, dtype=complex)
        if H.shape != (dim, dim):
            raise ValueError(f"H must have shape ({dim},{dim}), got {H.shape}")

        U = unitary_from_hamiltonian(H, dt)

        if self.backend == "statevector":
            for _ in range(steps):
                self.state = U @ self.state
        else:
            Udag = U.conj().T
            for _ in range(steps):
                self.rho = U @ self.rho @ Udag

        if hasattr(self, "history"):
            self.history.append(("EVOLVE_STEP", float(dt), int(steps)))

        return self.state if self.backend == "statevector" else self.rho


    def expectation_pauli(self, pauli: str, target: int) -> float:
        """
        Expectation value of a single-qubit Pauli (X/Y/Z) on `target`,
        embedded into the full system.
        """
        if not (0 <= target < self.num_qubits):
            raise ValueError(f"target must be in [0, {self.num_qubits - 1}], got {target}")

        p = pauli.upper().strip()
        from .gates import X, Y, Z  # uses existing module

        if p == "X":
            op = X
        elif p == "Y":
            op = Y
        elif p == "Z":
            op = Z
        else:
            raise ValueError("pauli must be one of: 'X', 'Y', 'Z'")

        big = expand_single_qubit_gate(op, target, self.num_qubits)
        return self.expectation_operator(big)

    def expectation_xyz(self, target: int = 0) -> tuple[float, float, float]:
        """
        Convenience: returns (⟨X⟩, ⟨Y⟩, ⟨Z⟩) for a target qubit.
        """
        return (
            self.expectation_pauli("X", target),
            self.expectation_pauli("Y", target),
            self.expectation_pauli("Z", target),
        )
def pauli_product_operator(num_qubits: int, ops: list[tuple[str, int]]) -> np.ndarray:
    """
    Build a full-system operator for a Pauli product term.

    ops: list of (pauli, target) e.g. [("Z",0), ("Z",1)] for Z0 ⊗ Z1
    Targets use the same indexing convention as the rest of the sim.
    """
    from .gates import I, X, Y, Z

    mat_for = {"I": I, "X": X, "Y": Y, "Z": Z}

    # map target -> pauli
    by_target: dict[int, str] = {}
    for p, t in ops:
        pp = p.upper().strip()
        if pp not in mat_for:
            raise ValueError("pauli must be one of: I X Y Z")
        if t < 0 or t >= num_qubits:
            raise ValueError(f"target must be in [0, {num_qubits - 1}], got {t}")
        by_target[t] = pp

    # kron in qubit order used elsewhere (same as expand_single_qubit_gate)
    big = None
    for q in range(num_qubits):
        op = mat_for[by_target.get(q, "I")]
        big = op if big is None else np.kron(big, op)
    return big


def expectation_hamiltonian(circuit: "Circuit", terms: list[tuple[float, list[tuple[str, int]]]]) -> float:
    """
    Compute ⟨H⟩ where H = sum_k coeff_k * (Pauli-product term_k).

    terms example:
      [
        (0.5, [("Z",0)]),
        (0.8, [("Z",0), ("Z",1)]),
        (-0.3, [("X",1)]),
      ]
    """
    e = 0.0
    for coeff, ops in terms:
        A = pauli_product_operator(circuit.num_qubits, ops)
        e += float(coeff) * float(circuit.expectation_operator(A))
    return float(e)


def vqe_grid_search(
    build_circuit,
    h_terms: list[tuple[float, list[tuple[str, int]]]],
    grids: list[np.ndarray],
):
    """
    Very simple grid search over parameters.

    - build_circuit(params) -> Circuit (prepared state)
    - grids: list of 1D arrays, one per parameter

    Returns: (best_params, best_cost)
    """
    if len(grids) == 0:
        raise ValueError("grids must have at least one parameter grid")

    best_params = None
    best_cost = None

    # iterative cartesian product without itertools
    def _recurse(i: int, cur: list[float]):
        nonlocal best_params, best_cost
        if i == len(grids):
            c = build_circuit(cur)
            cost = expectation_hamiltonian(c, h_terms)
            if (best_cost is None) or (cost < best_cost):
                best_cost = cost
                best_params = cur.copy()
            return
        for v in grids[i]:
            cur.append(float(v))
            _recurse(i + 1, cur)
            cur.pop()

    _recurse(0, [])
    return best_params, float(best_cost)


def vqe_random_search(
    build_circuit,
    h_terms: list[tuple[float, list[tuple[str, int]]]],
    bounds: list[tuple[float, float]],
    iters: int = 200,
    seed: int | None = None,
):
    """
    Random search over parameters.

    bounds: [(lo, hi), ...] per parameter
    Returns: (best_params, best_cost)
    """
    if iters <= 0:
        raise ValueError("iters must be positive")
    if len(bounds) == 0:
        raise ValueError("bounds must be non-empty")

    rng = np.random.default_rng(seed)

    best_params = None
    best_cost = None

    for _ in range(int(iters)):
        params = [float(rng.uniform(lo, hi)) for (lo, hi) in bounds]
        c = build_circuit(params)
        cost = expectation_hamiltonian(c, h_terms)
        if (best_cost is None) or (cost < best_cost):
            best_cost = cost
            best_params = params

    return best_params, float(best_cost)


def _q_to_qiskit_index(q: int, n: int) -> int:
    """
    Our simulator treats qubit 0 as the MOST significant bit in basis ordering
    (see masks: 1 << (n-1-q)). Qiskit uses little-endian conventions in its
    statevector / bitstrings. To match our probabilities/bitstrings, we map:

      our qubit q  ->  qiskit qubit (n-1-q)

    This makes |q0 q1 ...> ordering align for comparisons.
    """
    return (n - 1 - int(q))


def circuit_to_qasm(c: "Circuit") -> str:
    """
    Export Circuit.history -> OpenQASM 2.0.
    This is intended for *unitary circuits* (no noise sampling, no mid-circuit measurement).

    Notes:
    - We map qubits with _q_to_qiskit_index() so that basis ordering matches our simulator.
    - Unknown/non-unitary entries (e.g. EVOLVE, *_err) are exported as QASM comments.
    """
    n = int(c.num_qubits)
    lines: list[str] = []
    lines.append('OPENQASM 2.0;')
    lines.append('include "qelib1.inc";')
    lines.append(f"qreg q[{n}];")

    hist = getattr(c, "history", [])
    for item in hist:
        if not item:
            continue

        op = str(item[0])

        # --- multi-qubit ops we store as tuples ---
        if op == "cnot":
            _, control, target = item
            qc = _q_to_qiskit_index(int(control), n)
            qt = _q_to_qiskit_index(int(target), n)
            lines.append(f"cx q[{qc}],q[{qt}];")
            continue

        if op == "SWAP":
            _, q1, q2 = item
            a = _q_to_qiskit_index(int(q1), n)
            b = _q_to_qiskit_index(int(q2), n)
            lines.append(f"swap q[{a}],q[{b}];")
            continue

        if op == "CZ":
            _, q1, q2 = item
            a = _q_to_qiskit_index(int(q1), n)
            b = _q_to_qiskit_index(int(q2), n)
            lines.append(f"cz q[{a}],q[{b}];")
            continue

        if op == "CPHASE":
            _, q1, q2, theta = item
            a = _q_to_qiskit_index(int(q1), n)
            b = _q_to_qiskit_index(int(q2), n)
            # qelib1 has cu1 / cp depending on version; QASM 2 in qelib1 supports 'cu1'
            # We'll emit 'cu1(theta) q[a],q[b];' which Qiskit accepts.
            lines.append(f"cu1({float(theta)}) q[{a}],q[{b}];")
            continue

        if op == "TOFFOLI":
            _, c1, c2, t = item
            a = _q_to_qiskit_index(int(c1), n)
            b = _q_to_qiskit_index(int(c2), n)
            tt = _q_to_qiskit_index(int(t), n)
            lines.append(f"ccx q[{a}],q[{b}],q[{tt}];")
            continue

        # --- single-qubit ops recorded as (label, target) ---
        if len(item) == 2:
            label, target = item
            label = str(label)
            target = int(target)
            qt = _q_to_qiskit_index(target, n)

            L = label.upper().strip()

            if L in ("I", "X", "Y", "Z", "H", "S", "T"):
                # QASM has no identity op; skip it
                if L == "I":
                    lines.append(f"// I q[{qt}];")
                else:
                    lines.append(f"{L.lower()} q[{qt}];")
                continue

            # RZ(theta) / RY(theta) labels are like "RZ(0.123)" from parser.py
            if L.startswith("RZ(") and L.endswith(")"):
                theta = float(label[label.find("(") + 1 : label.rfind(")")])
                lines.append(f"rz({theta}) q[{qt}];")
                continue

            if L.startswith("RY(") and L.endswith(")"):
                theta = float(label[label.find("(") + 1 : label.rfind(")")])
                lines.append(f"ry({theta}) q[{qt}];")
                continue

            # noise error labels like X_err, Z_err, etc.
            if "ERR" in L or "NOISE" in L:
                lines.append(f"// {label} q[{qt}];  (not exported as physical noise)")
                continue

            # anything else
            lines.append(f"// UNHANDLED label={label!r} target={target}")
            continue

        # --- other entries ---
        lines.append(f"// UNHANDLED history item: {item!r}")

    return "\n".join(lines) + "\n"


def circuit_to_qiskit(c: "Circuit"):
    """
    Build a qiskit.QuantumCircuit from our Circuit by going through QASM.
    Requires qiskit to be installed.
    """
    try:
        from qiskit import QuantumCircuit
    except Exception as e:
        raise ImportError(
            "Qiskit is not installed. Try:\n"
            "  pip install qiskit qiskit-aer\n"
        ) from e

    qasm = circuit_to_qasm(c)
    return QuantumCircuit.from_qasm_str(qasm)


def validate_with_qiskit(c: "Circuit", *, atol: float = 1e-6) -> dict:
    """
    Compare our simulator vs Qiskit (statevector) for:
      - probabilities in computational basis
      - <X>, <Y>, <Z> for each qubit (using our convention of qubit indices)

    This is intended for:
      - backend == "statevector"
      - no noise enabled
      - unitary gates only

    Returns a dict with diffs.
    """
    if getattr(c, "backend", "statevector") != "statevector":
        raise ValueError("validate_with_qiskit requires statevector backend (density/noise not supported here).")

    hist = getattr(c, "history", [])
    for item in hist:
        if not item:
            continue
        tag = str(item[0]).upper()
        # If user ran with noise sampling, labels like X_err/Y_err/Z_err appear
        if "ERR" in tag or "NOISE" in tag:
            raise ValueError("Circuit history contains noise error operations; disable noise for validation.")
        if tag in ("EVOLVE", "EVOLVE_STEP"):
            raise ValueError("EVOLVE/EVOLVE_STEP is not exported/validated here (different model).")

    try:
        from qiskit.quantum_info import Statevector
    except Exception as e:
        raise ImportError(
            "Qiskit is not installed. Try:\n"
            "  pip install qiskit qiskit-aer\n"
        ) from e

    qc = circuit_to_qiskit(c)
    sv = Statevector.from_instruction(qc).data  # numpy array

    # Our probs are ordered with qubit0 as MSB. With our mapping, Qiskit's sv ordering should match.
    p_qiskit = (np.abs(np.asarray(sv)) ** 2).astype(float)
    p_qiskit = p_qiskit / float(np.sum(p_qiskit)) if float(np.sum(p_qiskit)) > 0 else p_qiskit

    p_ours = np.asarray(c.probs(), dtype=float)
    p_ours = p_ours / float(np.sum(p_ours)) if float(np.sum(p_ours)) > 0 else p_ours

    prob_diff = float(np.max(np.abs(p_ours - p_qiskit)))

    # Expectations: use our existing helpers (already in Circuit)
    exp_diffs = []
    exp_rows = []
    for q in range(int(c.num_qubits)):
        ox, oy, oz = c.expectation_xyz(q)

        # compute Qiskit expectations by applying Pauli operators on the full statevector
        # We'll build big operators using our own expand_single_qubit_gate (same convention),
        # but we must apply them to the statevector that already matches our basis ordering.
        from .gates import X, Y, Z
        Xq = expand_single_qubit_gate(X, q, c.num_qubits)
        Yq = expand_single_qubit_gate(Y, q, c.num_qubits)
        Zq = expand_single_qubit_gate(Z, q, c.num_qubits)

        v = np.asarray(sv, dtype=complex)
        qx = float(np.real(np.vdot(v, Xq @ v)))
        qy = float(np.real(np.vdot(v, Yq @ v)))
        qz = float(np.real(np.vdot(v, Zq @ v)))

        dx = abs(ox - qx)
        dy = abs(oy - qy)
        dz = abs(oz - qz)
        exp_diffs.extend([dx, dy, dz])
        exp_rows.append((q, ox, oy, oz, qx, qy, qz, dx, dy, dz))

    exp_max_diff = float(max(exp_diffs)) if exp_diffs else 0.0

    ok = (prob_diff <= float(atol)) and (exp_max_diff <= float(atol))

    return {
        "ok": bool(ok),
        "atol": float(atol),
        "prob_max_abs_diff": prob_diff,
        "exp_max_abs_diff": exp_max_diff,
        "exp_rows": exp_rows,
        "qasm": circuit_to_qasm(c),
    }

# ----------------------------
# Phase 4.6 — Circuit generators (programmatic)
# ----------------------------

def _ensure_qubit_list(qubits):
    """Accept either int (count) or list[int] (explicit qubits). Return list[int]."""
    if isinstance(qubits, int):
        if qubits <= 0:
            raise ValueError("qubits (int) must be positive")
        return list(range(qubits))
    return list(qubits)


def _default_n_from_qubits(qubits: list[int]) -> int:
    if len(qubits) == 0:
        raise ValueError("qubits list is empty")
    m = max(int(q) for q in qubits)
    if m < 0:
        raise ValueError("qubit indices must be >= 0")
    return m + 1


def circuit_to_scl(c: "Circuit") -> str:
    """
    Export Circuit.history -> SCL text.

    Notes:
    - This is best-effort. Unhandled items become comments.
    - RZ/RY labels are stored like "RZ(0.123)" by parser, and we convert back.
    """
    n = int(c.num_qubits)
    lines: list[str] = []
    lines.append(f"# Generated SCL (num_qubits={n})")

    hist = getattr(c, "history", [])
    for item in hist:
        if not item:
            continue

        op = str(item[0])

        # Multi-qubit ops
        if op == "cnot":
            _, control, target = item
            lines.append(f"CNOT {int(control)} {int(target)}")
            continue

        if op == "SWAP":
            _, q1, q2 = item
            lines.append(f"SWAP {int(q1)} {int(q2)}")
            continue

        if op == "CZ":
            _, q1, q2 = item
            lines.append(f"CZ {int(q1)} {int(q2)}")
            continue

        if op == "CPHASE":
            _, q1, q2, theta = item
            lines.append(f"CPHASE {int(q1)} {int(q2)} {float(theta)}")
            continue

        if op == "TOFFOLI":
            _, c1, c2, t = item
            lines.append(f"TOFFOLI {int(c1)} {int(c2)} {int(t)}")
            continue

        # Single-qubit ops recorded as (label, target)
        if len(item) == 2:
            label, target = item
            label = str(label).strip()
            target = int(target)

            L = label.upper().strip()

            if L in ("I", "X", "Y", "Z", "H", "S", "T"):
                lines.append(f"{L} {target}")
                continue

            # Convert "RZ(theta)" -> "RZ target theta"
            if L.startswith("RZ(") and L.endswith(")"):
                theta = float(label[label.find("(") + 1 : label.rfind(")")])
                lines.append(f"RZ {target} {theta}")
                continue

            if L.startswith("RY(") and L.endswith(")"):
                theta = float(label[label.find("(") + 1 : label.rfind(")")])
                lines.append(f"RY {target} {theta}")
                continue

            # Noise / errors -> comments
            if "ERR" in L or "NOISE" in L:
                lines.append(f"# {label} {target}  (noise/error not expressible as SCL gate)")
                continue

            lines.append(f"# UNHANDLED 1q label={label!r} target={target}")
            continue

        # Other entries
        lines.append(f"# UNHANDLED history item: {item!r}")

    return "\n".join(lines) + "\n"


def bell_pair(q0: int, q1: int, *, backend: str = "statevector", as_scl: bool = False):
    """
    Build a Bell pair on qubits (q0, q1):
      H q0
      CNOT q0 q1
    Returns either Circuit or SCL text.
    """
    q0 = int(q0); q1 = int(q1)
    n = max(q0, q1) + 1

    if as_scl:
        return f"H {q0}\nCNOT {q0} {q1}\n"

    from . import gates as g
    c = Circuit(n, backend=backend)
    c.apply_gate(g.H, q0, label="H")
    c.apply_cnot(q0, q1)
    return c


def ghz(n: int, *, backend: str = "statevector", as_scl: bool = False):
    """
    Build GHZ(n):
      H 0
      CNOT 0 1
      CNOT 0 2
      ...
    Returns either Circuit or SCL text.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        # GHZ(1) is just |+> if you follow the same pattern (H on 0)
        if as_scl:
            return "H 0\n"
        from . import gates as g
        c = Circuit(1, backend=backend)
        c.apply_gate(g.H, 0, label="H")
        return c

    if as_scl:
        lines = ["H 0"]
        for t in range(1, n):
            lines.append(f"CNOT 0 {t}")
        return "\n".join(lines) + "\n"

    from . import gates as g
    c = Circuit(n, backend=backend)
    c.apply_gate(g.H, 0, label="H")
    for t in range(1, n):
        c.apply_cnot(0, t)
    return c


def uniform_superposition(qubits, *, backend: str = "statevector", as_scl: bool = False):
    """
    Apply H to a set of qubits.

    qubits can be:
      - int N  -> applies H on [0..N-1]
      - list[int] -> applies H on those indices

    Returns either Circuit or SCL text.
    """
    qs = _ensure_qubit_list(qubits)
    n = _default_n_from_qubits(qs)

    if as_scl:
        return "".join([f"H {int(q)}\n" for q in qs])

    from . import gates as g
    c = Circuit(n, backend=backend)
    for q in qs:
        c.apply_gate(g.H, int(q), label="H")
    return c


def random_clifford_ish(
    seed: int,
    *,
    num_qubits: int = 2,
    depth: int = 12,
    backend: str = "statevector",
    as_scl: bool = False,
):
    """
    OPTIONAL: a simple 'Clifford-ish' random circuit generator (not a true uniform Clifford sampler).

    - Uses only gates you already have: H, S, X, Z, plus CNOT.
    - Mixes 1q layers with occasional CNOTs.

    Returns either Circuit or SCL text.
    """
    num_qubits = int(num_qubits)
    depth = int(depth)
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")

    rng = np.random.default_rng(int(seed))
    oneq = ["H", "S", "X", "Z"]

    if as_scl:
        lines: list[str] = []
        for _ in range(depth):
            # 1q layer
            for q in range(num_qubits):
                if rng.random() < 0.6:
                    gate = oneq[int(rng.integers(0, len(oneq)))]
                    lines.append(f"{gate} {q}")

            # occasional entangler
            if num_qubits >= 2 and rng.random() < 0.5:
                a = int(rng.integers(0, num_qubits))
                b = int(rng.integers(0, num_qubits - 1))
                if b >= a:
                    b += 1
                lines.append(f"CNOT {a} {b}")

        return "\n".join(lines) + "\n"

    from . import gates as g
    c = Circuit(num_qubits, backend=backend)

    gate_map = {"H": g.H, "S": g.S, "X": g.X, "Z": g.Z}

    for _ in range(depth):
        for q in range(num_qubits):
            if rng.random() < 0.6:
                name = oneq[int(rng.integers(0, len(oneq)))]
                c.apply_gate(gate_map[name], q, label=name)

        if num_qubits >= 2 and rng.random() < 0.5:
            a = int(rng.integers(0, num_qubits))
            b = int(rng.integers(0, num_qubits - 1))
            if b >= a:
                b += 1
            c.apply_cnot(a, b)

    return c




    
        
                
