import numpy as np

from .circuit import Circuit, expand_single_qubit_gate
from . import gates as g

GATE_MAP = {
    "I": g.I,
    "X": g.X,
    "Y": g.Y,
    "Z": g.Z,
    "H": g.H,
    "S": g.S,
    "T": g.T,
}


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def run_program(
    program: str,
    num_qubits: int,
    shots: int | None = None,
    seed: int | None = None,
    circuit: Circuit | None = None,
) -> dict:
    """
    SCL - Simple circuit language
    """
    
    if seed is not None:
        np.random.seed(seed)

    c = circuit if circuit is not None else Circuit(num_qubits)

    if c.num_qubits != num_qubits:
        raise ValueError(f"Provided circuit has num_qubits={c.num_qubits}, but num_qubits={num_qubits} was requested")

    results: list[tuple[str, str]] = []
    counts: dict[str, int] = {}

    """
    SCL - Simple circuit language

    Supported instructions:
      - 1-qubit gates: I/X/Y/Z/H/S/T <target>
      - parameterized: RZ <target> <theta>
                      RY <target> <theta>
      - CNOT <control> <target>
      - SWAP <q1> <q2>
      - CZ <q1> <q2>
      - CPHASE <q1> <q2> <theta>
      - TOFFOLI <q1> <q2> <target>
      - MEASUREALL
      - MEASURE  <q>
      - MEASUREQS <q0> <q1> ...
      - EXPECT <X|Y|Z> <target>
      - EVOLVE / EVOLVE_STEP ...
      - RUN <shots>
      - RESET
      - PRINTSTATE
    """

    lines = program.splitlines()

    for raw in lines:
        line = _strip_comment(raw)
        if not line:
            continue

        parts = line.split()
        op = parts[0].upper()
        args = parts[1:]

        if op in GATE_MAP:
            if len(args) != 1:
                raise ValueError(f"{op} expects 1 arg: {op} <target>")
            target = int(args[0])
            c.apply_gate(GATE_MAP[op], target, label=op)
            continue

        if op in ("RZ", "RY"):
            if len(args) != 2:
                raise ValueError(f"{op} expects 2 args: {op} <target> <theta>")

            target = int(args[0])
            theta = float(args[1])

            if op == "RZ":
                gate = g.RZ(theta)
            else:
                gate = g.RY(theta)

            c.apply_gate(gate, target, label=f"{op}({theta})")
            continue

        if op == "CNOT":
            if len(args) != 2:
                raise ValueError("CNOT expect 2 args: CNOT <control> <target>")
            c.apply_cnot(int(args[0]), int(args[1]))
            continue

        if op == "SWAP":
            if len(args) != 2:
                raise ValueError("SWAP expect 2 args: SWAP <q1> <q2>")
            c.apply_swap(int(args[0]), int(args[1]))
            continue

        if op == "CZ":
            if len(args) != 2:
                raise ValueError("CZ expect 2 args: CZ <q1> <q2>")
            c.apply_cz(int(args[0]), int(args[1]))
            continue

        if op == "CPHASE":
            if len(args) != 3:
                raise ValueError("CPHASE expects 3 args: CPHASE <q1> <q2> <theta>")
            q1, q2 = int(args[0]), int(args[1])
            theta = float(args[2])
            c.apply_cphase(q1, q2, theta)
            continue

        if op == "TOFFOLI":
            if len(args) != 3:
                raise ValueError("TOFFOLI expects 3 args: TOFFOLI <c1> <c2> <target>")
            c.apply_toffoli(int(args[0]), int(args[1]), int(args[2]))
            continue

        if op == "MEASUREALL":
            r = c.measure_all()
            results.append(("MEASUREALL", r))
            continue

        if op == "MEASURE":
            if len(args) != 1:
                raise ValueError("MEASURE expects 1 arg: MEASURE <q>")
            q = int(args[0])
            r = c.measure_qubit(q)
            results.append((f"MEASURE {q}", r))
            continue

        if op == "MEASUREQS":
            if len(args) < 1:
                raise ValueError("MEASUREQS expects >=1 qubit: MEASUREQS <q0> <q1> ...")
            qs = [int(a) for a in args]
            r = c.measure_qubits(qs)
            results.append((f"MEASURE {qs}", r))
            continue

        if op == "RUN":
            if len(args) != 1:
                raise ValueError("RUN expects 1 arg: RUN <shots>")
            s = int(args[0])
            counts = c.run(s)
            continue

        if op == "RESET":
            c.state = np.zeros_like(c.state)
            c.state[0] = 1.0 + 0.0j

            if hasattr(c, "history"):
                c.history.append(("RESET",))
            continue

        if op == "PRINTSTATE":
            if hasattr(c, "describe"):
                c.describe()
            else:
                print("describe() not available")
            continue

        if op == "EXPECT":
            # EXPECT <X|Y|Z> <target>
            if len(args) != 2:
                raise ValueError("EXPECT expects 2 args: EXPECT <X|Y|Z> <target>")
            pauli = args[0].upper()
            q = int(args[1])
            if hasattr(c, "expectation_pauli"):
                val = c.expectation_pauli(pauli, q)
            else:
                raise ValueError("Circuit is missing expectation_pauli()")
            results.append((f"EXPECT {pauli} {q}", f"{val:.6f}"))
            continue

        if op == "EVOLVE":
            # EVOLVE <X|Y|Z> <target> <t>
            # or: EVOLVE <X|Y|Z> <target> <scale> <t>  (H = scale * Pauli(target))
            if len(args) not in (3, 4):
                raise ValueError("EVOLVE expects 3 or 4 args: EVOLVE <X|Y|Z> <target> <t>  OR  EVOLVE <X|Y|Z> <target> <scale> <t>")

            p = args[0].upper()
            q = int(args[1])

            if p == "X":
                base = g.X
            elif p == "Y":
                base = g.Y
            elif p == "Z":
                base = g.Z
            else:
                raise ValueError("EVOLVE pauli must be one of: X Y Z")

            if len(args) == 3:
                scale = 1.0
                t = float(args[2])
            else:
                scale = float(args[2])
                t = float(args[3])

            H = scale * expand_single_qubit_gate(base, q, num_qubits)
            c.evolve(H, t)
            results.append((f"EVOLVE {p} {q}", f"t={t}"))
            continue

        if op == "EVOLVE_STEP":
            # EVOLVE_STEP <X|Y|Z> <target> <dt> <steps>
            # or: EVOLVE_STEP <X|Y|Z> <target> <scale> <dt> <steps>
            if len(args) not in (4, 5):
                raise ValueError("EVOLVE_STEP expects 4 or 5 args: EVOLVE_STEP <X|Y|Z> <target> <dt> <steps>  OR  EVOLVE_STEP <X|Y|Z> <target> <scale> <dt> <steps>")

            p = args[0].upper()
            q = int(args[1])

            if p == "X":
                base = g.X
            elif p == "Y":
                base = g.Y
            elif p == "Z":
                base = g.Z
            else:
                raise ValueError("EVOLVE_STEP pauli must be one of: X Y Z")

            if len(args) == 4:
                scale = 1.0
                dt = float(args[2])
                steps = int(args[3])
            else:
                scale = float(args[2])
                dt = float(args[3])
                steps = int(args[4])

            H = scale * expand_single_qubit_gate(base, q, num_qubits)
            c.evolve_step(H, dt, steps)
            results.append((f"EVOLVE_STEP {p} {q}", f"dt={dt}, steps={steps}"))
            continue


        raise ValueError(f"Unknown instruction: {op}")

    if shots is not None and not counts:
        counts = c.run(shots)

    return {"circuit": c, "results": results, "counts": counts}
