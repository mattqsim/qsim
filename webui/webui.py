import os, sys
import numpy as np
import streamlit as st

# --- make sure src is importable ---
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from qsim.circuit import Circuit, circuit_to_qasm, validate_with_qiskit
from qsim.parser import run_program
from qsim.noise import NoiseModel
from qsim.viz import plot_statevector_probs, plot_counts, plot_bloch_sphere
from qsim.circuit import vqe_grid_search, vqe_random_search
from qsim import gates as g




st.set_page_config(page_title="Quantum Circuit Sim (SCL Web UI)", layout="wide")

st.title("Quantum Circuit Sim — SCL Web UI")
st.caption("Statevector & Density-Matrix backends • Noise model • Measurement histograms • Bloch sphere")


# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Run settings")

    num_qubits = st.number_input("Number of qubits", min_value=1, max_value=8, value=1, step=1)
    shots = st.number_input("Shots", min_value=1, max_value=200000, value=2000, step=100)

    seed = st.number_input("Seed (optional)", min_value=0, max_value=10_000_000, value=0, step=1)
    use_seed = st.checkbox("Use seed", value=False)

    st.divider()
    st.header("Backend")
    use_density = st.checkbox("Use density matrix backend", value=False)

    st.divider()
    st.header("Noise (optional)")
    st.caption("Your qsim.noise implements: bit-flip, phase-flip, depolarizing.")

    noise_enabled = st.sidebar.checkbox("Apply noise", value=False)
    p_bitflip = st.slider("Bit-flip p", 0.0, 1.0, 0.0, 0.01)
    p_phaseflip = st.slider("Phase-flip p", 0.0, 1.0, 0.0, 0.01)
    p_depol = st.slider("Depolarizing p", 0.0, 1.0, 0.0, 0.01)

    st.divider()
    st.header("UI")
    show_state = st.checkbox("Show state", value=True)
    show_rho = st.checkbox("Show density matrix (if available)", value=True)
    show_bloch = st.checkbox("Show Bloch sphere (1 qubit only)", value=True)


# -----------------------
# SCL editor
# -----------------------
default_program = """# Example SCL
# 1 qubit:
# H 0
# RUN 2000
#
# 2 qubits:
# H 0
# CNOT 0 1
# RUN 2000

H 0
RUN 2000
"""

program = st.text_area("SCL program", value=default_program, height=260)

colA, colB = st.columns([1, 1])
with colA:
    run_clicked = st.button("Run", type="primary")
with colB:
    clear_clicked = st.button("Clear last results")

if clear_clicked:
    st.session_state.pop("last", None)
    st.success("Cleared last results.")

def _parse_h_terms(text: str) -> list[tuple[float, list[tuple[str, int]]]]:
    """Parse Hamiltonian terms from a simple text format.

    Format: one term per line:
      <coeff> <PauliIndex> [<PauliIndex> ...]

    Examples:
      1.0 Z0
      0.5 Z0 Z1
      -0.3 X0

    PauliIndex tokens look like: X0, Y1, Z2 (case-insensitive).
    Lines starting with # are ignored.
    """
    terms: list[tuple[float, list[tuple[str, int]]]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace("*", " ").split()
        if len(parts) < 2:
            raise ValueError(f"Bad Hamiltonian line: {raw!r}. Expected: <coeff> <PauliIndex> ...")
        coeff = float(parts[0])
        ops: list[tuple[str, int]] = []
        for tok in parts[1:]:
            t = tok.strip()
            if len(t) < 2:
                raise ValueError(f"Bad Pauli token {tok!r} in line {raw!r}")
            p = t[0].upper()
            q = int(t[1:])
            ops.append((p, q))
        terms.append((coeff, ops))

    if len(terms) == 0:
        raise ValueError("Hamiltonian is empty. Add at least one line like: 1.0 Z0")
    return terms





st.divider()
st.markdown("## Variational circuits (VQE-style)")

with st.expander("Run a simple VQE demo (grid / random search)", expanded=False):
    st.caption("Currently: 1-qubit ansatz RY(θ) and optional RZ(φ). Cost = ⟨H⟩ for a Hamiltonian written as Pauli terms.")

    if int(num_qubits) != 1:
        st.warning("VQE demo UI currently supports 1 qubit. Set 'Number of qubits' to 1 in the sidebar.")
    else:
        ansatz = st.selectbox("Ansatz", ["RY(θ)", "RY(θ) then RZ(φ)"], index=0)

        st.markdown("**Hamiltonian H (one term per line):**")
        h_text = st.text_area("", value="1.0 Z0", height=110, key="vqe_h")

        opt = st.selectbox("Optimizer", ["Grid search", "Random search"], index=0)

        if opt == "Grid search":
            steps = st.number_input("Grid steps per parameter", min_value=5, max_value=2001, value=361, step=10)
            lo = st.number_input("θ min", value=0.0)
            hi = st.number_input("θ max", value=float(2 * np.pi))
            if ansatz == "RY(θ) then RZ(φ)":
                lo2 = st.number_input("φ min", value=0.0)
                hi2 = st.number_input("φ max", value=float(2 * np.pi))
        else:
            iters = st.number_input("Iterations", min_value=10, max_value=20000, value=400, step=50)
            seed_vqe = st.number_input("VQE RNG seed (optional)", min_value=0, max_value=10_000_000, value=0, step=1)
            use_seed_vqe = st.checkbox("Use VQE seed", value=False)
            lo = st.number_input("θ lower", value=0.0, key="vqe_lo")
            hi = st.number_input("θ upper", value=float(2 * np.pi), key="vqe_hi")
            if ansatz == "RY(θ) then RZ(φ)":
                lo2 = st.number_input("φ lower", value=0.0, key="vqe_lo2")
                hi2 = st.number_input("φ upper", value=float(2 * np.pi), key="vqe_hi2")

        run_vqe = st.button("Run VQE", type="primary", key="run_vqe")

        if run_vqe:
            try:
                h_terms = _parse_h_terms(h_text)

                def build(params):
                    c = Circuit(1)
                    theta = float(params[0])
                    c.apply_gate(g.RY(theta), 0, label="RY")
                    if ansatz == "RY(θ) then RZ(φ)":
                        phi = float(params[1])
                        c.apply_gate(g.RZ(phi), 0, label="RZ")
                    return c

                if opt == "Grid search":
                    grid1 = np.linspace(float(lo), float(hi), int(steps))
                    grids = [grid1]
                    if ansatz == "RY(θ) then RZ(φ)":
                        grid2 = np.linspace(float(lo2), float(hi2), int(steps))
                        grids.append(grid2)
                    best_params, best_cost = vqe_grid_search(build, h_terms, grids)
                else:
                    bounds = [(float(lo), float(hi))]
                    if ansatz == "RY(θ) then RZ(φ)":
                        bounds.append((float(lo2), float(hi2)))
                    seed_arg = int(seed_vqe) if use_seed_vqe else None
                    best_params, best_cost = vqe_random_search(build, h_terms, bounds, iters=int(iters), seed=seed_arg)

                st.session_state["vqe_last"] = {
                    "ansatz": ansatz,
                    "opt": opt,
                    "h": h_text,
                    "best_params": best_params,
                    "best_cost": float(best_cost),
                }
            except Exception as e:
                st.error(f"VQE failed: {e}")

        vqe_last = st.session_state.get("vqe_last")
        if vqe_last:
            st.markdown("### Best found")
            st.write("Params:", vqe_last["best_params"])
            st.write("Cost ⟨H⟩:", vqe_last["best_cost"])

            # Show state for best params
            c_best = Circuit(1)
            theta = float(vqe_last["best_params"][0])
            c_best.apply_gate(g.RY(theta), 0, label="RY")
            if vqe_last["ansatz"] == "RY(θ) then RZ(φ)":
                phi = float(vqe_last["best_params"][1])
                c_best.apply_gate(g.RZ(phi), 0, label="RZ")

            fig = plot_statevector_probs(c_best.state, 1)
            st.pyplot(fig, clear_figure=True)


# -----------------------
# Helpers
# -----------------------
def _as_bitstring(i: int, n: int) -> str:
    return format(i, f"0{n}b")


def _probs_from_rho_diag(rho: np.ndarray) -> np.ndarray:
    """Computational-basis measurement probabilities p(x)=rho_xx (sanitised)."""
    diag = np.diag(rho)
    probs = np.real(diag).astype(float)

    # clamp tiny negatives from numerical noise
    probs[probs < 0] = 0.0

    s = float(np.sum(probs))
    if s > 0:
        probs = probs / s
    return probs


def _label_short(label: str) -> str:
    s = str(label).strip()
    # e.g. "RZ(0.123)" -> "RZ"
    if "(" in s:
        s = s[: s.find("(")]
    s = s.upper()
    # Keep it compact for ASCII diagram
    return s[:3] if len(s) > 3 else s


def circuit_ops_text(c: "Circuit") -> str:
    """Human-readable operations from c.history."""
    lines = []
    hist = getattr(c, "history", [])
    for item in hist:
        if not item:
            continue
        op = str(item[0])

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

        # 1-qubit style: (label, target)
        if len(item) == 2:
            label, target = item
            lines.append(f"{str(label)} {int(target)}")
            continue

        lines.append(f"UNHANDLED: {item!r}")

    return "\n".join(lines) + ("\n" if lines else "")


def circuit_ascii(c: "Circuit") -> str:
    """
    Very small ASCII circuit diagram from c.history.

    - 1q gates show as ─H─, ─RZ─ etc (truncated to 3 chars)
    - CNOT shows control ● and target ⊕ with vertical │ between wires
    """
    n = int(getattr(c, "num_qubits", 0))
    if n <= 0:
        return "(empty circuit)\n"

    hist = getattr(c, "history", [])
    if not hist:
        return "(no operations)\n"

    W = 5  # column width per operation chunk

    def blank():
        return "─" * W

    cols = []
    for item in hist:
        if not item:
            continue
        op = str(item[0])

        col = [blank() for _ in range(n)]

        if op == "cnot":
            _, control, target = item
            control = int(control); target = int(target)
            lo, hi = (control, target) if control <= target else (target, control)

            for q in range(lo + 1, hi):
                col[q] = "──│──"
            col[control] = "──●──"
            col[target] = "──⊕──"
            cols.append(col)
            continue

        # 1q gate recorded as (label, target)
        if len(item) == 2:
            label, target = item
            target = int(target)
            name = _label_short(label)
            # center the name in 3 chars: " H " / "RZ " etc
            mid = f"{name:^3}"
            col[target] = f"─{mid}─"
            cols.append(col)
            continue

        # fallback: show unknown op on q0 if possible
        if n > 0:
            name = _label_short(op)
            col[0] = f"─{name:^3}─"
        cols.append(col)

    # Build lines
    lines = []
    for q in range(n):
        parts = [f"q{q}:"] + [col[q] for col in cols]
        lines.append(" ".join(parts))

    return "\n".join(lines) + "\n"


def _probs_from_statevector(state: np.ndarray) -> np.ndarray:
    probs = (np.abs(state) ** 2).astype(float)
    s = float(np.sum(probs))
    if s > 0:
        probs = probs / s
    return probs


def _counts_from_probs(probs: np.ndarray, shots: int, rng: np.random.Generator, n: int) -> dict:
    idx = rng.choice(len(probs), size=shots, p=probs)
    counts = {}
    for i in idx:
        b = _as_bitstring(int(i), n)
        counts[b] = counts.get(b, 0) + 1
    return counts



def _rho_from_circuit(c: Circuit) -> np.ndarray:
    """Always return a valid density matrix (never None)."""
    if getattr(c, "backend", "statevector") == "density":
        if getattr(c, "rho", None) is not None:
            return c.rho
        # fallback: build rho from state if rho wasn't initialised
        ket = np.asarray(c.state, dtype=complex).reshape(-1, 1)
        return ket @ ket.conj().T

    ket = np.asarray(c.state, dtype=complex).reshape(-1, 1)
    return ket @ ket.conj().T



def _state_metrics_from_rho(rho: np.ndarray) -> tuple[float, float]:
    purity = float(np.real(np.trace(rho @ rho)))
    off = rho.copy()
    np.fill_diagonal(off, 0.0)
    coherence_l1 = float(np.sum(np.abs(off)))
    return purity, coherence_l1


# -----------------------
# Run logic
# -----------------------
if run_clicked:
    noise_enabled = bool(noise_enabled) and ((p_bitflip > 0.0) or (p_phaseflip > 0.0) or (p_depol > 0.0))


    backend = "density" if (use_density or noise_enabled) else "statevector"

    if use_seed:
        np.random.seed(int(seed))

    with st.spinner("Running SCL..."):
        # Build a pre-configured circuit so EXPECT/RUN inside SCL sees backend+noise.
        c0 = Circuit(int(num_qubits), backend=backend)
    
        if noise_enabled:
            c0.set_noise(
                NoiseModel(
                    bit_flip_p=float(p_bitflip),
                    phase_flip_p=float(p_phaseflip),
                    depolarizing_p=float(p_depol),
                )
            )
    
        out = run_program(
            program,
            num_qubits=int(num_qubits),
            shots=None,
            seed=int(seed) if use_seed else None,
            circuit=c0,
        )
    
        c = out["circuit"]

        st.divider()
        st.markdown("### Circuit view")

        with st.expander("Show circuit (ASCII + op list)", expanded=True):
            st.code(circuit_ascii(c), language="text")
            st.caption("Operation list (from Circuit.history)")
            st.code(circuit_ops_text(c), language="text")


        # Run measurements
        counts = c.run(int(shots))

        # Probabilities (correct for both backends)
        probs = c.probs()
        avg_probs = {
            format(i, f"0{c.num_qubits}b"): float(probs[i])
            for i in range(len(probs))
        }

        # Metrics (already implemented in Circuit)
        purity = c.purity()
        coherence_l1 = c.coherence_l1_offdiag()

        st.session_state["last"] = {
            "c": c,
            "results": out["results"],
            "counts": counts,
            "avg_probs": avg_probs,
            "purity": purity,
            "coherence_l1": coherence_l1,
            "backend": backend,
        }

# -----------------------
# Display last results
# -----------------------
if "last" in st.session_state:
    last = st.session_state["last"]
    c = last["c"]
    results = last["results"]
    counts = last["counts"]
    avg_probs = last["avg_probs"]
    purity = last["purity"]
    coherence_l1 = last["coherence_l1"]
    backend = last["backend"]

    st.subheader("Results")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Measurement histogram (counts)")
        fig_counts = plot_counts(counts)
        st.pyplot(fig_counts, clear_figure=True)

        st.markdown("### Average probabilities")
        st.json(avg_probs)

    with col2:
        st.markdown(f"### Backend: `{backend}`")

        if show_state:
            st.markdown("### State (pre-measurement)")
            if backend == "density":
                st.code("(density backend: statevector not shown)", language="text")
            else:
                # If your Circuit has pretty_state()
                if hasattr(c, "pretty_state"):
                    st.code(c.pretty_state(), language="text")
                else:
                    st.code(str(c.state), language="text")

        if show_rho and backend == "density":
            st.markdown("### Density matrix ρ")
            if hasattr(c, "pretty_rho"):
                st.code(c.pretty_rho(), language="text")
            else:
                st.code(str(c.rho), language="text")

            st.markdown("### Density probabilities (diag(ρ))")
            probs = _probs_from_rho_diag(_rho_from_circuit(c))
            probs_dict = {_as_bitstring(i, int(num_qubits)): float(probs[i]) for i in range(len(probs))}
            fig_probs = plot_statevector_probs(c.state, c.num_qubits)
            st.pyplot(fig_probs, clear_figure=True)

        st.markdown("### State quality metrics (ensemble / pre-measurement)")
        st.write(f"Purity Tr(ρ²): **{purity:.6f}**")
        st.write(f"Off-diagonal coherence L1: **{coherence_l1:.6f}**")

        if show_bloch and int(num_qubits) == 1:
            st.markdown("### Bloch sphere")

            # We can compute Bloch either via Circuit helper or directly from rho.
            if hasattr(c, "bloch_sphere") and backend != "density":
                fig_bloch, (rx, ry, rz) = c.bloch_sphere()
                st.pyplot(fig_bloch, clear_figure=True)
                st.write(f"⟨X⟩={rx:.4f}, ⟨Y⟩={ry:.4f}, ⟨Z⟩={rz:.4f}")
            else:
                # density-safe calculation: rx = Tr(rho X), etc.
                rx, ry, rz = c.expectation_xyz(0)


                fig_bloch = plot_bloch_sphere(c.state)
                st.pyplot(fig_bloch, clear_figure=True)
                st.write(f"⟨X⟩={rx:.4f}, ⟨Y⟩={ry:.4f}, ⟨Z⟩={rz:.4f}")

    st.divider()
    st.markdown("### SCL output")
    st.code(results, language="text")

    st.divider()
    st.markdown("### Interop (QASM / optional Qiskit)")

    with st.expander("Export QASM / Validate against Qiskit", expanded=False):
        qasm_text = circuit_to_qasm(c)
        st.code(qasm_text, language="text")

        st.download_button(
            label="Download QASM (.qasm)",
            data=qasm_text,
            file_name="circuit.qasm",
            mime="text/plain",
        )

        st.caption("Qiskit validation is optional. It requires: pip install qiskit qiskit-aer")

        # Only makes sense for pure unitary statevector circuits.
        if backend != "statevector":
            st.warning("Validation requires statevector backend. Disable 'Use density matrix backend' and noise.")
        else:
            if st.button("Validate with Qiskit", key="validate_qiskit_btn"):
                try:
                    report = validate_with_qiskit(c, atol=1e-6)
                    if report.get("ok"):
                        st.success("Qiskit validation: OK")
                    else:
                        st.error("Qiskit validation: MISMATCH")

                    st.write("prob max abs diff:", report.get("prob_max_abs_diff"))
                    st.write("exp  max abs diff:", report.get("exp_max_abs_diff"))

                    # exp_rows: (q, ox, oy, oz, qx, qy, qz, dx, dy, dz)
                    rows = report.get("exp_rows") or []
                    if rows:
                        st.markdown("#### Per-qubit expectations (ours vs Qiskit)")
                        st.table([
                            {
                                "q": r[0],
                                "ours ⟨X⟩": r[1], "ours ⟨Y⟩": r[2], "ours ⟨Z⟩": r[3],
                                "qiskit ⟨X⟩": r[4], "qiskit ⟨Y⟩": r[5], "qiskit ⟨Z⟩": r[6],
                                "|ΔX|": r[7], "|ΔY|": r[8], "|ΔZ|": r[9],
                            }
                            for r in rows
                        ])
                except ImportError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Validation failed: {e}")


else:
    st.info("Run a program to see results.")
