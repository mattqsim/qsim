import numpy as np 
import matplotlib.pyplot as plt 

def _basis_labels(num_qubits: int) -> list[str]:
    dim = 2 ** num_qubits
    return [format(i, f"0{num_qubits}b") for i in range(dim)]


def plot_statevector_probs(state: np.ndarray, num_qubits: int, *, title: str = "Statevector probabilities"):

    state = np.asarray(state, dtype=complex)
    probs = np.abs(state) ** 2
    labels = _basis_labels(num_qubits)

    fig = plt.figure()
    plt.bar(range(len(probs)), probs)
    plt.xticks(range(len(probs)), labels, rotation=90)
    plt.ylabel("Probability") 
    plt.title(title) 
    plt.tight_layout() 
    return fig 


def plot_circuit_state_probs(circuit, *, title: str = "Circuit state probabilities"): 
    return plot_statevector_probs(circuit.state, circuit.num_qubits, title=title) 


def plot_counts(counts: dict[str, int], *, title: str = "Measurement counts", sort: str = "bitstring"):

    if not counts: 
        raise ValueError("counts is empty")

    items = list(counts.items())
    if sort == "bitstring":
        items.sort(key=lambda kv: kv[0])
    elif sort == "count":
        items.sort(key=lambda kv: kv[1], reverse=True)
    elif sort == "none":
        pass
    else:
        raise ValueError("sort must be one of: 'bitstring', 'count', 'none'")

    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig = plt.figure() 
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=90) 
    plt.ylabel("Counts") 
    plt.title(title)
    plt.tight_layout()
    return fig 


def bloch_vector_from_state(state_1q: np.ndarray) -> np.ndarray: 

    state_1q = np.asarray(state_1q, dtype=complex).reshape(-1) 
    if state_1q.shape[0] != 2: 
        raise ValueError("bloch_vector_from_state expects a 1-qubit statevector of length 2") 

    alpha, beta= state_1q[0], state_1q[1] 


    x = 2.0 * np.real(np.conjugate(alpha) * beta) 
    y = 2.0 * np.imag(np.conjugate(alpha) * beta) 
    z = (np.abs(alpha) ** 2) - (np.abs(beta) ** 2)

    return np.array([x, y, z], dtype=float)        


def plot_bloch_sphere(state_1q: np.ndarray, *, title: str = "bloch sphere"): 

    v = bloch_vector_from_state(state_1q)
    x, y, z = v.tolist()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2*np.pi, 60) 
    t = np.linspace(0, np.pi, 60) 
    xs = np.outer(np.cos(u), np.sin(t)) 
    ys = np.outer(np.sin(u), np.sin(t))
    zs = np.outer(np.ones_like(u), np.cos(t))
    ax.plot_surface(xs, ys, zs, alpha=0.15, linewidth=0) 

    ax.plot([-1, 1], [0, 0], [0, 0])
    ax.plot([0, 0], [-1, 1], [0, 0])
    ax.plot([0, 0], [0, 0], [-1, 1])


    ax.quiver(0, 0, 0, x, y, z, length=1.0, normalize=False) 

    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1]) 
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()
    return fig 
