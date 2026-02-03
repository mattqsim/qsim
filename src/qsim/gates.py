import numpy as np 

I = np.array([
    [1, 0],
    [0, 1],
], dtype=complex)

X = np.array([
    [0, 1],
    [1, 0],
], dtype=complex)

Y = np.array([
    [0, -1j],
    [1j, 0],
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1],
], dtype=complex)

H = (1 / np.sqrt(2)) * np.array([
    [1, 1],
    [1, -1],
], dtype=complex)

S = np.array([
    [1, 0],
    [0, 1j],
], dtype=complex)

T = np.array([
    [1, 0],
    [0, np.exp(1j * np.pi / 4)],
], dtype=complex)


def RZ(theta: float) -> np.ndarray:
    """
    RZ(theta) = exp(-i theta Z / 2) = [[e^{-iθ/2}, 0], [0, e^{+iθ/2}]]
    """
    t = float(theta) / 2.0
    return np.array([
        [np.exp(-1j * t), 0.0],
        [0.0, np.exp(1j * t)],
    ], dtype=complex)


def RY(theta: float) -> np.ndarray:
    """
    RY(theta) = exp(-i theta Y / 2) =
      [[cos(θ/2), -sin(θ/2)],
       [sin(θ/2),  cos(θ/2)]]
    """
    t = float(theta) / 2.0
    c = np.cos(t)
    s = np.sin(t)
    return np.array([
        [c, -s],
        [s,  c],
    ], dtype=complex)
