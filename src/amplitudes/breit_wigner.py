import numpy as np


def BWA(s: np.ndarray, M: float, Gamma: float) -> np.ndarray:
    """Single Breit–Wigner amplitude: M Γ / (M² − s − i M Γ)."""
    return (M * Gamma) / (M**2 - s - 1j * M * Gamma)


def generate_breit_wigner(
    s: np.ndarray, resonances: list[tuple[float, float]]
) -> np.ndarray:
    """Sum of Breit–Wigner amplitudes: T = Σ BWA(s, M_r, Γ_r)."""
    t_bw = np.zeros_like(s, dtype=np.complex128)
    for M, Gamma in resonances:
        t_bw += BWA(s, M, Gamma)
    return t_bw
