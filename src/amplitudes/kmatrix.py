import numpy as np


def generate_kmatrix(
    s: np.ndarray,
    rho: np.ndarray,
    poles: list[tuple[float, float]],
    background: float = 0.0,
) -> np.ndarray:
    """Single-channel K-matrix: K = Σ g²/(m²−s) + c, T = K / (1 − i ρ K)."""
    k_s = np.full_like(s, background, dtype=np.complex128)

    for m, g in poles:
        k_s += (g**2) / (m**2 - s)

    return k_s / (1.0 - 1j * rho * k_s)
