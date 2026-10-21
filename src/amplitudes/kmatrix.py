import numpy as np


def K(s: np.ndarray, g: float, M: float) -> np.ndarray:
    """Single K-matrix pole: K = g² / (M² − s)."""
    return (g**2) / (M**2 - s)


def generate_kmatrix(s: np.ndarray, poles: list[tuple[float, float]]) -> np.ndarray:
    """K-matrix amplitude: T = K / (1 − i K), K = Σ g_r² / (M_r² − s)."""
    k_s = np.zeros_like(s, dtype=np.complex128)
    for M, g in poles:
        k_s += K(s, g, M)
    return k_s / (1.0 - 1j * k_s)
