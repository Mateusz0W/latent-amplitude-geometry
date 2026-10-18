import numpy as np


def generate_nd(
    s: np.ndarray,
    rho: np.ndarray,
    poles: list[tuple[float, float]],
    background: float = 0.0,
    slope: float = 0.0,
) -> np.ndarray:
    """Toy one-channel N/D with the same poles as a paired K-matrix.

    N(s) = Σ g_r²/(m_r² − s) + c + b·s,  T = N / (1 − i ρ N).

    Same unitaryization family as K-matrix, but a linear term makes the
    on-shell amplitude a different parameterization of related physics (E3).
    """
    n_s = np.full_like(s, background, dtype=np.complex128)
    n_s = n_s + slope * s

    for m, g in poles:
        n_s += (g**2) / (m**2 - s)

    return n_s / (1.0 - 1j * rho * n_s)
