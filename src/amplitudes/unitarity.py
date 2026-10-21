import numpy as np


def scattering_matrix(t_array: np.ndarray, rho: np.ndarray | float | None = None) -> np.ndarray:
    """S = 1 + 2i ρ T. If rho is None, uses the convention S = 1 + 2i T (as in T = K/(1−iK))."""
    if rho is None:
        return 1.0 + 2j * t_array
    return 1.0 + 2j * rho * t_array


def unitarity_violation(
    t_array: np.ndarray, rho: np.ndarray | float | None = None
) -> float:
    s = scattering_matrix(t_array, rho)
    return float(np.mean(np.abs(np.abs(s) - 1.0)))
