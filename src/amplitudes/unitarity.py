import numpy as np

def scattering_matrix(t_array: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return 1. + 2j * rho * t_array

def unitarity_violation(t_array: np.ndarray, rho: np.ndarray) -> float:
    s = scattering_matrix(t_array, rho)
    return np.mean(np.abs(np.abs(s) - 1.))