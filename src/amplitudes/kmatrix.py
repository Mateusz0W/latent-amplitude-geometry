import numpy as np

def generate_kmatrix(s: np.ndarray, rho: np.ndarray, poles: list[tuple[float, float]]) -> np.ndarray:
    k_s = np.zeros_like(s, dtype=np.complex128)
    
    for m, g in poles:
        k_s += (g**2) / (m**2 - s)
        
    t_k = k_s / (1.0 - 1j * rho * k_s)
    
    return t_k