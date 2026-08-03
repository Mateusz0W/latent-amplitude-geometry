import numpy as np

def generate_mix(t_k: np.ndarray, t_bw: np.ndarray, lam: float) -> np.ndarray:
    return (1.0 - lam) * t_k + lam * t_bw