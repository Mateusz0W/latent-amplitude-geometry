import numpy as np

def generate_breit_wigner(s: np.ndarray, resonances: list[tuple[float, float, float]]) -> np.ndarray:
    t_bw = np.zeros_like(s, dtype=np.complex128)
    
    for m, gamma, c in resonances:
        t_bw += c / (m**2 - s - 1j * m * gamma)
        
    return t_bw