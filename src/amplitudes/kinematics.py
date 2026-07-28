import numpy as np

def get_grid(m_pi: float, s_max: float, n_points: int, delta: float = 1e-3) -> tuple[float, np.ndarray]:
    s_th = (2 * m_pi) ** 2
    s = np.linspace(s_th + delta, s_max, n_points)

    return s_th, s

def get_phase_space(s_th: float, s: np.ndarray) -> np.ndarray:
    return np.sqrt(1 - s_th / s)

def t_to_vector(t_array: np.ndarray) -> np.ndarray:
    return np.concatenate([np.real(t_array), np.imag(t_array)])

def vector_to_t(vec: np.ndarray) -> np.ndarray:
    n = len(vec) // 2
    return vec[:n] + 1j * vec[n:]