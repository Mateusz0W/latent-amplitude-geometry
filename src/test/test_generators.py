# scripts/test_generators.py
import numpy as np
import matplotlib.pyplot as plt

from src.amplitudes.kinematics import get_grid
from src.amplitudes.unitarity import unitarity_violation
from src.amplitudes.kmatrix import generate_kmatrix
from src.amplitudes.breit_wigner import generate_breit_wigner

m_pi = 0.13957
s_max = 1.5
n_points = 64
_s_th, s = get_grid(m_pi, s_max, n_points)

# Shared (M, Gamma); g = sqrt(M * Gamma)
resonances = [(0.7, 0.15), (1.1, 0.2)]
poles = [(M, float(np.sqrt(M * Gamma))) for M, Gamma in resonances]

t_k = generate_kmatrix(s, poles)
t_bw = generate_breit_wigner(s, resonances)

loss_k = unitarity_violation(t_k, rho=None)
loss_bw = unitarity_violation(t_bw, rho=None)
print(f"K-matrix unitarity: {loss_k:.6e}")
print(f"BW unitarity:       {loss_bw:.6e}")
print(f"poles (M, g): {poles}")

plt.figure(figsize=(15, 5))
for t, label in [(t_k, "K-matrix"), (t_bw, "Breit-Wigner")]:
    plt.subplot(1, 3, 1)
    plt.plot(s, np.real(t), label=label)
    plt.subplot(1, 3, 2)
    plt.plot(s, np.imag(t), label=label)
    plt.subplot(1, 3, 3)
    plt.plot(s, np.abs(t) ** 2, label=label)

plt.subplot(1, 3, 1)
plt.title(r"$\Re T$")
plt.xlabel("s")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.title(r"$\Im T$")
plt.xlabel("s")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.title(r"$|T|^2$")
plt.xlabel("s")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n--- Sprawdzenie kryteriów ---")
print(f"K-matrix jest unitarna (miara ~0): {loss_k < 1e-10}")
print(f"BW jest nieunitarna (miara >0): {loss_bw > 1e-2}")
