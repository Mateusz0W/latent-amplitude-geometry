# scripts/test_generators.py
import numpy as np
import matplotlib.pyplot as plt

from src.amplitudes.kinematics import get_grid, get_phase_space
from src.amplitudes.unitarity import unitarity_violation
from src.amplitudes.kmatrix import generate_kmatrix
from src.amplitudes.breit_wigner import generate_breit_wigner
from src.amplitudes.mix import generate_mix

# 1. Konfiguracja siatki
m_pi = 0.13957
s_max = 1.5
n_points = 64
s_th, s = get_grid(m_pi, s_max, n_points)
rho = get_phase_space(s_th, s)

# 2. Definicja parametrów rezonansów (zabawkowy model)
# K-matrix: 2 bieguny (masa, sprzężenie)
poles_k = [(0.7, 0.4), (1.1, 0.5)]

# Breit-Wigner: 2 rezonanse (masa, szerokość, amplituda sprzężenia)
res_bw = [(0.7, 0.15, 0.5), (1.1, 0.2, 0.4)]

# 3. Generacja amplitud
t_k = generate_kmatrix(s, rho, poles_k)
t_bw = generate_breit_wigner(s, res_bw)

# 4. Obliczenie miary unitarności dla różnych wartości lambda
lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
losses = []

plt.figure(figsize=(15, 5))

for lam in lambdas:
    t_mix = generate_mix(t_k, t_bw, lam)
    loss = unitarity_violation(t_mix, rho)
    losses.append(loss)
    print(f"Lambda {lam:.2f} | Miara unitarności: {loss:.6f}")
    
    # Rysowanie sanity-plotów
    plt.subplot(1, 3, 1)
    plt.plot(s, np.real(t_mix), label=f'$\lambda={lam}$')
    plt.title('Część Rzeczywista ($\Re T$)')
    plt.xlabel('s')
    
    plt.subplot(1, 3, 2)
    plt.plot(s, np.imag(t_mix), label=f'$\lambda={lam}$')
    plt.title('Część Urojona ($\Im T$)')
    plt.xlabel('s')
    
    plt.subplot(1, 3, 3)
    plt.plot(s, np.abs(t_mix)**2, label=f'$\lambda={lam}$')
    plt.title('Moduł do kwadratu ($|T|^2$)')
    plt.xlabel('s')

# Formatowanie wykresów
for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
# Zapisz wykres do pliku (upewnij się, że masz folder results/sanity/)
# plt.savefig('results/sanity/generators_plot.png') 
plt.show()

# Weryfikacja warunków MVP
print("\n--- Sprawdzenie kryteriów ---")
print(f"K-matrix jest unitarna (miara ~0): {losses[0] < 1e-10}")
print(f"BW jest nieunitarna (miara >0): {losses[-1] > 1e-2}")
print(f"Miara rośnie z lambda: {all(x < y for x, y in zip(losses, losses[1:]))}")