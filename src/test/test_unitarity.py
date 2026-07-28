# scripts/test_unitarity.py
import numpy as np
from src.amplitudes.kinematics import get_grid, get_phase_space
from src.amplitudes.unitarity import unitarity_violation

# Definicja parametrów (np. pion)
m_pi = 0.13957
s_max = 1.0
n_points = 64

# Krok 1: Inicjalizacja siatki i kinematyki
s_th, s = get_grid(m_pi, s_max, n_points)
rho = get_phase_space(s_th, s)

# Krok 2: Przygotowanie sztucznej unitarnej T
# Z wykorzystaniem dowolnej dynamiki przesunięcia fazowego (delta)
delta = np.linspace(0.1, 1.2, n_points)
t_unitary = (np.exp(1j * delta) * np.sin(delta)) / rho

# Krok 3: Walidacja
loss = unitarity_violation(t_unitary, rho)
print(f"Miara naruszenia unitarności: {loss:.10e}")
# Wynik w konsoli rzędu e-16 spełnia kryterium: dla sztucznej unitarnej T miara ≈ 0[cite: 1]