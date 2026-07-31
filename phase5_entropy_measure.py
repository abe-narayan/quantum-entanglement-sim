"""
Phase 5 — entropy vs interaction strength.

Two particles in a harmonic trap with an adjustable interaction.

This part only sets up the grid, particles, wavefunction,
potential, and kinetic energy before the split operator is added.
"""

import numpy as np
import matplotlib.pyplot as plt

h_bar = 1.0
m = 1.0
omega = 1.0

x_min = -8.0
x_max = 8.0
N = 128

x = np.linspace(x_min,  x_max, N)

dx = x[1] - x[0]

# Momentum grid

k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# Initial particle positions

x_0_1 = 2.0
x_0_2 = -2.0

sigma_0 = 1.0

# Initial particle momentum

k_0_1 = 0.0
k_0_2 = 0.0

# Create particle 1 wavefunction

psi_1 = np.exp(
    -(x -x_0_1) **2 /
    (2 * sigma_0**2)
)

psi_1 = psi_1 * np.exp(
    1j * k_0_1*x
)

# Create particle 2 wavefunction

psi_2 = np.exp(
    -(x - x_0_2) **2 /
    (2 * sigma_0 **2)
)

psi_2 = psi_2 * np.exp(
    1j * k_0_2 *x
)

# Combine both particles into one wavefunction

Psi = np.outer(
    psi_1,
    psi_2
)

# Normalize two particle wavefunction

probability = np.abs(Psi) **2

norm = np.sum(probability) * dx * dx

Psi = Psi / np.sqrt(norm)

# Create two particle coordinate grid

x1 = np.zeros((N,N))
x2 = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        x1[i][j] = x[i]
        x2[i][j] = x[j]

# Harmonic trap potential

V_particle_1 = 0.5 * m * omega **2*x1 **2

V_particle_2 = 0.5 * m * omega **2 * x2 **2

V_trap = V_particle_1 + V_particle_2

# Interaction setup

a = 0.5

interaction_strengths = [
    0.0,
    0.25,
    0.5,
    1.0,
    2.0
]

# Time setup

dt = 0.01

total_time = 5.0

num_steps = int(total_time / dt)

# Create momentum grids for kinetic energy

k1 = np.zeros((N,N))
k2 = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        k1[i][j] = k[i]
        k2[i][j] = k[j]

# Kinetic energy for both particles

kinetic_energy = (
    h_bar**2 *
    (k1**2 + k2**2)
    /
    (2*m)
)