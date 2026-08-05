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

k = 2 * np.pi *np.fft.fftfreq(N, d=dx)

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

x1, x2 = np.meshgrid(x, x, indexing="ij")

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

K_factor = np.exp(-1j *kinetic_energy * dt/h_bar)


def do_one_step(wavefunction, potential_factor):

    wavefunction = potential_factor * wavefunction
    wavefunction = np.fft.fft2(wavefunction)
    wavefunction = K_factor * wavefunction
    wavefunction = np.fft.ifft2(wavefunction)
    wavefunction = potential_factor * wavefunction

    return wavefunction


def get_entropy(wavefunction):

    schmidt_values = np.linalg.svd(wavefunction * dx, compute_uv=False)
    probabilities = schmidt_values**2
    probabilities = probabilities[probabilities > 1e-16]
    entropy = -np.sum(probabilities * np.log(probabilities))
    
    return entropy


strengths_used = []
final_entropies = []

for strength in interaction_strengths:

    # soft coulomb interaction for this strength
    separation_squared = (x1 - x2)**2
    interaction = strength / np.sqrt(separation_squared + a**2)
    V_total = V_trap + interaction

    potential_factor = np.exp(-1j *V_total*dt / (2*h_bar))

    # fresh product state each run
    current_wavefunction = Psi.copy()

    step = 0
    while step < num_steps:
        current_wavefunction = do_one_step(current_wavefunction, potential_factor)
        step = step + 1

    entropy_at_end = get_entropy(current_wavefunction)

    strengths_used.append(strength)
    
    final_entropies.append(entropy_at_end)

    print("strength:", strength, "  final entropy:", entropy_at_end)


# entropy vs interaction strength
plt.figure(figsize=(8, 5))

plt.plot(strengths_used, final_entropies, marker="o", markersize=8, 
         linestyle="-", linewidth=2.5, color="#e377c2", markerfacecolor="#8c564b")

plt.xlabel("Interaction Strength (g)", fontsize=12)
plt.ylabel(f"Entanglement Entropy at t={total_time}", fontsize=12)
plt.title("Final Entanglement Entropy vs Interaction Strength", fontsize=14)

plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(interaction_strengths)
plt.ylim(bottom=0.0)

plt.tight_layout()
plt.savefig("plots/phase5_entropy_vs_strength.png", dpi=150)
plt.close()
