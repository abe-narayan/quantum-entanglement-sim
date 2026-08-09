"""
Classical comparison — same setup, no quantum mechanics.

Solving Newton's equations for the same trap + soft-Coulomb force, so we
have something classical to hold the quantum trajectories up against.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

h_bar = 1.0
m = 1.0
omega = 1.0
 
x_min = -8.0
x_max = 8.0
N = 128
 
g = 1.0     # interaction strength (k is taken by the momentum grid)
a = 0.5     # softening length
 
dt = 0.005
total_time = 12.0
num_steps = int(total_time / dt)

#copy of simulation from phase 3
# endpoint=False so the domain matches the period the FFT assumes
x = np.linspace(x_min, x_max, N, endpoint=False)
dx = x[1] - x[0]
 
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
 
x1, x2 = np.meshgrid(x, x, indexing="ij")
k1, k2 = np.meshgrid(k, k, indexing="ij")
 
 
## Initial state, a product so entropy starts at zero
 
psi_1 = np.exp(-(x - 2.0)**2 / 2)
psi_2 = np.exp(-(x + 2.0)**2 / 2)
 
Psi = np.outer(psi_1, psi_2)
Psi = Psi / np.sqrt(np.sum(np.abs(Psi)**2) * dx * dx)
 
 
## Operators
 
V = 0.5 * m * omega**2 * (x1**2 + x2**2) + g / np.sqrt((x1 - x2)**2 + a**2)
 
kinetic_energy = h_bar**2 * (k1**2 + k2**2) / (2 * m)
 
PV = np.exp(-1j * V * dt / (2 * h_bar))
K_factor = np.exp(-1j * kinetic_energy * dt / h_bar)
 
 
def split(Psi):
    Psi = PV * Psi
    Psi = np.fft.ifft2(K_factor * np.fft.fft2(Psi))
    Psi = PV * Psi
    return Psi
times = np.arange(num_steps + 1) * dt

x1_quantum_expected = np.empty(num_steps + 1)
x2_quantum_expected = np.empty(num_steps + 1)

density = np.abs(Psi)**2

x1_quantum_expected[0] = np.sum(x1 * density) * dx * dx
x2_quantum_expected[0] = np.sum(x2 * density) * dx * dx

for step in range(1, num_steps + 1):

    Psi = split(Psi)

    density = np.abs(Psi)**2

    x1_quantum_expected[step] = (np.sum(x1 * density) * dx * dx)

    x2_quantum_expected[step] = (np.sum(x2 * density) * dx * dx)


def equations_of_motion(t, y): # required by solve_ivp, sets up dy/dt=f(t,y)

    x1c = y[0]
    v1 = y[1]
    x2c = y[2]
    v2 = y[3]

    dx = x1c - x2c

    interaction = g * dx / (dx**2 + a**2)**1.5

    F1 = -m * omega**2 * x1c + interaction # classical force, trap+interaction
    F2 = -m * omega**2 * x2c - interaction

    a1 = F1 / m
    a2 = F2 / m

    return [v1, a1, v2, a2]

initial = [2.0, 0.0, -2.0, 0.0] #initial position and momentum

solution = solve_ivp(equations_of_motion, [0, total_time], initial, t_eval=times)

x1_classical = solution.y[0] 
x2_classical = solution.y[2]

# comparison plot
plt.figure(figsize=(9, 5))

plt.plot(times, x1_quantum_expected, label=r"Quantum $\langle x_1 \rangle$", color="#1f77b4", linewidth=4, alpha=0.6)
plt.plot(times, x2_quantum_expected, label=r"Quantum $\langle x_2 \rangle$", color="#ff7f0e", linewidth=4, alpha=0.6)

plt.plot(times, x1_classical, "--", label=r"Classical $x_1$", color="black", linewidth=1.5)
plt.plot(times, x2_classical, "-.", label=r"Classical $x_2$", color="black", linewidth=1.5)

plt.xlabel("Time", fontsize=12)
plt.ylabel("Position", fontsize=12)
plt.title("Quantum Expectation vs. Classical Trajectories", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right", frameon=True, shadow=True, fontsize=10)
plt.tight_layout()
plt.savefig("plots/classical_vs_quantum_trajectories.png", dpi=150)
plt.close()
