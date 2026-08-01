"""
Classical comparison — same setup, no quantum mechanics.

Solving Newton's equations for the same trap + soft-Coulomb force, so we
have something classical to hold the quantum trajectories up against.
"""

# TODO: pull <x1>(t), <x2>(t) out of the phase3 wavefunction
# TODO: write the classical force (trap term + derivative of the softened Coulomb term)
# TODO: integrate with solve_ivp, using the same starting position/momentum as the quantum wavepacket
# TODO: plot quantum vs classical trajectories on the same axes
# TODO: note where they agree/diverge — good for the writeup
# TODO: save the comparison plot to plots/

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
total_time = 11.5 # changed total time from 5 to 11.5
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
times = [0.0]

density = np.abs(Psi)**2

x1_quantum_expected = [np.sum(x1 * density) * dx * dx] # expected positions
x2_quantum_expected = [np.sum(x2 * density) * dx * dx]

for step in range(1, num_steps + 1):

    Psi = split(Psi)

    density = np.abs(Psi)**2

    x1_quantum_expected.append(np.sum(x1 * density) * dx * dx) 
    x2_quantum_expected.append(np.sum(x2 * density) * dx * dx)

    times.append(step * dt)

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
plt.figure()

plt.plot(times, x1_quantum_expected, label="Quantum <x1>")
plt.plot(times, x2_quantum_expected, label="Quantum <x2>")
plt.plot(times, x1_classical, "--",label="Classical x1")
plt.plot(times, x2_classical, "--", label="Classical x2")
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Quantum vs Classical Two Particle Trajectories")
plt.grid()
plt.legend()
plt.savefig("plots/classical_vs_quantum_trajectories.png")
plt.close
