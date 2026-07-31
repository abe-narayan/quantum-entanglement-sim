"""
Phase 1 — single particle in a harmonic trap.

The "does this even work" sanity check. No interactions, no second
particle yet — just making sure the split-operator propagator behaves
before we build anything on top of it.
"""

# TODO: set up the grid (N points, x range, dx) and the matching k-space grid
# TODO: define V(x) = 0.5 * m * omega**2 * x**2
# TODO: build the initial Gaussian wavepacket
# TODO: implement one split-operator step (half potential, FFT, kinetic, iFFT, half potential)
# TODO: loop it over the full simulation time
# TODO: check norm stays ~1 the whole way through
# TODO: compare center-of-mass motion to the known analytic solution
# TODO: save a plot to plots/

import numpy as np
import matplotlib.pyplot as plt

h_bar = 1.0
m= 1.0
omega = 1.0
x_min = -10.0
x_max = 10.0
N = 1024

x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]

k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

x_0 = 2.0
sigma_0 = 1.0

k_0 = 0.0
psi = np.exp(-(x - x_0)**2 /(2 * sigma_0**2)) *np.exp(1j * k_0 *x)

norm = np.sum(np.abs(psi)**2) * dx
psi = psi / np.sqrt(norm)

plt.figure()
plt.plot(x, np.abs(psi)**2)
plt.xlabel("x")
plt.ylabel("|psi(x)|^2")
plt.title("Initial wavepacket")
plt.savefig("plots/initial_wavepacket.png")
plt.close()


V = 0.5 * m * omega**2 * x**2

dt =0.01
total_time = 10.0
steps = int(total_time / dt)

potential_half = np.exp(-1j * V * dt / (2* h_bar))

kinetic_energy = h_bar * k**2 / (2 * m)
kinetic = np.exp(-1j * kinetic_energy *dt/ h_bar)

times = []
center_positions = []
analytic_positions = []
norms = []

for step in range(steps):

    psi = potential_half * psi

    psi_k = np.fft.fft(psi)

    psi_k = kinetic * psi_k

    psi = np.fft.ifft(psi_k)

    psi = potential_half * psi

    probability = np.abs(psi) ** 2

    norm = np.sum(probability) * dx
    norms.append(norm)

    average_x = np.sum(x * probability)
    average_x = average_x * dx

    current_time = (step+1) * dt

    expected_x = x_0 * np.cos(omega * current_time)

    center_positions.append(average_x)
    analytic_positions.append(expected_x)
    times.append(current_time)

maximum_norm = max(norms)

minimum_norm = min(norms)


print("Maximum norm:", maximum_norm)
print("Minimum norm:", minimum_norm)
print("Norm difference:", maximum_norm - minimum_norm)


plt.figure()
plt.plot(times, center_positions, label="Simulation")
plt.plot(times, analytic_positions, "--", label="Analytic")
plt.xlabel("Time")
plt.ylabel("Center Position")
plt.legend()
plt.savefig("plots/center_of_mass.png")
plt.close()



plt.figure()
plt.plot(times, norms)
plt.xlabel("Time")
plt.ylabel("Norm")
plt.savefig("plots/norm_vs_time.png")
plt.close()



final_probability = np.abs(psi) ** 2



plt.figure()
plt.plot(x, final_probability)
plt.xlabel("x")
plt.ylabel("|psi(x)|^2")
plt.title("Final wavepacket")
plt.savefig("plots/final_wavepacket.png")
plt.close()