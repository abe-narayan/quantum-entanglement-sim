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
sigma_0 = 1.0/np.sqrt(2.0)

k_0 = 0.0
psi = np.exp(-(x - x_0)**2 / (2 * sigma_0**2)) * np.exp(1j * k_0 * x)

norm = np.sum(np.abs(psi)**2) * dx
psi = psi / np.sqrt(norm)

plt.figure()
plt.plot(x, np.abs(psi)**2)
plt.xlabel("x")
plt.ylabel("|psi(x)|^2")
plt.title("Initial wavepacket")
plt.savefig("plots/initial_wavepacket.png")
plt.close()