"""
Phase 3 — turn the interaction on.

This is the one we actually care about. Same setup as Phase 2, but now
with soft-Coulomb between the particles, so we should finally see
entanglement entropy climb away from zero.
"""

# TODO: add Vint = k / sqrt((x1 - x2)**2 + a**2)
# TODO: pick values for k (interaction strength) and a (softening length)
# TODO: total potential = V(x1) + V(x2) + Vint(x1,x2)
# TODO: propagate with the same 2D split-operator as Phase 2
# TODO: reuse the entropy function from phase2
# TODO: confirm entropy grows from 0 over time
# TODO: save entropy plot + a couple density snapshots to plots/

import numpy as np
import matplotlib.pyplot as plt
 
h_bar = 1.0
m = 1.0
omega = 1.0
 
x_min = -8.0
x_max = 8.0
N = 128
 
g = 1.0     # interaction strength (k is taken by the momentum grid)
a = 0.5     # softening length
 
dt = 0.005
total_time = 5.0
num_steps = int(total_time / dt)
 
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
 
 
def entropy(Psi):
    # the dx makes the Schmidt coefficients sum to 1
    s = np.linalg.svd(Psi * dx, compute_uv=False)
    p = s**2
    p = p[p > 1e-16]
    return -np.sum(p * np.log(p))
 
 
times = [0.0]
entropies = [entropy(Psi)]
snapshots = [(0.0, np.abs(Psi)**2)]
 
for step in range(1, num_steps + 1):
 
    Psi = split(Psi)
 
    times.append(step * dt)
    entropies.append(entropy(Psi))
 
    if step % (num_steps // 5) == 0:
        snapshots.append((step * dt, np.abs(Psi)**2))
 
 
print("Entropy at start:", entropies[0])
print("Entropy at end:", entropies[-1])
print("Final norm:", np.sum(np.abs(Psi)**2) * dx * dx)
 
 
## Heat maps
 
vmax = max(density.max() for t, density in snapshots)
 
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
 
for ax, (t, density) in zip(axes.flat, snapshots):
    ax.imshow(density.T, origin="lower", extent=[x_min, x_max, x_min, x_max],
              cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title("t = " + str(round(t, 2)))
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
 
plt.tight_layout()
plt.savefig("plots/phase3_heatmap.png")
plt.close()
 
 
## Entropy vs time
 
plt.figure()
plt.plot(times, entropies)
plt.xlabel("Time")
plt.ylabel("Entanglement entropy")
plt.savefig("plots/phase3_entropy.png")
plt.close()
 