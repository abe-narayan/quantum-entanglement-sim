"""
Phase 3 — turn the interaction on.

This is the one we actually care about. Same setup as Phase 2, but now
with soft-Coulomb between the particles, so we should finally see
entanglement entropy climb away from zero.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
hbar = 1.0
m = 1.0
omega = 1.0
 
x_min = -8.0
x_max = 8.0
N = 128
 
g = 1.0    # interaction strength
a = 0.5    # softening length
 
dt = 0.005
num_steps = 2400
sample_every = 20
 
# endpoint=False so the domain matches the period the FFT assumes
x = np.linspace(x_min, x_max, N, endpoint=False)
dx = x[1] - x[0]
dA = dx * dx
k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
 
# (N,1) against (1,N) broadcasts to the full grid without storing meshgrids
x1, x2 = x[:, None], x[None, :]
k1, k2 = k[:, None], k[None, :]
 
V = 0.5 * m * omega**2 * (x1**2 + x2**2) + g / np.sqrt((x1 - x2)**2 + a**2)
T = hbar**2 * (k1**2 + k2**2) / (2.0 * m)
 
half_V = np.exp(-0.5j * V * dt / hbar)
full_K = np.exp(-1j * T * dt / hbar)
 
 
def split_step(psi):
    psi = half_V * psi
    psi = np.fft.ifft2(full_K * np.fft.fft2(psi))
    return half_V * psi
 
 
def entropy(psi):
    # psi * dx makes the Schmidt coefficients sum to 1
    p = np.linalg.svd(psi * dx, compute_uv=False)**2
    p = p[p > 1e-16]
    return -np.sum(p * np.log(p))
 
 
def energy(psi):
    weight = np.abs(np.fft.fft2(psi))**2
    return np.sum(T * weight) / np.sum(weight) + np.sum(V * np.abs(psi)**2) * dA
 
 
# product of two displaced coherent states, so the entropy starts at zero
Psi = np.outer(np.exp(-(x - 2.0)**2 / 2), np.exp(-(x + 2.0)**2 / 2)).astype(complex)
Psi /= np.sqrt(np.sum(np.abs(Psi)**2) * dA)
 
n_samples = num_steps // sample_every + 1
times = np.arange(n_samples) * sample_every * dt
entropies = np.empty(n_samples)
density = np.empty((n_samples, N, N))
 
entropies[0] = entropy(Psi)
density[0] = np.abs(Psi)**2
energy_start = energy(Psi)
 
for step in range(1, num_steps + 1):
    Psi = split_step(Psi)
    if step % sample_every == 0:
        i = step // sample_every
        entropies[i] = entropy(Psi)
        density[i] = np.abs(Psi)**2
 
print(f"norm:    {np.sum(density[-1]) * dA:.12f}")
print(f"energy:  {energy_start:.6f} -> {energy(Psi):.6f}")
print(f"entropy: {entropies[0]:.2e} -> {entropies[-1]:.4f}")
 
fig = plt.figure(figsize=(7, 6))
image = plt.imshow(density[0].T, origin="lower", extent=[x_min, x_max, x_min, x_max],
                  cmap="magma", vmin=0.0, vmax=density.max())
plt.xlabel(r"Particle 1 Position ($x_1$)", fontsize=12)
plt.ylabel(r"Particle 2 Position ($x_2$)", fontsize=12)
title = plt.title("Interacting Wavepacket: t = 0.00", fontsize=14)
plt.colorbar(label=r"$|\Psi(x_1, x_2)|^2$")
plt.tight_layout()

def update(i):
    image.set_data(density[i].T)
    title.set_text(f"Interacting Wavepacket: t = {times[i]:.2f}")
    return image, title

ani = animation.FuncAnimation(fig, update, frames=n_samples, interval=100)
ani.save("plots/phase3_density.gif", writer="pillow", fps=15)
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(times, entropies, color="#9467bd", linewidth=2.5)
plt.fill_between(times, entropies, color="#9467bd", alpha=0.2)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Von Neumann Entropy", fontsize=12)
plt.title("Growth of Entanglement via Soft Coulomb Interaction", fontsize=14)
plt.xlim(0.0, times[-1])
plt.ylim(bottom=0.0)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("plots/phase3_entropy.png", dpi=150)
plt.close()
 