"""
Phase 2 — two particles, no interaction yet.

Point of this one is to isolate bugs: if entanglement entropy comes out
nonzero here, the bug's in the entropy code, not the physics, since a
product state literally can't entangle.
"""

import numpy as np
import matplotlib.pyplot as plt

# Setting up...
h_bar = 1.0
m = 1.0
omega = 1.0
N = 128
x_min = -10.0
x_max = 10.0
x = np.linspace(x_min,x_max,N,endpoint=False)
dx = x[1] - x[0]
k = 2.0 * np.pi * np.fft.fftfreq(N,d=dx)


## Matrix graph thing

x1,x2 = np.meshgrid(x,x,indexing="ij")
k1,k2 = np.meshgrid(k,k,indexing="ij")

# Particle 1
x1_0 = 2.0
sigma1 = 1.0 / np.sqrt(2.0)
k1_0 = 0.0

# Particle 2
x2_0 = -2.0
sigma2 = 1.0 / np.sqrt(2.0)
k2_0 = 0.0

# Wave equations

psi1 = np.exp(-(x - x1_0)**2 / (2.0 * sigma1**2)) * np.exp(1j * k1_0 * x)
psi2 = np.exp(-(x - x2_0)**2 / (2.0 * sigma2**2)) * np.exp(1j * k2_0 * x)

psi1 = psi1 / np.sqrt( np.sum(np.abs(psi1)**2) * dx )
psi2 = psi2 / np.sqrt( np.sum(np.abs(psi2)**2) * dx )
                      
PSI = psi1[:,np.newaxis] * psi2[np.newaxis,:] 

## Normalizing

PSI = PSI / np.sqrt(np.sum(np.abs(PSI)**2) * dx**2)

# Potential

V = 0.5 * m * omega**2 * (x1**2 + x2**2)

# Meshgrid

k1,k2 = np.meshgrid(k,k,indexing="ij")
KE = h_bar**2 / (2.0 * m) * (k1**2 + k2**2)
dt = 0.01
t_f = 10.0
num_steps = int(t_f / dt)
times = np.arange(num_steps + 1) * dt

# Split Operator, refer to phase 1

def Split_2d_non(PSI):
  PSI = np.exp(-1j * V * dt / (2.0 * h_bar)) * PSI
  PSI_k = np.fft.fft2(PSI)
  PSI_k = np.exp(-1j * KE * dt / h_bar) * PSI_k
  PSI = np.fft.ifft2(PSI_k)
  PSI = np.exp(-1j * V * dt / (2.0 * h_bar)) * PSI
  return PSI

# Normalize in time

normz = np.zeros(num_steps + 1)

normz[0] = np.sum(np.abs(PSI)**2) * dx**2

for step in range(1, num_steps + 1):  
  PSI = Split_2d_non(PSI)
  normz[step] = np.sum(np.abs(PSI)**2) * dx**2

# Plotting 2D Density Product State
plt.figure(figsize=(7, 6))
density_2d = np.abs(PSI)**2
plt.imshow(density_2d.T, origin="lower", extent=[x_min, x_max, x_min, x_max], cmap="viridis")
plt.xlabel(r"Particle 1 Position ($x_1$)", fontsize=12)
plt.ylabel(r"Particle 2 Position ($x_2$)", fontsize=12)
plt.title("2D Probability Density (Not Interacting)", fontsize=14)
plt.colorbar(label=r"$|\Psi(x_1, x_2)|^2$")
plt.tight_layout()
plt.savefig("plots/phase2_2d_density.png", dpi=150)
plt.close()

# Entanglement/Entropy

def Entang_Entropy(PSI,dx,cutoff=1e-20):
  schmidt = np.linalg.svd(PSI * dx, compute_uv=False)
  prob = schmidt**2
  prob = prob[prob > cutoff]
  entropy = -np.sum(prob * np.log(prob))
  return entropy

# For a more specific cse, use PSI * np.sqrt(dx1 * dx2)
# If we do use this, please define dx1 & dx2

# SVD

entropy_times = []
entropies = []
for step in range(1, num_steps + 1):
  PSI = Split_2d_non(PSI)
  normz[step] = np.sum(np.abs(PSI)**2) * dx**2
  entropy_times.append(step * dt)
  entropies.append(Entang_Entropy(PSI, dx))

# Plotting Entropy
plt.figure(figsize=(8, 4))
plt.plot(entropy_times, entropies, color="#ea0909", linewidth=2)
plt.axhline(0.0, color="black", linestyle="--", alpha=0.5)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Von Neumann Entropy", fontsize=12)
plt.title("Entanglement Check (Should be 0)", fontsize=14)
plt.ylim(-1e-15, max(1e-15, max(entropies)*1.1)) # Force y-axis to show it's near zero
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("plots/phase2_entropy_vs_time.png", dpi=150)
plt.close()
