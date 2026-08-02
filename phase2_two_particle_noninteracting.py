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
x_min = -10
x_max = 10
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
psi2 = np.exp(-(x - x2_0)**2 / (2.0 * sigma1**2)) * np.exp(1j * k2_0 * x)

psi1 = psi1 / np.sqrt( np.sum(np.abs(psi1)**2) * dx
psi2 = psi2 / np.sqrt( np.sum(np.abs(psi2)**2) * dx
                      
PSI = psi1[:,np.newaxis] * psi2[np.newaxis,:] 

## Normalizing

PSI = PSI / np.sqrt(np.sum(np.abs(PSI)**2) * dx**2)

# Potential

V = 0.5 * m * omega**2 * (x1**2 + x2**2)

# Meshgrid

k1,k2 = np.meshgrid(k,k,indexing="ij")
KE = h_bar**2 / (2.0 * m) * (k1 + k2)
dt = 0.01
t_f = 10.0
num_steps = int(t_f / dt)
times = np.arange(num_steps + 1) * dt

# Split Operator, refer to phase 1

def Split_2d_non(PSI):
  PSI = np.exp(-1j * V * dt / (2.0 * h_bar)) * PSI
  PSI_k = np.fft.fft(PSI)
  PSI_k = np.exp(-1j * KE * dt / h_bar) * PSI_k
  PSI = np.fft.ifft(PSI_k)
  PSI = np.exp(-1j * V * dt / (2.0 * h_bar)) * PSI
  return psi

# Normalize in time

normz = np.zeros(num_steps + 1)

normz[0] = np.sum(np.abs(PSI)**2) * dx**2

for step in range(1, num_steps + 1):  
  PSI = Split_2d_non(PSI)
  normz[step] = np.sum(np.abs(PSI)**2) * dx**2

# Plotting, please debug not sure if these are correct

plt.figure()
plt.plot(times, normz)
plt.xlabel("Time")
plt.ylabel("Normalized")
plt.title("Two Non-Interacting Particles Normalized")
plt.show()
plt.savefig("plots/2_Non_Interacting_Particles.png")
plt.close

# Entanglement/Entropy

def Entang_Entropy(PSI,dx,cutoff=1e-20)
  prob = np.linalg.svd(PSI * dx, compute_uv=False)
  entropy = -np.sum(prob * np.log(prob))
  return entropy

# For a more specific cse, use dx = np.sqrt(dx1 + dx2)**2
# If we do use this, please define dx1 & dx2

# SVD

entropy_times = []
entropies = []
for step in range(1, num_steps + 1):
  PSI = split_2d_non(PSI)
  normz[step] = np.sum(np.abs(PSI)**2) * dx**2
  entropy_times.append(step * dt)
  entropies.append(Entang_Entropy(PSI, dx))

# Plotting, again please debug & be careful as I'm not sure if this code might overload your devices

plt.figure()
plt.plot(entropy_times, entropies)
plt.axhline(0.0, linestyle="-")
plt.xlabel("Time")
plt.ylabel("Entanglement")
plt.title("Entanglement of Two Non-Interacting Particles")
plt.show()
plt.savefig("plots/phase2_entropy_vs_time.png")
plt.close()

# TODO: build a 2D grid for x1 and x2
# TODO: initial state as a product: Psi(x1,x2,0) = psi1(x1) * psi2(x2)
# TODO: potential is just V(x1) + V(x2), no interaction term
# TODO: propagate with the 2D split-operator
# TODO: write the entanglement entropy function (SVD of Psi reshaped over x1,x2)
# TODO: confirm entropy stays ~0 the whole run
# TODO: save entropy-vs-time plot to plots/
