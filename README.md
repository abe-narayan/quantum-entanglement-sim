Here's the raw text — copy this whole block and paste it straight into the GitHub editor (the fence marks below are just so it displays as plain text here, don't include those two lines):
# Two-Particle Quantum Dynamics & Entanglement

Two particles in a harmonic trap, with an interaction we can switch on
and off. Point of this is to watch entanglement show up once it's on,
and see how it stacks up against plain old classical physics.

## What's going on physically

Each particle sits in a bowl-shaped trap, `V(x) = ½mω²x²`. Left alone
they just oscillate, nothing links them together. Add soft-core Coulomb
repulsion (`k/√((x₁-x₂)²+a²)`, softened so it doesn't blow up on the
grid) and their states start getting tangled up with each other — that's
entanglement, and we track how much with entanglement entropy over time.

For propagating the wavefunction we're using split-operator with FFT.
Each step is a pure phase, so it stays unitary no matter what step size
we pick — no probability leaking out, which is really the main thing we
need the method to guarantee.

## Phases

Building this up in stages rather than all at once, so if something
breaks we've got a decent idea where to look.

| Phase | What it does | How we know it worked |
|---|---|---|
| 1 | One particle, just the trap | Norm holds, matches the known analytic solution |
| 2 | Two particles, no interaction | Entropy stays ~0 (sanity-checks the entropy code) |
| 3 | Interaction turned on | Entropy actually grows |
| 4 | Classical comparison | Quantum vs. classical trajectories, side by side |
| 5 (stretch) | Entropy vs. interaction strength | Only if we have time |

Don't skip Phase 2 — if entropy comes out nonzero there, it's the entropy
code that's broken, not the physics.

## Files

- `phase1_single_particle.py` — foundation, validates the propagator
- `phase2_two_particle_noninteracting.py` — validates the entropy code
- `phase3_interacting.py` — the one we actually care about
- `classical_comparison.py` — quantum vs. classical, side by side
- `plots/` — figures land here

Each script's got a short description and TODO list at the top.

## Running it

pip install numpy scipy matplotlib
python phase1_single_particle.py
python phase2_two_particle_noninteracting.py
python phase3_interacting.py
python classical_comparison.py

Run them in order, each one leans on the last.

## Worth agreeing on as a group first

Grid resolution/extent, the softening length `a`, interaction strength
`k`, trap frequency `ω`. Better to nail these down together before
anyone starts coding — changing them later means re-validating
everything.
