# Two Particle Quantum Dynamics & Entanglement

A simulation of two particles in a one dimensional harmonic trap with an
interaction we can turn on and off. The goal is to watch entanglement
appear once the interaction is active, quantify how it grows, and compare
the quantum behavior against the classical two body problem.

## What's going on physically

Each particle sits in a harmonic trap, `V(x) = ½mω²x²`. On their own the
two particles simply oscillate, with nothing connecting them. Once we add
a soft core Coulomb repulsion, `g/√((x₁-x₂)²+a²)`, their states begin to
correlate in a way that cannot be factored into two independent particles.
That correlation is entanglement, and we measure how much of it there is
using the von Neumann entanglement entropy, tracked over time.

The softening length `a` under the square root keeps the potential finite
when the particles overlap, which would otherwise blow up on the grid and
break the simulation.

To propagate the wavefunction we use the split operator method with the
FFT. Each step is a product of pure phase factors, so the propagator stays
unitary regardless of the step size. That is the property we most need:
total probability is conserved, and nothing leaks out as the state evolves.

## Phases

We build the project up in stages rather than all at once. Each phase is a
checkpoint, so if a result looks wrong we have a good idea of where the
problem was introduced.

| Phase | What it does | How we know it worked |
|---|---|---|
| 1 | One particle, trap only | Norm is conserved and the trajectory matches the known analytic solution |
| 2 | Two particles, no interaction | Entanglement entropy stays at zero, as a product state requires |
| 3 | Interaction turned on | Entropy grows from zero while energy and norm stay fixed |
| 4 | Classical comparison | Quantum expectation values plotted against the classical trajectories |
| 5 | Entropy vs. interaction strength | Sweep the coupling |

Phases 1 and 2 are deliberate validation steps. Phase 1 confirms the
propagator itself is correct before we trust it with anything harder, and
Phase 2 confirms the entropy calculation returns zero when it should,
which means any nonzero entropy in Phase 3 is real physics rather than a
bug.

## Files

- `phase1_single_particle.py` — foundation, validates the propagator
- `phase2_two_particle_noninteracting.py` — validates the entropy code
- `phase3_interacting.py` — the main result, entanglement under interaction
- `classical_comparison.py` — quantum expectation values against classical trajectories
- `plots/` — generated figures are written here

Each script has a short description and a TODO list at the top.

## Running it

pip install numpy scipy matplotlib
python phase1_single_particle.py
python phase2_two_particle_noninteracting.py
python phase3_interacting.py
python classical_comparison.py

Run them in order, since each phase builds on the validation done by the
previous one.

## Parameters to agree on as a group first

A few parameters should be fixed before anyone starts coding: the grid
resolution and extent, the softening length `a`, the interaction strength
`g`, and the trap frequency `ω`. These need to be consistent across every
script, because changing them later means re validating each phase from
the beginning.
