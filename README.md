Two-Particle Quantum Dynamics & Entanglement

Two particles in a harmonic trap. We turn on an interaction between them
and watch entanglement show up — then check it against what classical
physics would predict for the same setup.

What's going on physically

Each particle sits in a bowl-shaped trap, V(x) = ½mω²x². On their own
they just oscillate — nothing interesting happens between them. Once they
can push on each other (soft-core Coulomb repulsion, k/√((x₁-x₂)²+a²),
a softened version so it doesn't blow up on the grid), their states get
linked. That linkage is entanglement, and we track it with entanglement
entropy over time.

We're using split-operator with FFT to propagate the wavefunction — each
sub-step is a pure phase, so it's exactly unitary no matter the step
size. Probability doesn't leak, which is exactly what we need to trust.

Phases

We're building this up instead of writing it all at once, so bugs get
caught early instead of tangled together later.

PhaseWhat it doesHow we know it worked1One particle, just the trapNorm holds, matches the known analytic solution2Two particles, no interactionEntropy stays ~0 (sanity-checks the entropy code)3Interaction turned onEntropy actually grows4Classical comparisonQuantum vs. classical trajectories, side by side5 (stretch)Entropy vs. interaction strengthOnly if we have time

Phase 2's the important one to not skip — if entropy shows up nonzero
there, we know the bug's in our entropy code, not the physics.

Files


phase1_single_particle.py — the foundation, validates the propagator
phase2_two_particle_noninteracting.py — validates the entropy code
phase3_interacting.py — the one we actually care about
classical_comparison.py — quantum vs. classical, side by side
plots/ — where the figures land


Each script has a short description + TODO list at the top.

Running it

bashpip install numpy scipy matplotlib
python phase1_single_particle.py
python phase2_two_particle_noninteracting.py
python phase3_interacting.py
python classical_comparison.py

Run them in order — each one leans on the last.

Worth agreeing on as a group first

Grid resolution/extent, the softening length a, interaction strength
k, and trap frequency ω. Changing these later means re-validating
everything, so better to pick them together before anyone starts coding.
