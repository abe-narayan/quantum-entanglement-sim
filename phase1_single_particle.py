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
