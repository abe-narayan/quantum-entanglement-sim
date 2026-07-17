"""
Phase 2 — two particles, no interaction yet.

Point of this one is to isolate bugs: if entanglement entropy comes out
nonzero here, the bug's in the entropy code, not the physics, since a
product state literally can't entangle.
"""

# TODO: build a 2D grid for x1 and x2
# TODO: initial state as a product: Psi(x1,x2,0) = psi1(x1) * psi2(x2)
# TODO: potential is just V(x1) + V(x2), no interaction term
# TODO: propagate with the 2D split-operator
# TODO: write the entanglement entropy function (SVD of Psi reshaped over x1,x2)
# TODO: confirm entropy stays ~0 the whole run
# TODO: save entropy-vs-time plot to plots/
