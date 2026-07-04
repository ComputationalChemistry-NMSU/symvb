"""
Symmetry-projected two-electron matrix construction.

Build H_2 directly in the totally-symmetric (orbit-sum) SALC basis, without
materialising the full determinant-basis matrix first. The infrastructure
in symvb.symmetry (apply_orbital_permutation, generate_group,
totally_symmetric_basis) supplies the orbit decomposition; this module
contracts the matrix elements using G-invariance of H_2 to skip redundant
det-pair calls.

Algorithm
---------
A totally-symmetric SALC over orbit `orb_a` of size `s_a` is the uniform sum

    psi_a = (1/sqrt(s_a)) * sum_{i in orb_a} D_i

Because H_2 commutes with every group element (a precondition the user
asserts implicitly through their `subst` and `subst_2e` dictionaries
grouping equivalent integrals), the matrix element between SALCs

    <psi_a | H_2 | psi_b> = (1/sqrt(s_a s_b)) * sum_{i,j} <D_i | H_2 | D_j>

reduces via group invariance to a single sum:

    <psi_a | H_2 | psi_b> = sqrt(s_a / s_b) * sum_{m in orb_b} <D_seed_a | H_2 | D_m>

where seed_a is any representative of orb_a. The double sum collapses from
s_a * s_b det-pair calls to s_b (or s_a if we pick the other side as seed).

For benzene's A_{1g} block (38 orbits, 400 dets), this drops the work from
~80k det-pair calls (full upper-triangle build) to ~8k (single-orbit-side
sums per SALC pair).

Preconditions
-------------
1. The user's subst dictionaries must make H_2 commute with every generator
   in the supplied group. For uniformly-substituted PPP-type Hamiltonians
   on regular topologies (benzene ring, allyl chain), this is automatic.
   If the user uses inequivalent integrals on equivalent atoms, the
   resulting block matrix will silently differ from the U^T H_full U form.

2. `orbits` must come from `symvb.symmetry.totally_symmetric_basis(generators, N)`
   on the same `full_basis` length. Signed orbit sums (from
   `signed_totally_symmetric_basis`) are not supported in this routine
   yet -- the formula above assumes uniform +1 weights.

Validation against the full build is done by
test_o2_symmetric.TestSymmetricO2Agreement on small systems where both
paths are tractable: it constructs the full o2_matrix, projects via
U^T H U using the unsigned projector, and compares element-wise against
the direct symmetric build.
"""
from __future__ import annotations

import sympy as sp


def o2_matrix_symmetric(molecule, full_basis, orbits):
    """Build the o2 matrix in the totally-symmetric (orbit-sum) SALC basis.

    Parameters
    ----------
    molecule : symvb.Molecule
        Provides o2_det / o2 (works in either 'direct' or 'blocked' o2_method).
    full_basis : list of FixedPsi
        The full Sz-sector basis, typically `m.basis` after
        `m.generate_basis(Na, Nb, Norbs)`.
    orbits : list of list of int
        Orbit decomposition of full_basis under the symmetry group.
        Output of `symvb.symmetry.totally_symmetric_basis(generators, N)[1]`.

    Returns
    -------
    H : sympy.Matrix of shape (k, k) with k = len(orbits).
        Element [a, b] equals <psi_a | H_2 | psi_b> where psi_a is the
        unit-normalised orbit-sum SALC for orbit a. Equals U^T @ H_full @ U
        with U from totally_symmetric_basis.

    Notes
    -----
    The symmetric upper triangle is computed and mirrored. Each off-diagonal
    pair costs min(|orb_a|, |orb_b|) det-pair calls; the diagonal costs
    |orb_a|. For benzene A_{1g} (38 orbits, mean orbit size ~10): ~8k
    det-pair calls total versus ~80k for the full upper-triangle basis
    build.
    """
    k = len(orbits)
    H = sp.zeros(k)
    sizes = [len(orb) for orb in orbits]

    # Cache the seed SlaterDet for each orbit to avoid repeated FixedPsi -> det.
    def _seed_det(idx):
        fp = full_basis[orbits[idx][0]]
        return fp.dets[0]

    seeds = [_seed_det(a) for a in range(k)]

    for a in range(k):
        size_a = sizes[a]
        for b in range(a, k):
            size_b = sizes[b]
            # Pick the orbit with fewer elements for the inner sum to
            # minimise det-pair calls. Both choices yield the same value
            # by Hermiticity; we pick the cheaper one.
            if size_b <= size_a:
                # Sum over orb_b, fix seed_a; factor sqrt(size_a / size_b).
                seed = seeds[a]
                inner = sp.S.Zero
                for m in orbits[b]:
                    other = full_basis[m].dets[0]
                    inner = inner + molecule.o2_det(seed, other)
                factor = sp.sqrt(sp.Rational(size_a, size_b))
            else:
                # Sum over orb_a, fix seed_b; factor sqrt(size_b / size_a).
                seed = seeds[b]
                inner = sp.S.Zero
                for m in orbits[a]:
                    other = full_basis[m].dets[0]
                    inner = inner + molecule.o2_det(other, seed)
                factor = sp.sqrt(sp.Rational(size_b, size_a))
            H[a, b] = factor * inner
            if a != b:
                H[b, a] = H[a, b]
    return H
