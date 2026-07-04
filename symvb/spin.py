"""
Total-spin (S^2) machinery: construct the S^2 matrix over a Slater-det
basis with fixed Sz and project onto a chosen total-spin eigenspace.

For the Sz = 0 subspace,

    S^2  =  S_+ S_-                (the S_z^2 - S_z piece vanishes)
         =  sum_{i, j}  c^+_{i alpha} c_{i beta} c^+_{j beta} c_{j alpha}.

The action of S^2 on a Slater det |D> is computed by applying the four
second-quantized operators in sequence, tracking fermion signs through
a canonical spin-orbital ordering (alpha_1, beta_1, alpha_2, beta_2, ...).
The resulting matrix element <D'|S^2|D> is summed over all orbital
pairs (i, j) to build the full S^2 matrix.

Given an S^2 matrix over any Slater-det basis, `project_onto_S(H, S2, S)`
returns the Hamiltonian block projected onto the S(S+1) eigenspace.
"""
from __future__ import annotations

import numpy as np


def _canon_det(alpha_set, beta_set):
    """
    Canonical det string: sort alpha and beta sets, interleave in pairs
    matching symvb.functions.generate_det_strings.  Returns None if
    n_alpha != n_beta (our caller will work with balanced dets only).
    """
    a = sorted(alpha_set)
    b = sorted(beta_set)
    if len(a) != len(b):
        return None
    return ''.join(a[k] + b[k].upper() for k in range(len(a)))


def _spin_orbital_list(alpha_set, beta_set):
    """Canonical ordering: alpha_1, beta_1, alpha_2, beta_2, ... sorted by orbital."""
    orbs = sorted(alpha_set | beta_set)
    return [(orb, s) for orb in orbs for s in (0, 1) if
            (s == 0 and orb in alpha_set) or (s == 1 and orb in beta_set)]


def _apply_c(alpha, beta, orb, spin, create):
    """
    Apply c_{orb, spin} or c^+_{orb, spin} to the det specified by the
    alpha/beta orbital sets.  Returns (new_alpha, new_beta, sign) or
    (None, None, 0) if annihilated.  sign = (-1)^{# occupied spin-orbitals
    before (orb, spin) in the canonical ordering}.

    Canonical ordering: for each orbital in alphabetical order, alpha first
    then beta.
    """
    present = (orb in alpha) if spin == 0 else (orb in beta)
    if create == present:   # trying to create an occupied slot, or annihilate empty
        return None, None, 0

    # Jordan-Wigner sign: count occupied spin-orbitals strictly before (orb, spin)
    count = 0
    for x in sorted(alpha | beta):
        if x < orb:
            count += int(x in alpha) + int(x in beta)
        elif x == orb:
            if spin == 1 and x in alpha:
                count += 1
    sign = 1 if count % 2 == 0 else -1

    new_alpha = alpha.copy()
    new_beta  = beta.copy()
    if spin == 0:
        if create:
            new_alpha.add(orb)
        else:
            new_alpha.discard(orb)
    else:
        if create:
            new_beta.add(orb)
        else:
            new_beta.discard(orb)
    return new_alpha, new_beta, sign


def _apply_s_pl_s_mi(orbs, alpha, beta):
    """
    Apply S_+ S_- = sum_{i, j} c^+_{i a} c_{i b} c^+_{j b} c_{j a}  to |D>.
    Returns a dict  { canonical_det_string : signed_coefficient, ... }.
    `orbs` is the list of orbital labels to iterate over.
    """
    out = {}
    for j in orbs:
        # c_{j alpha}
        a1, b1, s1 = _apply_c(alpha, beta, j, 0, create=False)
        if a1 is None:
            continue
        for i in orbs:
            # c^+_{j beta}
            a2, b2, s2 = _apply_c(a1, b1, j, 1, create=True)
            if a2 is None:
                continue
            # c_{i beta}
            a3, b3, s3 = _apply_c(a2, b2, i, 1, create=False)
            if a3 is None:
                continue
            # c^+_{i alpha}
            a4, b4, s4 = _apply_c(a3, b3, i, 0, create=True)
            if a4 is None:
                continue
            sign = s1 * s2 * s3 * s4
            key = _canon_det(a4, b4)
            if key is None:
                continue
            out[key] = out.get(key, 0) + sign
    return out


def _symvb_to_canonical_sign(det_string, orbital_index):
    """
    Sign relating the symvb Slater det (creation order = det_string order)
    to the canonical Jordan-Wigner ordering (alphabetical orbital, alpha
    before beta for each orbital).

    For det_string 'x0 X1 x2 X3 ...' the creation order is
        (x0, alpha), (X1, beta), (x2, alpha), (X3, beta), ...
    To put in canonical ascending order we count inversions in the list
    of spin-orbital indices  (2 * orb_index + spin).
    """
    indices = []
    for pos, c in enumerate(det_string):
        orb = c.lower()
        spin = 0 if c.islower() else 1
        indices.append(2 * orbital_index[orb] + spin)
    inv = 0
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            if indices[a] > indices[b]:
                inv += 1
    return 1 if inv % 2 == 0 else -1


def s_squared_matrix(det_strings, orbs=None):
    """
    Build the S^2 matrix in a Slater-det basis at Sz = 0.

    Thin wrapper around `symvb.operators.s_squared(orbs).matrix(...)`;
    kept for backward compatibility. Returns a numpy float ndarray with
    eigenvalues S(S+1).
    """
    from symvb import operators as _op
    if orbs is None:
        orbs = sorted({ch.lower() for ds in det_strings for ch in ds})
    return np.array(_op.s_squared(orbs).matrix(det_strings), dtype=float)


def _apply_eta_pl_eta_mi(orbs, site_sign, alpha, beta):
    """
    Apply  eta_+ eta_-  =  sum_{i,j} s_i s_j c^+_{i alpha} c^+_{i beta}
                                            c_{j beta} c_{j alpha}
    to |D> (alpha, beta occupation sets), where s_i = site_sign[i] = +/-1.
    Returns a dict { canonical_det_string : signed_coefficient, ... }.
    """
    out = {}
    for j in orbs:
        # c_{j alpha}
        a1, b1, s1 = _apply_c(alpha, beta, j, 0, create=False)
        if a1 is None:
            continue
        # c_{j beta}
        a2, b2, s2 = _apply_c(a1, b1, j, 1, create=False)
        if a2 is None:
            continue
        for i in orbs:
            # c^+_{i beta}
            a3, b3, s3 = _apply_c(a2, b2, i, 1, create=True)
            if a3 is None:
                continue
            # c^+_{i alpha}
            a4, b4, s4 = _apply_c(a3, b3, i, 0, create=True)
            if a4 is None:
                continue
            sign = s1 * s2 * s3 * s4 * site_sign[i] * site_sign[j]
            key = _canon_det(a4, b4)
            if key is None:
                continue
            out[key] = out.get(key, 0) + sign
    return out


def eta_squared_matrix(det_strings, site_signs, orbs=None):
    """
    Build the eta-pseudospin squared matrix  eta^2  in a Slater-det basis
    at eta_z = (N - L) / 2 = 0  (half-filling on a bipartite lattice).

    The eta operators are
        eta_+ = sum_i  s_i  c^+_{i alpha} c^+_{i beta}
        eta_- = (eta_+)^dagger
        eta_z = (N_total - L) / 2
    where s_i = +/-1 is the sublattice sign.  On a bipartite nearest-
    neighbor hopping lattice at half-filling, [H, eta_+-] = 0; on adding
    Hubbard U this reduces to [H, eta_z] = 0 only.

    Parameters
    ----------
    det_strings : list of symvb Slater-det strings, all at N = L.
    site_signs : dict {orbital_label: +1 or -1} encoding the bipartition.
    orbs : optional list of orbital labels (inferred from site_signs if
           omitted).

    Returns
    -------
    eta2 : (N, N) ndarray.  Eigenvalues are eta(eta+1) for eta = 0, 1, 2, ...

    Thin wrapper around `symvb.operators.eta_squared(...).matrix(...)`.
    """
    from symvb import operators as _op
    return np.array(_op.eta_squared(site_signs).matrix(det_strings), dtype=float)


def project_onto_S(H, S2, target_S, tol=1e-8):
    """
    Project H onto the eigenspace of S^2 with eigenvalue target_S * (target_S + 1).

    Returns (H_block, U) where U has orthonormal columns spanning the
    target-S subspace and H_block = U^T H U.
    """
    target = target_S * (target_S + 1)
    evals, evecs = np.linalg.eigh((S2 + S2.T) / 2)
    mask = np.abs(evals - target) < tol
    if not mask.any():
        raise ValueError(f'no S^2 eigenvalues near S(S+1) = {target} '
                         f'(spectrum: {sorted(set(round(e, 6) for e in evals))})')
    U = evecs[:, mask]
    H_block = U.T @ H @ U
    return H_block, U
