"""
Spin-block-aware two-electron matrix element (Phase 1.5: half-pair cache).

The historical o2_det in molecule.py expands <D_A | H_2 | D_B> via Loewdin's
cofactor formula on the full spin-mixed (N-2)-electron overlap and recomputes
that overlap determinant for every (i, k, j, l) electron-position quadruple.
Because the AO overlap is spin-diagonal, the (N-2)-electron overlap is itself
block-diagonal in spin: it factors into per-spin-block determinants. Exploiting
that structure lets the matrix element be assembled from a small per-pair
precompute (per-spin-block 1-row and 2-row cofactor tables) plus three spin-
channel contractions (alpha-alpha, beta-beta, alpha-beta).

Half-pair caching (Phase 1.5)
-----------------------------
Per-spin-block cofactor tables depend only on the half-string of D_A and the
half-string of D_B for that spin. Across the basis, many (D_A, D_B) pairs
share their (alpha-half-pair, beta-half-pair) decomposition: for benzene
3a+3b in 6 orbitals, there are C(6,3)^2 = 400 dets but only C(6,3)^2 = 400
distinct alpha-half-strings and 400 distinct beta-half-strings, hence at
most 400 * 400 = 160,000 (D_A, D_B) full-pair entries spread across only
20*20 = 400 unique alpha-half-pairs and 400 unique beta-half-pairs. The
per-half-pair cofactor tables are therefore evaluated once and reused
~400-fold rather than rebuilt at every full-pair call.

The cache is lazy: built on demand inside o2_det_blocked, keyed on the
sorted (alphabetised-within-block) half-strings, attached to the Molecule
instance as ._o2_blocked_cache. It populates as the caller walks the basis
and hits all needed entries naturally during one m.o2_matrix(m.basis) sweep.

Algorithm
---------
With the user's det_string in possibly-arbitrary creation order, fold the
parities of (i) the spin-block permutation, (ii) the within-alpha sort,
and (iii) the within-beta sort into an overall sign and work in canonical
layout (alpha block first sorted alphabetically, then beta block likewise).
For the canonical layout:

  - alpha-alpha pair: D^{AB}_{ik,jl} = det(S_beta) * Gamma^alpha_{ik,jl}
  - beta-beta pair:   D^{AB}_{ik,jl} = det(S_alpha) * Gamma^beta_{ik,jl}
  - alpha-beta pair:  D^{AB}_{ik,jl} = gamma^alpha_{ij} * gamma^beta_{kl}

where gamma^sigma_{ij} = (-1)^{i+j} det(S^sigma with row i col j removed) and
Gamma^sigma_{ik,jl} = (-1)^{i+k+j+l} det(S^sigma with rows i,k cols j,l
removed). The cofactor tables are looked up from the half-pair cache;
the contraction loops then sum over per-block index ranges directly.

The Phase 2 agreement-test harness in symvb/test_o2_agreement.py validates
symbolic parity with the direct path on a battery of cases; that harness
re-runs unchanged after this refactor because the algorithmic content is
identical.
"""
from __future__ import annotations

import sympy as sp

from symvb.functions import sorti


_SP_ZERO = sp.Integer(0)
_SP_ONE = sp.Integer(1)


def o2_det_blocked(molecule, D1, D2):
    """Two-electron matrix element via the spin-block-aware path with
    half-pair caching.

    Must agree symbolically with Molecule.o2_det(D1, D2) in 'direct' mode
    on every input. See module docstring for algorithm.
    """
    # --- Sz selection ----------------------------------------------------
    if D1.Nel != D2.Nel:
        return _SP_ZERO
    if (len(D1.alpha_string) != len(D2.alpha_string) or
            len(D1.beta_string) != len(D2.beta_string)):
        return _SP_ZERO

    # --- Bring each det to canonical (spin-block + alphabetical within
    #     block) and capture all three sign contributions per det. -------
    A_alpha_s, A_alpha_inv = sorti(D1.alpha_string)
    A_beta_s, A_beta_inv = sorti(D1.beta_string)
    B_alpha_s, B_alpha_inv = sorti(D2.alpha_string)
    B_beta_s, B_beta_inv = sorti(D2.beta_string)

    sigma_A = (_spin_block_parity(D1.spins) *
               (1 if A_alpha_inv % 2 == 0 else -1) *
               (1 if A_beta_inv % 2 == 0 else -1))
    sigma_B = (_spin_block_parity(D2.spins) *
               (1 if B_alpha_inv % 2 == 0 else -1) *
               (1 if B_beta_inv % 2 == 0 else -1))
    overall_sign = sigma_A * sigma_B

    # --- Half-pair cache lookup (lazy; builds on first request) ---------
    alpha_data = _get_half_pair(molecule, 'alpha', A_alpha_s, B_alpha_s)
    beta_data = _get_half_pair(molecule, 'beta', A_beta_s, B_beta_s)

    # Lowercase canonical strings for integral-symbol queries.
    A_alpha = A_alpha_s.lower()
    A_beta = A_beta_s.lower()
    B_alpha = B_alpha_s.lower()
    B_beta = B_beta_s.lower()

    n_a = len(A_alpha)
    n_b = len(A_beta)

    det_S_alpha = alpha_data['det']
    det_S_beta = beta_data['det']
    gamma_alpha = alpha_data['gamma']
    gamma_beta = beta_data['gamma']
    Gamma_alpha = alpha_data['Gamma']
    Gamma_beta = beta_data['Gamma']

    # --- alpha-alpha channel ---------------------------------------------
    # Bucket Gamma cofactors by their (int_d - int_x) integrand. Skip
    # quadruples whose Gamma cofactor is structurally zero (common when
    # the spin-block overlap has zero rows from non-interacting AOs):
    # iv-tuple construction and integral lookup avoided entirely.
    T_aa = _SP_ZERO
    if n_a >= 2:
        aa_groups = {}
        for i in range(n_a):
            for k in range(i + 1, n_a):
                for j in range(n_a):
                    for l in range(j + 1, n_a):
                        cof = Gamma_alpha[(i, k, j, l)]
                        if cof == _SP_ZERO:
                            continue
                        iv_d = (A_alpha[i], A_alpha[k], B_alpha[j], B_alpha[l])
                        iv_x = (A_alpha[i], A_alpha[k], B_alpha[l], B_alpha[j])
                        int_d = molecule.get_o2_expr(iv_d)
                        int_x = molecule.get_o2_expr(iv_x)
                        integrand = int_d - int_x
                        if integrand == _SP_ZERO:
                            continue
                        prev = aa_groups.get(integrand)
                        aa_groups[integrand] = cof if prev is None else prev + cof
        for integrand, coef in aa_groups.items():
            T_aa = T_aa + integrand * coef
        T_aa = T_aa * det_S_beta

    # --- beta-beta channel -----------------------------------------------
    T_bb = _SP_ZERO
    if n_b >= 2:
        bb_groups = {}
        for i in range(n_b):
            for k in range(i + 1, n_b):
                for j in range(n_b):
                    for l in range(j + 1, n_b):
                        cof = Gamma_beta[(i, k, j, l)]
                        if cof == _SP_ZERO:
                            continue
                        iv_d = (A_beta[i], A_beta[k], B_beta[j], B_beta[l])
                        iv_x = (A_beta[i], A_beta[k], B_beta[l], B_beta[j])
                        int_d = molecule.get_o2_expr(iv_d)
                        int_x = molecule.get_o2_expr(iv_x)
                        integrand = int_d - int_x
                        if integrand == _SP_ZERO:
                            continue
                        prev = bb_groups.get(integrand)
                        bb_groups[integrand] = cof if prev is None else prev + cof
        for integrand, coef in bb_groups.items():
            T_bb = T_bb + integrand * coef
        T_bb = T_bb * det_S_alpha

    # --- alpha-beta channel (direct only; exchange forbidden by spin) ----
    # The integral set in symvb is small (5-10 distinct symbols at typical
    # PPP scale), so many AO quadruples in this 4-loop share the same
    # integral. Two-tier sparse iteration:
    #
    #   (a) Reorder loops so gamma_alpha[i][j] sits in the outer pair,
    #       letting us skip the entire (k, l) inner loop when gamma_alpha
    #       is structurally zero (common when the alpha overlap has zero
    #       rows from non-interacting AOs).
    #   (b) Same for gamma_beta inside.
    #
    # Then accumulate the gamma_alpha * gamma_beta products per integral-
    # symbol bucket and multiply once at the end: replaces ~n_a^2 * n_b^2
    # three-way SymPy multiplies with the same number of two-way multiplies
    # plus a handful of final symbol-bucket multiplies.
    T_ab = _SP_ZERO
    if n_a >= 1 and n_b >= 1:
        ab_groups = {}
        for i in range(n_a):
            ai = A_alpha[i]
            for j in range(n_a):
                gA = gamma_alpha[i][j]
                if gA == _SP_ZERO:
                    continue
                bj = B_alpha[j]
                for k in range(n_b):
                    ak = A_beta[k]
                    for l in range(n_b):
                        gB = gamma_beta[k][l]
                        if gB == _SP_ZERO:
                            continue
                        iv = (ai, ak, bj, B_beta[l])
                        int_d = molecule.get_o2_expr(iv)
                        if int_d == _SP_ZERO:
                            continue
                        product = gA * gB
                        prev = ab_groups.get(int_d)
                        ab_groups[int_d] = product if prev is None else prev + product
        for int_sym, coef in ab_groups.items():
            T_ab = T_ab + int_sym * coef

    result = overall_sign * (T_aa + T_bb + T_ab)
    if result == 0:
        return _SP_ZERO
    return result


# --- Half-pair cache -----------------------------------------------------

def _get_half_pair(molecule, spin, A_sorted, B_sorted):
    """Return cached cofactor tables for the (A_sorted, B_sorted) half-pair
    in the given spin block. Builds on demand; subsequent calls hit the
    cache. The cache is attached to the molecule instance and persists for
    its lifetime."""
    if not hasattr(molecule, '_o2_blocked_cache') or molecule._o2_blocked_cache is None:
        molecule._o2_blocked_cache = {'alpha': {}, 'beta': {}}
    bucket = molecule._o2_blocked_cache[spin]
    key = (A_sorted, B_sorted)
    hit = bucket.get(key)
    if hit is not None:
        return hit
    S = _build_overlap(molecule, A_sorted.lower(), B_sorted.lower())
    n = S.rows
    entry = {
        'det': _safe_det(S),
        'gamma': _one_row_cofactor_table(S) if n >= 1 else None,
        'Gamma': _two_row_cofactor_dict(S) if n >= 2 else None,
    }
    bucket[key] = entry
    return entry


# --- helpers -------------------------------------------------------------

def _spin_block_parity(spins):
    """Sign of the permutation that brings a spin pattern (string of '+' for
    alpha and '-' for beta) into canonical (all '+' first, then all '-')
    order. Equals (-1)^(number of beta-before-alpha inversions)."""
    inv = 0
    n = len(spins)
    for i in range(n):
        if spins[i] == '-':
            for j in range(i + 1, n):
                if spins[j] == '+':
                    inv += 1
    return 1 if inv % 2 == 0 else -1


def _build_overlap(molecule, A_orbs, B_orbs):
    """Per-spin-block overlap matrix of pairwise AO overlaps. Both lists
    must be lowercase spatial-orbital labels of equal length."""
    n = len(A_orbs)
    if n == 0:
        return sp.zeros(0, 0)
    return sp.Matrix(n, n, lambda i, j: molecule.get_o1_expr(A_orbs[i], B_orbs[j], 'S'))


def _safe_det(M):
    """det() that returns 1 for the empty (0x0) matrix without invoking
    sympy.det on it (which can be quirky on degenerate sizes)."""
    if M.rows == 0:
        return _SP_ONE
    return M.det()


def _one_row_cofactor_table(S):
    """gamma[i][j] = (-1)^{i+j} det(S with row i, col j removed)."""
    n = S.rows
    table = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sign = 1 if (i + j) % 2 == 0 else -1
            table[i][j] = sign * _minor(S, [i], [j])
    return table


def _two_row_cofactor_dict(S):
    """Gamma[(i,k,j,l)] = (-1)^{i+k+j+l} det(S with rows i,k, cols j,l removed)
    for i<k, j<l. Keys are quadruples of block-internal indices."""
    n = S.rows
    table = {}
    for i in range(n):
        for k in range(i + 1, n):
            for j in range(n):
                for l in range(j + 1, n):
                    sign = 1 if (i + k + j + l) % 2 == 0 else -1
                    table[(i, k, j, l)] = sign * _minor(S, [i, k], [j, l])
    return table


def _minor(S, rows_to_remove, cols_to_remove):
    """Determinant of S with the given rows and columns deleted. Returns 1
    for the resulting empty matrix."""
    rows_keep = [r for r in range(S.rows) if r not in rows_to_remove]
    cols_keep = [c for c in range(S.cols) if c not in cols_to_remove]
    if not rows_keep:
        return _SP_ONE
    return S.extract(rows_keep, cols_keep).det()
