"""
MO -> AO Slater-determinant projection.

Given a set of occupied MOs (rows of an AO-coefficient matrix) and an
ordered AO Slater-determinant basis (symvb's native `m.basis`), expand
the MO determinant in the AO basis via Slater--Laplace cofactor
expansion. For each AO determinant |T_alpha; T_beta>,

    amp(T_alpha, T_beta) = (sign_v * sign_ab) * det(C_alpha[:, T_alpha])
                                              * det(C_beta [:, T_beta])

where C_spin is the submatrix of `mo_coeffs` selecting the rows for
the spin's occupied MOs, and the two sign factors convert between
symvb's det-string creation order, spin-block ordering, and the
canonical interleaved (alpha_j, beta_j) ordering used inside the
cofactor formula.

This is the engine that lets a user write down a candidate eigenvector
of a 400-dim VB problem (or 22, or 9) without diagonalising it: pick
the desired MO occupation pattern, hand it to this function, and the
resulting vector is exact-symbolic in whatever field the MO coefficients
live in.
"""
from itertools import combinations

import numpy
import sympy

from symvb.spin import _symvb_to_canonical_sign


class EigenpairResidualError(AssertionError):
    """Raised by `verify_eigenpair` when (H - E*S)*v has any nonzero
    component after simplification. Carries the index of the first
    offending entry plus its raw and simplified residual expressions."""

    def __init__(self, index, residual, simplified):
        self.index = index
        self.residual = residual
        self.simplified = simplified
        super().__init__(
            f"residual entry [{index}] is nonzero: {simplified}")


def verify_eigenpair(H, S, v, E, simplify=None):
    """Prove that (H - E*S)*v == 0 as a polynomial identity.

    Forms the residual r = H*v - E*(S*v), simplifies each component
    with `simplify` (default sympy.cancel; pass sympy.simplify for
    surd cases). Returns True if every component is identically zero;
    otherwise raises EigenpairResidualError on the first offender.

    Parameters
    ----------
    H, S : sympy.Matrix
        n x n symbolic Hamiltonian and overlap.
    v : sympy.Matrix (n x 1) or convertible
        Candidate eigenvector.
    E : sympy expression
        Candidate eigenvalue (function of the same symbols as H, S).
    simplify : callable, optional
        Per-entry simplifier. Defaults to `sympy.cancel` (best for
        rational-function residuals); use `sympy.simplify` if surds
        are involved.
    """
    if simplify is None:
        simplify = sympy.cancel
    residual = sympy.Matrix(H) * sympy.Matrix(v) \
             - E * (sympy.Matrix(S) * sympy.Matrix(v))
    for i in range(residual.rows):
        raw = residual[i, 0]
        simp = simplify(sympy.together(raw))
        if simp != 0:
            raise EigenpairResidualError(i, raw, simp)
    return True


def _spin_orbital_interleave_sign(T_alpha, T_beta):
    """Parity of the permutation that brings the spin-block ordering
    [a_{i_1}, a_{i_2}, ..., b_{j_1}, b_{j_2}, ...] (all alphas first,
    then all betas, each block sorted by site) into the interleaved
    canonical ordering (alpha_0, beta_0, alpha_1, beta_1, ...).

    With T_alpha and T_beta sorted, intra-block inversions vanish, so
    only the cross-block term contributes: encoding alpha_j as 2j (even)
    and beta_j as 2j+1 (odd), an inversion across blocks happens iff
    j_alpha > j_beta. This is O(|T_alpha| * |T_beta|) instead of
    O((|T_alpha| + |T_beta|)^2)."""
    inv = sum(1 for j_a in T_alpha for j_b in T_beta if j_a > j_b)
    return 1 if inv % 2 == 0 else -1


def _is_sympy(M):
    return isinstance(M, sympy.MatrixBase)


def _site_index_map(basis_dets, site_labels):
    """Return dict {char -> int} from lowercase site letters to indices.
    If `site_labels` is given, use that order; otherwise infer from the
    lowercase characters of basis_dets in first-appearance order."""
    if site_labels is not None:
        return {c: i for i, c in enumerate(site_labels)}
    seen = []
    for ds in basis_dets:
        for c in ds:
            cl = c.lower()
            if cl not in seen:
                seen.append(cl)
    seen.sort()
    return {c: i for i, c in enumerate(seen)}


def mo_determinant_in_ao(mo_coeffs, occupation, basis_dets,
                         site_labels=None):
    """Expand an MO Slater determinant in the symvb AO det basis.

    Parameters
    ----------
    mo_coeffs : sympy.Matrix or numpy.ndarray, shape (n_mo, n_sites)
        Row k = MO k expanded over the AOs in `site_labels` order.
        Backend (sympy vs numpy) is auto-detected from this argument
        and propagates to the returned vector.
    occupation : (alpha_indices, beta_indices)
        Two iterables of MO row indices (into `mo_coeffs`) listing the
        occupied alpha and beta MOs. For closed-shell, the two are
        equal.
    basis_dets : sequence of str
        AO Slater-det strings in symvb's convention (lowercase = alpha,
        uppercase = beta, creation order = string order). Typically
        `[fp.dets[0].det_string for fp in m.basis]`.
    site_labels : optional sequence of n_sites lowercase chars
        Maps the AO label to a column index of `mo_coeffs`. If None,
        inferred from `basis_dets` (first-appearance, then sorted).

    Returns
    -------
    Column vector (sympy.Matrix or numpy.ndarray of float) of length
    len(basis_dets), giving the AO-det expansion coefficients of
    |occupation>.
    """
    alpha_indices = list(occupation[0])
    beta_indices  = list(occupation[1])
    n_alpha = len(alpha_indices)
    n_beta  = len(beta_indices)

    site_idx = _site_index_map(basis_dets, site_labels)
    n_sites = len(site_idx)

    use_sympy = _is_sympy(mo_coeffs)

    # Submatrices: occupied alpha MOs, occupied beta MOs, all AOs (cols).
    if use_sympy:
        C_alpha = mo_coeffs[alpha_indices, :]
        C_beta  = mo_coeffs[beta_indices,  :]
    else:
        C = numpy.asarray(mo_coeffs)
        C_alpha = C[alpha_indices, :]
        C_beta  = C[beta_indices,  :]

    # Memoize cofactor minors over distinct sorted T-tuples. Each T
    # has size n_alpha (resp. n_beta); for benzene closed-shell this
    # collapses 400 evaluations to C(6, 3) = 20 per spin.
    det_alpha_cache = {}
    det_beta_cache  = {}

    def det_alpha(T):
        if T in det_alpha_cache:
            return det_alpha_cache[T]
        if use_sympy:
            d = C_alpha[:, list(T)].det()
        else:
            d = numpy.linalg.det(C_alpha[:, list(T)])
        det_alpha_cache[T] = d
        return d

    def det_beta(T):
        if T in det_beta_cache:
            return det_beta_cache[T]
        if use_sympy:
            d = C_beta[:, list(T)].det()
        else:
            d = numpy.linalg.det(C_beta[:, list(T)])
        det_beta_cache[T] = d
        return d

    # Collect amplitudes in a Python list, then build the result vector
    # in one go. This avoids per-element sympy.Matrix.__setitem__ /
    # numpy item-assignment overhead, which dominates the inner loop
    # for the 400-dim benzene case.
    zero = sympy.Integer(0) if use_sympy else 0.0
    amps = [zero] * len(basis_dets)
    for I, ds in enumerate(basis_dets):
        T_a, T_b = [], []
        for c in ds:
            j = site_idx[c.lower()]
            (T_a if c.islower() else T_b).append(j)
        if len(T_a) != n_alpha or len(T_b) != n_beta:
            # AO det doesn't have the right electron count for this
            # occupation; amplitude is zero by orthogonality.
            continue
        T_a_sorted = tuple(sorted(T_a))
        T_b_sorted = tuple(sorted(T_b))

        sign_ab = _spin_orbital_interleave_sign(T_a_sorted, T_b_sorted)
        sign_v  = _symvb_to_canonical_sign(ds, site_idx)

        d_a = det_alpha(T_a_sorted)
        d_b = det_beta(T_b_sorted)

        amps[I] = sign_v * sign_ab * d_a * d_b

    if use_sympy:
        return sympy.Matrix(amps)
    arr = numpy.asarray(amps)
    if numpy.iscomplexobj(arr):
        arr = arr.real
    return arr.astype(float, copy=False)


def linear_combination_in_ao(mo_coeffs, terms, basis_dets, site_labels=None):
    """Project a linear combination of MO Slater determinants into the
    AO det basis.

    Returns sum_i c_i * mo_determinant_in_ao(mo_coeffs, occ_i, ...) for
    each (c_i, occ_i) pair in `terms`. Backend (sympy vs numpy) is
    inherited from `mo_coeffs`.

    Parameters
    ----------
    mo_coeffs : sympy.Matrix or numpy.ndarray, shape (n_mo, n_sites)
        Same as in `mo_determinant_in_ao`.
    terms : iterable of (coefficient, occupation) pairs
        Each `occupation` is `(alpha_indices, beta_indices)`. The
        coefficient is multiplied into the corresponding determinant.
    basis_dets, site_labels : same as in `mo_determinant_in_ao`.

    Returns
    -------
    Column vector (sympy.Matrix or numpy.ndarray) of length
    len(basis_dets), the AO-det expansion of the linear combination.
    """
    terms = list(terms)
    if not terms:
        raise ValueError("terms must be non-empty")
    use_sympy = _is_sympy(mo_coeffs)
    coef0, occ0 = terms[0]
    result = coef0 * mo_determinant_in_ao(
        mo_coeffs, occ0, basis_dets, site_labels=site_labels)
    for coef, occ in terms[1:]:
        result = result + coef * mo_determinant_in_ao(
            mo_coeffs, occ, basis_dets, site_labels=site_labels)
    return result
