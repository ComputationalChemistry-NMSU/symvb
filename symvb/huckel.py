"""
Symbolic Huckel solver.

Given a one-electron tight-binding graph (adjacency + uniform resonance
integral h, uniform overlap s), produce the molecular orbitals and
energies symbolically. The MO eigenvectors of the adjacency matrix are
simultaneously eigenvectors of the GHEP H c = eps S c whenever
S = I + (s/h) H; in that case the energies follow as

    eps_k(s) = h * lambda_k / (1 + s * lambda_k),  lambda_k = adjacency
                                                   eigenvalue.

This module returns those quantities in `HuckelResult`. Coefficients are
RAW integer / algebraic-number entries (no 1/sqrt(L) prefactor): the
global normalization is a scalar that cancels in any eigenvalue
identity, and rationals/surds keep SymPy fast.

Special cases are hand-coded for cyclic L in {3, 4, 5, 6} so that the
coefficient matrices agree with the manuscript scripts (matching
benzene's real-cos/sin Bloch basis, Cp's golden-ratio entries, etc).
General graphs fall through to `sympy.Matrix.eigenvects`.
"""
from dataclasses import dataclass, field
from typing import Optional

import sympy


class HuckelGHEPMismatchError(ValueError):
    """Raised when a user-supplied overlap matrix does not satisfy the
    S = I + (s/h) H precondition that lets MOs of H simultaneously
    diagonalize the GHEP."""


@dataclass(frozen=True)
class HuckelResult:
    """Symbolic output of a Huckel solve.

    Fields:
      site_labels   : tuple of site names (AO ordering used).
      eigenvalues   : tuple of adjacency-matrix eigenvalues lambda_k
                      (parameter-free sympy expressions; rationals or
                      algebraic numbers).
      energies      : tuple of MO energies eps_k(h, s) = h*lambda_k /
                      (1 + s*lambda_k), as sympy expressions in
                      (h_symbol, s_symbol).
      coefficients  : sympy Matrix of shape (n_mo, n_sites). Row k is
                      MO k expanded over the AO basis, in raw
                      integer/algebraic form (NOT AO-orthonormalized).
      h_symbol      : the sympy.Symbol for the resonance integral.
      s_symbol      : the sympy.Symbol for the AO overlap.
      point_group   : optional point-group label (e.g. 'D_6h').
      irrep_labels  : optional per-MO irrep tags (e.g. ('a_2u', 'e_1g',
                      'e_1g', 'e_2u', 'e_2u', 'b_2g')).
    """
    site_labels: tuple
    eigenvalues: tuple
    energies: tuple
    coefficients: sympy.Matrix
    h_symbol: sympy.Symbol
    s_symbol: sympy.Symbol
    point_group: Optional[str] = None
    irrep_labels: Optional[tuple] = None

    def energy_of_occupation(self, occupation):
        """Return the total one-electron energy for a given MO
        occupation. `occupation` is an iterable of (mo_index,
        n_electrons) pairs OR a flat iterable of length n_mo with
        per-MO electron counts."""
        occ = list(occupation)
        if occ and not isinstance(occ[0], (tuple, list)):
            assert len(occ) == len(self.energies)
            return sympy.Add(*(n * e for n, e in zip(occ, self.energies)))
        return sympy.Add(*(n * self.energies[k] for k, n in occ))


def _resolve_symbol(x, default_name):
    if isinstance(x, sympy.Symbol):
        return x
    return sympy.Symbol(x)


def _energies_from_eigenvalues(lambdas, h, s):
    return tuple(h * lam / (1 + s * lam) for lam in lambdas)


def _ring_special_case(L):
    """Return (eigenvalues, coefficients, irrep_labels) for L in
    {3, 4, 5, 6} with a hand-picked real Bloch basis whose entries are
    integers (or, for L=5, golden-ratio surds). Coefficients are NOT
    AO-orthonormalized; row scaling is chosen for memorable rationals.

    Conventions match the existing manuscript example scripts
    (benzene_1e_analytical_overlap.py, etc).
    """
    if L == 3:
        # C_3 triangle. lambda = {2, -1, -1}.
        # Real basis: totally symmetric, plus two e-type orthogonal vectors.
        eigs = (sympy.Integer(2), sympy.Integer(-1), sympy.Integer(-1))
        coeffs = sympy.Matrix([
            [1, 1, 1],
            [2, -1, -1],
            [0, 1, -1],
        ])
        irreps = ('a', 'e', 'e')
        return eigs, coeffs, irreps

    if L == 4:
        # C_4 square. lambda = {2, 0, 0, -2}.
        eigs = (sympy.Integer(2), sympy.Integer(0),
                sympy.Integer(0), sympy.Integer(-2))
        coeffs = sympy.Matrix([
            [1, 1, 1, 1],
            [1, 0, -1, 0],
            [0, 1, 0, -1],
            [1, -1, 1, -1],
        ])
        irreps = ('a_g', 'e_u', 'e_u', 'b_g')
        return eigs, coeffs, irreps

    if L == 5:
        # Pentagon. lambda = {2, 2cos(72), 2cos(72), 2cos(144), 2cos(144)}
        # = {2, (sqrt(5)-1)/2, (sqrt(5)-1)/2, -(sqrt(5)+1)/2, -(sqrt(5)+1)/2}
        sqrt5 = sympy.sqrt(5)
        phi_inv = (sqrt5 - 1) / 2     # 2 cos(72)
        neg_phi = -(sqrt5 + 1) / 2    # 2 cos(144)
        eigs = (sympy.Integer(2), phi_inv, phi_inv, neg_phi, neg_phi)
        # Real Bloch coefficients: cos(2 pi m j / 5) and sin(2 pi m j / 5).
        # Using sympy.cos/sin then simplifying gives surds in Q[sqrt(5)].
        coeffs_rows = [[sympy.Integer(1)] * 5]
        for m in (1, 2):
            cos_row = [sympy.simplify(sympy.cos(2 * sympy.pi * m * j / 5))
                       for j in range(5)]
            sin_row = [sympy.simplify(sympy.sin(2 * sympy.pi * m * j / 5))
                       for j in range(5)]
            coeffs_rows.append(cos_row)
            coeffs_rows.append(sin_row)
        coeffs = sympy.Matrix(coeffs_rows)
        irreps = ('a', 'e_1', 'e_1', 'e_2', 'e_2')
        return eigs, coeffs, irreps

    if L == 6:
        # Benzene hexagon. lambda = {2, 1, 1, -1, -1, -2}.
        # Real basis with sqrt(3)/2 absorbed into row scaling so all
        # entries are rational (matches benzene_1e_analytical_overlap.py).
        eigs = (sympy.Integer(2),
                sympy.Integer(1), sympy.Integer(1),
                sympy.Integer(-1), sympy.Integer(-1),
                sympy.Integer(-2))
        coeffs = sympy.Matrix([
            [1, 1, 1, 1, 1, 1],                                   # k=0,  a_2u
            [sympy.Rational(2), sympy.Integer(1), sympy.Integer(-1),
             sympy.Integer(-2), sympy.Integer(-1), sympy.Integer(1)],  # k=1c
            [sympy.Integer(0), sympy.Integer(1), sympy.Integer(1),
             sympy.Integer(0), sympy.Integer(-1), sympy.Integer(-1)],  # k=1s
            [sympy.Rational(2), sympy.Integer(-1), sympy.Integer(-1),
             sympy.Integer(2), sympy.Integer(-1), sympy.Integer(-1)],  # k=2c
            [sympy.Integer(0), sympy.Integer(1), sympy.Integer(-1),
             sympy.Integer(0), sympy.Integer(1), sympy.Integer(-1)],   # k=2s
            [sympy.Integer(1), sympy.Integer(-1), sympy.Integer(1),
             sympy.Integer(-1), sympy.Integer(1), sympy.Integer(-1)],  # k=3
        ])
        irreps = ('a_2u', 'e_1g', 'e_1g', 'e_2u', 'e_2u', 'b_2g')
        return eigs, coeffs, irreps

    return None  # caller falls through to generic path


def _generic_solve(adjacency_matrix, site_labels, h, s):
    """Diagonalize an arbitrary adjacency matrix symbolically via
    sympy.Matrix.eigenvects. Returns (eigenvalues, coefficients) with
    rows = MO eigenvectors, sorted by descending eigenvalue."""
    A = sympy.Matrix(adjacency_matrix)
    if A.rows != A.cols or A.rows != len(site_labels):
        raise ValueError(
            f"adjacency shape {A.shape} inconsistent with "
            f"{len(site_labels)} site labels")

    raw = A.eigenvects()  # list of (eigenvalue, multiplicity, basis)

    triples = []
    for eig, mult, basis in raw:
        for vec in basis:
            triples.append((sympy.simplify(eig), vec))

    def _sort_key(eig):
        try:
            return -float(eig)
        except (TypeError, ValueError):
            return -float(sympy.N(eig))

    triples.sort(key=lambda t: _sort_key(t[0]))

    eigs = tuple(t[0] for t in triples)
    rows = [list(t[1]) for t in triples]
    rows = [[entry[0] if hasattr(entry, '__iter__') else entry
             for entry in row] for row in rows]
    coeffs = sympy.Matrix(rows)
    return eigs, coeffs


def _check_ghep_precondition(adjacency, overlap, h, s):
    """If user supplies an explicit overlap matrix, verify
    overlap == I + (s/h)*adjacency. Raise HuckelGHEPMismatchError if not."""
    if overlap is None:
        return
    A = sympy.Matrix(adjacency)
    S_user = sympy.Matrix(overlap)
    if S_user.shape != A.shape:
        raise ValueError(f"overlap shape {S_user.shape} != adjacency {A.shape}")
    expected = sympy.eye(A.rows) + (s / h) * A
    diff = sympy.simplify(S_user - expected)
    if diff != sympy.zeros(*A.shape):
        raise HuckelGHEPMismatchError(
            "user-supplied overlap does not satisfy S = I + (s/h)*H. "
            "The MO-direct shortcut requires this form; for richer "
            "overlap structures, solve the full GHEP elsewhere.")


def solve_ring(L, h='h', s='s', basis='real', site_labels=None):
    """Solve the cyclic Huckel ring of L sites.

    Parameters
    ----------
    L : int
        Ring size.
    h : str | sympy.Symbol
        Resonance integral symbol (default 'h').
    s : str | sympy.Symbol
        AO overlap symbol (default 's').
    basis : 'real' | 'complex'
        Real (cos/sin) Bloch basis (default) or complex exponentials.
        Currently only 'real' is implemented.
    site_labels : optional sequence of L strings
        Defaults to first L lowercase letters ('a', 'b', ..., chr(96+L)).

    Returns
    -------
    HuckelResult
    """
    if basis != 'real':
        raise NotImplementedError("only basis='real' is implemented in v1")
    h_sym = _resolve_symbol(h, 'h')
    s_sym = _resolve_symbol(s, 's')
    if site_labels is None:
        site_labels = tuple(chr(ord('a') + j) for j in range(L))
    else:
        site_labels = tuple(site_labels)
        if len(site_labels) != L:
            raise ValueError(f"need {L} site_labels, got {len(site_labels)}")

    special = _ring_special_case(L)
    if special is not None:
        eigs, coeffs, irreps = special
        return HuckelResult(
            site_labels=site_labels,
            eigenvalues=eigs,
            energies=_energies_from_eigenvalues(eigs, h_sym, s_sym),
            coefficients=coeffs,
            h_symbol=h_sym,
            s_symbol=s_sym,
            point_group=f'D_{L}h',
            irrep_labels=irreps,
        )

    # general L: build adjacency of C_L and fall through
    adjacency = sympy.zeros(L, L)
    for j in range(L):
        adjacency[j, (j + 1) % L] = 1
        adjacency[(j + 1) % L, j] = 1
    eigs, coeffs = _generic_solve(adjacency, site_labels, h_sym, s_sym)
    return HuckelResult(
        site_labels=site_labels,
        eigenvalues=eigs,
        energies=_energies_from_eigenvalues(eigs, h_sym, s_sym),
        coefficients=coeffs,
        h_symbol=h_sym,
        s_symbol=s_sym,
        point_group=f'D_{L}h',
        irrep_labels=None,
    )


def solve(adjacency, site_labels=None, h='h', s='s', overlap=None):
    """Solve a general Huckel graph.

    Parameters
    ----------
    adjacency : matrix-like
        n x n symmetric adjacency matrix (entries are coefficients of
        h, typically 0/1 or symbolic per-edge weights).
    site_labels : optional sequence of n strings.
    h, s : str | sympy.Symbol
        Resonance and overlap symbols.
    overlap : optional matrix-like
        Explicit overlap matrix. If provided, must equal I + (s/h)*adjacency
        or HuckelGHEPMismatchError is raised. If None, the canonical
        S = I + (s/h)H form is assumed.

    Returns
    -------
    HuckelResult
    """
    h_sym = _resolve_symbol(h, 'h')
    s_sym = _resolve_symbol(s, 's')
    A = sympy.Matrix(adjacency)
    if site_labels is None:
        site_labels = tuple(chr(ord('a') + j) for j in range(A.rows))
    else:
        site_labels = tuple(site_labels)

    _check_ghep_precondition(A, overlap, h_sym, s_sym)

    eigs, coeffs = _generic_solve(A, site_labels, h_sym, s_sym)
    return HuckelResult(
        site_labels=site_labels,
        eigenvalues=eigs,
        energies=_energies_from_eigenvalues(eigs, h_sym, s_sym),
        coefficients=coeffs,
        h_symbol=h_sym,
        s_symbol=s_sym,
    )
