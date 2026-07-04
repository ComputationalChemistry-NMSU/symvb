"""
Automatic symmetry detection and symmetry-adapted basis construction.

Two detection strategies:

1. `degenerate_block_basis` — numerical eigenanalysis.
   Diagonalise a single (H, S) numerically and group the eigenvectors by
   degenerate eigenvalue. Each degenerate cluster spans an irrep subspace.
   Fully automatic, no group theory, immediately useful but only exposes the
   block structure at the specific parameter point used.

2. `detect_permutation_group` — graph-automorphism detection via pynauty.
   Encode the symbolic (H, S) as a vertex- and edge-coloured graph; pynauty
   (via Weisfeiler–Leman refinement + backtracking) returns a minimal set of
   generators for the group of basis permutations that preserve the matrices
   symbolically. Requires the optional dependency `pynauty`.

Also provided:

    generate_group(generators, N)       — enumerate a group from generators.
    totally_symmetric_basis(generators, N) — orbit-sum basis for the trivial
        irrep; any H commuting with the generators block-diagonalises here,
        and the ground state of benzene-like problems lives in this block.
    signed_totally_symmetric_basis(signed_generators, N) — numeric variant
        tracking fermion signs (needed above half filling).
    signed_totally_symmetric_basis_exact(signed_generators, N) — the same
        subspace in exact arithmetic (sympy columns, no tolerances).
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.linalg import eigh


def degenerate_block_basis(H_num, S_num=None, tol=1e-8):
    """
    Diagonalise (H, S) and group eigenvectors by degenerate eigenvalue.

    Parameters
    ----------
    H_num : ndarray of shape (N, N).
    S_num : ndarray of shape (N, N), optional. Generalized overlap metric.
    tol   : eigenvalue degeneracy tolerance.

    Returns
    -------
    evals  : (N,) ndarray of sorted eigenvalues.
    evecs  : (N, N) ndarray; columns are eigenvectors, already a
             symmetry-adapted basis up to within-block unitary freedom.
    blocks : list of (eigenvalue, column_indices) pairs, one per irrep cluster.
    """
    if S_num is None:
        evals, evecs = np.linalg.eigh(H_num)
    else:
        evals, evecs = eigh(H_num, S_num)

    blocks = []
    i = 0
    while i < len(evals):
        j = i + 1
        while j < len(evals) and abs(evals[j] - evals[i]) < tol:
            j += 1
        blocks.append((float(evals[i]), list(range(i, j))))
        i = j
    return evals, evecs, blocks


def _canon(expr):
    """Canonical hashable key for a sympy expression.

    Sympy `Expr` objects are already canonically comparable (equal expressions
    hash to the same value), so we use the expression itself wrapped for
    robustness. We only fall back to `sp.expand` when two expressions that
    might be equal don't hash the same.
    """
    return sp.sympify(expr)


def detect_permutation_group(H_sym, S_sym=None):
    """
    Find every basis permutation that leaves H_sym (and S_sym if given)
    unchanged as a symbolic expression, using pynauty.

    The matrices are encoded as a colored graph:
        vertex color of i = (H[i,i], S[i,i])
        edge color of (i,j) = (H[i,j], S[i,j])
    Edge colors are converted to vertex colors via auxiliary vertices.

    Parameters
    ----------
    H_sym : sympy Matrix of shape (N, N).
    S_sym : sympy Matrix of shape (N, N), optional.

    Returns
    -------
    generators : list of ndarray of shape (N,). A generating set for the
                 automorphism group; each array is a permutation of 0..N-1.
    group_order: int. Order of the full group.
    """
    try:
        import pynauty
    except ImportError as e:
        raise ImportError(
            "pynauty is required for detect_permutation_group. "
            "Install with `pip install pynauty` (may need `--no-binary :all:` "
            "on older CPUs without AVX2)."
        ) from e

    if not isinstance(H_sym, sp.Matrix):
        H_sym = sp.Matrix(H_sym)
    if S_sym is not None and not isinstance(S_sym, sp.Matrix):
        S_sym = sp.Matrix(S_sym)

    N = H_sym.shape[0]

    def vcol(i):
        if S_sym is None:
            return (_canon(H_sym[i, i]),)
        return (_canon(H_sym[i, i]), _canon(S_sym[i, i]))

    def ecol(i, j):
        if S_sym is None:
            return (_canon(H_sym[i, j]),)
        return (_canon(H_sym[i, j]), _canon(S_sym[i, j]))

    zero_key = ((_canon(0),) if S_sym is None
                else (_canon(0), _canon(0)))

    # Auxiliary vertex per non-zero edge, grouped by edge-color
    edge_class = {}
    edge_aux = {}
    aux = N
    for i in range(N):
        for j in range(i + 1, N):
            c = ecol(i, j)
            if c == zero_key:
                continue
            edge_aux[(i, j)] = aux
            edge_class.setdefault(c, []).append(aux)
            aux += 1

    total = aux

    g = pynauty.Graph(total)
    for (i, j), a in edge_aux.items():
        g.connect_vertex(a, [i, j])

    vclass = {}
    for i in range(N):
        vclass.setdefault(vcol(i), []).append(i)

    partition = [set(v) for v in vclass.values()] + [set(e) for e in edge_class.values()]
    g.set_vertex_coloring(partition)

    gens_raw, order, *_ = pynauty.autgrp(g)

    # Trim each generator to the first N entries. pynauty should keep aux
    # vertices within their own color class, so original vertices permute
    # among themselves.
    generators = []
    for gen in gens_raw:
        perm = np.asarray(gen[:N], dtype=int)
        if set(perm.tolist()) == set(range(N)):
            generators.append(perm)
    return generators, int(order)


def _compose(p, q):
    """Permutation composition (p o q)(i) = p[q[i]]."""
    return np.asarray([p[q[i]] for i in range(len(p))], dtype=int)


def generate_group(generators, N=None):
    """
    Enumerate every permutation in the group generated by `generators`.
    Returns a list of ndarrays.
    """
    if N is None:
        N = len(generators[0]) if generators else 0
    identity = tuple(range(N))
    seen = {identity}
    queue = [np.arange(N)]
    elements = [np.arange(N)]
    while queue:
        g = queue.pop()
        for gen in generators:
            new = _compose(gen, g)
            key = tuple(new.tolist())
            if key not in seen:
                seen.add(key)
                queue.append(new)
                elements.append(new)
    return elements


def apply_orbital_permutation(orbital_map, basis_dets, canon_fn):
    """
    Induce a basis permutation from an orbital-label permutation.

    Given a permutation of single-particle orbital labels (e.g. C_6 on
    {a,b,c,d,e,f} -> {b,c,d,e,f,a}) and a list of basis determinants,
    compute which basis index each det maps to and the associated fermion
    sign (from re-canonicalising the permuted string). Returns None if the
    permutation does not preserve the basis set (i.e. a permuted det is
    not representable in the given basis).

    Parameters
    ----------
    orbital_map : dict mapping each lowercase label to its image.
                  Uppercase (beta) images are derived automatically.
    basis_dets  : list of strings, one per basis det, in canonical form.
    canon_fn    : callable taking a det_string, returning (canonical_string,
                  sign). Typically wraps SlaterDet(s).get_sorted().

    Returns
    -------
    perm : ndarray of shape (N,). Basis index `i` maps to index `perm[i]`.
    signs: ndarray of shape (N,) with +/- 1 fermion signs.
    """
    lower = list(orbital_map.keys())
    upper_map = {k.upper(): v.upper() for k, v in orbital_map.items()}
    translate = str.maketrans({**orbital_map, **upper_map})

    index_of = {d: i for i, d in enumerate(basis_dets)}
    N = len(basis_dets)
    perm = np.empty(N, dtype=int)
    signs = np.empty(N, dtype=int)
    for i, d in enumerate(basis_dets):
        image = d.translate(translate)
        canon, sgn = canon_fn(image)
        if canon not in index_of:
            return None, None
        perm[i] = index_of[canon]
        signs[i] = sgn
    return perm, signs


def totally_symmetric_basis(generators, N):
    """
    Construct the orbit-sum basis for the trivial irrep: a column for each
    orbit, uniform weight 1/sqrt(orbit_size) on the orbit members, zero
    elsewhere. Columns are orthogonal (different orbits don't overlap) and
    unit-normalised.

    Any H commuting with every generator is block-diagonal in the resulting
    decomposition; this function returns only the totally-symmetric block,
    which contains the ground state of benzene-like problems.

    Parameters
    ----------
    generators : list of ndarray permutations of 0..N-1.
    N          : int.

    Returns
    -------
    U      : ndarray of shape (N, k) where k is the number of orbits.
    orbits : list of index lists.

    Notes
    -----
    This is an UNSIGNED orbit-sum projector and is correct only when every
    group operation acts on the basis with sign +1 (e.g. benzene at half-
    filling, sub-half-filled rings). For over-half-filled fermionic Slater
    determinants (e.g. C4H4 dianion: 6 electrons in 4 orbitals) the
    permutation representation carries -1 signs that the orbit sum
    misses; use `signed_totally_symmetric_basis` instead.
    """
    if not generators:
        U = np.eye(N)
        return U, [[i] for i in range(N)]

    elements = generate_group(generators, N)
    assigned = [False] * N
    orbits = []
    for i in range(N):
        if assigned[i]:
            continue
        orb = set()
        for g in elements:
            orb.add(int(g[i]))
        for j in orb:
            assigned[j] = True
        orbits.append(sorted(orb))

    U = np.zeros((N, len(orbits)))
    for k, orb in enumerate(orbits):
        for idx in orb:
            U[idx, k] = 1.0
        U[:, k] /= np.sqrt(len(orb))
    return U, orbits


def signed_totally_symmetric_basis(signed_generators, N, tol=1e-8,
                                   max_order=2048):
    """
    Sign-aware variant of `totally_symmetric_basis`.

    Build the trivial-irrep projector P = (1/|G|) Sum_g rho(g), where
    rho(g) is the SIGNED permutation representation on the basis (each
    basis element transforms with a +/- 1 fermion sign). Return an
    orthonormal basis of the +1-eigenspace of P.

    Use this in place of `totally_symmetric_basis` whenever the basis
    elements are fermionic Slater determinants and the chosen filling
    can introduce a -1 sign upon group action -- most notably for
    over-half-filled rings (e.g. C4H4 dianion, L=4 with N=6). For
    sub-half-filled / half-filled benzene-like systems the signs are
    uniformly +1 and this function recovers the same subspace as
    `totally_symmetric_basis`.

    Parameters
    ----------
    signed_generators : iterable of (perm, signs) tuples
        Each generator is a (perm, signs) pair as returned by
        `apply_orbital_permutation`. `perm[i]` is the basis index that
        element i maps to; `signs[i]` is the +/- 1 fermion sign.
    N : int
        Basis dimension.
    tol : float
        Tolerance for identifying +1 eigenvectors of the projector.
    max_order : int
        Safety cap on enumerated group order.

    Returns
    -------
    U : ndarray of shape (N, k)
        Orthonormal columns spanning the totally-symmetric subspace.
    group_order : int
        |G|, the number of group elements enumerated.
    """
    gens_mats = []
    for perm, signs in signed_generators:
        M = np.zeros((N, N))
        for i in range(N):
            M[int(perm[i]), i] = float(signs[i])
        gens_mats.append(M)

    def key(M):
        return tuple(np.asarray(M).flatten().round(10))

    identity = np.eye(N)
    seen = {key(identity)}
    elements = [identity]
    queue = [identity]
    while queue and len(elements) < max_order:
        g = queue.pop()
        for M in gens_mats:
            new = M @ g
            k = key(new)
            if k not in seen:
                seen.add(k)
                queue.append(new)
                elements.append(new)

    P = sum(elements) / len(elements)
    P_sym = 0.5 * (P + P.T)
    evals, evecs = np.linalg.eigh(P_sym)
    U = evecs[:, np.abs(evals - 1.0) < tol]
    return U, len(elements)


def signed_totally_symmetric_basis_exact(signed_generators, N,
                                         max_order=2048):
    """
    Exact-arithmetic variant of `signed_totally_symmetric_basis`.

    Enumerate the signed permutation group generated by `signed_generators`
    over the integers, form the (implicit) Reynolds projector
    P = (1/|G|) Sum_g rho(g), and return an exact orthonormal basis of its
    image, the totally-symmetric subspace. No floating point, no
    tolerances: the group is closed over integer (perm, sign) pairs and
    each basis column is a signed orbit sum with entries +-1/sqrt(orbit
    size), so the result lives in Q extended by square roots and can be
    fed directly into symbolic reductions (U.T * H * U) and `lambdify`.

    The basis ordering is deterministic: one column per sign-unfrustrated
    orbit, orbits taken in order of their smallest basis index. An orbit
    contributes no column when some group element fixes a member with
    fermion sign -1 (the orbit sum then cancels exactly); this is how the
    -1 signs of over-half-filled rings reduce the dimension relative to
    the unsigned `totally_symmetric_basis`. When every sign is +1 the
    columns coincide with the unsigned orbit sums.

    Parameters
    ----------
    signed_generators : iterable of (perm, signs) tuples
        Each generator is a (perm, signs) pair as returned by
        `apply_orbital_permutation`. `perm[i]` is the basis index that
        element i maps to; `signs[i]` is the +/- 1 fermion sign.
    N : int
        Basis dimension.
    max_order : int
        Safety cap on the enumerated group order; exceeding it raises
        RuntimeError (the group did not close within the cap).

    Returns
    -------
    U : sympy Matrix of shape (N, k)
        Exactly orthonormal columns spanning the totally-symmetric
        subspace.
    group_order : int
        |G|, the number of signed group elements enumerated.
    """
    gens = []
    for perm, signs in signed_generators:
        p = tuple(int(x) for x in perm)
        s = tuple(int(x) for x in signs)
        if sorted(p) != list(range(N)):
            raise ValueError(f"generator perm {p} is not a permutation "
                             f"of 0..{N - 1}")
        if any(x not in (-1, 1) for x in s):
            raise ValueError("generator signs must be +-1")
        gens.append((p, s))

    # Close the group over exact integer (perm, sign) pairs.
    # rho(g): e_i -> signs[i] * e_{perm[i]};  composing g2 after g1 gives
    # e_i -> s1[i]*s2[p1[i]] * e_{p2[p1[i]]}.
    identity = (tuple(range(N)), (1,) * N)
    seen = {identity}
    elements = [identity]
    queue = [identity]
    while queue:
        p1, s1 = queue.pop()
        for p2, s2 in gens:
            el = (tuple(p2[p1[i]] for i in range(N)),
                  tuple(s1[i] * s2[p1[i]] for i in range(N)))
            if el not in seen:
                if len(elements) >= max_order:
                    raise RuntimeError(
                        f"group did not close within max_order={max_order}")
                seen.add(el)
                queue.append(el)
                elements.append(el)

    # One candidate column per orbit: |G| * P e_i has integer entry
    # sum_{g: g(i)=j} signs_g(i) at position j. Within an orbit the entries
    # are uniformly +-|Stab(i)|, or all zero when the stabilizer character
    # is frustrated (some stabilizer element carries sign -1).
    assigned = [False] * N
    cols = []
    for i in range(N):
        if assigned[i]:
            continue
        coef = [0] * N
        orbit = set()
        for p, s in elements:
            coef[p[i]] += s[i]
            orbit.add(p[i])
        for j in orbit:
            assigned[j] = True
        magnitudes = {abs(coef[j]) for j in orbit}
        if magnitudes == {0}:
            continue
        if len(magnitudes) != 1:
            raise RuntimeError(
                f"non-uniform orbit sum {sorted(magnitudes)} on orbit of "
                f"index {i}; signed generators are inconsistent")
        m = magnitudes.pop()
        norm = m * sp.sqrt(len(orbit))
        v = sp.zeros(N, 1)
        for j in orbit:
            v[j] = sp.Integer(coef[j]) / norm
        cols.append(v)

    U = sp.Matrix.hstack(*cols) if cols else sp.zeros(N, 0)
    return U, len(elements)
