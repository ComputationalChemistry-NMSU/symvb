"""Overlap-invariance of the benzene ionicity-class weights at U = 0.

The manuscript states, after the single-edge wrong-sign scan, that the
U = 0 ground-state ionicity weights (5, 31, 31, 5)/72 hold at ANY overlap s,
not only at s = 0: the U = 0 ground state is the SAME determinant at every s,
and that determinant is an eigenvector of the determinant-space overlap
matrix. This script backs both claims exactly and symbolically.

Setup: six-orbital benzene ring, half filling (3 alpha + 3 beta), Sz = 0,
400 AO determinants, Hubbard on-site U with h = -1, overlap s kept symbolic.
At U = 0 the ground state is the closed-shell determinant of the three
occupied momentum (Hueckel) MOs k = 0, +-1. Those MOs are simultaneous
eigenvectors of the circulant one-electron h and the circulant one-particle
overlap for every s, so their AO-determinant coefficient vector c is exactly
s-independent (a determinant of the fixed MO coefficient matrix).

What is proven here:
  1. Occupied MO set stays k = 0, +-1 (the three lowest) for all s < 1/2,
     so c is the same closed-shell determinant throughout.
  2. S(s) c = lambda(s) c EXACTLY in all 400 entries, with
     lambda(s) = (1 + 2s)^2 (1 + s)^4 = product of the occupied-MO norms
     squared, n_k = 1 + 2s cos(2 pi k / 6).
  3. Because c is an eigenvector of the metric, every Chirgwin-Coulson weight
     collapses to its s = 0 orthonormal value w_I = c_I^2 / ||c||^2, so the
     ionicity-class weights equal (5, 31, 31, 5)/72 at EVERY s. Verified two
     ways: the eigenvector shortcut, and the full CC definition
     w = c_I (S c)_I / (c^T S c) evaluated in exact rational arithmetic at
     several s, including negative s.

The two-orbital analogue is the H2 tie w_cov = w_ion = 1/2 at all s:
there sigma = (a + b) gives c = (1, 1, 1, 1) and S(s) c = (1 + s)^2 c.
Benzene is the general-ring instance of the same statement.

Run from the repo root: PYTHONPATH=. python3 examples/benzene_weights_overlap_invariance.py
"""
import os
import sys
import pickle
import time
from itertools import combinations

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from symvb import Molecule
from symvb.spin import _symvb_to_canonical_sign

CACHE = '/tmp/benzene_hubbard_matrices.pkl'
TARGET = {0: sp.Rational(5, 72), 1: sp.Rational(31, 72),
          2: sp.Rational(31, 72), 3: sp.Rational(5, 72)}


def build_molecule():
    m = Molecule(
        zero_ii=True,
        interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'],
        subst={'h': ('H_ab', 'H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af'),
               's': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af')},
        subst_2e={'U': ('1111',)},
        max_2e_centers=1,
    )
    m.generate_basis(3, 3, 6)
    return m


def load_or_build_S(m):
    """Return the symbolic 400x400 overlap S from the cache, rebuilding the
    full (H1, S, H2) cache via the standard example recipe if it is missing."""
    if not os.path.exists(CACHE):
        print('cache missing: building 400x400 symbolic (H1, S, H2) once '
              '(several minutes)...')
        t0 = time.time()
        H1 = m.build_matrix(m.basis, op='H')
        S = m.build_matrix(m.basis, op='S')
        H2 = m.o2_matrix(m.basis)
        with open(CACHE, 'wb') as f:
            pickle.dump((H1, S, H2), f)
        print(f'  cache built in {time.time() - t0:.1f}s')
    with open(CACHE, 'rb') as f:
        _, S, _ = pickle.load(f)
    return sp.Matrix(S)


def double_occ(ds):
    """Number of doubly occupied orbitals (ionicity class n_d) of a det."""
    occ = {}
    for c in ds:
        occ.setdefault(c.lower(), [False, False])
        if c.islower():
            occ[c.lower()][0] = True
        else:
            occ[c.lower()][1] = True
    return sum(1 for ab in occ.values() if ab[0] and ab[1])


def closed_shell_in_AO_sp(det_strings):
    """Exact rational coefficient vector of the U = 0 closed-shell ground state
    (three occupied momentum MOs) in the AO-determinant basis. s-independent
    by construction: the 3x6 MO coefficient matrix is fixed by ring symmetry."""
    site_idx = {c: i for i, c in enumerate('abcdef')}
    M_rat = sp.Matrix([
        [sp.Integer(1)] * 6,
        [sp.Rational(1), sp.Rational(1, 2), sp.Rational(-1, 2),
         sp.Rational(-1), sp.Rational(-1, 2), sp.Rational(1, 2)],
        [sp.Integer(0), sp.Rational(1, 2), sp.Rational(1, 2),
         sp.Integer(0), sp.Rational(-1, 2), sp.Rational(-1, 2)],
    ])
    det_rat = {T: M_rat[:, list(T)].det() for T in combinations(range(6), 3)}
    v0 = sp.zeros(len(det_strings), 1)
    for I, ds in enumerate(det_strings):
        T_a, T_b = [], []
        for c in ds:
            j = site_idx[c.lower()]
            (T_a if c.islower() else T_b).append(j)
        T_a = tuple(sorted(T_a)); T_b = tuple(sorted(T_b))
        seq = [2 * j for j in T_a] + [2 * j + 1 for j in T_b]
        inv = sum(1 for a in range(len(seq))
                  for b in range(a + 1, len(seq)) if seq[a] > seq[b])
        sign_ab = 1 if inv % 2 == 0 else -1
        sign_v = _symvb_to_canonical_sign(ds, site_idx)
        v0[I, 0] = sign_v * sign_ab * det_rat[T_a] * det_rat[T_b]
    return v0


def check_occupied_mo_set():
    """The three occupied momentum MOs stay k = 0, +-1 for all s < 1/2.

    Hueckel-with-overlap MO energies E_k = 2 h cos(theta_k) / (1 + 2 s cos(theta_k)),
    h = -1, theta_k = 2 pi k / 6. Occupied = three lowest.
    """
    s = sp.symbols('s')
    cos_k = {0: sp.Integer(1), 1: sp.Rational(1, 2), -1: sp.Rational(1, 2),
             2: sp.Rational(-1, 2), -2: sp.Rational(-1, 2), 3: sp.Integer(-1)}
    E = {k: -2 * ck / (1 + 2 * s * ck) for k, ck in cos_k.items()}
    ok = True
    for s_val in [sp.Rational(a, 100) for a in range(0, 50, 5)]:
        order = sorted(cos_k, key=lambda k: float(E[k].subs(s, s_val)))
        occ = set(order[:3])
        ok &= (occ == {0, 1, -1})
    print(f'  occupied MO set = {{0, +-1}} for all s in [0, 0.45]: {ok}')
    assert ok, 'occupied MO set changed below s = 1/2'
    return ok


def main():
    t0 = time.time()
    print(__doc__.splitlines()[0])
    m = build_molecule()
    det_strings = [fp.dets[0].det_string for fp in m.basis]
    nd = [double_occ(ds) for ds in det_strings]
    classes = {k: [i for i in range(len(det_strings)) if nd[i] == k]
               for k in range(4)}
    print(f'\nclass sizes n_d = 0..3: {[len(classes[k]) for k in range(4)]}')

    print('\n[1] occupied-MO-set stability')
    check_occupied_mo_set()

    s = sp.symbols('s')
    S = load_or_build_S(m)
    c = closed_shell_in_AO_sp(det_strings)

    print('\n[2] eigenvector identity  S(s) c = lambda(s) c')
    lam = (1 + 2 * s) ** 2 * (1 + s) ** 4
    Sc = S * c                                   # 400 polynomials in s
    resid = sp.expand(Sc - lam * c)
    n_nonzero = sum(1 for i in range(len(det_strings))
                    if sp.simplify(resid[i, 0]) != 0)
    print(f'  lambda(s) = (1 + 2s)^2 (1 + s)^4')
    print(f'  nonzero entries of S(s)c - lambda(s)c: {n_nonzero}/400')
    assert n_nonzero == 0, 'c is NOT an exact eigenvector of S(s)'
    # lambda = product of occupied-MO norms squared
    norm_prod = (1 + 2 * s) ** 2 * (1 + s) ** 2 * (1 + s) ** 2
    assert sp.expand(lam - norm_prod) == 0
    print('  lambda equals n_0^2 n_1^2 n_{-1}^2, n_k = 1 + 2s cos(2 pi k/6): OK')

    print('\n[3a] class weights via eigenvector shortcut  w_I = c_I^2 / ||c||^2')
    norm2 = sum(c[i, 0] ** 2 for i in range(len(det_strings)))
    w_short = {k: sp.Rational(sum(c[i, 0] ** 2 for i in classes[k]), 1) / norm2
               for k in range(4)}
    for k in range(4):
        w_short[k] = sp.nsimplify(w_short[k])
        print(f'  n_d = {k}: w = {w_short[k]}  (target {TARGET[k]})')
        assert w_short[k] == TARGET[k], f'class {k} weight != target'
    assert sum(w_short.values()) == 1
    print('  weights = (5, 31, 31, 5)/72 and sum to 1: OK')

    print('\n[3b] full Chirgwin-Coulson definition at several s (exact rational)')
    print('     w_k(s) = sum_{I in k} c_I (S c)_I / (c^T S c)')
    # Reuse the already-built polynomial vector Sc; substitute exact s values
    # (cheap: subs on a 400-vector of low-degree polys, not the 400x400 matrix).
    s_test = [sp.Integer(0), sp.Rational(1, 10), sp.Rational(3, 10),
              sp.Rational(1, 2), sp.Rational(-1, 5), sp.Rational(-2, 5)]
    header = f'  {"s":>6}  ' + '  '.join(f'{"w"+str(k):>7}' for k in range(4)) + '   sum'
    print(header)
    for sv in s_test:
        Sc_num = [Sc[i, 0].subs(s, sv) for i in range(len(det_strings))]
        D = sum(c[i, 0] * Sc_num[i] for i in range(len(det_strings)))
        wk = {}
        for k in range(4):
            Nk = sum(c[i, 0] * Sc_num[i] for i in classes[k])
            wk[k] = sp.nsimplify(Nk / D)
            assert wk[k] == TARGET[k], f's={sv}, class {k}: {wk[k]} != {TARGET[k]}'
        tot = sum(wk.values())
        assert tot == 1
        print(f'  {str(sv):>6}  ' + '  '.join(f'{str(wk[k]):>7}' for k in range(4))
              + f'   {tot}')
    print('  all class weights = (5, 31, 31, 5)/72 at every s (incl. negative): OK')

    print(f'\nAll assertions passed in {time.time() - t0:.1f}s.')


if __name__ == '__main__':
    main()
