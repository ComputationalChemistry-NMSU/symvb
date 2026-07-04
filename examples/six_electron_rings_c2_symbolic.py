"""
Symbolic extraction of the MP2 coefficient c_2 in the PT series

    E_0(h, U) = c_0 + c_1 U + c_2 U^2 + ...

for all three cyclic 6pi-electron rings: L = 4, 5, 6.

Method
------
L = 4: the singlet-A_1 block is 3x3 with known characteristic polynomial
       lambda^3 - 7U lambda^2 + 16(U^2 - h^2) lambda - 12 U^3 + 40 U h^2 = 0.
       Implicit differentiation wrt U around the ground-state root gives
       c_1 and c_2 in closed form.

L = 5: 12-dim A_1 block.  Symbolic diagonalisation of H_0 (at U = 0, s = 0)
       and Rayleigh-Schrodinger on the sorted eigenbasis, with orbit-size
       D-metric normalisation (same pattern as examples/benzene_hubbard_pt.py).

L = 6: quoted from examples/benzene_hubbard_pt.py:  c_2 = -29/288.

All three are at h < 0, s = 0.
"""
import time
import numpy as np
import sympy as sp

from symvb import Molecule, SlaterDet, symmetry


def implicit_c2_for_L4():
    """L = 4: char-poly route.  Returns c_0, c_1, c_2 as sympy expressions."""
    print('-' * 72)
    print('L = 4  (cyclobutadiene dianion)')
    print('-' * 72)

    lam, U, h = sp.symbols('lambda U h', real=True)
    # Derived earlier from sign-aware A_1 block (verified against 16-dim FCI).
    P = (lam**3
         - 7*U*lam**2
         + 16*(U**2 - h**2)*lam
         - 12*U**3 + 40*U*h**2)
    print('Characteristic polynomial of singlet-A_1 block (3 dim):')
    print(f'  P(lambda; U, h) = lambda^3 - 7U*lambda^2 + 16(U^2 - h^2) lambda')
    print(f'                     - 12 U^3 + 40 U h^2')

    # Ground-state root at U = 0:  lambda^3 - 16 h^2 lambda = 0  ->  -4|h|
    # Choose h = -|h| (keep h symbolic, track sign via abs)
    # Implicit series: lambda(U) around U = 0 with lambda(0) = -4 |h|.
    # Substitute h -> -H with H > 0 to avoid signed radicals.
    H = sp.symbols('H', positive=True)
    P_H = P.subs(h, -H)
    lam_0 = -4 * H

    # First derivative
    dP_dlam = sp.diff(P_H, lam)
    dP_dU   = sp.diff(P_H, U)

    lam1 = -dP_dU / dP_dlam
    c1 = sp.simplify(lam1.subs({lam: lam_0, U: 0}))
    print(f'\n  c_1 = -dP/dU  /  dP/dlambda  |_(lambda=-4H, U=0)')
    print(f'      = {c1}')

    # Second derivative.  Differentiate P(lambda(U), U) = 0 twice:
    #   P_lambda_lambda (lambda')^2 + 2 P_lambda_U lambda' + P_UU
    #     + P_lambda lambda'' = 0
    d2P_dlam2  = sp.diff(P_H, lam, 2)
    d2P_dlamdU = sp.diff(P_H, lam, U)
    d2P_dU2    = sp.diff(P_H, U, 2)

    lam2_formula = -(d2P_dlam2 * lam1**2 + 2 * d2P_dlamdU * lam1 + d2P_dU2) / dP_dlam
    c2 = sp.simplify(lam2_formula.subs({lam: lam_0, U: 0}) / 2)
    print(f'\n  c_2 = lambda\'\'(0) / 2  (from implicit 2nd derivative)')
    print(f'      = {c2}')

    return sp.Integer(-4) * H, c1, c2


def _double_occ(ds):
    occ = {}
    for c in ds:
        occ.setdefault(c.lower(), [False, False])
        if c.islower():
            occ[c.lower()][0] = True
        else:
            occ[c.lower()][1] = True
    return sum(1 for ab in occ.values() if ab[0] and ab[1])


def symbolic_c2_via_pt(L, A1_dim_expected, flush=True):
    """
    L in {5, 6}: benzene-style symbolic A_1 block with D-metric
    (orbit-size diagonal) and Rayleigh-Schrodinger.  V_U is the
    diagonal double-occupancy count (site-local Hubbard), so we
    build it directly without calling o2_matrix.
    Returns (c_0, c_1, c_2) with h = -1, s = 0.
    """
    def p(*args, **kw):
        kw['flush'] = flush
        print(*args, **kw)

    p('-' * 72)
    p(f'L = {L}')
    p('-' * 72)
    orbs = [chr(ord('a') + i) for i in range(L)]
    edges = [''.join(sorted(orbs[i] + orbs[(i + 1) % L])) for i in range(L)]

    m = Molecule(
        zero_ii=True,
        interacting_orbs=edges,
        subst={'h': tuple(f'H_{e}' for e in edges),
               's': tuple(f'S_{e}' for e in edges)},
    )
    m.generate_basis(3, 3, L)
    det_strings = [fp.dets[0].det_string for fp in m.basis]
    N_full = len(det_strings)
    p(f'  full Sz=0 dim = {N_full}')

    t0 = time.time()
    H1 = m.build_matrix(m.basis, op='H')
    p(f'  symbolic H1 built in {time.time() - t0:.1f}s')

    def canon(ds):
        fp = SlaterDet(ds).get_sorted()
        return fp.dets[0].det_string, fp.coefs[0]

    C_L = {orbs[i]: orbs[(i + 1) % L] for i in range(L)}
    sigma = {orbs[i]: orbs[(-i) % L] for i in range(L)}
    perms = [symmetry.apply_orbital_permutation(om, det_strings, canon)[0]
             for om in (C_L, sigma)]
    _, orbits = symmetry.totally_symmetric_basis(perms, N_full)
    N = len(orbits)
    p(f'  A_1 block dim = {N}   (expected {A1_dim_expected})')

    # Integer orbit-sum projector (no sqrt factors)
    U_sp = sp.zeros(N_full, N)
    for col, orb in enumerate(orbits):
        for idx in orb:
            U_sp[idx, col] = 1
    orbit_sizes = [len(o) for o in orbits]
    D = sp.diag(*[sp.Integer(sz) for sz in orbit_sizes])

    h_sym, s_sym = sp.symbols('h s')
    H0_full = sp.Matrix(H1).subs({h_sym: -1, s_sym: 0})
    V_full = sp.diag(*[sp.Integer(_double_occ(d)) for d in det_strings])

    t0 = time.time()
    H0_red = sp.Matrix(U_sp.T * H0_full * U_sp)
    V_red  = sp.Matrix(U_sp.T * V_full  * U_sp)
    p(f'  projected in {time.time() - t0:.1f}s')

    # Generalised eigenvalue problem  H0_red x = E D x
    # equivalent to  D^{-1} H0_red x = E x.
    Dinv = sp.diag(*[sp.Rational(1, sz) for sz in orbit_sizes])
    M = Dinv * H0_red

    t0 = time.time()
    E_sym = sp.Symbol('E')
    cp = M.charpoly(E_sym).as_expr()
    p(f'  char-poly built in {time.time() - t0:.1f}s; solving...')
    t0 = time.time()
    eigenvalues = sorted(set(sp.solve(cp, E_sym)), key=lambda r: float(r))
    p(f'  solved in {time.time() - t0:.1f}s')
    p(f'  distinct A_1 eigenvalues: {eigenvalues}')
    E0 = eigenvalues[0]
    p(f'  ground-state energy E_0 = {E0}')

    # Solve generalised EVP H0_red v = E D v directly in high-precision mpmath,
    # which side-steps sympy Gram-Schmidt on nested radicals (known to fail).
    import mpmath
    mpmath.mp.dps = 80
    t0 = time.time()
    H0_mp = mpmath.matrix([[mpmath.mpf(sp.N(H0_red[i, j], 80))
                            for j in range(N)] for i in range(N)])
    D_mp = mpmath.matrix([[mpmath.mpf(D[i, j]) for j in range(N)]
                          for i in range(N)])
    # Convert to standard EVP:  D^{-1/2} H0 D^{-1/2} y = E y, v = D^{-1/2} y
    D_sqrt_inv = mpmath.matrix(N, N)
    for i in range(N):
        D_sqrt_inv[i, i] = 1 / mpmath.sqrt(D_mp[i, i])
    A = D_sqrt_inv * H0_mp * D_sqrt_inv
    # Hermitise for safety
    for i in range(N):
        for j in range(i):
            a = (A[i, j] + A[j, i]) / 2
            A[i, j] = A[j, i] = a
    evs_mp, Q = mpmath.eigsy(A)            # Hermitian eigensolve
    order = sorted(range(N), key=lambda k: float(evs_mp[k]))
    E_mp = [evs_mp[k] for k in order]
    # y_k column k;  v_k = D^{-1/2} y_k  (D-orthonormal)
    V_eig = D_sqrt_inv * mpmath.matrix(
        [[Q[i, order[j]] for j in range(N)] for i in range(N)])
    v_lists = [[V_eig[i, k] for i in range(N)] for k in range(N)]
    p(f'  mpmath eigensolve in {time.time() - t0:.1f}s')

    # D-normalisation check
    for k in range(N):
        dn = mpmath.fsum(mpmath.mpf(D[i, i]) * v_lists[k][i]**2
                         for i in range(N))
        if abs(dn - 1) > mpmath.mpf('1e-50'):
            p(f'  WARNING v[{k}] ||v||_D^2 = {dn}')

    # Map to sympy eigenvalues for presentation
    eig_data = [(E_mp[k], None) for k in range(N)]
    E0 = sp.nsimplify(E_mp[0], [sp.sqrt(5)] if L == 5 else [], rational=True,
                      tolerance=mpmath.mpf('1e-50'))
    p(f'  ground-state energy E_0 (clean) = {E0}')

    # Convert V_red to mpmath
    t0 = time.time()
    V_mp = [[mpmath.mpf(sp.N(V_red[i, j], 80))
             for j in range(V_red.cols)]
            for i in range(V_red.rows)]
    p(f'  V_red -> mpmath in {time.time() - t0:.1f}s')

    def mat_vec(M, v):
        n = len(v)
        return [mpmath.fsum(M[i][j] * v[j] for j in range(n)) for i in range(n)]

    def inner(u, v):
        return mpmath.fsum(u[j] * v[j] for j in range(len(u)))

    v0 = v_lists[0]
    Vv0 = mat_vec(V_mp, v0)
    c1_mp = inner(v0, Vv0)
    c1_rat = sp.nsimplify(c1_mp, rational=True, tolerance=mpmath.mpf('1e-50'))
    p(f'\n  c_1 (mpmath) = {c1_mp}')
    p(f'  c_1 (clean)  = {c1_rat}')

    # c_2 = sum_{k > 0}  |<k|V|0>|^2 / (E_0 - E_k)
    t0 = time.time()
    c2_mp = mpmath.mpf(0)
    E0_mp = E_mp[0]
    for k in range(1, len(eig_data)):
        if abs(E_mp[k] - E0_mp) < mpmath.mpf('1e-30'):
            continue
        Vk0 = inner(v_lists[k], Vv0)
        c2_mp += Vk0 * Vk0 / (E0_mp - E_mp[k])
    p(f'  c_2 summed in {time.time() - t0:.2f}s')
    p(f'  c_2 (mpmath, 60 dp) = {c2_mp}')

    t0 = time.time()
    if L == 5:
        # PSLQ:  find integers (a, b, c) with  a*c_2 + b + c*sqrt(5) = 0
        pslq = mpmath.pslq([c2_mp, mpmath.mpf(1), mpmath.sqrt(5)],
                           tol=mpmath.mpf('1e-60'))
        if pslq is None:
            p('  PSLQ failed to recognise c_2 in {1, sqrt(5)} basis')
            c2_clean = c2_mp
        else:
            a, b, c = pslq
            c2_clean = -(sp.Integer(b) + sp.Integer(c) * sp.sqrt(5)) / sp.Integer(a)
            p(f'  PSLQ: a*c_2 + b + c*sqrt(5) = 0  with (a, b, c) = {pslq}')
    else:
        c2_clean = sp.Rational(mpmath.nstr(c2_mp, 30))
    p(f'  c_2 (closed form) = {c2_clean}   (in {time.time() - t0:.2f}s)')
    residual = abs(mpmath.mpf(sp.N(c2_clean, 80)) - c2_mp)
    p(f'  residual = {residual}')

    # Similarly cleanup of E_0 for L = 5
    if L == 5:
        pslq = mpmath.pslq([E_mp[0], mpmath.mpf(1), mpmath.sqrt(5)],
                           tol=mpmath.mpf('1e-60'))
        if pslq is not None:
            a, b, c = pslq
            E0 = -(sp.Integer(b) + sp.Integer(c) * sp.sqrt(5)) / sp.Integer(a)
            p(f'  E_0 (clean) = {E0}')

    return E0, c1_rat, c2_clean


def main():
    print('=' * 72)
    print('Symbolic c_2 coefficient for cyclic 6pi-electron rings')
    print('(Rayleigh-Schrodinger, h = -1, s = 0)')
    print('=' * 72 + '\n')

    c0_4, c1_4, c2_4 = implicit_c2_for_L4()
    # L = 4 was derived at general h; specialise h = -1 for numerical compare
    H_sym = sp.symbols('H', positive=True)
    print(f'\n  [specialise |h|=1:]')
    print(f'    c_0 = {c0_4.subs(H_sym, 1)}')
    print(f'    c_1 = {c1_4.subs(H_sym, 1)}')
    print(f'    c_2 = {c2_4.subs(H_sym, 1)}')
    print(f'    c_2 (decimal) = {float(c2_4.subs(H_sym, 1))}')

    print('\n')
    c0_5, c1_5, c2_5 = symbolic_c2_via_pt(5, A1_dim_expected=12)

    print('\n' + '=' * 72)
    print('SUMMARY')
    print('=' * 72)
    print(f'L = 4:  c_0 = -4|h|,   c_1 = 9/4,   c_2 = {c2_4.subs(H_sym, 1)} '
          f'(per unit |h|)')
    print(f'L = 5:  c_0 = {c0_5},   c_1 = {c1_5},   c_2 = {c2_5}')
    print(f'L = 6:  c_0 = -8,       c_1 = 3/2,   c_2 = -29/288  '
          f'(from benzene_hubbard_pt.py)')


if __name__ == '__main__':
    main()
