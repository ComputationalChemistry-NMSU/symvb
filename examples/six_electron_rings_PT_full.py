"""
Full Hubbard perturbation series E_0(U) / |h|  =  sum_k  c_k (U/|h|)^k
for cyclic 6pi-electron rings L = 4, 5, 6 in closed form to several
orders.

Methods
-------
L = 4:  implicit series expansion of the known 3x3 characteristic
        polynomial of the singlet-A_1 block.  Every c_k is a rational
        number (signs from the Cardano structure).

L = 5:  Rayleigh-Schrodinger recursion inside the 12-dim A_1 block,
        using a generalised eigenvalue solver at 80-digit mpmath
        precision, followed by PSLQ recognition in the Q[sqrt(5)] basis.

L = 6:  taken verbatim from examples/benzene_hubbard_pt.py.

Convention: h is negative, U >= 0.  Results quoted at |h| = 1.
"""
import time
import numpy as np
import sympy as sp
import mpmath


# ---------------------------------------------------------------------------
# L = 4  via the singlet-A_1 cubic
# ---------------------------------------------------------------------------
def L4_series(n_max=6):
    lam, U, H = sp.symbols('lambda U H', real=True)
    # char poly at h = -H (so that E_0 at U=0 is -4H, the ground state)
    P = (lam**3 - 7*U*lam**2 + 16*(U**2 - H**2)*lam
         - 12*U**3 + 40*U*H**2)

    c = [sp.Symbol(f'c{k}', real=True) for k in range(n_max + 1)]
    lam_series = sum(c[k] * U**k for k in range(n_max + 1))
    expanded = sp.expand(P.subs(lam, lam_series))

    c_vals = {c[0]: -4*H}         # ground-state root at U = 0
    for n in range(1, n_max + 1):
        coeff_n = expanded.coeff(U, n).subs(c_vals)
        c_n = sp.solve(coeff_n, c[n])
        # pick the physical branch (real, analytic in U)
        c_vals[c[n]] = sp.simplify(c_n[0])
    return [sp.simplify(c_vals[c[k]].subs(H, 1)) for k in range(n_max + 1)]


# ---------------------------------------------------------------------------
# L = 5  via RSPT in the 12-dim A_1 block (mpmath + PSLQ)
# ---------------------------------------------------------------------------
def L5_series(n_max=6, dps=200):
    from symvb import Molecule, SlaterDet, symmetry

    orbs = list('abcde')
    edges = [''.join(sorted(orbs[i] + orbs[(i + 1) % 5])) for i in range(5)]
    m = Molecule(
        zero_ii=True,
        interacting_orbs=edges,
        subst={'h': tuple(f'H_{e}' for e in edges),
               's': tuple(f'S_{e}' for e in edges)},
    )
    m.generate_basis(3, 3, 5)
    det_strings = [fp.dets[0].det_string for fp in m.basis]
    N_full = len(det_strings)

    H1 = m.build_matrix(m.basis, op='H')

    def canon(ds):
        fp = SlaterDet(ds).get_sorted()
        return fp.dets[0].det_string, fp.coefs[0]

    C5 = {orbs[i]: orbs[(i + 1) % 5] for i in range(5)}
    sigma = {orbs[i]: orbs[(-i) % 5] for i in range(5)}
    perms = [symmetry.apply_orbital_permutation(om, det_strings, canon)[0]
             for om in (C5, sigma)]
    _, orbits = symmetry.totally_symmetric_basis(perms, N_full)
    N = len(orbits)

    # integer orbit-sum projector
    U_sp = sp.zeros(N_full, N)
    for col, orb in enumerate(orbits):
        for idx in orb:
            U_sp[idx, col] = 1
    D = sp.diag(*[sp.Integer(len(o)) for o in orbits])

    h_sym, s_sym = sp.symbols('h s')
    H0_full = sp.Matrix(H1).subs({h_sym: -1, s_sym: 0})

    def double_occ(ds):
        occ = {}
        for c in ds:
            occ.setdefault(c.lower(), [False, False])
            if c.islower():
                occ[c.lower()][0] = True
            else:
                occ[c.lower()][1] = True
        return sum(1 for ab in occ.values() if ab[0] and ab[1])

    V_full = sp.diag(*[sp.Integer(double_occ(d)) for d in det_strings])

    H0_red = sp.Matrix(U_sp.T * H0_full * U_sp)
    V_red  = sp.Matrix(U_sp.T * V_full  * U_sp)

    # generalised EVP in mpmath
    mpmath.mp.dps = dps
    H0_mp = mpmath.matrix(
        [[mpmath.mpf(sp.N(H0_red[i, j], dps)) for j in range(N)]
         for i in range(N)])
    D_mp  = mpmath.matrix(
        [[mpmath.mpf(D[i, j]) for j in range(N)] for i in range(N)])
    V_mp  = mpmath.matrix(
        [[mpmath.mpf(sp.N(V_red[i, j], dps)) for j in range(N)]
         for i in range(N)])

    D_sqrt_inv = mpmath.matrix(N, N)
    for i in range(N):
        D_sqrt_inv[i, i] = 1 / mpmath.sqrt(D_mp[i, i])
    A = D_sqrt_inv * H0_mp * D_sqrt_inv
    # symmetrise
    for i in range(N):
        for j in range(i):
            avg = (A[i, j] + A[j, i]) / 2
            A[i, j] = A[j, i] = avg
    evs, Q = mpmath.eigsy(A)
    order = sorted(range(N), key=lambda k: float(evs[k]))
    E_mp = [evs[k] for k in order]
    V_eig = D_sqrt_inv * mpmath.matrix(
        [[Q[i, order[j]] for j in range(N)] for i in range(N)])
    vecs = [[V_eig[i, k] for i in range(N)] for k in range(N)]

    def mv(M, v):
        return [mpmath.fsum(M[i, j] * v[j] for j in range(N))
                for i in range(N)]

    def inner(u, v):
        return mpmath.fsum(u[j] * v[j] for j in range(N))

    # RSPT recursion in the H_0 eigenbasis.  In that basis:
    #   H_0 is diagonal with entries E_mp[k]
    #   V has matrix elements  V_kl = <k| V |l>  = vecs[k]^T V_mp vecs[l]
    t0 = time.time()
    V_eigbasis = mpmath.matrix(N, N)
    # pre-compute V_mp * vecs[l] once per l
    V_vl = [mv(V_mp, vecs[l]) for l in range(N)]
    for k in range(N):
        for l in range(N):
            V_eigbasis[k, l] = inner(vecs[k], V_vl[l])
    print(f'  V in eigenbasis built in {time.time() - t0:.2f}s')

    # RSPT:
    #   |psi_n>  = R [(V - c_1) |psi_{n-1}>  -  sum_{k=2}^{n-1} c_k |psi_{n-k}>]
    #   c_{n+1} = <psi_0 | V | psi_n>
    # Working with coordinates |psi_n> = (alpha_{n,0}, ..., alpha_{n,N-1}) in
    # the H_0 eigenbasis.  R is diagonal: R_{kk} = 1/(E_0 - E_k)  for k != 0,
    # and R_{00} = 0 (projection onto the complement of |0>).
    E0_mp = E_mp[0]
    Rdiag = [mpmath.mpf(0)]
    for k in range(1, N):
        Rdiag.append(1 / (E0_mp - E_mp[k]))

    # |psi_0> = e_0
    psi = [[mpmath.mpf(1 if i == 0 else 0) for i in range(N)]]
    c_list = [E0_mp, V_eigbasis[0, 0]]

    for n in range(1, n_max):
        # rhs = V psi_{n-1}  -  sum_{k=1..n-1} c_{k+1} psi_{n-k}   ... careful:
        # Standard: rhs = (V - E_1) psi_{n-1} - sum_{k=2}^{n-1} E_k psi_{n-k}
        rhs = [mpmath.fsum(V_eigbasis[i, j] * psi[n - 1][j] for j in range(N))
               for i in range(N)]
        for i in range(N):
            rhs[i] -= c_list[1] * psi[n - 1][i]
        for k in range(2, n):
            for i in range(N):
                rhs[i] -= c_list[k] * psi[n - k][i]

        psi_n = [Rdiag[i] * rhs[i] for i in range(N)]
        psi.append(psi_n)
        c_new = mpmath.fsum(V_eigbasis[0, j] * psi_n[j] for j in range(N))
        c_list.append(c_new)

    return [E0_mp] + list(c_list[1:]), E_mp


def pslq_sqrt5(val, tol=mpmath.mpf('1e-60'), maxcoeff=10**40):
    if abs(val) < 1e-60:
        return sp.Integer(0)
    r = mpmath.pslq([val, mpmath.mpf(1), mpmath.sqrt(5)],
                    tol=tol, maxcoeff=maxcoeff, maxsteps=2000)
    if r is None:
        return None
    a, b, c = r
    if a == 0:
        return None
    return -(sp.Integer(b) + sp.Integer(c) * sp.sqrt(5)) / sp.Integer(a)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    n_max = 6
    print('=' * 72)
    print(f'Hubbard PT series for cyclic 6pi rings  (c_k in units of 1/|h|^(k-1))')
    print('=' * 72)

    t0 = time.time()
    c4 = L4_series(n_max)
    print(f'\nL = 4  (cyclobutadiene dianion)  -- cubic char-poly route')
    for k, val in enumerate(c4):
        print(f'   c_{k} = {val}   (decimal {float(val):+.10g})')
    print(f'  wall: {time.time() - t0:.1f}s')

    t0 = time.time()
    print(f'\nL = 5  (cyclopentadienyl anion)  -- 12-dim A_1 RSPT + PSLQ')
    c5_mp, E_spectrum = L5_series(n_max)
    for k, val in enumerate(c5_mp):
        # PSLQ-recognise in Q[sqrt(5)]
        rec = pslq_sqrt5(val)
        rec_str = f'{rec}' if rec is not None else '(PSLQ failed)'
        print(f'   c_{k} = {rec_str}   (decimal {float(val):+.10g})')
    print(f'  wall: {time.time() - t0:.1f}s')

    # L = 6 hardcoded from benzene_hubbard_pt.py
    print(f'\nL = 6  (benzene)  -- from benzene_hubbard_pt.py')
    c6 = [sp.Integer(-8),
          sp.Rational(3, 2),
          sp.Rational(-29, 288),
          sp.Integer(0),
          sp.Rational(-2855, 5971968),
          sp.Integer(0),
          None]        # c_6 not recorded; benzene script also does sixth-order
    for k, val in enumerate(c6):
        if val is None:
            continue
        print(f'   c_{k} = {val}   (decimal {float(val):+.10g})')

    # Side-by-side summary
    print('\n' + '=' * 72)
    print('SIDE-BY-SIDE SUMMARY')
    print('=' * 72)
    print(f'{"k":>2}  {"L = 4":>28}  {"L = 5":>28}  {"L = 6":>28}')
    print('-' * 94)
    for k in range(n_max + 1):
        l4 = f'{c4[k]}' if k < len(c4) else ''
        l5r = pslq_sqrt5(c5_mp[k]) if k < len(c5_mp) else None
        l5 = f'{l5r}' if l5r is not None else f'{float(c5_mp[k]):.6g}'
        l6 = f'{c6[k]}' if k < len(c6) and c6[k] is not None else '-'
        print(f'{k:>2}  {l4:>28}  {l5:>28}  {l6:>28}')


if __name__ == '__main__':
    main()
