"""
Benzene with the full (U, J, K, M) two-electron parameter set — the
three-index integral M is NOT dropped.  Tests whether the
(U - J) effective-parameter reduction (examples/benzene_hubbard_ujk.py)
survives when the ZDO approximation is relaxed.

Result (h = -1, s = 0):

    E_1 = (3/2) U + (27/2) J - 3 K + 6 M

    E_2 = -(29/288)(U - J)^2
          + (M/18 - 7K/36)(U - J)
          - K^2 + (4/9) K M - (16/9) M^2

Both the  J  and  J^2  coefficients vanish identically when E_2 is
re-expressed at fixed (U - J, K, M), so the (U - J) reduction
survives M retention: the identity  V_J|HF> = - V_U|HF>  is not
affected by adding V_M, which opens a third, independent
correlation channel with its own -(16/9) M^2 term.

Matrices are loaded from /tmp/benzene_ujk_matrices.pkl -- the
companion benzene_hubbard_ujk.py builds that cache.  This example
simply skips the  M: 0  substitution and re-runs the PT.
"""
import os
import pickle
import sys
import time

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, SlaterDet, symmetry


CACHE = '/tmp/benzene_ujk_matrices.pkl'
if not os.path.exists(CACHE):
    raise FileNotFoundError(
        f'{CACHE} missing; run examples/benzene_hubbard_ujk.py first '
        'to build the symbolic 400x400 matrices.'
    )
with open(CACHE, 'rb') as f:
    H1, S, H2 = pickle.load(f)

m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'],
    subst={'h': ('H_ab', 'H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af'),
           's': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af')},
    subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
              'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)
m.generate_basis(3, 3, 6)
det_strings = [fp.dets[0].det_string for fp in m.basis]


def canon(ds):
    fp = SlaterDet(ds).get_sorted()
    return fp.dets[0].det_string, fp.coefs[0]

C6  = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'e', 'e': 'f', 'f': 'a'}
sig = {'a': 'a', 'b': 'f', 'c': 'e', 'd': 'd', 'e': 'c', 'f': 'b'}
perms = [symmetry.apply_orbital_permutation(om, det_strings, canon)[0]
         for om in (C6, sig)]
U_mat, orbits = symmetry.totally_symmetric_basis(perms, 400)
Nblk = U_mat.shape[1]
orbit_sizes = [len(o) for o in orbits]
D = sp.diag(*[sp.Integer(sz) for sz in orbit_sizes])
Dinv = sp.diag(*[sp.Rational(1, sz) for sz in orbit_sizes])

U_sp = sp.zeros(400, Nblk)
for col, orb in enumerate(orbits):
    for idx in orb:
        U_sp[idx, col] = 1

h, s, U, J, K, M = sp.symbols('h s U J K M')
# Crucial: do NOT substitute M = 0 here; keep M as a live parameter
H_full = sp.Matrix(H1 + H2).subs({h: -1, s: 0})

print("Projecting to A_1g block ...")
t0 = time.time()
H_red = sp.Matrix(U_sp.T * H_full * U_sp)
print(f"  {time.time() - t0:.1f}s")

H0 = H_red.subs({U: 0, J: 0, K: 0, M: 0})
Mmat = Dinv * H0

E_sym = sp.Symbol('E')
roots_all = sorted(set(sp.solve(Mmat.charpoly(E_sym).as_expr(), E_sym)),
                   key=lambda r: float(r))
print(f"H_0 eigenvalues: {roots_all}")

eig = []
for r in roots_all:
    ker = (Mmat - r * sp.eye(Nblk)).nullspace()
    orth = []
    for v in ker:
        w = v
        for u in orth:
            w = w - ((u.T * D * w)[0, 0]) * u
        n2 = (w.T * D * w)[0, 0]
        if n2 == 0:
            continue
        orth.append(w / sp.sqrt(n2))
    for w in orth:
        eig.append((r, w))
Nd = len(eig)

V_red = sp.simplify(H_red - H0)
E0 = eig[0][0]
c0 = eig[0][1]
E_1 = sp.simplify((c0.T * V_red * c0)[0, 0])
print(f"\nE_1 = {E_1}")

print("\nBuilding V in eigenbasis (~4 min) ...")
t0 = time.time()
V_mn = sp.zeros(Nd, Nd)
for mi, (_, vm) in enumerate(eig):
    for ni, (_, vn) in enumerate(eig):
        V_mn[mi, ni] = sp.simplify((vm.T * V_red * vn)[0, 0])
print(f"  {time.time() - t0:.1f}s")

E_2 = sp.Rational(0)
for ni in range(1, Nd):
    denom = E0 - eig[ni][0]
    if denom == 0:
        continue
    E_2 += V_mn[0, ni] ** 2 / denom
E_2 = sp.expand(sp.simplify(E_2))
print(f"\nE_2 = {E_2}")

# Also compute E_3 via Wigner 2n+1 with |psi_1>
# |psi_1>[m] = V[m,0] / (E_0 - E_m) for m != 0
print("\nComputing E_3 (Feenberg recursion) ...")
r_diag = [sp.Rational(0)] * Nd
for n in range(1, Nd):
    denom = E0 - eig[n][0]
    if denom != 0:
        r_diag[n] = sp.Rational(1) / denom
psi1 = [sp.Rational(0)] * Nd
for mi in range(1, Nd):
    psi1[mi] = sp.simplify(r_diag[mi] * V_mn[mi, 0])
# E_3 = <psi_1|V|psi_1> - E_1 <psi_1|psi_1>
t0 = time.time()
psi1_V_psi1 = sp.simplify(sum(
    psi1[i] * V_mn[i, j] * psi1[j]
    for i in range(Nd) for j in range(Nd)
))
psi1_psi1 = sp.simplify(sum(psi1[i] ** 2 for i in range(Nd)))
E_3 = sp.simplify(psi1_V_psi1 - E_1 * psi1_psi1)
print(f"  {time.time() - t0:.1f}s")
print(f"\nE_3 = {E_3}")

# Test (U - J) reduction at E_3
x = sp.Symbol('x')
E3_fixed = sp.expand(E_3.subs(U, x + J))
print("\n(U - J) reduction test at 3rd order:")
for k in range(1, 4):
    print(f"  J^{k} coefficient at fixed (x=U-J, K, M): "
          f"{sp.simplify(E3_fixed.coeff(J, k))}")
E3_clean = sp.collect(sp.expand(E3_fixed), [x, K, M])
print(f"\nE_3 in terms of x = U - J:\n  {E3_clean}")

# Monomial inventory
print("\nMonomial coefficients of E_2:")
for mono in [U**2, J**2, K**2, M**2, U*J, U*K, U*M, J*K, J*M, K*M]:
    print(f"  {mono}:  {sp.simplify(E_2.coeff(mono))}")

# Test the (U - J) reduction
x = sp.Symbol('x')
E2_fixed = sp.expand(E_2.subs(U, x + J))
J_lin = sp.simplify(E2_fixed.coeff(J))
J_sq  = sp.simplify(E2_fixed.coeff(J, 2))
print(f"\nAt fixed  x = U - J:")
print(f"  coefficient of J^1:  {J_lin}")
print(f"  coefficient of J^2:  {J_sq}")
print(f"  => E_2 depends only on (x, K, M): {J_lin == 0 and J_sq == 0}")

E2_clean = sp.collect(sp.expand(E2_fixed), [x, K, M])
print(f"\nE_2 (expressed in terms of x = U - J):\n  {E2_clean}")
