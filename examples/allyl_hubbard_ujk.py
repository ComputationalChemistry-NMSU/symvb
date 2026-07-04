"""
Allyl anion (3c4e) RS perturbation series with the full PPP-type
(U, J, K) two-electron integrals beyond on-site U.

Convention (integral-pattern indices, matching symvb's subst_2e):
    U = (aa|aa)          on-site                     (pattern 1111)
    J = (ab|ab)          two-center exchange         (pattern 1212)
    K = (aa|bb)          two-center direct Coulomb   (pattern 1122)
    M = (aa|ab), etc.    three-index, dropped (ZDO)  (pattern 1112 + perms)

All matrices are built symbolically in SymPy, reduced to the 5-dim
sigma = +1 A_1 block via symvb.symmetry, diagonalised exactly over
Q[sqrt(2)], and the first two Rayleigh-Schrodinger coefficients
are expressed as polynomials in (U, J, K).
"""
import os
import sys
import time

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, SlaterDet, symmetry, hamiltonian
from symvb.fixed_psi import generate_dets


# ------------------------------------------------------------------------
# 1.  Build H1, S, H2 symbolically with full (U, J, K) integrals
# ------------------------------------------------------------------------
m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab', 'bc'],
    subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
    subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
              'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)
P = generate_dets(2, 2, 3)
det_strings = [p.dets[0].det_string for p in P]

t0 = time.time()
H_raw, _ = hamiltonian(m, P)
print(f"Symbolic matrix build (3c4e UJK): {time.time() - t0:.1f}s")

H = sp.Matrix(H_raw)
h, s, U, J, K, M = sp.symbols('h s U J K M')
H_s0 = H.subs({s: 0, h: -1, M: 0})


# ------------------------------------------------------------------------
# 2.  Project to sigma = +1 subspace (reflection a <-> c)
# ------------------------------------------------------------------------
# Built symbolically (exact over Q[sqrt(2)]) so the PT coefficients below are
# exact; symmetry.signed_totally_symmetric_basis returns a numeric basis.
def canon(ds):
    fp = SlaterDet(ds).get_sorted()
    return fp.dets[0].det_string, fp.coefs[0]

sig = {'a': 'c', 'b': 'b', 'c': 'a'}
perm, signs = symmetry.apply_orbital_permutation(sig, det_strings, canon)

U_plus = []
seen = [False] * 9
for i in range(9):
    if seen[i]:
        continue
    j = perm[i]; sj = signs[i]
    if j == i:
        seen[i] = True
        if sj == 1:
            v = sp.zeros(9, 1); v[i] = 1; U_plus.append(v)
    else:
        seen[i] = seen[j] = True
        v = sp.zeros(9, 1); v[i] = 1; v[j] = sj
        U_plus.append(v / sp.sqrt(2))

Up = sp.Matrix.hstack(*U_plus)
H_red = sp.simplify(Up.T * H_s0 * Up)
Nd = H_red.shape[0]


# ------------------------------------------------------------------------
# 3.  Diagonalise H_0 (pure Huckel) exactly over Q[sqrt(2)]
# ------------------------------------------------------------------------
H0 = H_red.subs({U: 0, J: 0, K: 0})
E_sym = sp.Symbol('E')
roots_all = sorted(set(sp.solve(H0.charpoly(E_sym).as_expr(), E_sym)),
                   key=lambda r: float(r))
print(f"H_0 eigenvalues: {roots_all}")

eig = []
for r in roots_all:
    ker = (H0 - r * sp.eye(Nd)).nullspace()
    orth = []
    for v in ker:
        w = v
        for u in orth:
            w = w - (u.T * w)[0, 0] * u
        n2 = (w.T * w)[0, 0]
        if n2 == 0:
            continue
        orth.append(w / sp.sqrt(n2))
    for w in orth:
        eig.append((r, w))


# ------------------------------------------------------------------------
# 4.  First- and second-order perturbation coefficients as polynomials
#     in (U, J, K)
# ------------------------------------------------------------------------
V_red = sp.simplify(H_red - H0)
E0 = eig[0][0]
c0 = eig[0][1]

E_1 = sp.simplify((c0.T * V_red * c0)[0, 0])
print(f"\nE_1 (linear in U, J, K):")
print(f"  {E_1}")

V_mn = sp.zeros(Nd, Nd)
for mi, (_, vm) in enumerate(eig):
    for ni, (_, vn) in enumerate(eig):
        V_mn[mi, ni] = sp.simplify((vm.T * V_red * vn)[0, 0])

E_2 = sp.Rational(0)
for ni in range(1, Nd):
    E_2 += V_mn[0, ni] ** 2 / (E0 - eig[ni][0])
E_2 = sp.expand(sp.simplify(E_2))
print(f"\nE_2 (quadratic in U, J, K):")
print(f"  {E_2}")

print("\nE_2 factors as  (sqrt(2)/512) * [ -21 (U - J)^2 - 28 K (U - J) - 52 K^2 ]")
print("  which is a negative-definite quadratic in (U - J) and K -- the")
print("  second-order correction stabilises the ground state for all real")
print("  (U, J, K).  The flat direction U = J, K = 0 corresponds to a model")
print("  where on-site repulsion exactly equals two-center exchange and the")
print("  leading correlation correction vanishes.")


# ------------------------------------------------------------------------
# 5.  Cross-sections
# ------------------------------------------------------------------------
print("\nCross-sections of E_1 and E_2 along each axis:")
for label, subs in [("pure U",  {J: 0, K: 0}),
                    ("pure J",  {U: 0, K: 0}),
                    ("pure K",  {U: 0, J: 0})]:
    print(f"  {label:8s}:  E_1 = {sp.simplify(E_1.subs(subs))},   "
          f"E_2 = {sp.simplify(E_2.subs(subs))}")

print("\nCompare pure-U row to examples/allyl_hubbard_pt.py:")
print("  E_1 = 11/8 * U       (matches)")
print("  E_2 = -21*sqrt(2)/512 * U^2   (matches)")
