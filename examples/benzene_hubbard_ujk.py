"""
Benzene with full PPP-type (U, J, K) two-electron integrals beyond
on-site U.

Same recipe as examples/benzene_hubbard_pt.py extended to three
perturbation parameters:

    U = (aa|aa)          on-site                     (pattern 1111)
    J = (ab|ab)          two-center exchange         (pattern 1212)
    K = (aa|bb)          two-center direct Coulomb   (pattern 1122)
    M = (aa|ab), etc.    three-index, dropped (ZDO)  (pattern 1112 + perms)

(The symbol names follow symvb's integral-pattern convention; in MO
language our J is exchange and our K is direct Coulomb.)

Matrices are cached in /tmp/benzene_ujk_matrices.pkl -- if the cache
is missing, uncomment the "build" block or run the companion
/tmp/benzene_ujk_bg.py generator.

Results (h = -1, s = 0, M = 0):

    E_1  =  (3/2) U  +  (27/2) J  -  3 K

    E_2  =  -(1/288) [ 29 (U - J)^2  -  56 K (U - J)  +  288 K^2 ]

The (U - J) combination acts as an effective single parameter, with
the K channel mixing into it through the cross term.  The quadratic
form in (U - J, K) is negative-definite -- second-order correlation
stabilises the ground state for all real (U, J, K).

Run from the repo root:  PYTHONPATH=. python3 examples/benzene_hubbard_ujk.py
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
    print(f'building {CACHE} (~8 minutes) ...')
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
    t0 = time.time()
    H1 = m.build_matrix(m.basis, op='H')
    S  = m.build_matrix(m.basis, op='S')
    H2 = m.o2_matrix(m.basis)
    print(f'  built in {time.time() - t0:.0f}s')
    with open(CACHE, 'wb') as f:
        pickle.dump((H1, S, H2), f)

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


# ------------------------------------------------------------------------
# A_1g projection (same as examples/benzene_hubbard_pt.py)
# ------------------------------------------------------------------------
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
H_full = sp.Matrix(H1 + H2).subs({h: -1, s: 0, M: 0})

print(f"A_1g block dimension: {Nblk}")

print("Projecting to A_1g block ...")
t0 = time.time()
H_red = sp.Matrix(U_sp.T * H_full * U_sp)
print(f"  {time.time() - t0:.1f}s")

# H_0: set U = J = K = 0
H0 = H_red.subs({U: 0, J: 0, K: 0})
Mmat = Dinv * H0

# Diagonalise H_0 over Q
E_sym = sp.Symbol('E')
roots_all = sorted(set(sp.solve(Mmat.charpoly(E_sym).as_expr(), E_sym)),
                   key=lambda r: float(r))
print(f"H_0 eigenvalues: {roots_all}")

# D-orthonormal eigenvectors
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


# ------------------------------------------------------------------------
# Perturbation coefficients
# ------------------------------------------------------------------------
V_red = sp.simplify(H_red - H0)
E0 = eig[0][0]
c0 = eig[0][1]

E_1 = sp.simplify((c0.T * V_red * c0)[0, 0])
print(f"\nE_1 (linear in U, J, K):\n  {E_1}")

print("\nBuilding V in eigenbasis ...")
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
print(f"\nE_2 (quadratic):\n  {E_2}")
print(f"\nfactored:  {sp.factor(E_2)}")

x = sp.Symbol('x')
E_2_in_x = sp.simplify(E_2.subs(U, x + J))  # substitute U = J + x
print(f"\nIn terms of x = U - J:\n  {sp.collect(sp.expand(E_2_in_x), [x, K])}")

print("\n--- Cross-sections ---")
for label, subs in [("pure U",  {J: 0, K: 0}),
                    ("pure J",  {U: 0, K: 0}),
                    ("pure K",  {U: 0, J: 0})]:
    print(f"  {label}:  E_1 = {sp.simplify(E_1.subs(subs))},  "
          f"E_2 = {sp.simplify(E_2.subs(subs))}")

print("\n" + "=" * 66)
print("Comparison with allyl (examples/allyl_hubbard_ujk.py):")
print("  allyl    E_1 = (11/8) U + (37/8) J - (3/4) K")
print("  allyl    E_2 = (sqrt(2)/512) [-21 (U-J)^2 - 28 K(U-J) - 52 K^2]")
print("  benzene  E_1 = (3/2) U  + (27/2) J - 3 K")
print("  benzene  E_2 = -(1/288) [29 (U-J)^2 - 56 K(U-J) + 288 K^2]")
print()
print("Both systems share the (U - J) effective-parameter structure and")
print("the negative-definite quadratic form in (U - J, K).  The benzene")
print("rational coefficients are the multi-variable generalisation of the")
print("familiar  E_2 = -29/288 U^2  from the pure-Hubbard case.")
