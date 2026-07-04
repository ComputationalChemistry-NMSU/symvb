"""
Rayleigh-Schrodinger perturbation series for the 3-centre, 4-electron
linear chain (allyl anion) with on-site Hubbard U.

Same recipe as examples/benzene_hubbard_pt.py applied to a smaller
system so every step can be inspected by eye.

Setup:
    - 3 orbitals a, b, c with couplings h on (a,b) and (b,c) only
    - 4 electrons (2 alpha + 2 beta), Sz = 0 subspace has 9 dets
    - orthogonal AOs  (s = 0),  h = -1,  on-site U

Key differences from benzene:
    1. Odd orders do NOT vanish (4/3-filling, no particle-hole symmetry).
    2. Coefficients alternate between pure rationals and  Q * sqrt(2):
       even orders carry sqrt(2), odd orders do not.
    3. The mean-field coefficient E_1 = 11/8 decodes cleanly as a sum
       over sites of  (density/2)^2  in the Huckel reference.
"""
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, SlaterDet, symmetry, hamiltonian
from symvb.fixed_psi import generate_dets


# ---------------------------------------------------------------------------
# 1. Build H_1e, S, H_2e for 3c4e chain
# ---------------------------------------------------------------------------
m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab', 'bc'],
    subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
    subst_2e={'U': ('1111',)},
    max_2e_centers=1,
)

P = generate_dets(2, 2, 3)
det_strings = [p.dets[0].det_string for p in P]
print(f"Basis ({len(P)} determinants):")
for d in det_strings:
    print(f"  |{d}|")

H_raw, S_raw = hamiltonian(m, P)
H  = sp.Matrix(H_raw)
S_mat = sp.Matrix(S_raw)

h, s, U = sp.symbols('h s U')
H_s0 = H.subs({s: 0, h: -1})
assert S_mat.subs({s: 0}) == sp.eye(9), "S at s=0 should be identity"


# ---------------------------------------------------------------------------
# 2. Project to the sigma = +1 subspace (reflection a <-> c)
# ---------------------------------------------------------------------------
# Built symbolically here (exact over Q[sqrt(2)]) so the RS series below is
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
    j = perm[i]
    sj = signs[i]
    if j == i:
        seen[i] = True
        if sj == 1:
            v = sp.zeros(9, 1); v[i] = 1
            U_plus.append(v)
    else:
        seen[i] = seen[j] = True
        v = sp.zeros(9, 1); v[i] = 1; v[j] = sj
        U_plus.append(v / sp.sqrt(2))

Up = sp.Matrix.hstack(*U_plus)
Nd = Up.shape[1]
print(f"\nsigma = +1 block dimension: {Nd}")

H_red = sp.simplify(Up.T * H_s0 * Up)
print("\nH in the sigma = +1 block (h = -1, s = 0):")
sp.pprint(H_red)


# ---------------------------------------------------------------------------
# 3. Diagonalise H_0 = H_red|_{U=0}  exactly over Q[sqrt(2)]
# ---------------------------------------------------------------------------
H0_red = H_red.subs(U, 0)
V_red  = sp.simplify(H_red - H0_red).subs(U, 1)

E_sym = sp.Symbol('E')
cp = H0_red.charpoly(E_sym).as_expr()
roots_all = sorted(set(sp.solve(cp, E_sym)), key=lambda r: float(r))
print(f"\nEigenvalues of H_0 in sigma=+1 block: {roots_all}")

eig = []
for r in roots_all:
    ker = (H0_red - r * sp.eye(Nd)).nullspace()
    orth = []
    for v in ker:
        w = v
        for u in orth:
            w = w - (u.T * w)[0, 0] * u
        norm2 = (w.T * w)[0, 0]
        if norm2 == 0:
            continue
        orth.append(w / sp.sqrt(norm2))
    for w in orth:
        eig.append((r, w))

# V in eigenbasis
V_mn = sp.zeros(Nd, Nd)
for mi, (_, vm) in enumerate(eig):
    for ni, (_, vn) in enumerate(eig):
        V_mn[mi, ni] = sp.simplify((vm.T * V_red * vn)[0, 0])


# ---------------------------------------------------------------------------
# 4. RS perturbation recursion
# ---------------------------------------------------------------------------
E0 = eig[0][0]
r_diag = [sp.Rational(0)] * Nd
for n in range(1, Nd):
    denom = E0 - eig[n][0]
    if denom != 0:
        r_diag[n] = sp.Rational(1) / denom

n_max = 5
psi = [None] * (n_max + 1)
psi[0] = [sp.Rational(0)] * Nd
psi[0][0] = sp.Rational(1)

E_coef = [sp.Rational(0)] * (n_max + 2)
E_coef[0] = E0
E_coef[1] = sp.simplify(V_mn[0, 0])

for n in range(1, n_max + 1):
    rhs = [sum(V_mn[m, l] * psi[n-1][l] for l in range(Nd)) for m in range(Nd)]
    for m in range(Nd):
        rhs[m] = sp.simplify(rhs[m] - E_coef[1] * psi[n-1][m])
    for k in range(2, n):
        for m in range(Nd):
            rhs[m] = sp.simplify(rhs[m] - E_coef[k] * psi[n-k][m])

    psi[n] = [sp.simplify(r_diag[m] * rhs[m]) if r_diag[m] != 0 else sp.Rational(0)
              for m in range(Nd)]
    E_coef[n + 1] = sp.simplify(sum(V_mn[0, m] * psi[n][m] for m in range(Nd)))


print("\n=== RS perturbation series for allyl anion (h=-1, s=0) ===")
for k in range(len(E_coef)):
    c = sp.simplify(E_coef[k])
    print(f"  E_{k} = {c}    ({float(c):+.10f})")


# ---------------------------------------------------------------------------
# 5. Decode E_1 via Huckel MOs
# ---------------------------------------------------------------------------
# Huckel MO coefficients for 3-chain with h = -1:
#   psi_1 = (1/2, sqrt(2)/2, 1/2)         E = -sqrt(2)
#   psi_2 = (1/sqrt(2), 0, -1/sqrt(2))    E =  0
#   psi_3 = (1/2, -sqrt(2)/2, 1/2)        E = +sqrt(2)
print("\n=== Decoding E_1 from Huckel MO densities ===")
C1 = (sp.Rational(1, 2), sp.sqrt(2) / 2, sp.Rational(1, 2))
C2 = (1 / sp.sqrt(2), sp.Integer(0), -1 / sp.sqrt(2))
densities = [sp.simplify(2 * (C1[k]**2 + C2[k]**2)) for k in range(3)]
print(f"  Huckel densities (alpha+beta per site): "
      f"rho_a = {densities[0]}, rho_b = {densities[1]}, rho_c = {densities[2]}")
E1_decoded = sum((rho / 2) ** 2 for rho in densities)
E1_decoded = sp.simplify(E1_decoded)
print(f"  E_1 = sum_k (rho_k / 2)^2 = {E1_decoded}")
print(f"  symvb PT result:             {E_coef[1]}")
print(f"  Match: {E1_decoded == E_coef[1]}")


# ---------------------------------------------------------------------------
# 6. Comparison with benzene
# ---------------------------------------------------------------------------
print("\n=== Comparison with benzene (6c6e, half-filling) ===")
print(f"{'order':>6}  {'allyl (3c4e)':>28}  {'benzene (6c6e)':>28}")
print(f"{'n':>6}  {'E_n':>28}  {'E_n':>28}")
print("-" * 72)
benzene_coef = [
    ('-8', -8.0),
    ('3/2', 1.5),
    ('-29/288', -0.10069),
    ('0', 0.0),
    ('-2855/5971968', -4.78e-4),
    ('0', 0.0),
    ('855791/61917364224', 1.38e-5),
]
for k in range(7):
    ec = sp.simplify(E_coef[k])
    ec_num = float(ec)
    bz_str, bz_num = benzene_coef[k]
    print(f"  {k:>4}  {str(ec):>28}  {bz_str:>28}")

print("\nObservations:")
print("  * allyl: odd orders survive (E_1, E_3, E_5 all != 0)")
print("  * benzene: E_3 = E_5 = 0 by particle-hole symmetry at half-filling")
print("  * allyl coefficients carry sqrt(2) on even orders (Huckel gap scale)")
print("  * benzene coefficients are pure rational (Huckel gaps are integers)")
