"""Full-FCI single-edge weakening of benzene vs on-site Hubbard U (s = 0.2).

The covalent-only Rumer model has a wrong-sign maximum in E(lambda); the exact
FCI is monotone at small U. The covalent five-structure space is the
U -> infinity projection of the ring (ionic determinants frozen out by on-site
repulsion), so the FCI inherits the same maximum at strong correlation: it is
monotone up to U/|h| ~ 8.25 and develops the bump above it, approaching the
covalent-only curve as U grows.

H = H1(h, s, H_ab) + U * H2(s), with H2 the on-site (1111) two-electron matrix
built by o2_matrix; H is linear in both H_ab and U, so three numerical builds
span the (lambda, U) plane. Builds are cached in /tmp (first H1 build ~ minutes;
both caches are shared with examples/make_figures.py).

Run from the repo root:  PYTHONPATH=. python3 examples/benzene_fci_vs_U_onset.py
Runtime ~7 min with the /tmp caches present (four dense 400x400 symbolic
substitutions, then a fine U-bisection of the full-FCI lowest eigenvalue).
"""
import os
import pickle
import numpy as np
from scipy.linalg import eigh
import sympy as sp
from symvb import Molecule

h, s = sp.symbols('h s')
Hab = sp.Symbol('H_ab')
Usym = sp.Symbol('U')
H_VAL, S_VAL = -1.0, 0.2
SUBST = dict(
    zero_ii=True,
    subst={'s': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af'),
           'h': ('H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af')},   # H_ab kept free
    interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'])

C_H1 = '/tmp/benzene_full_aromaticity_HS.pkl'         # (H1, S), one-electron
C_H2 = '/tmp/benzene_aromaticity_H2_onsite.pkl'       # on-site U two-electron
if os.path.exists(C_H1):
    H1, S = pickle.load(open(C_H1, 'rb'))
else:
    mf = Molecule(**SUBST)
    mf.generate_basis(3, 3, 6)
    H1 = mf.build_matrix(mf.basis, op='H')
    S = mf.build_matrix(mf.basis, op='S')
    pickle.dump((H1, S), open(C_H1, 'wb'))
if os.path.exists(C_H2):
    H2 = pickle.load(open(C_H2, 'rb'))
else:
    mu = Molecule(subst_2e={'U': ('1111',)}, max_2e_centers=1, **SUBST)
    mu.generate_basis(3, 3, 6)
    H2 = mu.o2_matrix(mu.basis)
    pickle.dump(H2, open(C_H2, 'wb'))

A0 = np.array(H1.subs({h: H_VAL, s: S_VAL, Hab: 0.0}).tolist(), float)
dA = np.array(H1.subs({h: H_VAL, s: S_VAL, Hab: H_VAL}).tolist(), float) - A0
BU = np.array(H2.subs({s: S_VAL, Usym: 1.0}).tolist(), float)     # per unit U
Sf = np.array(S.subs({s: S_VAL}).tolist(), float)
assert np.abs(BU).max() > 0, "on-site U did not enter H2"
lam = np.linspace(1.0, 0.0, 401)


def fci(Uv):
    return np.array([eigh(A0 + l * dA + Uv * BU, Sf, eigvals_only=True,
                          subset_by_index=[0, 0])[0] for l in lam])


print(f"{'U/|h|':>6} | {'E(1)':>9} | {'E(0)':>9} | {'lam*':>6} | "
      f"{'bump/|h|':>9} | monotone?")
print("-" * 64)
for Uv in [0, 2, 4, 8, 16, 32, 64, 128]:
    E = fci(Uv)
    im = int(np.argmax(E))
    interior = 0 < im < len(lam) - 1
    print(f"{Uv:>6} | {E[0]:>9.4f} | {E[-1]:>9.4f} | "
          f"{(lam[im] if interior else float('nan')):>6.3f} | "
          f"{(E[im] - E[-1] if interior else 0.0):>9.2e} | {not interior}")

lo, hi = 0.0, 64.0
for _ in range(40):
    mid = 0.5 * (lo + hi)
    im = int(np.argmax(fci(mid)))
    if 0 < im < len(lam) - 1:
        hi = mid
    else:
        lo = mid
print(f"\nFCI wrong-sign bump onset:  U/|h| = {hi:.2f}")
assert 8.0 < hi < 8.6, "onset moved"
