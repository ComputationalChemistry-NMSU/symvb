"""Covalent-only benzene single-edge weakening: the wrong-sign bump is a pure
overlap (non-orthogonality) effect.

  * At s = 0 the covalent five-structure Hamiltonian is exactly the zero matrix,
    so the covalent-only energy is independent of the weakened-edge integral
    h_ab (indeed of every integral): there is no maximum.
  * At s > 0 a shallow maximum appears near lambda ~ 0.3 and grows with s.

Supports the benzene mechanism paragraph and the Figure 5A inset.

Run from the repo root:  PYTHONPATH=. python3 examples/benzene_wrongsign_overlap_scan.py
"""
import numpy as np
from scipy.linalg import eigh, eigvalsh
import sympy as sp
from symvb import Molecule, FixedPsi

m = Molecule(
    zero_ii=True,
    subst={'s': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af'),
           'h': ('H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af')},   # H_ab kept free
    interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'])
PARENT = 'aBcDeF'
rumer = [
    FixedPsi(PARENT, coupled_pairs=[(0, 1), (2, 3), (4, 5)]),   # Kek1 pairs a-b
    FixedPsi(PARENT, coupled_pairs=[(0, 5), (1, 2), (3, 4)]),   # Kek2
    FixedPsi(PARENT, coupled_pairs=[(0, 1), (2, 5), (3, 4)]),   # Dew1
    FixedPsi(PARENT, coupled_pairs=[(0, 3), (1, 2), (4, 5)]),   # Dew2
    FixedPsi(PARENT, coupled_pairs=[(0, 5), (1, 4), (2, 3)])]   # Dew3
h, s = sp.symbols('h s')
Hab = sp.Symbol('H_ab')
Hs = m.build_matrix(rumer, op='H')
Ss = m.build_matrix(rumer, op='S')

# --- Part 1: symbolic proof that the covalent block is h_ab-independent at s=0
Hs0 = sp.simplify(Hs.subs(s, 0))
assert Hs0.is_zero_matrix, "expected zero covalent H at s=0"
assert sp.simplify(sp.diff(Hs, Hab).subs(s, 0)).is_zero_matrix
print("s = 0:  covalent 5x5 H is the zero matrix  ->  E_cov independent of h_ab")

# --- Part 2: the maximum vs overlap
H_VAL = -1.0
lam = np.linspace(1.0, 0.0, 1001)
fH = sp.lambdify((h, s, Hab), Hs, 'numpy')
fS = sp.lambdify((h, s, Hab), Ss, 'numpy')
print(f"\n{'s':>4} | {'Smin':>8} | {'lam*':>6} | {'rise(0->max)/|h|':>16} | bump?")
print("-" * 56)
for sv in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    E = np.empty(len(lam)); smin = np.inf
    for i, l in enumerate(lam):
        Hn = np.asarray(fH(H_VAL, sv, l * H_VAL), float)
        Sn = np.asarray(fS(H_VAL, sv, l * H_VAL), float)
        smin = min(smin, float(eigvalsh(Sn)[0]))
        E[i] = eigh(Hn, Sn, eigvals_only=True)[0]
    im = int(np.argmax(E)); interior = 0 < im < len(lam) - 1 and np.ptp(E) > 1e-9
    rise = E[im] - E[-1] if interior else 0.0
    print(f"{sv:>4.1f} | {smin:>8.2e} | "
          f"{(lam[im] if interior else float('nan')):>6.3f} | {rise:>16.4e} | {interior}")

# anchor against the manuscript / Figure 5 (s = 0.2)
E02 = np.array([eigh(np.asarray(fH(H_VAL, 0.2, l * H_VAL), float),
                     np.asarray(fS(H_VAL, 0.2, l * H_VAL), float),
                     eigvals_only=True)[0] for l in lam])
im = int(np.argmax(E02))
assert abs(lam[im] - 0.322) < 0.01 and abs((E02[im] - E02[-1]) - 0.0142) < 5e-4
print("\ns = 0.2 matches Figure 5:  lam* = %.3f,  rise = %.4f |h|"
      % (lam[im], E02[im] - E02[-1]))
