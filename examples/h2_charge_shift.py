"""Charge-shift decomposition of the symmetric 2c2e bond versus AO overlap s.

Build the covalent/ionic 2x2 structure Hamiltonian of the 2c2e bond directly with
symvb (cov = aB + bA, ion = aA + bB), and split the bond energy (separated atoms = 0):
    D_full = -E_full   full 2x2 ground state
    D_cov  = -E_cov     covalent structure alone (= 2|h|s/(1+s^2), overlap-driven)
    RE_CS  = E_cov - E_full = D_full - D_cov    covalent-ionic resonance

The covalent diagonal is proportional to s while the off-diagonal resonance 2h
is overlap-independent, so reducing the overlap (the minimal-model proxy for the
more compact atoms of an electronegative element such as F) weakens the covalent
structure while leaving the resonance intact: the bond turns charge-shift. The
projected 2x2 is checked against manuscript eq (5).

Run from the repo root:  PYTHONPATH=. python3 examples/h2_charge_shift.py
"""
import numpy as np
import sympy as sp
from scipy.linalg import eigh
from symvb import Molecule, FixedPsi, System

m = Molecule(zero_ii=True, interacting_orbs=['ab'],
             subst={'h': ('H_ab',), 's': ('S_ab',)},
             subst_2e={'U': ('1111',)}, max_2e_centers=1)
h, s, U = sp.symbols('h s U')

# covalent / ionic structures, fed straight to the facade; it builds the
# {cov, ion} 2x2 with the two-electron block folded into H (no U^2 footgun).
cov = FixedPsi('aB'); cov.add_str_det('bA', coef=1)      # one electron per atom
ion = FixedPsi('aA'); ion.add_str_det('bB', coef=1)      # doubly occupied atom
H2, S2 = System.from_structures(m, [cov, ion]).hamiltonian()
H2, S2 = sp.simplify(H2), sp.simplify(S2)

# cross-check the normalized 2x2 against manuscript eq (5)
nrm = sp.diag(1/sp.sqrt(S2[0, 0]), 1/sp.sqrt(S2[1, 1]))
H_A1 = sp.Matrix([[2*h*s, 2*h], [2*h, U + 2*h*s]]) / (1 + s**2)
S_A1 = sp.Matrix([[1, 2*s/(1+s**2)], [2*s/(1+s**2), 1]])
assert sp.simplify(nrm*H2*nrm - H_A1).is_zero_matrix
assert sp.simplify(nrm*S2*nrm - S_A1).is_zero_matrix
print("projected 2x2 matches manuscript eq (5)\n")

H_VAL = -1.0
for U_VAL in (4.0, 8.0):
    print(f"=============  U = {U_VAL:.0f} |h|  (h = -1)  =============")
    print(f"{'s':>5} | {'E_cov':>7} | {'E_ion':>7} | {'E_full':>7} | "
          f"{'D_cov':>6} | {'RE_CS':>6} | {'RE_CS/D':>8} | {'w_ion':>6}")
    print("-" * 74)
    rows = {}
    for sv in (0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
        sub = {h: H_VAL, s: sv, U: U_VAL}
        Hn = np.array(H2.subs(sub).evalf().tolist(), float)
        Sn = np.array(S2.subs(sub).evalf().tolist(), float)
        E_cov, E_ion = Hn[0, 0]/Sn[0, 0], Hn[1, 1]/Sn[1, 1]
        ev, vc = eigh(Hn, Sn)
        E_full, c0 = ev[0], vc[:, 0]
        w = c0 * (Sn @ c0)
        RE, D_full = E_cov - E_full, -E_full
        rows[sv] = (E_cov, E_ion, E_full, -E_cov, RE, RE/D_full, w[1])
        print(f"{sv:>5.2f} | {E_cov:>7.3f} | {E_ion:>7.3f} | {E_full:>7.3f} | "
              f"{-E_cov:>6.3f} | {RE:>6.3f} | {RE/D_full:>7.1%} | {w[1]:>6.3f}")
    if U_VAL == 4.0:                              # manuscript values
        assert abs(rows[0.5][3] - 0.80) < 0.01 and rows[0.5][5] < 0.20
        assert abs(rows[0.1][3] - 0.20) < 0.01 and abs(rows[0.1][4] - 0.70) < 0.01
        assert abs(rows[0.0][5] - 1.0) < 1e-6     # s=0 : bond is charge-shift in full
    print()
print("manuscript charge-shift values reproduced")
