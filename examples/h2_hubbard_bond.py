"""
How a Hubbard U reshapes a covalent bond: the H2 example.

The basis at s=0, zero_ii=True is the 4-determinant Sz=0 space; the standard
valence-bond pictures are the symmetric singlet combinations

    |cov>  = (|aB| + |bA|)        Heitler-London singlet
    |ion>  = (|aA| + |bB|)        symmetric ionic combination

The A_1 block on {cov, ion} has eigenvalues  E = U/2 +/- sqrt((U/2)^2 + 4 h^2),
and the covalent/ionic weights cross over near U ~ 4|h|.

This script uses the high-level `symvb.System` facade. One call builds the
{cov, ion} Hamiltonian with the two-electron block folded in, one call solves
the generalized eigenproblem, and one call returns the Chirgwin-Coulson
weights -- compare with the hand-rolled basis transform, characteristic
polynomial, manual root selection, and weight algebra this used to need.
"""
import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, FixedPsi, System
from symvb.system import ground_state, chirgwin_coulson

h, s, U = sp.symbols('h s U')


# ------------------------------------------------------------------------
# 1. The H2 model and its covalent / ionic structures
# ------------------------------------------------------------------------
m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab'],
    subst={'h': ('H_ab',), 's': ('S_ab',)},
    subst_2e={'U': ('1111',)},
    max_2e_centers=1,                       # on-site U only
)
cov = FixedPsi('aB'); cov.add_str_det('bA', coef=1)    # Heitler-London singlet
ion = FixedPsi('aA'); ion.add_str_det('bB', coef=1)    # symmetric ionic
sysm = System.from_structures(m, [cov, ion])

# build_matrix + o2_matrix + combine, all in one call (no U^2 footgun):
H, S = sysm.hamiltonian()
print("H over {cov, ion} (two-electron block folded in):")
sp.pprint(sp.simplify(H))
print("\nS over {cov, ion}:")
sp.pprint(sp.simplify(S))


# ------------------------------------------------------------------------
# 2. Ground state and weights at s = 0 -- solved by the facade
# ------------------------------------------------------------------------
H0, S0 = H.subs(s, 0), S.subs(s, 0)
E_gs, c = ground_state(H0, S0)              # picks the bonding root automatically
w_cov, w_ion = chirgwin_coulson(c, S0)      # metric-correct weights

print(f"\nGround state (closed form):  E(U, h) = {sp.simplify(E_gs)}")
print(f"  -> E(U, h=-1) = {sp.simplify(E_gs.subs(h, -1))}")

# cross-check against the textbook two-configuration Hubbard result
assert sp.simplify(E_gs - (U / 2 - sp.sqrt((U / 2) ** 2 + 4 * h ** 2))) == 0
assert sp.simplify(w_cov.subs(U, 0)) == sp.Rational(1, 2)
print("checks: E_gs matches U/2 - sqrt((U/2)^2 + 4h^2);  w_cov(U=0) = 1/2  -> OK")


# ------------------------------------------------------------------------
# 3. The physical story: U-dependent bond character
# ------------------------------------------------------------------------
print("\n" + "=" * 66)
print("H2 ground state vs U (at t = 1, orthogonal orbitals)")
print("=" * 66)
print(f"{'U':>6}  {'E_gs':>10}  {'w_cov':>8}  {'w_ion':>8}  interpretation")
for Uval in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 100.0]:
    E_num = float(E_gs.subs({U: Uval, h: -1}))
    wc = float(w_cov.subs({U: Uval, h: -1}))
    wi = 1 - wc
    tag = ("HF-like (50/50)"        if abs(wc - 0.5) < 0.05 else
           "predominantly covalent" if wc > 0.8 else
           "mixed covalent/ionic")
    print(f"  {Uval:4.1f}  {E_num:+10.6f}  {wc:7.4f}  {wi:7.4f}  {tag}")

print("\nLimits:")
print("  U = 0       w_cov = 1/2, w_ion = 1/2     (restricted HF result)")
print("  U -> infty  w_cov -> 1, w_ion -> 0       (pure Heitler-London)")
print("  crossover at U ~ 4 t  (ionic penalty equals the resonance)")


# ------------------------------------------------------------------------
# 4. Taylor series for H2 -- for comparison with the benzene case
# ------------------------------------------------------------------------
print("\n" + "=" * 66)
print("Taylor series of E_gs(U) at t = 1 -- compare to benzene's")
print("=" * 66)
series_H2 = sp.series(E_gs.subs(h, -1), U, 0, 9).removeO()
print(f"  H2:        E(U) = {series_H2}")
print("\nH2 has a clean closed form  E(U) = U/2 - sqrt((U/2)^2 + 4 t^2)")
print("because the A_1 block is only 2 x 2; benzene's 38-dim A_1g block gives")
print("rational Taylor coefficients but no elementary closed form.")
