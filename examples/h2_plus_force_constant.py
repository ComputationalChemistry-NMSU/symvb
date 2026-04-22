"""H2+ force constant from symbolic VBT.

Pipeline:
  1. vbt3 builds the 2x2 one-electron Hamiltonian and overlap matrices for
     {|a>, |b>} and we read off the ground-state energy

         E_+(R) = (h(R) + h_ab(R)) / (1 + s(R)).

  2. Differentiate E_tot(R) = E_+(R) + 1/R twice and use the equilibrium
     condition dE_tot/dR = 0 to collapse the expression into

         k = [h'' + h_ab'' - 2 s'/R_eq^2 - E_+ s''] / (1 + s) + 2/R_eq^3
                                                         (all at R = R_eq).

  3. Plug analytic Slater-1s integrals (zeta = 1) into h, h_ab, s as
     functions of R, locate R_eq, and evaluate k. Verify the compact
     form by comparing to a direct d^2E_tot/dR^2 via sympy.

  4. Decompose k into the four physical contributions so the *source*
     of the stiffness is visible.

All in atomic units.  Literature checks for zeta = 1 LCAO-MO H2+:
R_eq ~ 2.49 bohr, D_e ~ 0.065 Ha ~ 1.76 eV, omega ~ 2300 cm^-1.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import sympy as sp
from scipy.optimize import brentq

from vbt3 import Molecule
from vbt3.fixed_psi import generate_dets


# ---------------------------------------------------------------------
# 1. Symbolic 2x2 GHEP from vbt3
# ---------------------------------------------------------------------
m = Molecule(
    zero_ii=False,
    interacting_orbs=['ab'],
    subst={'h': ('H_aa', 'H_bb'), 'h_ab': ('H_ab',), 's': ('S_ab',)},
)
P = generate_dets(1, 0, 2)                        # 1 alpha, 0 beta, 2 AOs
print("Basis:", [p.dets[0].det_string for p in P])

H1 = m.build_matrix(P, op='H')
S  = m.build_matrix(P, op='S')
print("\nH1 ="); sp.pprint(H1)
print("\nS  ="); sp.pprint(S)

h_sym, hab_sym, s_sym, E = sp.symbols('h h_ab s E')
chi = (H1 - E * S).det()
roots = sp.solve(sp.simplify(chi), E)
Egs_sym = sp.simplify((h_sym + hab_sym) / (1 + s_sym))
assert any(sp.simplify(r - Egs_sym) == 0 for r in roots), \
    "vbt3 GHEP did not reproduce E_+ = (h + h_ab)/(1 + s)"
print("\nE_+ =", Egs_sym, "    <- vbt3 matches the closed form")


# ---------------------------------------------------------------------
# 2. Symbolic differentiation: two equivalent expressions for k
# ---------------------------------------------------------------------
R = sp.Symbol('R', positive=True)
hR   = sp.Function('h')(R)
habR = sp.Function('h_ab')(R)
sR   = sp.Function('s')(R)

Eplus = (hR + habR) / (1 + sR)
Etot  = Eplus + 1 / R

# compact form (valid at R_eq only; we verify numerically below)
k_compact = (sp.diff(hR, R, 2) + sp.diff(habR, R, 2)
             - 2 * sp.diff(sR, R) / R**2
             - Eplus * sp.diff(sR, R, 2)) / (1 + sR) + 2 / R**3


# ---------------------------------------------------------------------
# 3. Slater 1s integrals at zeta = 1, substitute into h(R), h_ab(R), s(R)
# ---------------------------------------------------------------------
#   S(R)  = <a|b>               = (1 + R + R^2/3) e^-R
#   J(R)  = <a|1/r_b|a>         = (1 - (1+R) e^-2R) / R
#   K(R)  = <a|1/r_a|b>         = (1 + R) e^-R
# kinetic: -1/2 nabla^2 on 1s gives (-1/2 + 1/r_b), so T_ab = -S/2 + K
# leading to h = -1/2 - J,  h_ab = -S/2 - K.
S_R   = sp.exp(-R) * (1 + R + R**2 / 3)
J_R   = (1 - (1 + R) * sp.exp(-2 * R)) / R
K_R   = (1 + R) * sp.exp(-R)
h_R   = -sp.Rational(1, 2) - J_R
hab_R = -sp.Rational(1, 2) * S_R - K_R

subs_map = {hR: h_R, habR: hab_R, sR: S_R,
            sp.diff(hR, R):    sp.diff(h_R, R),
            sp.diff(habR, R):  sp.diff(hab_R, R),
            sp.diff(sR, R):    sp.diff(S_R, R),
            sp.diff(hR, R, 2):   sp.diff(h_R, R, 2),
            sp.diff(habR, R, 2): sp.diff(hab_R, R, 2),
            sp.diff(sR, R, 2):   sp.diff(S_R, R, 2)}

Etot_num    = Etot.subs(subs_map)
k_direct    = sp.diff(Etot_num, R, 2)
k_compact_n = k_compact.subs(subs_map)

dE_fn       = sp.lambdify(R, sp.diff(Etot_num, R), 'numpy')
E_fn        = sp.lambdify(R, Etot_num,              'numpy')
k_direct_fn = sp.lambdify(R, k_direct,              'numpy')
k_compact_fn = sp.lambdify(R, k_compact_n,          'numpy')

R_eq = brentq(dE_fn, 1.5, 4.0)
E_eq = float(E_fn(R_eq))
De   = -(E_eq - (-0.5))
k1   = float(k_direct_fn(R_eq))
k2   = float(k_compact_fn(R_eq))

print("\n" + "=" * 62)
print("H2+ at zeta = 1 (LCAO-MO with Slater 1s AOs)")
print("=" * 62)
print(f"  R_eq = {R_eq:.4f} bohr          (lit. ~ 2.49)")
print(f"  E    = {E_eq:+.5f} Ha")
print(f"  D_e  = {De:.5f} Ha  =  {De * 27.2114:.3f} eV     (lit. ~ 1.76 eV)")
print(f"  k (full d2E/dR2)   = {k1:+.6f} Ha/bohr^2")
print(f"  k (compact form)   = {k2:+.6f} Ha/bohr^2")
print(f"  difference         = {k1 - k2:+.2e}   <- should be ~ 0")

# H2+ reduced mass: two protons, m_p = 1836.15 m_e
mu = 1836.15 / 2
omega_au = np.sqrt(k1 / mu)
print(f"  omega              = {omega_au * 219474.6:.0f} cm^-1   "
      f"(lit. ~2300 cm^-1)")


# ---------------------------------------------------------------------
# 4. Decomposition of k at R_eq
# ---------------------------------------------------------------------
def _ev(expr):
    return float(sp.lambdify(R, expr.subs(subs_map), 'numpy')(R_eq))

term1 = _ev((sp.diff(hR, R, 2) + sp.diff(habR, R, 2)) / (1 + sR))
term2 = _ev(-2 * sp.diff(sR, R) / (R**2 * (1 + sR)))
term3 = _ev(-Eplus * sp.diff(sR, R, 2) / (1 + sR))
term4 = 2 / R_eq**3

print("\nWhere the stiffness comes from (Ha/bohr^2):")
print(f"  (h''  + h_ab'')/(1+s)       = {term1:+.5f}   bare curvature of numerator")
print(f"  -2 s'/[R_eq^2 (1+s)]         = {term2:+.5f}   overlap-slope x nuc. repulsion cross")
print(f"  -E_+ s''/(1+s)               = {term3:+.5f}   overlap curvature, weighted by E_+")
print(f"  +2 / R_eq^3                  = {term4:+.5f}   pure nuclear repulsion curvature")
print(f"                         sum   = {term1 + term2 + term3 + term4:+.5f}")
print(f"  (should match k above:       {k1:+.5f})")
