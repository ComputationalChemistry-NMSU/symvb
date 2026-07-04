"""(H2)2+ dimer cation: 2x2 diabatic sigma-hole model with vibronic analysis.

Four nuclei, three electrons.  Two localized sigma-hole diabatics:

    |L> = H2(r1) * H2+(r2)     hole on the right fragment
    |R> = H2+(r1) * H2(r2)     hole on the left  fragment

Under the assumption  t = t(R)  only  (no intra-fragment dependence),

           | E_H2(r1) + E_H2+(r2)                     t(R)
    H_2x2 |
           | t(R)                     E_H2+(r1) + E_H2(r2)

Expansion around r1 = r2 = r*  (the compromise bond length where
E_H2'(r*) + E_H2+'(r*) = 0) in mass-weighted normal-mode-like coordinates

    eta = (r1 + r2)/sqrt(2)      symmetric intra stretch
    xi  = (r1 - r2)/sqrt(2)      antisymmetric intra stretch

yields a clean three-mode decoupling

    k_sym  = k_0 = (E_H2''(r*) + E_H2+''(r*)) / 2
    k_asym = k_0 - 2 alpha^2 / |t(R)|          alpha = E_H2'(r*)
    k_R    comes from  d2|t|/dR^2              (inter-fragment, separate)

The -2 alpha^2/|t| term is the second-order Jahn-Teller softening from
vibronic mixing with the charge-transfer excited state.  It drives a
Robin-Day Class III -> Class II crossover at

    |t(R_c)| = 2 alpha^2 / k_0

Below |t_c| the symmetric r1 = r2 geometry is a saddle and the hole
localizes onto one fragment.

Fragment energies use Morse potentials fit to literature values for H2 and
H2+ (atomic units throughout); the 2x2 closed form is derived in sympy.

Run from the repo root:  PYTHONPATH=. python3 examples/h2h2_plus_diabatic.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import sympy as sp
from scipy.optimize import brentq


# ---------------------------------------------------------------------
# 1. Morse potentials for H2 and H2+
# ---------------------------------------------------------------------
# Literature equilibrium data (atomic units):
# H2 : D_e = 0.1745 Ha, r_e = 1.401 bohr, k = 0.370 Ha/bohr^2   (omega 4401 cm^-1)
# H2+: D_e = 0.1026 Ha, r_e = 1.997 bohr, k = 0.103 Ha/bohr^2   (omega 2322 cm^-1)
# Asymptotes: H2  -> -1     Ha (2 H atoms)
#             H2+ -> -0.5   Ha (H + H+)
H2_DE,   H2_RE,   H2_K,   H2_ASY   = 0.1745, 1.401, 0.370, -1.0
H2P_DE, H2P_RE, H2P_K, H2P_ASY     = 0.1026, 1.997, 0.103, -0.5

r = sp.Symbol('r', positive=True)

def morse(De, re, k, asymptote):
    a = sp.sqrt(k / (2 * De))
    return asymptote + De * (1 - sp.exp(-a * (r - re)))**2 - De

E_H2  = morse(H2_DE,  H2_RE,  H2_K,  H2_ASY)
E_H2p = morse(H2P_DE, H2P_RE, H2P_K, H2P_ASY)


# ---------------------------------------------------------------------
# 2. Compromise bond length r*
# ---------------------------------------------------------------------
dE_sum = sp.diff(E_H2, r) + sp.diff(E_H2p, r)
f = sp.lambdify(r, dE_sum, 'numpy')
r_star = brentq(f, H2_RE, H2P_RE)

alpha   = float(sp.lambdify(r, sp.diff(E_H2,  r))(r_star))
alpha_p = float(sp.lambdify(r, sp.diff(E_H2p, r))(r_star))
k0_sym = (sp.diff(E_H2, r, 2) + sp.diff(E_H2p, r, 2)) / 2
k0     = float(sp.lambdify(r, k0_sym)(r_star))

print("=" * 66)
print("Compromise geometry of (H2)2+ (2x2 diabatic model)")
print("=" * 66)
print(f"  r_e(H2)  = {H2_RE:.3f} bohr        r_e(H2+) = {H2P_RE:.3f} bohr")
print(f"  r* (compromise)                 = {r_star:.4f} bohr")
print(f"  E_H2'(r*)   =  {alpha:+.5f} Ha/bohr  ('promotion slope' alpha)")
print(f"  E_H2+'(r*)  =  {alpha_p:+.5f} Ha/bohr  (should equal -alpha)")
print(f"  k_0 = (E_H2'' + E_H2+'')/2      = {k0:.5f} Ha/bohr^2")


# ---------------------------------------------------------------------
# 3. Symbolic 2x2 diabatic analysis: derive k_sym, k_asym
# ---------------------------------------------------------------------
xi, eta, t_sym, a_sym, k0_s = sp.symbols('xi eta t alpha k0', real=True,
                                          positive=True)
# Mean diagonal contains the symmetric quadratic part in both modes.
# Splitting H_LL - H_RR is linear in xi only: 2*sqrt(2)*alpha*xi.
mean_diag  = k0_s * (xi**2 + eta**2) / 2       # relative to epsilon(r*)
half_split = sp.sqrt(2 * a_sym**2 * xi**2 + t_sym**2)
E_minus    = mean_diag - half_split

k_sym_sym  = sp.simplify(sp.diff(E_minus, eta, 2).subs({xi: 0, eta: 0}))
k_asym_sym = sp.simplify(sp.diff(E_minus, xi,  2).subs({xi: 0, eta: 0}))

print("\n" + "=" * 66)
print("Symbolic expansion of the GS eigenvalue around r1 = r2 = r*")
print("=" * 66)
print(f"  k_sym   =  d2E/d(eta)^2 |_0  =  {k_sym_sym}")
print(f"  k_asym  =  d2E/d(xi)^2  |_0  =  {k_asym_sym}")
assert sp.simplify(k_sym_sym  - k0_s) == 0
assert sp.simplify(k_asym_sym - (k0_s - 2 * a_sym**2 / t_sym)) == 0
print("  -> matches k_sym = k0,  k_asym = k0 - 2 alpha^2/|t|")


# ---------------------------------------------------------------------
# 4. Class II / Class III crossover
# ---------------------------------------------------------------------
t_c = 2 * alpha**2 / k0
print("\n" + "=" * 66)
print("Critical hopping for the antisymmetric stretch")
print("=" * 66)
print(f"  |t_c| = 2 alpha^2 / k_0 = {t_c:.5f} Ha  =  {t_c * 27.2114:.4f} eV")
print("  |t(R)| >  |t_c|  :  Class III, delocalized hole, symmetric minimum")
print("  |t(R)| <  |t_c|  :  Class II,  localized hole, broken symmetry")


# ---------------------------------------------------------------------
# 5. Scan R with a physically-motivated t(R)
# ---------------------------------------------------------------------
# Hopping ~ sigma-sigma one-electron resonance, which at long R tracks the
# 1s-1s exchange integral K(R) = (1 + R) exp(-R). Pre-factor chosen so
# t_c falls around R_c ~ 6 bohr (typical (H2)2+ dimer range).
T_SCALE = 0.30      # Ha, amplitude
def t_of_R(R):
    return -T_SCALE * (1 + R) * np.exp(-R)

# Solve for R_c
R_c = brentq(lambda R: abs(t_of_R(R)) - t_c, 1.0, 20.0)
print(f"\n  With t(R) = -{T_SCALE} * (1+R) e^-R :    R_c = {R_c:.3f} bohr\n")


mu_HH = 1836.15 / 2                             # proton-proton reduced mass
TO_CM = 219474.6

print(f"{'R':>5}  {'|t(R)|':>9}  {'k_sym':>8}  {'k_asym':>9}  "
      f"{'omega_sym':>10}  {'omega_asym':>12}  class")
for R_val in [3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0]:
    t_val = abs(t_of_R(R_val))
    k_sym  = k0
    k_asym = k0 - 2 * alpha**2 / t_val
    w_sym  = np.sqrt(max(k_sym,  0) / mu_HH) * TO_CM
    if k_asym > 0:
        w_asym_str = f"{np.sqrt(k_asym / mu_HH) * TO_CM:>12.1f}"
        cls = "III"
    else:
        # imaginary frequency along xi -> saddle
        w_asym_str = f"{'i ' + f'{np.sqrt(-k_asym / mu_HH) * TO_CM:.1f}':>12}"
        cls = "II"
    print(f"{R_val:>5.1f}  {t_val:>9.5f}  {k_sym:>8.4f}  {k_asym:>+9.4f}  "
          f"{w_sym:>10.1f}  {w_asym_str}  {cls}")


# ---------------------------------------------------------------------
# 6. Inter-fragment mode (R) force constant
# ---------------------------------------------------------------------
# Under the assumption, the R-dependence enters E_GS only through -|t(R)|:
#   E_GS(r*, R) = epsilon(r*) - |t(R)|
# so k_R = -d^2|t|/dR^2  evaluated at the R-minimum of E_GS.  The
# R-minimum is where d|t|/dR = 0, which for t(R) = -T0 (1+R) e^-R occurs
# at R -> infinity (|t| is monotonically decreasing).  So in this minimal
# model there is NO bound R-mode: the two fragments don't bind via the
# hole alone; one needs charge-dipole or dispersion terms (outside the
# 2x2 diabatic model). This is itself revealing:
print("\n" + "=" * 66)
print("Inter-fragment (R) stretch")
print("=" * 66)
print("  With only the 2x2 diabatic term, |t(R)| is monotonic in R so")
print("  there is no binding along R.  The (H2)2+ dimer bond comes from")
print("  charge-induced-dipole + dispersion, NOT from hole delocalization")
print("  per se -- experimentally, (H2)2+ is weakly bound (~ 1.7 kcal/mol)")
print("  with a long 6-8 bohr fragment separation.")


# ---------------------------------------------------------------------
# 7. Interpretation
# ---------------------------------------------------------------------
print("\n" + "=" * 66)
print("Interpretation")
print("=" * 66)
print(f"""
  Under  t = t(R) only  (no dependence on r1, r2), the 2x2 diabatic
  Hamiltonian decouples cleanly into three modes:

    eta  (symmetric intra)   k_sym  = k_0              = {k0:.4f} Ha/bohr^2
    xi   (antisymmetric)     k_asym = k_0 - 2 alpha^2/|t|    <-  varies with R
    R    (inter-fragment)    set by d^2|t|/dR^2 + electrostatics

  Key quantities at r* = {r_star:.3f} bohr:
    alpha = E_H2'(r*)   = {alpha:.4f} Ha/bohr  (cost of stretching H2 past r_e)
    k_0                 = {k0:.4f} Ha/bohr^2

  The antisymmetric stretch is soft, and becomes unstable (saddle) when
    |t(R)| < |t_c| = 2 alpha^2 / k_0 = {t_c:.4f} Ha
  crossing the Class II / Class III boundary at R_c = {R_c:.2f} bohr for our
  choice of t(R).

  Why the assumption 't(R) only' buys so much:
    * diagonal splitting is LINEAR in xi only (not in eta)
    * off-diagonal is independent of both xi and eta
    -> no bilinear eta-xi or xi-R or eta-R terms at leading order.
    -> the three modes block-diagonalize.
    -> the vibronic softening of xi is captured by a 2-state formula,
       with alpha, k_0, and t(R) as the only inputs.

  If t depended on r1, r2 (e.g. because the dimer AO overlap is affected
  by intra-fragment distortion), all of these terms would couple and
  the analysis loses its clean closed form.  In practice the assumption
  is quantitatively reliable at long R (>5-6 bohr) where the inter-
  fragment overlap is the tail of 1s-1s exponentials and its dependence
  on intra-r is subleading in exp(-R).
""")
