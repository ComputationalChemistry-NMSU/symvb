"""(H2)2+ elastic constants resolved into 1- and 2-electron integrals.

Companion to examples/h2h2_plus_diabatic.py, where the fragment energies
E_H2(r) and E_H2+(r) are taken from Morse fits to literature constants and
the inter-fragment hopping t(R) is parameterised phenomenologically.  Here
every input to the elastic-constant formulae

    k_sym  = k_0 = (E_H2''(r*) + E_H2+''(r*)) / 2
    k_asym = k_0 - 2 alpha^2 / |t(R)|         alpha = E_H2'(r*)

is built explicitly from 1- and 2-electron integrals of Slater 1s
orbitals (zeta = 1) using symvb:

  Quantity     symvb closed form                 Integral content
  ---------    -------------------------------  -----------------------
  E_H2+(r)     (h + h_ab)/(1 + s) + 1/r         1e:  h, h_ab, s
  E_H2(r)      4-det GHEP, lowest root + 1/r    1e:  h, h_ab, s
               at s = 0 and M = 0:              2e:  U=(aa|aa), J=(aa|bb),
                  2h + (U+J+2K)/2                    K=(ab|ab), M=(aa|ab)
                  - sqrt((U-J)^2/4 + 4 h_ab^2)
  t(R)         <sigma_L|h|sigma_R>              1e:  inter-fragment
               with sigma = (a+b)/sqrt(2(1+s))       h_ac, h_ad,
                                                     h_bc, h_bd

The 2e content of E_H2(r) is the *only* place where two-electron physics
enters the elastic constants in the leading 2x2 diabatic model.  k_R
(inter-fragment stretch) requires inter-fragment 2e integrals
(charge-induced-dipole, dispersion) — these live outside the 2x2 model
and are not resolved here.

Numbers won't reproduce experimental H2/H2+ Morse data because the LCAO-
STO ansatz at zeta = 1 gets D_e and r_e wrong by ~30%.  The point is to
expose the *integral content* of the elastic constants, not to compete
with the Morse fit.

Numerics: 1e integrals are exact closed forms at zeta = 1; 2e integrals
use a pyscf STO-6G expansion with exponents rescaled to zeta = 1 (so
(aa|aa) = 5/8 to numerical precision).

Run from the repo root:
    PYTHONPATH=. python3 examples/h2h2_plus_elastic_resolved.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import sympy as sp
from scipy.linalg import eigh
from scipy.optimize import brentq
from pyscf import gto

from symvb import Molecule, hamiltonian
from symvb.fixed_psi import generate_dets


# ---------------------------------------------------------------------
# 1.  SYMBOLIC: symvb closed forms for the three building blocks
# ---------------------------------------------------------------------
print("=" * 72)
print(" 1. Closed forms produced by symvb")
print("=" * 72)

h_, hab_, s_, U_, J_, K_, M_, E_ = sp.symbols('h h_ab s U J K M E')

# --- E_H2+(r) : 1e in 2 AOs --------------------------------------------
m1 = Molecule(zero_ii=False, interacting_orbs=['ab'],
              subst={'h': ('H_aa', 'H_bb'),
                     'h_ab': ('H_ab',),
                     's': ('S_ab',)})
P1 = generate_dets(1, 0, 2)
roots1 = sp.solve(sp.simplify((m1.build_matrix(P1, 'H')
                               - E_ * m1.build_matrix(P1, 'S')).det()), E_)
E_H2p = sp.simplify([r for r in roots1
                     if sp.simplify(r.subs({h_: -sp.Rational(1, 2),
                                            hab_: -sp.Rational(1, 2),
                                            s_: 0}) + 1) == 0][0])
print("\n  E_H2+(r) = (h + h_ab) / (1 + s)")
print("           =", E_H2p, "   <- symvb 1e GHEP, lowest root")

# --- E_H2(r) : 2e in 2 AOs, full PPP UJKM ------------------------------
m2 = Molecule(zero_ii=False, interacting_orbs=['ab'],
              subst={'h': ('H_aa', 'H_bb'),
                     'h_ab': ('H_ab',),
                     's': ('S_ab',)},
              subst_2e={'U': ('1111',),
                        'J': ('1212',),    # (aa|bb), direct two-center Coulomb
                        'K': ('1122',),    # (ab|ab), two-center exchange
                        'M': ('1112', '1121', '1222')},
              max_2e_centers=2)
P2 = generate_dets(1, 1, 2)
H_h2_sym, S_h2_sym = hamiltonian(m2, P2)   # 2e block folded into H_h2_sym
print(f"\n  E_H2(r): symvb builds 4x4 GHEP (basis {[p.dets[0].det_string for p in P2]})")
print("           At s = 0 and M = 0 the lowest root collapses to")
print()
print("  E_H2(r) = 2 h(r) + (U + J + 2K) / 2")
print("                   - sqrt( (U - J)^2 / 4 + 4 h_ab(r)^2 )")
print()
print("  Verification of the closed form vs symvb GHEP at random integral")
print("  values (s = 0, M = 0):")
E_H2_cf = 2 * h_ + (U_ + J_ + 2 * K_) / 2 - sp.sqrt((U_ - J_) ** 2 / 4
                                                    + 4 * hab_ ** 2)
H_s0 = H_h2_sym.subs({s_: 0, M_: 0})
S_s0 = S_h2_sym.subs({s_: 0, M_: 0})
for trial in [{h_: -0.7, hab_: -0.4, U_: 0.625, J_: 0.4, K_: 0.15},
              {h_: -0.5, hab_: -0.3, U_: 1.0, J_: 0.2, K_: 0.05}]:
    Hn = np.array(H_s0.subs(trial).tolist(), dtype=float)
    Sn = np.array(S_s0.subs(trial).tolist(), dtype=float)
    E_num = float(eigh(Hn, Sn, eigvals_only=True)[0])
    E_cf = float(E_H2_cf.subs(trial))
    print(f"    {trial}:  E_GHEP = {E_num:+.6f}, E_closed = {E_cf:+.6f}")

# --- t(R) : sigma-projected inter-fragment hopping (1e) ----------------
print("\n  t(R) = <sigma_L | h | sigma_R>")
print("       = (h_ac + h_ad + h_bc + h_bd) / [2 sqrt((1 + s_ab)(1 + s_cd))]")
print("    sigma_L = (a + b)/sqrt(2(1+s_ab)),  sigma_R = (c + d)/sqrt(2(1+s_cd))")


# ---------------------------------------------------------------------
# 2.  NUMERICS: Slater 1s integrals at zeta = 1
# ---------------------------------------------------------------------
# 1e: exact closed forms (Sugiura).  2e: pyscf STO-6G with rescaled
# exponents so that (aa|aa) = 5/8 exactly (trick from
# h2_two_electron_integrals.py).
ZETA_PYSCF = 1.24
EXPS_124 = [35.52322122, 6.513143725, 1.822142904,
            0.625955266, 0.243076747, 0.100112428]
COEFS    = [0.00916359628, 0.04936149294, 0.16853830490,
            0.37056279970, 0.41649152980, 0.13033408410]
EXPS_1   = [a / ZETA_PYSCF**2 for a in EXPS_124]
BASIS_Z1 = {'H': [[0] + list(zip(EXPS_1, COEFS))]}


# 1e closed forms at zeta = 1 (Slater 1s)
def s_intra(r):  return np.exp(-r) * (1 + r + r**2 / 3)
def J_attr(r):   return (1 - (1 + r) * np.exp(-2 * r)) / r
def K_res(r):    return (1 + r) * np.exp(-r)
def h_aa(r):     return -0.5 - J_attr(r)
def h_ab(r):     return -0.5 * s_intra(r) - K_res(r)


def two_e_h2(r):
    """All four PPP 2e integrals U, J=(aa|bb), K=(ab|ab), M=(aa|ab) at H-H = r."""
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {r}',
                basis=BASIS_Z1, unit='Bohr', verbose=0)
    eri = mol.intor('int2e')
    return (eri[0, 0, 0, 0],   # U  = (aa|aa)
            eri[0, 0, 1, 1],   # J  = (aa|bb)
            eri[0, 1, 0, 1],   # K  = (ab|ab)
            eri[0, 0, 0, 1])   # M  = (aa|ab)


def E_H2(r):
    """Full 4x4 GHEP ground state, all overlaps and 2e integrals included."""
    Uv, Jv, Kv, Mv = two_e_h2(r)
    subs = {h_: h_aa(r), hab_: h_ab(r), s_: s_intra(r),
            U_: Uv, J_: Jv, K_: Kv, M_: Mv}
    Hn = np.array(H_h2_sym.subs(subs).tolist(), dtype=float)
    Sn = np.array(S_h2_sym.subs(subs).tolist(), dtype=float)
    return float(eigh(Hn, Sn, eigvals_only=True)[0]) + 1.0 / r


def E_H2plus(r):
    return (h_aa(r) + h_ab(r)) / (1 + s_intra(r)) + 1.0 / r


# 5-point stencil derivatives (the integrals are smooth in r)
def deriv(f, x, n, h=1e-3):
    if n == 1:
        return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12*h)
    return (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h)
            - f(x - 2*h)) / (12 * h**2)


# ---------------------------------------------------------------------
# 3.  Compromise geometry r* and the building blocks (alpha, k_0)
# ---------------------------------------------------------------------
print()
print("=" * 72)
print(" 2. Compromise geometry of (H2)2+ from symvb fragment energies")
print("=" * 72)
r_star = brentq(lambda r: deriv(E_H2, r, 1) + deriv(E_H2plus, r, 1), 1.4, 2.4)
alpha   = deriv(E_H2,    r_star, 1)
alpha_p = deriv(E_H2plus, r_star, 1)
k_H2    = deriv(E_H2,    r_star, 2)
k_H2p   = deriv(E_H2plus, r_star, 2)
k0      = (k_H2 + k_H2p) / 2

print(f"  r*                         = {r_star:.4f} bohr")
print(f"  E_H2'(r*)   =  alpha       = {alpha:+.5f} Ha/bohr")
print(f"  E_H2+'(r*)  =  -alpha      = {alpha_p:+.5f} Ha/bohr  (check)")
print(f"  E_H2''(r*)                 = {k_H2:+.5f} Ha/bohr^2")
print(f"  E_H2+''(r*)                = {k_H2p:+.5f} Ha/bohr^2")
print(f"  k_0 = (E_H2''+E_H2+'')/2   = {k0:+.5f} Ha/bohr^2")

# Show the 1e/2e content at r*.
Uv, Jv, Kv, Mv = two_e_h2(r_star)
print(f"\n  Integral content at r* = {r_star:.4f}:")
print(f"     1e (closed-form Slater 1s):")
print(f"        h    = (a|h|a) = {h_aa(r_star):+.5f}")
print(f"        h_ab = (a|h|b) = {h_ab(r_star):+.5f}")
print(f"        s    = <a|b>   = {s_intra(r_star):+.5f}")
print(f"     2e (pyscf STO-6G, rescaled to zeta=1):")
print(f"        U = (aa|aa)    = {Uv:.5f}    "
      f"(R-independent; exact 5/8 = {5/8:.5f})")
print(f"        J = (aa|bb)    = {Jv:.5f}    (direct, ~ 1/r at large r)")
print(f"        K = (ab|ab)    = {Kv:.5f}    (exchange, ~ exp(-2r))")
print(f"        M = (aa|ab)    = {Mv:.5f}    (hybrid, dropped in ZDO/PPP)")


# ---------------------------------------------------------------------
# 4.  Hellmann-Feynman decomposition of alpha at the full GHEP
# ---------------------------------------------------------------------
# alpha = dE_H2/dr|_{r*} = sum_x  (dE/dx) * (dx/dr),  x in {h, h_ab, s,
# U, J, K, M}, where dE/dx is computed at the actual r* (full overlap,
# full M) by finite-differencing the GHEP solver while holding other
# integrals fixed.  This is the Hellmann-Feynman picture: each integral
# carries an additive contribution to alpha, set by its sensitivity
# (dE/dx) and its r-slope (dx/dr).
print()
print("=" * 72)
print(" 3. Hellmann-Feynman decomposition of alpha at r* (full GHEP)")
print("=" * 72)


def E_H2_from_integrals(h_v, hab_v, s_v, U_v, J_v, K_v, M_v):
    subs = {h_: h_v, hab_: hab_v, s_: s_v,
            U_: U_v, J_: J_v, K_: K_v, M_: M_v}
    Hn = np.array(H_h2_sym.subs(subs).tolist(), dtype=float)
    Sn = np.array(S_h2_sym.subs(subs).tolist(), dtype=float)
    return float(eigh(Hn, Sn, eigvals_only=True)[0])


def deriv_int(f, r, n=1, dx=1e-3):
    return ((-f(r+2*dx) + 8*f(r+dx) - 8*f(r-dx) + f(r-2*dx)) / (12*dx) if n == 1
            else (-f(r+2*dx) + 16*f(r+dx) - 30*f(r) + 16*f(r-dx)
                  - f(r-2*dx)) / (12 * dx**2))


star = dict(h_v=h_aa(r_star), hab_v=h_ab(r_star), s_v=s_intra(r_star),
            U_v=Uv, J_v=Jv, K_v=Kv, M_v=Mv)
e_star = E_H2_from_integrals(**star)


def dE_dx(key, eps=1e-4):
    plus  = dict(star); plus[key]  += eps
    minus = dict(star); minus[key] -= eps
    return (E_H2_from_integrals(**plus) - E_H2_from_integrals(**minus)) / (2 * eps)


dE = {k: dE_dx(k) for k in star}
dx_dr = {
    'h_v':   deriv_int(h_aa, r_star),
    'hab_v': deriv_int(h_ab, r_star),
    's_v':   deriv_int(s_intra, r_star),
    'U_v':   deriv_int(lambda r: two_e_h2(r)[0], r_star),
    'J_v':   deriv_int(lambda r: two_e_h2(r)[1], r_star),
    'K_v':   deriv_int(lambda r: two_e_h2(r)[2], r_star),
    'M_v':   deriv_int(lambda r: two_e_h2(r)[3], r_star),
}
labels = {'h_v': 'h    (1e, atomic)',
          'hab_v': 'h_ab (1e, off-diag)',
          's_v': 's    (1e, overlap)',
          'U_v': 'U    (2e, on-site)',
          'J_v': 'J    (2e, direct)',
          'K_v': 'K    (2e, exchange)',
          'M_v': 'M    (2e, hybrid)'}

print(f"  alpha = dE_H2/dr|_{{r*}} = sum_x (dE/dx) * (dx/dr) + 1/r contribution")
print()
print(f"  {'integral x':<22s}  {'dE/dx':>10s}  {'dx/dr':>10s}  {'contrib':>10s}")
total = 0.0
for k in ['h_v', 'hab_v', 's_v', 'U_v', 'J_v', 'K_v', 'M_v']:
    contrib = dE[k] * dx_dr[k]
    total += contrib
    print(f"  {labels[k]:<22s}  {dE[k]:>+10.5f}  {dx_dr[k]:>+10.5f}  {contrib:>+10.5f}")
nuc = -1.0 / r_star**2
total += nuc
print(f"  {'+ d(1/r)/dr':<22s}  {'':>10s}  {'':>10s}  {nuc:>+10.5f}")
print(f"  {'-' * 60}")
print(f"  {'sum':<22s}  {'':>10s}  {'':>10s}  {total:>+10.5f}"
      f"  vs alpha = {alpha:+.5f}")

print()
print("  Reading at r* = {:.3f} bohr, s = {:.2f}:".format(r_star, star['s_v']))
order = sorted([(k, abs(dE[k] * dx_dr[k])) for k in dE], key=lambda kv: -kv[1])
top = order[:3]
print(f"    Three biggest contributions: "
      + ", ".join(labels[k].split(' ')[0] for k, _ in top))
print(f"    1e content (h, h_ab, s):  "
      f"{sum(dE[k]*dx_dr[k] for k in ['h_v','hab_v','s_v']):+.5f} Ha/bohr")
print(f"    2e content (U, J, K, M):  "
      f"{sum(dE[k]*dx_dr[k] for k in ['U_v','J_v','K_v','M_v']):+.5f} Ha/bohr")
print()
print("  In words: the H2 fragment slope at the compromise geometry is")
print("  set by competition between 1e bonding (h_ab pulling the bond")
print("  shorter through the overlap denominator) and 2e nuclear-screening")
print("  (J and K both decreasing as the bond stretches).  U is")
print("  R-independent so dU/dr = 0 and U does not contribute to alpha")
print("  even though dE/dU != 0 (it shifts E_H2 uniformly across r).")


# ---------------------------------------------------------------------
# 5.  Inter-fragment hopping t(R) from cross 1e integrals
# ---------------------------------------------------------------------
def t_of_R(R, r_intra=None):
    """sigma-projected inter-fragment hopping for collinear (H2)(H2+).
       Both fragments at intra-fragment bond length r_intra (default r*).
       R is the centre-to-centre distance."""
    r_intra = r_intra if r_intra is not None else r_star
    a = (-R/2 - r_intra/2, 0, 0)
    b = (-R/2 + r_intra/2, 0, 0)
    c = (+R/2 - r_intra/2, 0, 0)
    d = (+R/2 + r_intra/2, 0, 0)
    mol = gto.M(atom=[('H', a), ('H', b), ('H', c), ('H', d)],
                basis=BASIS_Z1, unit='Bohr', verbose=0)
    h1 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    s1 = mol.intor('int1e_ovlp')
    s_L = s1[0, 1]; s_R = s1[2, 3]
    cross = h1[0, 2] + h1[0, 3] + h1[1, 2] + h1[1, 3]
    return cross / (2 * np.sqrt((1 + s_L) * (1 + s_R)))


print()
print("=" * 72)
print(" 4. t(R) and resulting elastic constants for collinear (H2)(H2+)")
print("=" * 72)
print(f"  Both fragments held at r* = {r_star:.4f} bohr (compromise geometry).")
print()
t_c = 2 * alpha**2 / k0
print(f"  Critical |t_c| = 2 alpha^2 / k_0 = {t_c:.5f} Ha"
      f"  ({t_c * 27.2114:.3f} eV)")
mu_HH = 1836.15 / 2
TO_CM = 219474.6
print()
print(f"  {'R':>5}  {'|t(R)|':>9}  {'k_sym':>9}  {'k_asym':>10}  "
      f"{'omega_sym':>10}  {'omega_asym':>13}  class")
for R in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
    tR = abs(t_of_R(R))
    k_sym  = k0
    k_asym = k0 - 2 * alpha**2 / tR
    w_sym  = np.sqrt(max(k_sym, 0) / mu_HH) * TO_CM
    if k_asym > 0:
        w_str = f"{np.sqrt(k_asym / mu_HH) * TO_CM:>13.1f}"
        cls = "III"
    else:
        w_str = f"{'i ' + f'{np.sqrt(-k_asym / mu_HH) * TO_CM:.1f}':>13}"
        cls = "II"
    print(f"  {R:>5.1f}  {tR:>9.5f}  {k_sym:>+9.5f}  {k_asym:>+10.5f}  "
          f"{w_sym:>10.1f}  {w_str}  {cls}")

# Critical R from the symvb t(R)
def f_tc(R):  return abs(t_of_R(R)) - t_c
R_c = brentq(f_tc, 2.0, 30.0)
print(f"\n  R_c (Class III -> II crossover): {R_c:.3f} bohr")
print()
print("  k_R (inter-fragment stretch) requires d^2|t|/dR^2 plus charge-")
print("  induced-dipole and dispersion contributions, i.e. inter-fragment")
print("  2e integrals (aa|cd), (ab|cd), etc.  Those are outside the 2x2")
print("  diabatic model; the closing of the binding well in real (H2)2+")
print("  comes from these 2e cross terms, not from |t(R)| alone.")


# ---------------------------------------------------------------------
# 6.  Comparison to the Morse-fit diabatic script
# ---------------------------------------------------------------------
print()
print("=" * 72)
print(" 5. Comparison to h2h2_plus_diabatic.py (Morse fits to literature)")
print("=" * 72)
print(f"  {'quantity':<14s} {'this script':>14s}   {'Morse fits':>12s}")
print(f"  {'r*':<14s} {r_star:>14.4f}   {'1.5712':>12s}")
print(f"  {'alpha':<14s} {alpha:>+14.5f}   {'+0.09439':>12s}")
print(f"  {'k_0':<14s} {k0:>+14.5f}   {'+0.23693':>12s}")
print(f"  {'|t_c|':<14s} {t_c:>14.5f}   {'~0.0752':>12s}")
print()
print("  Numbers differ by ~30% because LCAO at zeta=1 doesn't reproduce")
print("  experimental D_e and r_e (the manuscript's H2+ force-constant")
print("  example, h2_plus_force_constant.py, has the same caveat).  What")
print("  this script DOES show is the *integral content* of every entry on")
print("  the left column: which of (h, h_ab, s, U, J, K, M) drives which")
print("  piece of the elastic spectrum.")
