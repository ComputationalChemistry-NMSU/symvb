"""(H2)2+ Class III -> II transition resolved into AO integrals.

Builds on examples/h2h2_plus_elastic_resolved.py.  The Robin-Day
class-III/class-II boundary in the 2x2 diabatic model is

         |t(R_c)|  =  2 alpha^2 / k_0,

with

      alpha  =  E_H2'(r*),     k_0  =  (E_H2'' + E_H2+'')/2 |_{r*},

where r* is the compromise geometry that satisfies E_H2'(r*) + E_H2+'(r*) = 0.
This script writes both sides of that equation purely in terms of distance-
dependent 1e and 2e integrals over Slater 1s AOs (zeta = 1):

  RHS  (intra-fragment, evaluated at r*)
        h(r)   = (a|h|a)         1e atomic
        h_ab(r)= (a|h|b)         1e off-diagonal
        s(r)   = <a|b>           1e overlap
        U      = (aa|aa) = 5/8   2e on-site (R-independent)
        J(r)   = (aa|bb)         2e direct Coulomb
        K(r)   = (ab|ab)         2e exchange
        M(r)   = (aa|ab)         2e hybrid (PPP-dropped)

  LHS  (inter-fragment, evaluated at R)
        h_ac(R), h_ad(R), h_bc(R), h_bd(R)  1e cross integrals
        s_ab(r*), s_cd(r*)                  intra overlaps for normalization

The script:
  (1) computes r*, alpha, k_0 from the intra-AO-integral inputs;
  (2) solves |t(R_c)| = 2 alpha^2 / k_0 from the cross 1e integrals;
  (3) prints the integral content of both sides at R_c;
  (4) plots the antisymmetric-stretch energy E-(xi) for three R values
      that bracket R_c, making the single-well -> flat -> double-well
      transition visible.

The figure is saved to ../vbt-3/figures/ for the manuscript.

Run from the repo root:
    PYTHONPATH=. python3 examples/h2h2_plus_class_transition.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.optimize import brentq
from pyscf import gto

from symvb import Molecule, hamiltonian
from symvb.fixed_psi import generate_dets


# ---------------------------------------------------------------------
# 1.  Integral functions of distance (1e closed form, 2e via pyscf)
# ---------------------------------------------------------------------
ZETA_PYSCF = 1.24
EXPS_124 = [35.52322122, 6.513143725, 1.822142904,
            0.625955266, 0.243076747, 0.100112428]
COEFS    = [0.00916359628, 0.04936149294, 0.16853830490,
            0.37056279970, 0.41649152980, 0.13033408410]
EXPS_1   = [a / ZETA_PYSCF**2 for a in EXPS_124]
BASIS_Z1 = {'H': [[0] + list(zip(EXPS_1, COEFS))]}


def s_intra(r): return np.exp(-r) * (1 + r + r**2 / 3)
def J_attr(r):  return (1 - (1 + r) * np.exp(-2 * r)) / r
def K_res(r):   return (1 + r) * np.exp(-r)
def h_aa(r):    return -0.5 - J_attr(r)
def h_ab(r):    return -0.5 * s_intra(r) - K_res(r)


def two_e_h2(r):
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {r}',
                basis=BASIS_Z1, unit='Bohr', verbose=0)
    eri = mol.intor('int2e')
    return (eri[0,0,0,0], eri[0,0,1,1], eri[0,1,0,1], eri[0,0,0,1])


def cross_1e_collinear(R, r_intra):
    """Inter-fragment 1e integrals (h_ac, h_ad, h_bc, h_bd) plus the
    intra-fragment overlaps for sigma-MO normalization, in a collinear
    H-H ... H-H geometry: a---b ... c---d."""
    a = (-R/2 - r_intra/2, 0, 0)
    b = (-R/2 + r_intra/2, 0, 0)
    c = (+R/2 - r_intra/2, 0, 0)
    d = (+R/2 + r_intra/2, 0, 0)
    mol = gto.M(atom=[('H', a), ('H', b), ('H', c), ('H', d)],
                basis=BASIS_Z1, unit='Bohr', verbose=0)
    h1 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    s1 = mol.intor('int1e_ovlp')
    return dict(h_ac=h1[0,2], h_ad=h1[0,3], h_bc=h1[1,2], h_bd=h1[1,3],
                s_ab=s1[0,1], s_cd=s1[2,3])


def t_of_R(R, r_intra):
    cx = cross_1e_collinear(R, r_intra)
    cross = cx['h_ac'] + cx['h_ad'] + cx['h_bc'] + cx['h_bd']
    return cross / (2 * np.sqrt((1 + cx['s_ab']) * (1 + cx['s_cd'])))


# ---------------------------------------------------------------------
# 2.  symvb GHEP for E_H2(r) and closed form for E_H2+(r)
# ---------------------------------------------------------------------
h_, hab_, s_, U_, J_, K_, M_ = sp.symbols('h h_ab s U J K M')

m2 = Molecule(zero_ii=False, interacting_orbs=['ab'],
              subst={'h': ('H_aa', 'H_bb'), 'h_ab': ('H_ab',), 's': ('S_ab',)},
              subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
                        'M': ('1112', '1121', '1222')},
              max_2e_centers=2)
P2 = generate_dets(1, 1, 2)
H_h2_sym, S_h2_sym = hamiltonian(m2, P2)   # 2e block folded into H_h2_sym


def E_H2(r):
    Uv, Jv, Kv, Mv = two_e_h2(r)
    subs = {h_: h_aa(r), hab_: h_ab(r), s_: s_intra(r),
            U_: Uv, J_: Jv, K_: Kv, M_: Mv}
    Hn = np.array(H_h2_sym.subs(subs).tolist(), dtype=float)
    Sn = np.array(S_h2_sym.subs(subs).tolist(), dtype=float)
    return float(eigh(Hn, Sn, eigvals_only=True)[0]) + 1.0 / r


def E_H2plus(r):
    return (h_aa(r) + h_ab(r)) / (1 + s_intra(r)) + 1.0 / r


def deriv(f, x, n, h=1e-3):
    if n == 1:
        return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12*h)
    return (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h)
            - f(x - 2*h)) / (12 * h**2)


# ---------------------------------------------------------------------
# 3.  Compromise r* and the RHS  2 alpha^2 / k_0
# ---------------------------------------------------------------------
print("=" * 70)
print(" 1. Intra-fragment AO-integral inputs at the compromise geometry")
print("=" * 70)

r_star = brentq(lambda r: deriv(E_H2, r, 1) + deriv(E_H2plus, r, 1), 1.4, 2.4)
alpha  = deriv(E_H2, r_star, 1)
k_0    = (deriv(E_H2, r_star, 2) + deriv(E_H2plus, r_star, 2)) / 2
RHS    = 2 * alpha**2 / k_0

Uv, Jv, Kv, Mv = two_e_h2(r_star)
print(f"  r* = {r_star:.4f} bohr   (E_H2'(r*) + E_H2+'(r*) = 0)")
print(f"  Intra 1e:  h    = {h_aa(r_star):+.5f}")
print(f"             h_ab = {h_ab(r_star):+.5f}")
print(f"             s    = {s_intra(r_star):+.5f}")
print(f"  Intra 2e:  U    = {Uv:.5f}   (R-independent)")
print(f"             J    = {Jv:.5f}")
print(f"             K    = {Kv:.5f}")
print(f"             M    = {Mv:.5f}")
print()
print(f"  alpha  = E_H2'(r*)            = {alpha:+.5f} Ha/bohr")
print(f"  k_0    = (E_H2''+E_H2+'')/2   = {k_0:+.5f} Ha/bohr^2")
print(f"  RHS    = 2 alpha^2 / k_0      = {RHS:.5f} Ha")


# ---------------------------------------------------------------------
# 4.  Solve |t(R_c)| = RHS for R_c using inter-fragment 1e integrals
# ---------------------------------------------------------------------
print()
print("=" * 70)
print(" 2. Class-III -> II transition from inter-fragment 1e integrals")
print("=" * 70)

R_c = brentq(lambda R: abs(t_of_R(R, r_star)) - RHS, 2.0, 30.0)
cx_c = cross_1e_collinear(R_c, r_star)
denom = 2 * np.sqrt((1 + cx_c['s_ab']) * (1 + cx_c['s_cd']))
cross_sum = cx_c['h_ac'] + cx_c['h_ad'] + cx_c['h_bc'] + cx_c['h_bd']
t_c_check = cross_sum / denom

print(f"  R_c = {R_c:.4f} bohr   (|t(R_c)| = 2 alpha^2 / k_0)")
print()
print(f"  Inter-fragment 1e at R_c (collinear a-b ... c-d, r_intra = r*):")
print(f"      h_ac = {cx_c['h_ac']:+.5f}")
print(f"      h_ad = {cx_c['h_ad']:+.5f}")
print(f"      h_bc = {cx_c['h_bc']:+.5f}")
print(f"      h_bd = {cx_c['h_bd']:+.5f}")
print(f"  -----------------------------")
print(f"   sum = {cross_sum:+.5f}    (numerator)")
print(f"      s_ab(r*) = s_cd(r*) = {cx_c['s_ab']:.5f}")
print(f"      denom = 2 sqrt((1+s_ab)(1+s_cd)) = {denom:.5f}")
print()
print(f"  |t(R_c)| = |sum| / denom = {abs(t_c_check):.5f} Ha")
print(f"  RHS      = 2 alpha^2/k_0 = {RHS:.5f} Ha   (match)")


# ---------------------------------------------------------------------
# 5.  E_-(xi) at three R values, showing the single-well -> flat ->
#     double-well crossover
# ---------------------------------------------------------------------
def E_minus(xi, R):
    """Antisymmetric-stretch GS energy of the 2x2 diabatic Hamiltonian
       at R, with eta = 0.  All inputs are AO integrals."""
    tR = abs(t_of_R(R, r_star))
    return 0.5 * k_0 * xi**2 - np.sqrt(2 * alpha**2 * xi**2 + tR**2)


# Use mass-weighted xi (in bohr).  R offsets chosen to bracket R_c with
# visible single-well, flat, double-well shapes.
dR = 1.5
R_vals = [R_c - dR, R_c, R_c + dR]
labels = [f"R = {R_c - dR:.2f} bohr  (Class III)",
          f"R = {R_c:.2f} bohr  (R = R_c, flat)",
          f"R = {R_c + dR:.2f} bohr  (Class II)"]

xi_grid = np.linspace(-1.6, 1.6, 401)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

# --- Panel (a): |t(R)| vs R, with horizontal line at 2 alpha^2 / k_0
R_grid = np.linspace(3.0, 12.0, 80)
t_grid = np.array([abs(t_of_R(R, r_star)) for R in R_grid])
axes[0].semilogy(R_grid, t_grid, 'C0-', lw=2.0,
                 label=r'$|t(R)|$  from cross 1e integrals')
axes[0].axhline(RHS, color='C3', lw=1.6, ls='--',
                label=r'$2\alpha^2/k_0$  from intra-AO integrals at $r^*$')
axes[0].axvline(R_c, color='k', lw=0.8, ls=':')
axes[0].plot(R_c, RHS, 'ko', ms=6)
axes[0].annotate(f'$R_c = {R_c:.2f}$ bohr', (R_c, RHS),
                 xytext=(R_c + 0.3, RHS * 2.2), fontsize=10,
                 arrowprops=dict(arrowstyle='-', lw=0.6))
axes[0].set_xlabel(r'$R$ (bohr)')
axes[0].set_ylabel('Ha (log)')
axes[0].set_title(r'(a)  Class III $\leftrightarrow$ II crossing condition')
axes[0].grid(alpha=0.3, which='both')
axes[0].legend(fontsize=9, loc='lower left')

# --- Panel (b): E_-(xi) at three R values
colors = ['C0', 'k', 'C3']
styles = ['-', '-', '-']
lws    = [2.0, 1.6, 2.0]
for R, lab, c, ls, lw in zip(R_vals, labels, colors, styles, lws):
    E = np.array([E_minus(x, R) for x in xi_grid])
    # Subtract value at xi=0 to align curves
    E0 = E_minus(0.0, R)
    axes[1].plot(xi_grid, (E - E0) * 1000, c + ls, lw=lw, label=lab)
axes[1].axhline(0, color='gray', lw=0.4)
axes[1].axvline(0, color='gray', lw=0.4)
axes[1].set_xlabel(r'antisymmetric stretch  $\xi$  (bohr)')
axes[1].set_ylabel(r'$E_-(\xi) - E_-(0)$  (mHa)')
axes[1].set_title(r'(b)  Antisymmetric-stretch potential across $R_c$')
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=9, loc='upper center')

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'vbt-3', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, 'fig_h2h2_plus_class_transition.png')
out_pdf = os.path.join(out_dir, 'fig_h2h2_plus_class_transition.pdf')
plt.savefig(out_png, dpi=140)
plt.savefig(out_pdf)
plt.close()
print()
print(f"  Figure saved to {out_png}")
print(f"                  {out_pdf}")


# ---------------------------------------------------------------------
# 5b.  3D bifurcation surface E_-(xi, R) showing the Class III -> II
#      transition as a pitchfork bifurcation in xi at R = R_c
# ---------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def xi_min(R):
    """Class-II equilibrium |xi| where dE/dxi = 0 (zero for R <= R_c)."""
    tR = abs(t_of_R(R, r_star))
    if k_0 - 2 * alpha**2 / tR >= 0:
        return 0.0
    rhs = (2 * alpha**2 / k_0)**2 - tR**2
    return np.sqrt(rhs / (2 * alpha**2))


fig3d = plt.figure(figsize=(14, 6.2))

# --- Panel (a): 3D surface of E_-(xi, R) - E_-(0, R) -----------------
ax3d = fig3d.add_subplot(1, 2, 1, projection='3d')

xi_range = np.linspace(-1.6, 1.6, 81)
R_range  = np.linspace(R_c - 2.0, R_c + 2.0, 61)
XI, RR = np.meshgrid(xi_range, R_range)

# Cache |t(R)| on the R grid (the pyscf 4-atom call dominates the cost)
t_cache = np.array([abs(t_of_R(R, r_star)) for R in R_range])
TT = np.broadcast_to(t_cache[:, None], RR.shape)

E_surface = 0.5 * k_0 * XI**2 - np.sqrt(2 * alpha**2 * XI**2 + TT**2)
E_at_origin = -t_cache[:, None]
DELTA = (E_surface - E_at_origin) * 1000  # mHa, height above E(xi=0,R)

surf = ax3d.plot_surface(XI, RR, DELTA, cmap='RdBu_r', linewidth=0,
                         antialiased=True, alpha=0.9,
                         vmin=-25, vmax=80, rcount=60, ccount=60)
# Project iso-energy contours onto the floor for a cleaner read
floor = DELTA.min() - 18
ax3d.contour(XI, RR, DELTA, levels=np.linspace(-20, 80, 11),
             zdir='z', offset=floor, cmap='RdBu_r', linewidths=0.6,
             alpha=0.7)

# Mark the R = R_c isoline (the boundary cross-section)
i_c = np.argmin(np.abs(R_range - R_c))
ax3d.plot(xi_range, np.full_like(xi_range, R_c),
          DELTA[i_c, :], 'k-', lw=2.6,
          label=fr'$R = R_c = {R_c:.2f}$ bohr')
# Symmetric-point loci (xi = 0, valley floor below R_c, ridge above)
ax3d.plot(np.zeros_like(R_range), R_range,
          DELTA[:, len(xi_range) // 2], 'k--', lw=1.2, alpha=0.7)

# Pitchfork loci: traced minima for R > R_c
R_pf = np.linspace(R_c, R_c + 2.0, 50)
xi_pf = np.array([xi_min(R) if R >= R_c else 0.0 for R in R_pf])
def E_at(xi, R):
    tR = abs(t_of_R(R, r_star))
    return 0.5 * k_0 * xi**2 - np.sqrt(2 * alpha**2 * xi**2 + tR**2) + tR
E_pf = np.array([E_at(x, R) * 1000 for x, R in zip(xi_pf, R_pf)])
ax3d.plot(+xi_pf, R_pf, E_pf, 'k-', lw=2.0)
ax3d.plot(-xi_pf, R_pf, E_pf, 'k-', lw=2.0,
          label=r'$\xi_{\rm eq}^\pm(R)$ pitchfork branches')

ax3d.set_xlabel(r'$\xi$ (bohr)', labelpad=4)
ax3d.set_ylabel(r'$R$ (bohr)', labelpad=4)
ax3d.set_zlabel(r'$E_-(\xi,R) - E_-(0,R)$  (mHa)', labelpad=6)
ax3d.set_zlim(floor, max(80, DELTA.max() * 1.05))
ax3d.set_title(r'(a)  Bifurcation surface  $E_-(\xi, R) - E_-(0,R)$  ($\eta = 0$)',
               fontsize=11)
ax3d.view_init(elev=28, azim=-62)
ax3d.legend(fontsize=9, loc='upper left')

# --- Panel (b): contour map of the same surface, with valley loci -----
ax2d = fig3d.add_subplot(1, 2, 2)
cf = ax2d.contourf(XI, RR, DELTA, levels=np.linspace(-30, 30, 21),
                   cmap='RdBu_r', extend='both')
ax2d.contour(XI, RR, DELTA, levels=[0.0], colors='k', linewidths=0.6)
fig3d.colorbar(cf, ax=ax2d, label=r'$E_-(\xi,R)-E_-(0,R)$  (mHa)')

R_zoom = np.linspace(R_c, R_c + 2.0, 50)
xi_eq  = np.array([xi_min(R) for R in R_zoom])
ax2d.plot(+xi_eq, R_zoom, 'k-', lw=1.8, label=r'$\xi_{\rm eq}(R)$ (Class II)')
ax2d.plot(-xi_eq, R_zoom, 'k-', lw=1.8)
ax2d.axhline(R_c, color='k', lw=0.6, ls=':')
ax2d.text(1.0, R_c - 0.05, r'$R_c$', fontsize=10, va='top')
ax2d.set_xlabel(r'$\xi$ (bohr)')
ax2d.set_ylabel(r'$R$ (bohr)')
ax2d.set_title(r'(b)  Pitchfork: $\xi=0$ stable for $R<R_c$,'
               '\n      ' r'splits into $\pm\xi_{\rm eq}(R)$ for $R>R_c$',
               fontsize=11)
ax2d.legend(fontsize=9, loc='upper right')
ax2d.grid(alpha=0.25)

plt.tight_layout()
out3d_png = os.path.join(out_dir, 'fig_h2h2_plus_class_transition_3d.png')
out3d_pdf = os.path.join(out_dir, 'fig_h2h2_plus_class_transition_3d.pdf')
plt.savefig(out3d_png, dpi=140)
plt.savefig(out3d_pdf)
plt.close()
print(f"  3D figure saved to {out3d_png}")
print(f"                     {out3d_pdf}")


# ---------------------------------------------------------------------
# 6.  Numerical check: curvature at xi = 0 should change sign at R_c
# ---------------------------------------------------------------------
def curvature_at_origin(R):
    return k_0 - 2 * alpha**2 / abs(t_of_R(R, r_star))


# ---------------------------------------------------------------------
# 6.5  Closed-form transcendental criterion using 2-atom h_ab(d)
# ---------------------------------------------------------------------
# Approximate the 4-atom cross 1e integrals h_ac, h_ad, h_bc, h_bd by
# the 2-atom Slater-1s formula h_ab(d) = -S(d)/2 - K(d) at the actual
# inter-atom distance d.  In the collinear (a-b ... c-d) geometry the
# four cross distances are (R-r*, R, R, R+r*) for (h_bc, h_ac, h_bd, h_ad)
# respectively.  The dominant term (smallest distance) is h_bc(R - r*).
#
# Keeping ONLY h_bc and using h_ab(d) = -e^(-d) (d^2 + 9 d + 9)/6 gives
# a clean transcendental equation for R_c in terms of (alpha, k_0, s(r*))
# alone:
#
#       (x^2 + 9 x + 9) e^(-x)  =  24 alpha^2 (1 + s(r*)) / k_0,
#                                       x = R_c - r*.
#
# Including the h_ac and h_bd contributions (both at distance R,
# multiplicity 2) refines this to
#
#       [(x^2+9x+9) + 2 (R^2+9R+9) e^(-(R-x))] e^(-x)
#                =  24 alpha^2 (1 + s(r*)) / k_0
#
# which is also transcendental in R only.  The 4-atom pyscf calculation
# above includes additionally the 3-center nuclear-attraction integrals
# (V_a and V_d acting between b and c, etc.) -- these are NOT in the
# 2-atom h_ab(d) formula and account for the residual ~30% gap in |t|.
print()
print("=" * 70)
print(" 3. Closed-form transcendental R_c from 2-atom h_ab(d)")
print("=" * 70)


def h_ab_2atom(d):
    """Slater 1s zeta=1, two-atom h_ab(d) = -S(d)/2 - K(d)."""
    return -(np.exp(-d) * (1 + d + d**2 / 3)) / 2 - (1 + d) * np.exp(-d)


# Leading-only:  |h_bc(R-r*)| / [2 (1 + s(r*))] = 2 alpha^2 / k_0
def lhs_leading(R):
    return abs(h_ab_2atom(R - r_star)) / (2 * (1 + s_intra(r_star)))


def lhs_three_term(R):
    """Add h_ac and h_bd (both at distance R)."""
    cross = h_ab_2atom(R - r_star) + 2 * h_ab_2atom(R)
    return abs(cross) / (2 * (1 + s_intra(r_star)))


def lhs_four_term(R):
    """Full 2-atom cross-sum (still drops 3-center corrections)."""
    cross = (h_ab_2atom(R - r_star) + 2 * h_ab_2atom(R)
             + h_ab_2atom(R + r_star))
    return abs(cross) / (2 * (1 + s_intra(r_star)))


R_c_lead = brentq(lambda R: lhs_leading(R)    - RHS, 2.0, 20.0)
R_c_3t   = brentq(lambda R: lhs_three_term(R) - RHS, 2.0, 20.0)
R_c_4t   = brentq(lambda R: lhs_four_term(R)  - RHS, 2.0, 20.0)

print("  Approximations to t(R) at zeta=1, dropping 3-center corrections:")
print(f"    leading (h_bc only):        |t| = |h_ab(R-r*)| / [2(1+s)]")
print(f"      transcendental:  (x^2+9x+9) e^(-x) = "
      f"24 alpha^2 (1+s)/k_0  with  x = R-r*")
print(f"      24 alpha^2 (1+s) / k_0 = "
      f"{24 * alpha**2 * (1 + s_intra(r_star)) / k_0:.5f}")
print(f"      R_c (leading)       = {R_c_lead:.4f} bohr")
print(f"    + h_ac, h_bd (3-term):  R_c = {R_c_3t:.4f} bohr")
print(f"    + h_ad      (4-term):   R_c = {R_c_4t:.4f} bohr")
print(f"    full pyscf (with 3-center attractions): R_c = {R_c:.4f} bohr")
print()
print(f"  Closed-form (leading) R_c is low by"
      f" {R_c - R_c_lead:.3f} bohr ({(R_c_lead - R_c)/R_c*100:+.1f}%).")
print(f"  Adding the other 2-atom cross terms (h_ac, h_bd, h_ad) closes only")
print(f"  ~half the gap.  The remainder is the 3-center nuclear-attraction")
print(f"  integrals  < phi_b | V_a + V_d | phi_c >, etc., which the 2-atom")
print(f"  h_ab(d) formula cannot reproduce -- they appear because the b-c")
print(f"  matrix element of h sees the OTHER fragment's nuclei too, and at")
print(f"  R_c those distant nuclei are still close enough (~7-9 bohr) to")
print(f"  contribute meaningfully to <phi_b|V|phi_c>.")
print()
print(f"  Asymptotic form: at large x the criterion is dominated by x^2 e^(-x),")
print(f"  giving the Lambert-W solution")
print(f"     R_c - r*  ~  -2 W_(-1)( -sqrt(C/2) ),  C = 24 alpha^2 (1+s)/k_0,")
print(f"  i.e. R_c grows logarithmically with k_0 / alpha^2.")


print()
print("=" * 70)
print(" 4. xi-curvature numerical check  (k_asym = k_0 - 2 alpha^2/|t|)")
print("=" * 70)
print(f"  {'R (bohr)':>10}  {'|t(R)|':>10}  {'k_asym':>11}  class")
for R in [R_c - 1.5, R_c - 0.5, R_c - 0.05, R_c, R_c + 0.05, R_c + 0.5, R_c + 1.5]:
    tR = abs(t_of_R(R, r_star))
    ka = k_0 - 2 * alpha**2 / tR
    cls = "III" if ka > 1e-6 else ("II" if ka < -1e-6 else "boundary")
    print(f"  {R:>10.4f}  {tR:>10.6f}  {ka:>+11.6f}  {cls}")
print()
print(f"  At R = R_c = {R_c:.4f}, k_asym = "
      f"{curvature_at_origin(R_c):+.6f} Ha/bohr^2  (zero).")
print(f"  This is the Robin-Day Class III -> II boundary, expressed entirely")
print(f"  in 1e and 2e AO integrals.")
