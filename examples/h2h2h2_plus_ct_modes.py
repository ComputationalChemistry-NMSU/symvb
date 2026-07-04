"""(H2)3+ linear chain: systematic construction of charge-transfer coordinates.

Six nuclei, five electrons, chain topology.  Three localized sigma-hole
diabatics |I> (I = 1, 2, 3) with hole on fragment I and two other fragments
neutral.

Intra-fragment displacements u_I = r_I - r*  (I = 1, 2, 3).
Inter-fragment hoppings: only nearest-neighbor t_{12} = t_{23} = t(R) (chain,
no 1-3 direct coupling), under the assumption t = t(R).

This script shows how the two CT coordinates emerge systematically.  Recipe:

  1. Compromise geometry  r*  from  E_H2+'(r*) + (n-1) E_H2'(r*) = 0.
     For n = 3 this shifts r* toward r_e(H2).

  2. Linear vibronic model on the n = 3 diabatic manifold:
         H_el(u) = epsilon(U) * I  +  T  -  n alpha * diag(q_1, q_2, q_3)
     where q_I = u_I - U/n  (n-1 = 2 independent CT coordinates).

  3. Normal modes of {u_I} split by the mirror symmetry sigma_v (1 <-> 3):
        Q_s = (u_1 + u_2 + u_3)/sqrt(3)     a_1 symmetric, not a CT mode
        Q_m = (u_1 - 2 u_2 + u_3)/sqrt(6)   a_1 middle-vs-ends        (CT)
        Q_e = (u_1 - u_3)/sqrt(2)           b_2 end-to-end asymmetric (CT)

  4. Electronic eigenstates of T for the 3-site chain (E_m = 2t cos(m pi/4)):
        |v_1>  E_1 = t sqrt(2)   (a_1, GS for t < 0)
        |v_2>  E_2 = 0           (b_2, 1st excited)
        |v_3>  E_3 = -t sqrt(2)  (a_1, 2nd excited)

  5. Symmetry selection rules pick one (CT-mode, excited-state) pair each:
        Q_e (b_2)  <->  |v_2> (b_2),  gap  =  |t| sqrt(2)
        Q_m (a_1)  <->  |v_3> (a_1),  gap  =  2 |t| sqrt(2)
     All other couplings vanish by symmetry.  The two CT modes are
     INDEPENDENT 2-state problems.

  6. Softened force constants (generalization of k_asym = k0 - 2 alpha^2/|t|):
        K_Qe = k_0  -  9 sqrt(2) alpha^2 / (4 |t|)
        K_Qm = k_0  -  27 sqrt(2) alpha^2 / (16 |t|)

     Q_e (end-end) softens FIRST as |t| shrinks (i.e. as R grows):
        |t_c^e| = 9 sqrt(2) alpha^2 / (4 k_0)
        |t_c^m| = 27 sqrt(2) alpha^2 / (16 k_0)   <   |t_c^e|
     So the chain first breaks mirror symmetry (hole localizes on one end)
     via Q_e, and only later also loses the totally-symmetric diagonal mode
     via Q_m.

All derivations below are symbolic (sympy); formulas are verified numerically
by diagonalizing the 3 x 3 electronic Hamiltonian on a grid in (Q_e, Q_m).

Run from the repo root:  PYTHONPATH=. python3 examples/h2h2h2_plus_ct_modes.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import sympy as sp
from scipy.optimize import brentq


# ---------------------------------------------------------------------
# 1. Morse potentials for H2 and H2+ (same as h2h2_plus_diabatic.py)
# ---------------------------------------------------------------------
H2_DE,  H2_RE,  H2_K,  H2_ASY  = 0.1745, 1.401, 0.370, -1.0
H2P_DE, H2P_RE, H2P_K, H2P_ASY = 0.1026, 1.997, 0.103, -0.5

r = sp.Symbol('r', positive=True)


def morse(De, re, k, asymptote):
    a = sp.sqrt(k / (2 * De))
    return asymptote + De * (1 - sp.exp(-a * (r - re)))**2 - De


E_H2  = morse(H2_DE,  H2_RE,  H2_K,  H2_ASY)
E_H2p = morse(H2P_DE, H2P_RE, H2P_K, H2P_ASY)


# ---------------------------------------------------------------------
# 2. Compromise bond length for n = 3:  E_H2+'(r*) + 2 E_H2'(r*) = 0
# ---------------------------------------------------------------------
n_frag = 3
compromise = sp.diff(E_H2p, r) + (n_frag - 1) * sp.diff(E_H2, r)
r_star = brentq(sp.lambdify(r, compromise, 'numpy'), H2_RE, H2P_RE)

alpha       = float(sp.lambdify(r, sp.diff(E_H2,  r))(r_star))
alpha_plus  = float(sp.lambdify(r, sp.diff(E_H2p, r))(r_star))
f_double    = float(sp.lambdify(r, sp.diff(E_H2p, r, 2)
                                   + (n_frag - 1) * sp.diff(E_H2, r, 2))(r_star))
k0 = f_double / n_frag

print("=" * 70)
print("Compromise geometry for (H2)3+ chain")
print("=" * 70)
print(f"  r*                                = {r_star:.4f} bohr    "
      f"(was 1.621 for n = 2; now closer to r_e(H2) = {H2_RE})")
print(f"  alpha   = E_H2'(r*)               = {alpha:+.5f} Ha/bohr")
print(f"  alpha+  = E_H2+'(r*)              = {alpha_plus:+.5f} Ha/bohr  "
      f"(check: -2 alpha = {-2*alpha:+.5f})")
print(f"  k_0    = [E_H2+'' + 2 E_H2'']/3   = {k0:.5f} Ha/bohr^2")


# ---------------------------------------------------------------------
# 3. Symbolic construction of CT normal modes
# ---------------------------------------------------------------------
u1, u2, u3 = sp.symbols('u1 u2 u3', real=True)

# Nuclear normal modes (orthonormal linear combinations of u_I)
Q_s = (u1 + u2 + u3) / sp.sqrt(3)               # a_1, symmetric
Q_m = (u1 - 2*u2 + u3) / sp.sqrt(6)             # a_1, middle-vs-ends (CT)
Q_e = (u1 - u3) / sp.sqrt(2)                    # b_2, end-to-end asymm (CT)

# Inverse transform: u_I in terms of Q's
Qs, Qm, Qe = sp.symbols('Q_s Q_m Q_e', real=True)
transform = sp.solve([sp.Eq(Qs, Q_s), sp.Eq(Qm, Q_m), sp.Eq(Qe, Q_e)],
                     [u1, u2, u3])
u_of_Q = sp.Matrix([transform[u1], transform[u2], transform[u3]])

# q_I = u_I - U/3  where U = u_1 + u_2 + u_3 = sqrt(3) * Q_s
q_vec = u_of_Q - sp.ones(3, 1) * (u_of_Q[0] + u_of_Q[1] + u_of_Q[2]) / 3
print("\n" + "=" * 70)
print("CT coordinates q_I = u_I - U/3 in the nuclear-mode basis")
print("=" * 70)
print(f"  q_1 = {sp.simplify(q_vec[0])}")
print(f"  q_2 = {sp.simplify(q_vec[1])}")
print(f"  q_3 = {sp.simplify(q_vec[2])}")
print("  -> Q_s drops out as expected (it does NOT drive CT); only Q_m, Q_e remain")


# ---------------------------------------------------------------------
# 4. Electronic Hamiltonian T and its eigenstates
# ---------------------------------------------------------------------
t_sym = sp.Symbol('t', real=True, negative=True)
T = sp.Matrix([[0,      t_sym, 0    ],
               [t_sym,  0,     t_sym],
               [0,      t_sym, 0    ]])
eigs = T.eigenvects()
eigs_sorted = sorted(eigs, key=lambda x: float(x[0].subs(t_sym, -1)))
print("\n" + "=" * 70)
print("Electronic eigenstates of T (3-site chain, nearest-neighbor t)")
print("=" * 70)
irrep_labels = {'v_1': 'a_1 (GS)', 'v_2': 'b_2 (1st exc)', 'v_3': 'a_1 (2nd exc)'}
V_cols = []
E_vals = []
for i, (eval_, mult, evecs) in enumerate(eigs_sorted):
    v = evecs[0]
    v = v / v.norm()
    V_cols.append(v)
    E_vals.append(eval_)
    label = f"v_{i+1}"
    print(f"  {label}  E = {sp.simplify(eval_)}   vec = {v.T}   "
          f"irrep: {irrep_labels[label]}")


# ---------------------------------------------------------------------
# 5. Linear vibronic coupling and symbolic selection rules
# ---------------------------------------------------------------------
# Full diabatic coupling: V_LVC = -n * alpha * diag(q_1, q_2, q_3)
alpha_s = sp.Symbol('alpha', real=True, positive=True)
V_LVC = -n_frag * alpha_s * sp.diag(q_vec[0], q_vec[1], q_vec[2])

# Transform to electronic-eigenstate basis
def _me(i, j):
    m = V_cols[i].T * V_LVC * V_cols[j]
    return sp.simplify(m[0, 0])

V_basis = sp.Matrix(3, 3, lambda i, j: _me(i, j))

print("\n" + "=" * 70)
print("Vibronic coupling in electronic-eigenstate basis  <v_i | V_LVC | v_j>")
print("=" * 70)
for i in range(3):
    for j in range(3):
        expr = V_basis[i, j]
        if expr != 0:
            print(f"  <v_{i+1} | V_LVC | v_{j+1}>  =  {expr}")

# Pull out coupling coefficients to GS: lambda_Q = d <v_1|V|v_m> / dQ
Q_e_coup_v2 = sp.simplify(sp.diff(V_basis[0, 1], Qe))
Q_m_coup_v3 = sp.simplify(sp.diff(V_basis[0, 2], Qm))
Q_e_coup_v3 = sp.simplify(sp.diff(V_basis[0, 2], Qe))
Q_m_coup_v2 = sp.simplify(sp.diff(V_basis[0, 1], Qm))

print("\nSelection rules:")
print(f"  d<v_1|V|v_2>/dQ_e = {Q_e_coup_v2}         (b_2 x b_2 -> a_1: allowed)")
print(f"  d<v_1|V|v_2>/dQ_m = {Q_m_coup_v2}         (a_1 x b_2 -> b_2: forbidden)")
print(f"  d<v_1|V|v_3>/dQ_m = {Q_m_coup_v3}         (a_1 x a_1 -> a_1: allowed)")
print(f"  d<v_1|V|v_3>/dQ_e = {Q_e_coup_v3}         (b_2 x a_1 -> b_2: forbidden)")


# ---------------------------------------------------------------------
# 6. Closed-form softened force constants
# ---------------------------------------------------------------------
# K_Q = k_0 - 2 * |lambda_Q|^2 / (E_m - E_1)
k0_s = sp.Symbol('k_0', positive=True)
gap_e = E_vals[1] - E_vals[0]                           # = -t sqrt(2) = |t| sqrt(2)
gap_m = E_vals[2] - E_vals[0]                           # = -2 t sqrt(2)
K_Qe_sym = sp.simplify(k0_s - 2 * Q_e_coup_v2**2 / gap_e)
K_Qm_sym = sp.simplify(k0_s - 2 * Q_m_coup_v3**2 / gap_m)

print("\n" + "=" * 70)
print("Softened force constants for the two CT modes")
print("=" * 70)
print(f"  K(Q_e) = {K_Qe_sym}")
print(f"  K(Q_m) = {K_Qm_sym}")

# Critical hopping for each mode (where K -> 0)
t_c_e = sp.solve(K_Qe_sym.subs(t_sym, -sp.Symbol('tabs', positive=True)),
                 sp.Symbol('tabs', positive=True))[0]
t_c_m = sp.solve(K_Qm_sym.subs(t_sym, -sp.Symbol('tabs', positive=True)),
                 sp.Symbol('tabs', positive=True))[0]
print(f"\n  |t_c^e|  (Q_e goes imaginary) = {t_c_e}")
print(f"  |t_c^m|  (Q_m goes imaginary) = {t_c_m}")

t_c_e_num = float(t_c_e.subs({alpha_s: alpha, k0_s: k0}))
t_c_m_num = float(t_c_m.subs({alpha_s: alpha, k0_s: k0}))
print(f"\n  numerically:   |t_c^e| = {t_c_e_num:.5f} Ha    "
      f"|t_c^m| = {t_c_m_num:.5f} Ha")
print(f"  ratio                    = {t_c_e_num/t_c_m_num:.3f}    "
      f"(Q_e softens first)")


# ---------------------------------------------------------------------
# 7. Numerical verification: diagonalize the 3x3 H_el directly
# ---------------------------------------------------------------------
def H_el_num(Qe_val, Qm_val, t_val):
    # u from Q (Q_s = 0 is fine; it doesn't affect CT)
    u1v = Qm_val / np.sqrt(6) + Qe_val / np.sqrt(2)
    u2v = -2 * Qm_val / np.sqrt(6)
    u3v = Qm_val / np.sqrt(6) - Qe_val / np.sqrt(2)
    U_val = u1v + u2v + u3v                       # should be ~ 0
    q1v = u1v - U_val / 3
    q2v = u2v - U_val / 3
    q3v = u3v - U_val / 3
    H = np.array([[0.0, t_val, 0.0],
                  [t_val, 0.0, t_val],
                  [0.0, t_val, 0.0]]) \
        + (-n_frag * alpha) * np.diag([q1v, q2v, q3v])
    return np.linalg.eigvalsh(H)[0]


# finite-difference curvature check at t = -0.03 Ha  (near |t_c^e|)
t_test = -0.03
delta = 1e-3
E_00 = H_el_num(0,      0,      t_test)
E_pe = H_el_num(+delta, 0,      t_test)
E_me = H_el_num(-delta, 0,      t_test)
E_pm = H_el_num(0,      +delta, t_test)
E_mm = H_el_num(0,      -delta, t_test)
K_Qe_fd = (E_pe + E_me - 2 * E_00) / delta**2 + k0
K_Qm_fd = (E_pm + E_mm - 2 * E_00) / delta**2 + k0
K_Qe_an = float(K_Qe_sym.subs({alpha_s: alpha, k0_s: k0, t_sym: t_test}))
K_Qm_an = float(K_Qm_sym.subs({alpha_s: alpha, k0_s: k0, t_sym: t_test}))
print("\n" + "=" * 70)
print(f"Numerical check at t = {t_test} Ha")
print("=" * 70)
print(f"  K(Q_e): analytic = {K_Qe_an:+.6f}   finite-diff = {K_Qe_fd:+.6f}")
print(f"  K(Q_m): analytic = {K_Qm_an:+.6f}   finite-diff = {K_Qm_fd:+.6f}")
print("  (finite-diff adds +k_0 by hand since H_el has no bare curvature)")


# ---------------------------------------------------------------------
# 8. Scan over R using same t(R) = -0.3 (1+R) e^-R as the n=2 script
# ---------------------------------------------------------------------
T_SCALE = 0.30


def t_of_R(R_val):
    return -T_SCALE * (1 + R_val) * np.exp(-R_val)


R_c_e = brentq(lambda R_: abs(t_of_R(R_)) - t_c_e_num, 1.0, 20.0)
R_c_m = brentq(lambda R_: abs(t_of_R(R_)) - t_c_m_num, 1.0, 20.0)

mu_HH = 1836.15 / 2
TO_CM = 219474.6

print("\n" + "=" * 70)
print("Scan over inter-fragment separation R")
print("=" * 70)
print(f"  R_c(Q_e) = {R_c_e:.3f} bohr   (Q_e goes imaginary -> end-end asymm)")
print(f"  R_c(Q_m) = {R_c_m:.3f} bohr   (Q_m goes imaginary -> mid-ends split)")
print(f"  Both modes stable when R < {R_c_m:.2f} ; Class III, hole delocalized.")
print(f"  R_c(Q_m) < R < R_c(Q_e): Q_e unstable only; C_2v breaks, hole on end.")
print(f"  R > R_c(Q_e): both unstable; strongly localized Class II.")

print(f"\n{'R':>5}  {'|t|':>9}  {'K(Q_e)':>10}  {'K(Q_m)':>10}  "
      f"{'omega(Q_e)':>12}  {'omega(Q_m)':>12}  phase")
for R_val in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0]:
    tval = t_of_R(R_val)
    Ke = float(K_Qe_sym.subs({alpha_s: alpha, k0_s: k0, t_sym: tval}))
    Km = float(K_Qm_sym.subs({alpha_s: alpha, k0_s: k0, t_sym: tval}))
    we = (f"{np.sqrt(Ke/mu_HH)*TO_CM:>12.1f}" if Ke > 0
          else f"{'i '+f'{np.sqrt(-Ke/mu_HH)*TO_CM:.1f}':>12}")
    wm = (f"{np.sqrt(Km/mu_HH)*TO_CM:>12.1f}" if Km > 0
          else f"{'i '+f'{np.sqrt(-Km/mu_HH)*TO_CM:.1f}':>12}")
    if Ke > 0 and Km > 0:
        phase = "III (sym)"
    elif Ke < 0 and Km > 0:
        phase = "II-end"
    else:
        phase = "II-multi"
    print(f"{R_val:>5.2f}  {abs(tval):>9.5f}  {Ke:>+10.5f}  {Km:>+10.5f}  "
          f"{we}  {wm}  {phase}")


# ---------------------------------------------------------------------
# 9. What the user takes away
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("Take-away")
print("=" * 70)
print(f"""
  The two CT coordinates for (H2)3+ chain are the n-1 = 2 nuclear normal
  modes orthogonal to the symmetric stretch:

      Q_e = (u_1 - u_3)/sqrt(2)           b_2 end-to-end asymmetric
      Q_m = (u_1 - 2 u_2 + u_3)/sqrt(6)   a_1 middle-vs-ends

  They arise SYSTEMATICALLY: take the n-1 eigenvectors of the electronic
  hopping matrix T orthogonal to the GS, pair each with a nuclear mode of
  matching irrep under the molecular point group. For C_2v (linear chain):

      b_2 electronic excited (v_2) <--pairs--> b_2 nuclear mode (Q_e)
      a_1 electronic excited (v_3) <--pairs--> a_1 nuclear mode (Q_m)

  Each pair is an independent 2-state vibronic problem; the softening is
  the 2-state formula with appropriate coupling coefficient and gap:

      K_Q  =  k_0  -  2 |lambda_Q|^2 / (E_m - E_1)

  Numerically, |t_c^e| = {t_c_e_num:.4f} Ha  and  |t_c^m| = {t_c_m_num:.4f} Ha,
  so as R grows Q_e destabilises first: the mirror-symmetric geometry
  loses stability to an END-TO-END asymmetric distortion, localizing the
  hole on one of the end fragments (chain becomes asymmetric H2+...H2...H2
  or H2...H2...H2+).

  The same recipe extends to any n and any topology: diagonalise T,
  read off the n-1 CT coordinates from symmetry-adapted nuclear modes,
  and each (CT mode, excited electronic state) pair gives an independent
  softening formula.
""")


# ---------------------------------------------------------------------
# 10. 2D pseudo-color plot of E_GS(Q_e, Q_m) at selected R
# ---------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def E_gs_2d(Qe_v, Qm_v, t_v):
    """Full GS energy = nuclear harmonic + lowest electronic eigenvalue."""
    u1v =      Qm_v / np.sqrt(6) + Qe_v / np.sqrt(2)
    u2v = -2 * Qm_v / np.sqrt(6)
    u3v =      Qm_v / np.sqrt(6) - Qe_v / np.sqrt(2)
    U_v = u1v + u2v + u3v
    q1v, q2v, q3v = u1v - U_v/3, u2v - U_v/3, u3v - U_v/3
    H = np.array([[0.0, t_v, 0.0],
                  [t_v, 0.0, t_v],
                  [0.0, t_v, 0.0]]) + (-n_frag * alpha) * np.diag([q1v, q2v, q3v])
    E_el = np.linalg.eigvalsh(H)[0]
    return E_el + 0.5 * k0 * (Qe_v**2 + Qm_v**2)


R_PLOTS = [3.0, 4.5, 5.5, 7.0]
Q_MAX   = 1.2
N_GRID  = 240

q_grid = np.linspace(-Q_MAX, Q_MAX, N_GRID)
Qe_mesh, Qm_mesh = np.meshgrid(q_grid, q_grid)

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for ax, R_val in zip(axes.flat, R_PLOTS):
    t_v = t_of_R(R_val)
    # vectorized evaluation
    E_grid = np.empty_like(Qe_mesh)
    for i in range(N_GRID):
        for j in range(N_GRID):
            E_grid[i, j] = E_gs_2d(Qe_mesh[i, j], Qm_mesh[i, j], t_v)
    E_grid -= E_grid.min()

    im = ax.pcolormesh(Qe_mesh, Qm_mesh, E_grid, shading='auto', cmap='viridis')
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(r'$E - E_{\min}$  (Ha)', fontsize=9)

    # white contours to aid reading
    lvls = np.percentile(E_grid, [5, 15, 30, 50, 70, 90])
    ax.contour(Qe_mesh, Qm_mesh, E_grid, levels=lvls,
               colors='white', alpha=0.35, linewidths=0.5)

    Ke = float(K_Qe_sym.subs({alpha_s: alpha, k0_s: k0, t_sym: t_v}))
    Km = float(K_Qm_sym.subs({alpha_s: alpha, k0_s: k0, t_sym: t_v}))
    cls = ("III: delocalized"       if Ke > 0 and Km > 0 else
           "II-end: $C_{2v}$ broken" if Ke < 0 and Km > 0 else
           "II-multi: localized")
    ax.set_title(f"R = {R_val:.1f} bohr,  $|t|$ = {abs(t_v):.4f} Ha\n"
                 f"Class {cls}", fontsize=10.5)
    ax.set_xlabel(r'$Q_e = (u_1 - u_3)/\sqrt{2}$   (bohr)')
    ax.set_ylabel(r'$Q_m = (u_1 - 2u_2 + u_3)/\sqrt{6}$   (bohr)')
    ax.axhline(0, color='w', lw=0.4, alpha=0.35)
    ax.axvline(0, color='w', lw=0.4, alpha=0.35)
    ax.set_aspect('equal')

fig.suptitle(r'(H$_2$)$_3^+$ linear chain: GS potential on the CT plane '
             r'($Q_s = 0$, same $r^*$ center)', fontsize=12, y=1.005)
plt.tight_layout()

outpath = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'vbt-3', 'figures', 'h2h2h2_plus_CT_PES.png'))
os.makedirs(os.path.dirname(outpath), exist_ok=True)
plt.savefig(outpath, dpi=140, bbox_inches='tight')
print(f"\nPES plot saved to: {outpath}")
