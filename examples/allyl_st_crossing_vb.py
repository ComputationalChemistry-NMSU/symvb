"""
Singlet-triplet crossing of the 3c4e allyl anion in VB language.

Framework:  sweep K at fixed (U, J) = (4, 0.5), M = 0, tracking the lowest
SINGLET and lowest TRIPLET eigenstates of the 9-det H.  For each K:

    * E_singlet, E_triplet (locates the crossing K*)
    * VB weights (w_ab, w_bc, w_ac) of the lowest singlet in the 3
      covalent Rumer structures defined in allyl_long_bond_vb.py
    * VB weight of the "triplet long-bond" structure T_ac =
      ( HL(a,c)_triplet_Ms=0 ) * b^2  -- a sigma = -1 structure living
      in the triplet sector, only det-disjoint from the singlet Phi_ac.

At s = 0, the singlet Phi_ac and the triplet T_ac populate the SAME two
dets (idx 2 and 6, b-doubly-occ with a/c open-shell) but with opposite
relative sign.  So the K dependence of their respective weights in the
lowest-S and lowest-T states tracks how the open-shell pair on (a, c)
reorganises its spin-pairing across the crossing.
"""
import os
import sys

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import FixedPsi, Molecule, hamiltonian, structure_vector
from symvb.fixed_psi import generate_dets
from symvb.spin import s_squared_matrix


# ------------------------------------------------------------------------
# 1. Build the 9-det H (reusing the setup from allyl_long_bond_vb.py)
# ------------------------------------------------------------------------
m = Molecule(
    zero_ii=True, interacting_orbs=['ab', 'bc'],
    subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
    subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
              'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)
P = generate_dets(2, 2, 3)
det_strings = [p.dets[0].det_string for p in P]
H_det, _ = hamiltonian(m, P)
H_det = sp.Matrix(H_det)
h, s, U, J, K, M = sp.symbols('h s U J K M')
H_det_s0 = H_det.subs({s: 0, h: -1})
H_det_fn = sp.lambdify((U, J, K, M), H_det_s0, 'numpy')
S2_9 = s_squared_matrix(det_strings, orbs='abc')


# ------------------------------------------------------------------------
# 2. Singlet covalent VB structures (same as allyl_long_bond_vb.py):
#    Phi_ab, Phi_bc, Phi_ac all in sigma = +1.
#    Triplet long-bond structure:  T_ac = triplet-HL(a,c) * b^2,
#    lives in sigma = -1 (Ms = 0 triplet).
# ------------------------------------------------------------------------
def sv(fp):
    """Normalised expansion of a VB structure over the 9-det basis."""
    v = np.array(structure_vector(fp, det_strings), float).ravel()
    return v / np.linalg.norm(v)

# Singlet HL-type covalent structures (same as allyl_long_bond_vb.py)
vhat_ab = sv(FixedPsi('aBcC', coupled_pairs=[(0, 1)]))
vhat_bc = sv(FixedPsi('aAbC', coupled_pairs=[(2, 3)]))
vhat_ac = sv(FixedPsi('abBC', coupled_pairs=[(0, 3)]))

# Triplet long-bond structure T_ac = (HL-triplet on a,c) * b^2. HL-triplet Ms=0
# is (a_alpha c_beta + a_beta c_alpha)/sqrt(2); times b^2 it populates the SAME
# two dets as the singlet Phi_ac but with opposite relative sign, so build it
# from the two determinant vectors directly.
_d = lambda d: np.array(structure_vector(FixedPsi(d), det_strings), float).ravel()
_vT = _d('abBC') - _d('cbBA')
vhat_Tac = _vT / np.linalg.norm(_vT)


# ------------------------------------------------------------------------
# 3. Solver: lowest singlet, lowest triplet, VB weights
# ------------------------------------------------------------------------
def analyse(Uv, Jv, Kv, Mv):
    Hn = np.array(H_det_fn(Uv, Jv, Kv, Mv), dtype=float)
    Hn = 0.5 * (Hn + Hn.T)
    # Full diagonalisation, classify by <S^2>
    ev, vec_all = np.linalg.eigh(Hn)
    s2_all = np.array([vec_all[:, i] @ S2_9 @ vec_all[:, i] for i in range(9)])
    # Lowest singlet
    s_mask = np.abs(s2_all - 0.0) < 0.3
    i_s = np.where(s_mask)[0][0]
    E_s = ev[i_s]; psi_s = vec_all[:, i_s]
    # Lowest triplet
    t_mask = np.abs(s2_all - 2.0) < 0.3
    i_t = np.where(t_mask)[0][0] if t_mask.any() else None
    if i_t is not None:
        E_t = ev[i_t]; psi_t = vec_all[:, i_t]
    else:
        E_t = np.nan; psi_t = np.zeros(9)
    # VB weights in lowest singlet
    w_ab = (vhat_ab @ psi_s) ** 2
    w_bc = (vhat_bc @ psi_s) ** 2
    w_ac = (vhat_ac @ psi_s) ** 2
    # VB weight of TRIPLET long-bond in lowest triplet
    w_Tac = (vhat_Tac @ psi_t) ** 2 if i_t is not None else np.nan
    return dict(E_s=E_s, E_t=E_t, gap=E_t - E_s,
                w_ab=w_ab, w_bc=w_bc, w_ac=w_ac, w_cov=w_ab + w_bc + w_ac,
                w_Tac=w_Tac, psi_s=psi_s, psi_t=psi_t)


# ------------------------------------------------------------------------
# 4. K sweep at U=4, J=0.5, M=0
# ------------------------------------------------------------------------
U_fix, J_fix, M_fix = 4.0, 0.5, 0.0
Ks = np.linspace(0, 3.0, 301)
recs = [analyse(U_fix, J_fix, Kv, M_fix) for Kv in Ks]

E_s = np.array([r['E_s'] for r in recs])
E_t = np.array([r['E_t'] for r in recs])
gaps = E_t - E_s
# Find crossing K*
sign_change = np.where(np.diff(np.sign(gaps)))[0]
if sign_change.size:
    i = sign_change[0]
    # Linear interpolate crossing
    K_star = Ks[i] - gaps[i] * (Ks[i+1] - Ks[i]) / (gaps[i+1] - gaps[i])
else:
    K_star = np.nan
print(f"\nS-T crossing at U={U_fix}, J={J_fix}, M=0:  K* = {K_star:.4f}")
print(f"  K*/U = {K_star / U_fix:.4f}")

# VB composition at K*, at K=0, and past crossing
print(f"\n{'K':>6s}  {'E_s':>9s}  {'E_t':>9s}  {'gap':>8s}  "
      f"{'w_ab':>6s}  {'w_bc':>6s}  {'w_ac':>6s}  {'w_cov':>6s}  {'w_T_ac':>7s}")
for Kv in [0, 0.5, 1.0, K_star, K_star + 0.3, 2.0, 3.0]:
    if np.isnan(Kv): continue
    o = analyse(U_fix, J_fix, Kv, M_fix)
    print(f"  {Kv:>6.3f}  {o['E_s']:>+9.4f}  {o['E_t']:>+9.4f}  "
          f"{o['gap']:>+8.4f}  {o['w_ab']:>6.4f}  {o['w_bc']:>6.4f}  "
          f"{o['w_ac']:>6.4f}  {o['w_cov']:>6.4f}  {o['w_Tac']:>7.4f}")


# ------------------------------------------------------------------------
# 5. Extended sweep: K*(U) trajectory at J = 0.5, M = 0
# ------------------------------------------------------------------------
print("\n" + "=" * 60)
print("K*(U) trajectory at J = 0.5, M = 0")
print("=" * 60)
print(f"  {'U':>6s}  {'K*':>7s}  {'K*/U':>6s}  "
      f"{'w_ac at K=0':>11s}  {'w_ac at K*':>11s}")
K_fine = np.linspace(0, 8, 801)
for Uv in [2, 4, 8, 16, 32, 64]:
    gaps_u = np.array([analyse(Uv, J_fix, Kv, M_fix)['gap'] for Kv in K_fine])
    ch = np.where(np.diff(np.sign(gaps_u)))[0]
    if ch.size:
        i = ch[0]
        Ks_u = K_fine[i] - gaps_u[i] * (K_fine[i+1] - K_fine[i]) / (gaps_u[i+1] - gaps_u[i])
    else:
        Ks_u = np.nan
    w_ac_0 = analyse(Uv, J_fix, 0.0, M_fix)['w_ac']
    w_ac_at_star = analyse(Uv, J_fix, Ks_u - 1e-4, M_fix)['w_ac'] if not np.isnan(Ks_u) else np.nan
    print(f"  {Uv:>6g}  {Ks_u:>7.4f}  {Ks_u/Uv:>6.4f}  "
          f"{w_ac_0:>11.4f}  {w_ac_at_star:>11.4f}")


# ------------------------------------------------------------------------
# 6. Figure: S-T crossing + VB decomposition of lowest singlet
# ------------------------------------------------------------------------
w_ab = np.array([r['w_ab'] for r in recs])
w_bc = np.array([r['w_bc'] for r in recs])
w_ac = np.array([r['w_ac'] for r in recs])
w_cov = np.array([r['w_cov'] for r in recs])
w_Tac = np.array([r['w_Tac'] for r in recs])

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

# Panel (a): energies
ax[0].plot(Ks, E_s, 'C0-', lw=2.0, label='lowest singlet')
ax[0].plot(Ks, E_t, 'C3-', lw=2.0, label='lowest triplet')
ax[0].axvline(K_star, color='k', lw=0.8, ls='--', alpha=0.7)
ax[0].text(K_star + 0.05, E_s[0] + 0.5, rf'$K^* = {K_star:.3f}$',
           fontsize=10, ha='left')
ax[0].set_xlabel(r'$K / |h|$'); ax[0].set_ylabel(r'$E$ (units of $|h|$)')
ax[0].set_title(rf'(a)  S-T crossing at $U={U_fix}$, $J={J_fix}$')
ax[0].legend(fontsize=9, loc='lower right'); ax[0].grid(alpha=0.3)

# Panel (b): singlet VB composition
ax[1].plot(Ks, w_ab, 'C0-',  lw=1.6, label=r'$w_{ab}$')
ax[1].plot(Ks, w_bc, 'C0--', lw=1.4, label=r'$w_{bc}$')
ax[1].plot(Ks, w_ac, 'C3-',  lw=2.2, label=r'$w_{ac}$ (long-bond)')
ax[1].plot(Ks, w_cov, 'k:',  lw=1.6, label='total covalent')
ax[1].axvline(K_star, color='k', lw=0.8, ls='--', alpha=0.7)
ax[1].set_xlabel(r'$K / |h|$'); ax[1].set_ylabel('VB weight in lowest singlet')
ax[1].set_title('(b)  Singlet VB composition vs $K$')
ax[1].legend(fontsize=9, loc='center right'); ax[1].grid(alpha=0.3)
ax[1].set_ylim(0, 1.05)

# Panel (c): singlet vs triplet long-bond amplitude
ax[2].plot(Ks, w_ac,  'C3-',  lw=2.2, label=r'singlet $w_{ac}$')
ax[2].plot(Ks, w_Tac, 'C4--', lw=2.0, label=r'triplet $w_{T_{ac}}$')
ax[2].axvline(K_star, color='k', lw=0.8, ls='--', alpha=0.7)
ax[2].axhline(0.5, color='k', lw=0.3, alpha=0.3)
ax[2].set_xlabel(r'$K / |h|$'); ax[2].set_ylabel('long-bond weight')
ax[2].set_title('(c)  Singlet vs triplet long-bond weight')
ax[2].legend(fontsize=9, loc='center right'); ax[2].grid(alpha=0.3)
ax[2].set_ylim(0, 1.05)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', 'figures', 'fig_allyl_st_crossing_vb.png')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
plt.savefig(outpath, dpi=140)
plt.close()
print(f"\nFigure saved: {outpath}")
