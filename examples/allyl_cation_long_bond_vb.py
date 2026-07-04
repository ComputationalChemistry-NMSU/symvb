"""
Allyl cation (3c2e) VB analysis: the long-bond structure HL(a, c) in a
2-electron system, and the parallel to the 4-electron anion findings.

Six singlet VB structures span the S=0 subspace:

    HL(a, b)   = [a . b]_s                 (covalent a-b bond)
    HL(b, c)   = [b . c]_s                 (covalent b-c bond)
    HL(a, c)   = [a . c]_s                 (long-bond, no b electron)
    a^2, b^2, c^2                          (ionic / closed-shell)

where [p . q]_s = (p_alpha q_beta - p_beta q_alpha) / sqrt(2).

Key physics differences vs the 3c4e anion (see allyl_long_bond_vb.py):

  * No electron on the "bridge" atom for the long-bond structure.  HL(a,c)
    is empty on b; in the anion Phi_ac = HL(a,c) * b^2 has b doubly
    occupied.  Despite this, HL(a,c) still couples to HL(a,b) and HL(b,c)
    via h_bc and h_ab respectively (single-SO differences in the
    respective det pair).

  * Kekule-Kekule direct coupling is zero at s=0 (as in the anion), but
    the role of Coulomb and direct hopping matters differently in 3c2e.

  * At large U the ionic dets are suppressed as in the anion, and the
    pure-HL 3x3 block has the same spectrum {-sqrt(2), 0, +sqrt(2)} as
    the anion covalent block.  Asymptotic weights:
        w(HL_ab) = w(HL_bc) = 1/4,    w(HL_ac) = 1/2
    identical to the anion.

  * The Huckel baseline (U = J = K = M = 0) is ALSO the same for the
    long-bond:  w(HL_ac) = 1/8,  w(HL_ab) = w(HL_bc) = 1/4,  ionic = 3/8
    (ionic partition differs from anion: here (1/16, 1/4, 1/16) vs
    anion's (1/8, 1/8, 1/8)).
"""
import os
import sys
import time

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import FixedPsi, Molecule, hamiltonian, structure_vector
from symvb.fixed_psi import generate_dets
from symvb.spin import s_squared_matrix, project_onto_S


# ------------------------------------------------------------------------
# 1. Molecule + 9-det basis for 3c2e (Nalpha=Nbeta=1)
# ------------------------------------------------------------------------
m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab', 'bc'],
    subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
    subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
              'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)
P = generate_dets(1, 1, 3)
det_strings = [p.dets[0].det_string for p in P]
print(f"3c2e 9-det basis: {det_strings}")

t0 = time.time()
H_det, S_det = hamiltonian(m, P)
H_det = sp.Matrix(H_det)
S_det = sp.Matrix(S_det)
h, s, U, J, K, M = sp.symbols('h s U J K M')
H_det_s0 = H_det.subs({s: 0, h: -1})
S_det_s0 = S_det.subs({s: 0})
print(f"9-det H build: {time.time()-t0:.1f}s")


# ------------------------------------------------------------------------
# 2. Build 6 singlet VB structures as 9-dim coefficient vectors
# ------------------------------------------------------------------------
# At orthogonal AOs all 9 dets are orthonormal; the 6 singlet structures
# below are mutually orthogonal (disjoint det supports, normalisation 1/sqrt(2)
# or 1 for ionic).  structure_vector expands each into the 9-det basis with the
# correct fermion sign: the ionic p^2 structures are single determinants
# (p_alpha p_beta), the HL singlets couple an alpha/beta pair (un-normalised,
# norm^2 = 2).
structures = [FixedPsi('aA'), FixedPsi('bB'), FixedPsi('cC'),
              FixedPsi('aB', coupled_pairs=[(0, 1)]),
              FixedPsi('bC', coupled_pairs=[(0, 1)]),
              FixedPsi('aC', coupled_pairs=[(0, 1)])]
V = sp.Matrix.hstack(*[structure_vector(st, det_strings) for st in structures])
labels = ['a^2', 'b^2', 'c^2', 'HL(a,b)', 'HL(b,c)', 'HL(a,c)']


# ------------------------------------------------------------------------
# 3. Build 6x6 H_VB = V^T H_det V
# ------------------------------------------------------------------------
assert S_det_s0 == sp.eye(9), "orthonormal at s=0"
H_vb_s0 = sp.simplify(V.T * H_det_s0 * V)
S_vb_s0 = sp.simplify(V.T * S_det_s0 * V)
# Norms: ionic = 1, HL = sqrt(2)
norms_sq = [S_vb_s0[i, i] for i in range(6)]
norms = [sp.sqrt(n) for n in norms_sq]
print(f"\nStructure norms^2: {norms_sq}")

# Normalise to orthonormal basis
D_inv = sp.diag(*[1 / n for n in norms])
H_vb_orth = sp.simplify(D_inv * H_vb_s0 * D_inv)
print("\nOrthonormal H_VB (s=0, h=-1), basis = [a^2, b^2, c^2, HL(ab), HL(bc), HL(ac)]:")
sp.pprint(H_vb_orth)


# ------------------------------------------------------------------------
# 4. Numerical FCI + VB analysis with singlet filter
# ------------------------------------------------------------------------
H_det_fn = sp.lambdify((U, J, K, M), H_det_s0, 'numpy')
V_np = np.array(V, dtype=float)
V_hat = V_np / np.array([float(n) for n in norms])[None, :]   # 9 x 6, orthonormal columns

S2_9 = s_squared_matrix(det_strings, orbs='abc')

def analyse(Uv, Jv, Kv, Mv):
    Hn = np.array(H_det_fn(Uv, Jv, Kv, Mv), dtype=float)
    Hn = 0.5 * (Hn + Hn.T)
    evals, evecs = np.linalg.eigh(Hn)
    E_fci = evals[0]
    psi_fci = evecs[:, 0]
    s2_fci = float(psi_fci @ S2_9 @ psi_fci)
    # Project to singlet, lowest state
    H_sing, U_sing = project_onto_S(Hn, S2_9, target_S=0)
    es, vs = np.linalg.eigh(0.5 * (H_sing + H_sing.T))
    E_sing = es[0]
    psi_sing = U_sing @ vs[:, 0]
    w = (V_hat.T @ psi_sing) ** 2
    return dict(
        E_fci=E_fci, s2_fci=s2_fci,
        E_sing=E_sing, psi_sing=psi_sing,
        w_a2=float(w[0]), w_b2=float(w[1]), w_c2=float(w[2]),
        w_ab=float(w[3]), w_bc=float(w[4]), w_ac=float(w[5]),
        w_ion=float(w[0] + w[1] + w[2]),
        w_cov=float(w[3] + w[4] + w[5]),
    )


# ------------------------------------------------------------------------
# 5. Huckel baseline check (U=J=K=M=0)
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Huckel baseline (3c2e cation, U=J=K=M=0)")
print("=" * 70)
o = analyse(0, 0, 0, 0)
print(f"  E_FCI = {o['E_fci']:+.6f}   (exact psi_1^2 = -2*sqrt(2) = {-2*np.sqrt(2):.6f})")
print(f"  S^2   = {o['s2_fci']:.4f}")
print(f"  Ionic weights:     a^2={o['w_a2']:.4f}  b^2={o['w_b2']:.4f}  c^2={o['w_c2']:.4f}")
print(f"  Covalent weights:  HL(a,b)={o['w_ab']:.4f}  HL(b,c)={o['w_bc']:.4f}  "
      f"HL(a,c)={o['w_ac']:.4f}")
print(f"  Totals:  ionic={o['w_ion']:.4f}   covalent={o['w_cov']:.4f}")
print(f"  Expected:  a^2=c^2=1/16=0.0625, b^2=1/4, HL(ab)=HL(bc)=1/4, HL(ac)=1/8")


# ------------------------------------------------------------------------
# 6. Strong-U asymptote (HL-only block)
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Strong-U asymptote: HL-only 3x3 block diagonalisation")
print("=" * 70)
H_hl = H_vb_orth[3:6, 3:6]
print("  HL block (orthonormal basis):")
sp.pprint(H_hl)
# Eigenvalues at J=K=0 for clean sorting
H_hl0 = H_hl.subs({J: 0, K: 0, M: 0})
eigs0 = sorted(sp.Matrix(H_hl0).eigenvals().keys(), key=lambda r: float(r))
print(f"  eigenvalues (J=K=M=0): {[sp.nsimplify(e) for e in eigs0]}")
# Ground-state eigenvector at J=K=M=0
M_hl = sp.Matrix(H_hl0)
v = sp.symbols('c_ab c_bc c_ac')
eqs = [sum(M_hl[i, j] * v[j] for j in range(3)) - eigs0[0] * v[i] for i in range(3)]
sol = sp.solve(eqs + [v[0]**2 + v[1]**2 + v[2]**2 - 1], list(v), dict=True)
print(f"  strong-U GS eigenvector candidates: {sol}")


# ------------------------------------------------------------------------
# 7. Sweep U vs weights; compare to anion
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Pure-Hubbard sweep (3c2e cation)")
print("=" * 70)
print(f"  {'U':>8s}  {'E_FCI':>10s}  {'S^2':>5s}  "
      f"{'w(a^2)':>7s} {'w(b^2)':>7s} {'w(c^2)':>7s}  "
      f"{'w_ab':>7s} {'w_bc':>7s} {'w_ac':>7s}  {'w_cov':>7s}")
for Uv in [0, 0.5, 1, 2, 4, 8, 16, 64, 256, 4096]:
    o = analyse(Uv, 0, 0, 0)
    print(f"  {Uv:>8g}  {o['E_fci']:>+10.4f}  {o['s2_fci']:>5.2f}  "
          f"{o['w_a2']:>7.4f} {o['w_b2']:>7.4f} {o['w_c2']:>7.4f}  "
          f"{o['w_ab']:>7.4f} {o['w_bc']:>7.4f} {o['w_ac']:>7.4f}  {o['w_cov']:>7.4f}")


# ------------------------------------------------------------------------
# 8. Side-by-side comparison with anion (pure-Hubbard, Huckel + strong-U)
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Cation (3c2e) vs Anion (3c4e) VB weights -- pure Hubbard")
print("=" * 70)
print(f"  {'system':>12s}  {'U':>6s}  {'w_ab':>6s} {'w_bc':>6s} {'w_LB':>6s}  "
      f"{'w_ion':>6s}")
print(f"  (Huckel and strong-U limits)")
# Cation
oc0 = analyse(0, 0, 0, 0)
oc_inf = analyse(16384, 0, 0, 0)
print(f"  {'cation':>12s}  {'0':>6s}  {oc0['w_ab']:>6.4f} {oc0['w_bc']:>6.4f} "
      f"{oc0['w_ac']:>6.4f}  {oc0['w_ion']:>6.4f}")
print(f"  {'cation':>12s}  {'inf':>6s}  {oc_inf['w_ab']:>6.4f} {oc_inf['w_bc']:>6.4f} "
      f"{oc_inf['w_ac']:>6.4f}  {oc_inf['w_ion']:>6.4f}")
print(f"  {'anion':>12s}  {'0':>6s}  {0.25:>6.4f} {0.25:>6.4f} {0.125:>6.4f}  {0.375:>6.4f}")
print(f"  {'anion':>12s}  {'inf':>6s}  {0.25:>6.4f} {0.25:>6.4f} {0.5:>6.4f}  {0:>6.4f}")


# ------------------------------------------------------------------------
# 9. Figure: cation long-bond alongside anion
# ------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

U_grid = np.logspace(-2, 4, 200)
recs_c = [analyse(Uv, 0, 0, 0) for Uv in U_grid]
arr = lambda key: np.array([r[key] for r in recs_c])

# Also rebuild the anion (3c4e) for the side-by-side panel: its 9-det H and the
# three covalent Rumer structures (Kekule a-b, Kekule b-c, long bond a-c).
m2 = Molecule(zero_ii=True, interacting_orbs=['ab', 'bc'],
              subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
              subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
                        'M': ('1112', '1121', '1222')},
              max_2e_centers=2)
P2 = generate_dets(2, 2, 3)
ds2 = [p.dets[0].det_string for p in P2]
Hd2, _ = hamiltonian(m2, P2)
Hd2 = sp.Matrix(Hd2).subs({s: 0, h: -1})

V2 = sp.Matrix.hstack(*[structure_vector(fp, ds2) for fp in (
    FixedPsi('aBcC', coupled_pairs=[(0, 1)]),
    FixedPsi('aAbC', coupled_pairs=[(2, 3)]),
    FixedPsi('abBC', coupled_pairs=[(0, 3)]))])
norms2 = [sp.sqrt(V2[:, i].dot(V2[:, i])) for i in range(3)]
V2_np = np.array(V2, dtype=float) / np.array([float(n) for n in norms2])[None, :]
Hd2_fn = sp.lambdify((U, J, K, M), Hd2, 'numpy')

def analyse_anion(Uv, Jv, Kv, Mv):
    Hn = np.array(Hd2_fn(Uv, Jv, Kv, Mv), dtype=float)
    Hn = 0.5 * (Hn + Hn.T)
    ev, evc = np.linalg.eigh(Hn)
    psi = evc[:, 0]
    w = (V2_np.T @ psi) ** 2
    return dict(w_ab=float(w[0]), w_bc=float(w[1]), w_ac=float(w[2]),
                w_cov=float(w.sum()), E_fci=ev[0])

recs_a = [analyse_anion(Uv, 0, 0, 0) for Uv in U_grid]
arr_a = lambda key: np.array([r[key] for r in recs_a])


fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel (a): cation weights
ax[0].plot(U_grid, arr('w_ab'),  'C0-',  lw=1.8, label=r'$w_{ab}$ (Kekulé)')
ax[0].plot(U_grid, arr('w_bc'),  'C0--', lw=1.4, label=r'$w_{bc}$ (Kekulé)')
ax[0].plot(U_grid, arr('w_ac'),  'C3-',  lw=2.2, label=r'$w_{ac}$ (long-bond)')
ax[0].plot(U_grid, arr('w_ion'), 'C7:',  lw=1.6, label='ionic (total)')
ax[0].plot(U_grid, arr('w_b2'),  'C2-.', lw=1.0, alpha=0.7, label=r'$w_{b^2}$')
for y in (0.125, 0.25, 0.5):
    ax[0].axhline(y, color='k', lw=0.3, alpha=0.3)
ax[0].set_xscale('log'); ax[0].set_xlabel(r'$U / |h|$')
ax[0].set_ylabel('Chirgwin–Coulson weight (singlet GS)')
ax[0].set_title(r'(a)  3c2e cation: VB weights vs $U$')
ax[0].legend(fontsize=9, loc='center left')
ax[0].set_ylim(0, 0.6); ax[0].grid(alpha=0.3)

# Panel (b): long-bond weight for cation vs anion
ax[1].plot(U_grid, arr('w_ac'),   'C3-',  lw=2.0, label=r'cation $w_{ac}$ (3c2e)')
ax[1].plot(U_grid, arr_a('w_ac'), 'C1--', lw=2.0, label=r'anion $w_{ac}$ (3c4e)')
ax[1].plot(U_grid, arr('w_ab'),   'C0-',  lw=1.4, alpha=0.7, label=r'cation $w_{ab}$')
ax[1].plot(U_grid, arr_a('w_ab'), 'C0:',  lw=1.4, alpha=0.7, label=r'anion $w_{ab}$')
for y in (0.125, 0.25, 0.5):
    ax[1].axhline(y, color='k', lw=0.3, alpha=0.3)
ax[1].set_xscale('log'); ax[1].set_xlabel(r'$U / |h|$')
ax[1].set_ylabel('VB weight')
ax[1].set_title('(b)  Long-bond: cation and anion trace identical limits')
ax[1].legend(fontsize=9, loc='center left')
ax[1].set_ylim(0, 0.6); ax[1].grid(alpha=0.3)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', 'figures', 'fig_allyl_cation_long_bond.png')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
plt.savefig(outpath, dpi=140)
plt.close()
print(f"\nFigure saved: {outpath}")
