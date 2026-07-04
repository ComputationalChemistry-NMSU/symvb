"""
Allyl anion 3c4e: long-bond / biradical VB-structure analysis.

Three covalent VB structures span the spin-paired singlet sector of the
3c4e chain (orthogonal AOs, s = 0):

    Phi_ab = [a . b]_s * c^2     Kekule (a-b bond, c lone pair)
    Phi_bc = [b . c]_s * a^2     Kekule (b-c bond, a lone pair)
    Phi_ac = [a . c]_s * b^2     long-bond (a-c HL pair, b lone pair)

where [p . q]_s = (p_alpha q_beta - p_beta q_alpha)/sqrt(2).  Phi_ac is
the VB signature of biradical character: an a-c bond can only pair two
electrons that are separated in space by a full bond length, so its
weight in the ground state measures a/c spin-pair delocalisation.

At s = 0 the three structures are mutually orthogonal (they populate
disjoint det pairs in the 9-det Sz=0 basis), so the covalent-only H in
VB basis is a symmetric 3 x 3 matrix with no overlap correction.

Diagnostics:

  (1) Variational gain from adding Phi_ac
      E_2 = lowest eigenvalue of {Phi_ab, Phi_bc}-only 2 x 2 block
      E_3 = lowest eigenvalue of full 3 x 3 covalent block
      Delta_lb = E_3 - E_2   <= 0  always

  (2) Chirgwin-Coulson weight of Phi_ac in the FCI ground state
      (at s=0, structures orthogonal, so CC = Lowdin).  Total covalent
      weight w_cov = sum_i <Phi_i|Psi>^2 complements the ionic weight
      in the 2+2+0 closed-shell dets (|aAbB>, |aAcC>, |bBcC>).

  (3) Second-order self-energy estimate
      Using Psi_ref = ground state of 2 x 2 {Phi_ab, Phi_bc} block,
      Sigma_ac = <Psi_ref | H | Phi_ac>^2 / (E_ref - H_ac_ac).

Sweeps:
  - Pure-Hubbard axis (U only), to see biradical emergence with U.
  - Representative PPP points (U, J_adj, K_adj) to land in physical regime.

Companion context: examples/allyl_biradical.py established the
natural-orbital (7/4, 3/2, 3/4) strong-U asymptote.  That diagnostic
lives in the MO basis; the present script translates the same physics
into VB / Rumer language.
"""
import os
import sys
import time

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import FixedPsi, Molecule, hamiltonian, operators as op, structure_vector
from symvb.fixed_psi import generate_dets
from symvb.spin import s_squared_matrix, project_onto_S


# ------------------------------------------------------------------------
# 1. Molecule + det basis + symbolic H  (pattern-unified U, J, K, M)
# ------------------------------------------------------------------------
m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab', 'bc'],
    subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
    subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
              'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)
P = generate_dets(2, 2, 3)
det_strings = [p.dets[0].det_string for p in P]
print(f"9-det basis: {det_strings}")

t0 = time.time()
H_det, S_det = hamiltonian(m, P)
print(f"9-det symbolic H build: {time.time()-t0:.1f}s")

H_det = sp.Matrix(H_det)
S_det = sp.Matrix(S_det)
h, s, U, J, K, M = sp.symbols('h s U J K M')


# ------------------------------------------------------------------------
# 2. Build the three covalent VB structures as FixedPsi
# ------------------------------------------------------------------------
#   Parent dets are chosen so that couple_orbitals(o1, o2) pairs the
#   desired alpha/beta pair into a singlet HL bond.  Creation-order
#   signs are handled by symvb when build_matrix canonicalises.
#
#   Phi_ab: parent 'aBcC' = a_alpha b_beta c_alpha c_beta;
#            couple positions (0, 1)  -> HL(a,b) x c^2
#   Phi_bc: parent 'aAbC' = a_alpha a_beta b_alpha c_beta;
#            couple positions (2, 3)  -> HL(b,c) x a^2
#   Phi_ac: parent 'abBC' = a_alpha b_alpha b_beta c_beta;
#            couple positions (0, 3)  -> HL(a,c) x b^2
Phi_ab = FixedPsi('aBcC', coupled_pairs=[(0, 1)])
Phi_bc = FixedPsi('aAbC', coupled_pairs=[(2, 3)])
Phi_ac = FixedPsi('abBC', coupled_pairs=[(0, 3)])

print("\nVB structures (raw, before canonicalisation):")
print(f"  Phi_ab = {Phi_ab}")
print(f"  Phi_bc = {Phi_bc}")
print(f"  Phi_ac = {Phi_ac}")

# Canonicalise to get coefficients in the 9-det standard basis
Phi_ab_c = FixedPsi(Phi_ab); Phi_ab_c.canonicalize()
Phi_bc_c = FixedPsi(Phi_bc); Phi_bc_c.canonicalize()
Phi_ac_c = FixedPsi(Phi_ac); Phi_ac_c.canonicalize()
print("\nVB structures (canonicalised, 9-det expansion):")
print(f"  Phi_ab = {Phi_ab_c}")
print(f"  Phi_bc = {Phi_bc_c}")
print(f"  Phi_ac = {Phi_ac_c}")

# Expand each VB structure over the 9-det basis. structure_vector canonicalises
# the raw structure and tracks the fermion sign of the reorder into standard
# (interleaved) determinant order, so the uuLL pattern from couple_pairs(0, 3)
# on the 'abBC' parent lands with the correct sign.
V = sp.Matrix.hstack(*[structure_vector(fp, det_strings)
                       for fp in (Phi_ab, Phi_bc, Phi_ac)])   # 9 x 3


# ------------------------------------------------------------------------
# 3. 3x3 H and S in VB basis, formed as V^T H_det V: a projection of the
#    9-det Hamiltonian onto the three (un-normalised, norm^2 = 2) structures.
# ------------------------------------------------------------------------
H_det_s0 = H_det.subs({s: 0, h: -1})
S_det_s0 = S_det.subs({s: 0})
assert S_det_s0 == sp.eye(9), "det basis should be orthonormal at s=0"

t0 = time.time()
H_vb_s0 = sp.simplify(V.T * H_det_s0 * V)
S_vb_s0 = sp.simplify(V.T * S_det_s0 * V)
print(f"\n3x3 VB matrix via V^T H V: {time.time()-t0:.1f}s")
assert sp.simplify(S_vb_s0 - 2 * sp.eye(3)) == sp.zeros(3, 3), \
    f"expected S_vb = 2 I but got {S_vb_s0}"

print("\nH_vb at s=0, h=-1 (unnormalised VB basis, norm^2 = 2):")
sp.pprint(H_vb_s0)


# ------------------------------------------------------------------------
# 4. Numerical solver: FCI, 3-VB, 2-VB, weights
# ------------------------------------------------------------------------
H_det_fn = sp.lambdify((U, J, K, M), H_det_s0, 'numpy')
H_vb_fn  = sp.lambdify((U, J, K, M), H_vb_s0,  'numpy')
V_np = np.array(V, dtype=float)   # 9 x 3, columns = un-normalised Phi_i
norms = np.sqrt(np.diag(V_np.T @ V_np))  # sqrt(2) each
V_hat = V_np / norms[None, :]     # 9 x 3, columns = orthonormal Phi_i

# S^2 matrix on 9-det basis (for singlet projection)
S2_9 = s_squared_matrix(det_strings, orbs='abc')


def analyse(Uv, Jv, Kv, Mv):
    Hn = np.array(H_det_fn(Uv, Jv, Kv, Mv), dtype=float)
    Hn = 0.5 * (Hn + Hn.T)
    evals, evecs = np.linalg.eigh(Hn)
    E_fci = evals[0]
    psi = evecs[:, 0]
    s2_fci = float(psi @ S2_9 @ psi)
    # Lowest singlet state (may differ from FCI GS past the triplet crossing)
    H_sing, U_sing = project_onto_S(Hn, S2_9, target_S=0)
    evs_s, evc_s = np.linalg.eigh(0.5 * (H_sing + H_sing.T))
    E_sing = evs_s[0]
    psi_sing = U_sing @ evc_s[:, 0]
    # projections onto orthonormal VB structures (use singlet state for weights)
    c_fci  = V_hat.T @ psi          # length 3
    c_sing = V_hat.T @ psi_sing
    w_fci  = c_fci  ** 2
    w_sing = c_sing ** 2
    # 3-VB CI (in orthonormal VB basis)
    Hvb = np.array(H_vb_fn(Uv, Jv, Kv, Mv), dtype=float) / 2.0  # divide norm^2
    Hvb = 0.5 * (Hvb + Hvb.T)
    ev3, vc3 = np.linalg.eigh(Hvb)
    E_3 = ev3[0]; psi3 = vc3[:, 0]
    # 2-VB CI (drop Phi_ac = last row/col)
    Hvb2 = Hvb[:2, :2]
    ev2, vc2 = np.linalg.eigh(Hvb2)
    E_2 = ev2[0]; psi2 = vc2[:, 0]
    # self-energy of Phi_ac relative to Psi_ref (lift psi2 into 3D)
    psi_ref3 = np.array([psi2[0], psi2[1], 0.0])
    V_coupling = psi_ref3 @ Hvb @ np.array([0, 0, 1.0])
    E_phi_ac = Hvb[2, 2]
    denom = ev2[0] - E_phi_ac
    sigma_ac = (V_coupling ** 2) / denom if abs(denom) > 1e-12 else np.nan
    return dict(
        E_fci=E_fci, psi_fci=psi, s2_fci=s2_fci,
        E_sing=E_sing, psi_sing=psi_sing,
        w_ab=float(w_fci[0]),  w_bc=float(w_fci[1]),  w_ac=float(w_fci[2]),
        w_cov=float(w_fci.sum()),
        w_ab_s=float(w_sing[0]), w_bc_s=float(w_sing[1]), w_ac_s=float(w_sing[2]),
        w_cov_s=float(w_sing.sum()),
        E_3cov=E_3, psi_3cov=psi3,
        E_2cov=E_2, psi_2cov=psi2,
        Delta_lb=E_3 - E_2,
        V_ref_ac=V_coupling, H_ac_ac=E_phi_ac,
        sigma_ac=sigma_ac,
    )


# ------------------------------------------------------------------------
# 5. Baseline check at U=J=K=M=0 (Huckel)
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Huckel baseline (U=J=K=M=0)")
print("=" * 70)
o = analyse(0, 0, 0, 0)
print(f"  E_FCI   = {o['E_fci']:+.6f}     (exact -2*sqrt(2) = {-2*np.sqrt(2):+.6f})")
print(f"  E_3cov  = {o['E_3cov']:+.6f}")
print(f"  E_2cov  = {o['E_2cov']:+.6f}")
print(f"  Delta_lb (variational gain from Phi_ac) = {o['Delta_lb']:+.6f}")
print(f"  VB weights in FCI:  Phi_ab = {o['w_ab']:.4f}   "
      f"Phi_bc = {o['w_bc']:.4f}   Phi_ac = {o['w_ac']:.4f}")
print(f"  Total covalent weight = {o['w_cov']:.4f}   "
      f"(ionic: {1 - o['w_cov']:.4f})")
print(f"  Self-energy Sigma_ac = {o['sigma_ac']:+.6f}   "
      f"(coupling V = {o['V_ref_ac']:+.4f}, H_ac_ac = {o['H_ac_ac']:+.4f})")


# ------------------------------------------------------------------------
# 6. Pure-Hubbard sweep (J = K = M = 0)
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Pure-Hubbard axis: Phi_ac stabilisation and weight vs U")
print("=" * 70)
print(f"  {'U':>8s}  {'E_FCI':>10s}  {'S^2':>5s}  {'E_2cov':>10s}  "
      f"{'E_3cov':>10s}  {'Delta_lb':>10s}  {'w_ab':>7s}  {'w_bc':>7s}  "
      f"{'w_ac':>7s}  {'w_cov':>7s}")
for Uv in [0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 128, 1024]:
    o = analyse(Uv, 0, 0, 0)
    print(f"  {Uv:>8g}  {o['E_fci']:>+10.4f}  {o['s2_fci']:>5.2f}  "
          f"{o['E_2cov']:>+10.4f}  {o['E_3cov']:>+10.4f}  "
          f"{o['Delta_lb']:>+10.4f}  "
          f"{o['w_ab']:>7.4f}  {o['w_bc']:>7.4f}  {o['w_ac']:>7.4f}  "
          f"{o['w_cov']:>7.4f}")


# ------------------------------------------------------------------------
# 7. Physical PPP-ish points
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Physical PPP points (pattern-unified J, K)")
print("=" * 70)
points = [
    ("Huckel",              0,  0,   0, 0),
    ("weak U",              2,  0,   0, 0),
    ("PPP-like (U=10,K=3)", 10, 0.5, 3, 0),
    ("PPP-like (U=10,K=7)", 10, 0.5, 7, 0),
    ("strong-U",            64, 0,   0, 0),
]
print(f"  {'label':>22s}  {'E_FCI':>8s}  {'S^2':>4s}  {'E_sing':>8s}  "
      f"{'E_3cov':>8s}  {'Delta_lb':>9s}  {'w_ac(s)':>7s}  {'w_cov(s)':>8s}")
print("  (singlet-projected weights w_ab(s), w_ac(s) shown; FCI GS may be triplet)")
for label, Uv, Jv, Kv, Mv in points:
    o = analyse(Uv, Jv, Kv, Mv)
    print(f"  {label:>22s}  {o['E_fci']:>+8.3f}  {o['s2_fci']:>4.2f}  "
          f"{o['E_sing']:>+8.3f}  {o['E_3cov']:>+8.3f}  {o['Delta_lb']:>+9.4f}  "
          f"{o['w_ac_s']:>7.4f}  {o['w_cov_s']:>8.4f}")


# ------------------------------------------------------------------------
# 8. Closed forms at Huckel limit (U=J=K=M=0)
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Closed forms at U=J=K=M=0")
print("=" * 70)
H_vb_0 = H_vb_s0.subs({U: 0, J: 0, K: 0, M: 0})
# Convert to unit-norm VB basis by dividing by 2 (since <Phi_i|Phi_i> = 2)
H_vb_0_norm = sp.Rational(1, 2) * H_vb_0
print("  H_vb_orthonorm =")
sp.pprint(H_vb_0_norm)
ev3_sym = sorted(sp.Matrix(H_vb_0_norm).eigenvals().keys(), key=lambda r: float(r))
ev2_sym = sorted(sp.Matrix(H_vb_0_norm[:2, :2]).eigenvals().keys(), key=lambda r: float(r))
print(f"\n  E_3cov eigenvalues : {[sp.nsimplify(e) for e in ev3_sym]}")
print(f"  E_2cov eigenvalues : {[sp.nsimplify(e) for e in ev2_sym]}")
Delta_sym = sp.simplify(sp.nsimplify(ev3_sym[0] - ev2_sym[0]))
print(f"  Delta_lb (closed form) = {Delta_sym} = {float(Delta_sym):+.6f}")


# ------------------------------------------------------------------------
# 9. Strong-U asymptote and the (7/4, 3/2, 3/4) signature
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Strong-U asymptote: VB composition of the biradical ground state")
print("=" * 70)
for Uv in [16, 64, 256, 1024, 4096]:
    o = analyse(Uv, 0, 0, 0)
    psi3 = o['psi_3cov']
    if psi3.sum() < 0:
        psi3 = -psi3
    print(f"  U={Uv:>6g}: Phi_3cov = "
          f"{psi3[0]:>+.4f} Phi_ab + {psi3[1]:>+.4f} Phi_bc + {psi3[2]:>+.4f} Phi_ac   "
          f"(|ac|/|ab| = {abs(psi3[2])/abs(psi3[0]):.4f})")

# Closed-form strong-U eigenstate (covalent 3 x 3, diagonal = U shift)
print("\n  Symbolic eigen-analysis of the covalent 3x3 H_cov:")
ev3_U = sorted(sp.Matrix(H_vb_s0 / 2).subs({J: 0, K: 0, M: 0}).eigenvals().keys(),
               key=lambda r: float(sp.simplify(r - U).subs(U, 0)))
print(f"    eigenvalues  =  {[sp.nsimplify(e) for e in ev3_U]}")
print("    (ground state = U - sqrt(2) ; excited = U ; U + sqrt(2))")
print("    -> strong-U covalent Phi_GS = (Phi_ab + Phi_bc - sqrt(2) Phi_ac)/2")
print("    -> w_ab = w_bc = 1/4, w_ac = 1/2 (exact, U-independent in covalent sector)")

# Connection to NO (natural orbital) occupations
print("\n  Natural-orbital occupations of the strong-U GS:")
# Build 1-RDM in AO basis at the strong-U point and diagonalise
o_big = analyse(16384, 0, 0, 0)
psi = o_big['psi_fci']
# AO 1-RDM (spin-summed) via second-quantized c†_p c_q from symvb.operators.
rho_ao_9 = np.zeros((3, 3, 9, 9))
for ip, p in enumerate('abc'):
    for iq, q in enumerate('abc'):
        c_pq = (op.cdag(p, 'alpha') @ op.c(q, 'alpha')
                + op.cdag(p, 'beta') @ op.c(q, 'beta'))
        rho_ao_9[ip, iq] = np.array(c_pq.matrix(det_strings), dtype=float)
gamma_ao = np.array([[psi @ rho_ao_9[i, j] @ psi for j in range(3)] for i in range(3)])
gamma_ao = 0.5 * (gamma_ao + gamma_ao.T)
nat_occs = np.sort(np.linalg.eigvalsh(gamma_ao))[::-1]
r2 = np.sqrt(2.0)
C_mo = np.array([[0.5, 1/r2, 0.5], [1/r2, 0.0, -1/r2], [0.5, -1/r2, 0.5]]).T
huck_occ = np.diag(C_mo.T @ gamma_ao @ C_mo)
print(f"    NO occupations (sorted, U=16384):  {nat_occs.round(4)}")
print(f"    Huckel-MO occupations:              n(psi_1)={huck_occ[0]:.4f}, "
      f"n(psi_2)={huck_occ[1]:.4f}, n(psi_3)={huck_occ[2]:.4f}")
print(f"    Expected strong-U: (7/4, 3/2, 3/4) = (1.7500, 1.5000, 0.7500)")
print(f"    VB long-bond weight w_ac = {o_big['w_ac']:.4f}   (exact 1/2)")


# ------------------------------------------------------------------------
# 10. Figure: VB weights, energies, and NO-connection vs U
# ------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# fonttype 42 keeps PDF text as editable TrueType text, not Type-3 outlines
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

U_grid = np.logspace(-2, 4, 200)
records = [analyse(Uv, 0, 0, 0) for Uv in U_grid]
arr = lambda key: np.array([r[key] for r in records])

def no_psi3(psi):
    gamma = np.array([[psi @ rho_ao_9[i, j] @ psi for j in range(3)]
                      for i in range(3)])
    gamma = 0.5 * (gamma + gamma.T)
    return float(np.diag(C_mo.T @ gamma @ C_mo)[2])

bir_no = 0.5 * np.array([no_psi3(r['psi_fci']) for r in records])

plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw={'wspace': 0.32})

# Panel (a): VB weights
ax[0].plot(U_grid, arr('w_ab'), 'C0-',  lw=1.8, label=r'$w_{ab}$ (Kekulé)')
ax[0].plot(U_grid, arr('w_bc'), 'C0--', lw=1.4, label=r'$w_{bc}$ (Kekulé)')
ax[0].plot(U_grid, arr('w_ac'), 'C3-',  lw=2.2, label=r'$w_{ac}$ (long-bond)')
ax[0].plot(U_grid, 1 - arr('w_cov'), 'C7:', lw=1.6, label='ionic (remainder)')
for y in (0.125, 0.25, 0.375, 0.5):
    ax[0].axhline(y, color='k', lw=0.3, alpha=0.3)
ax[0].set_xscale('log')
ax[0].set_xlabel(r'$U / |h|$')
ax[0].set_ylabel('Chirgwin–Coulson weight')
ax[0].set_title('(A)  VB weights vs $U$')
ax[0].legend(fontsize=6.2, loc='upper right', labelspacing=0.25,
             handlelength=1.5, borderpad=0.3, framealpha=0.9)
ax[0].set_ylim(0, 0.86); ax[0].grid(alpha=0.3)
ax[0].text(2e-2, 0.127, '1/8', fontsize=8, color='gray', va='bottom')
ax[0].text(2e-2, 0.252, '1/4', fontsize=8, color='gray', va='bottom')
ax[0].text(2e-2, 0.377, '3/8', fontsize=8, color='gray', va='bottom')
ax[0].text(2e-2, 0.502, '1/2', fontsize=8, color='gray', va='bottom')

# Panel (b): Energies relative to U (removes the trivial shift)
ax[1].plot(U_grid, arr('E_fci')  - U_grid, 'k-',  lw=2.0, label=r'$E_{\rm FCI} - U$')
ax[1].plot(U_grid, arr('E_3cov') - U_grid, 'C3-', lw=1.8,
           label=r'$E_{\rm 3-cov} - U$  (with $\Phi_{ac}$)')
ax[1].plot(U_grid, arr('E_2cov') - U_grid, 'C0--', lw=1.6,
           label=r'$E_{\rm 2-cov} - U$  (no $\Phi_{ac}$)')
ax[1].axhline(-np.sqrt(2), color='C3', lw=0.6, alpha=0.6)
ax[1].axhline(-2 * np.sqrt(2), color='k', lw=0.6, alpha=0.6)
ax[1].set_xscale('log')
ax[1].set_xlabel(r'$U / |h|$')
ax[1].set_ylabel(r'$E - U$  (units of $|h|$)')
ax[1].set_title(r'(B)  $\Delta_{\rm lb} = -\sqrt{2}\,|h|$')
ax[1].legend(fontsize=6.2, loc='center right',
             labelspacing=0.25, handlelength=1.5, borderpad=0.3,
             framealpha=0.9)
ax[1].grid(alpha=0.3)
ax[1].text(2e-2, -np.sqrt(2) + 0.16, r'$-\sqrt{2}$', color='C3', fontsize=8, va='bottom')
ax[1].text(2e-2, -2 * np.sqrt(2) + 0.16, r'$-2\sqrt{2}$', color='k',
           fontsize=8, va='bottom')

# Panel (c): biradical diagnostics from VB vs NO
ax[2].plot(U_grid, arr('w_ac'), 'C3-', lw=2.2, label=r'$w_{ac}$ (VB long-bond)')
ax[2].plot(U_grid, bir_no,     'C2--', lw=1.8, label=r'$n_3/2$ (biradical index)')
ax[2].plot(U_grid, arr('w_ac') - bir_no, 'C7:', lw=1.4,
           label=r'difference  $w_{ac} - n_3/2$')
ax[2].axhline(0.5, color='C3', lw=0.6, alpha=0.6)
ax[2].axhline(3/8, color='C2', lw=0.6, alpha=0.6)
ax[2].axhline(1/8, color='C7', lw=0.6, alpha=0.6)
ax[2].set_xscale('log')
ax[2].set_xlabel(r'$U / |h|$')
ax[2].set_ylabel('biradical diagnostic')
ax[2].set_title('(C)  VB vs NO: two biradical scales')
ax[2].set_ylim(0, 0.62)
ax[2].legend(fontsize=6.2, loc='upper left', bbox_to_anchor=(0.02, 0.575),
             labelspacing=0.25, handlelength=1.5, borderpad=0.3,
             framealpha=0.9)
ax[2].grid(alpha=0.3)
ax[2].text(2e3, 0.56, r'$1/2$', color='C3', fontsize=8, va='bottom')
ax[2].text(2e3, 0.43, r'$3/8$', color='C2', fontsize=8, va='bottom')
ax[2].text(2e3, 0.18, r'$1/8$', color='C7', fontsize=8, va='bottom')

plt.tight_layout()
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', '..', 'vbt-3', 'figures', 'fig5_allyl_long_bond.png')
plt.savefig(outpath, dpi=450, bbox_inches='tight')
plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {outpath}")
