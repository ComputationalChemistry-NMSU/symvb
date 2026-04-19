"""Verify whether |psi_Huckel> = |psi_1^2 psi_2^2> is literally an
eigenstate of the 9-dim CI Hamiltonian along U=J (K=M=0, s=0, h=-1).

Strategy:  at several (U=J) values, diagonalize H numerically and check:
(1) what the actual numerical GS vector is (compare against the
    expansion of |psi_1^2 psi_2^2>);
(2) whether the GS energy as a function of U=J is linear (which would be
    the signature of a single-config eigenstate with no correlation
    admixture).
"""
import os, sys
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from vbt3 import Molecule, SlaterDet, symmetry
from vbt3.fixed_psi import generate_dets

m = Molecule(
    zero_ii=True, interacting_orbs=['ab', 'bc'],
    subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
    subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
              'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)
P = generate_dets(2, 2, 3)
det_strings = [p.dets[0].det_string for p in P]
H1 = m.build_matrix(P, op='H')
H2 = m.o2_matrix(P)
H_full = sp.Matrix(H1 + H2)
h, s, U, J, K, M = sp.symbols('h s U J K M')
H_full = H_full.subs({s: 0, h: -1, K: 0, M: 0})
Hline_fn = sp.lambdify((U, J), H_full, 'numpy')

# Strategy: diagonalize H(U=J) numerically at several values and see if
# GS energy is exactly linear in U.  If yes -> Huckel eigenstate.
# If no -> there's correlation admixture.
print("=== Numerical diagonalization of full 9x9 H at U=J (K=M=0) ===")
print(f"  {'U':>5s}  {'E_GS':>11s}  {'E_1st':>11s}  {'gap':>8s}  "
      f"{'E_GS - (aU+b)':>14s}")

Us = np.array([0, 0.5, 1, 2, 4, 8, 16])
E_gs = []
for Uv in Us:
    Hn = np.array(Hline_fn(Uv, Uv), dtype=float)
    Hn = 0.5*(Hn+Hn.T)
    ev, _ = np.linalg.eigh(Hn)
    E_gs.append(ev[0])
    print(f"  {Uv:>5.1f}  {ev[0]:>11.6f}  {ev[1]:>11.6f}  {ev[1]-ev[0]:>8.4f}")

E_gs = np.array(E_gs)
# Fit E_GS = a*U + b
a, b = np.polyfit(Us, E_gs, 1)
print(f"\n  Linear fit:  E_GS(U=J, U) = {a:.6f} * U + {b:.6f}")
print(f"  Residuals (E_GS - a*U - b):")
for Uv, E in zip(Us, E_gs):
    resid = E - a*Uv - b
    print(f"    U={Uv:>5.1f}: {resid:+.2e}")

# Theoretical Huckel energy: -2*sqrt(2) + (11/8 + ?)*U at 1st order
# If E_GS is linear -> Huckel IS an eigenstate
# If not -> NOT an eigenstate

# Second test: project the GS vector onto the Huckel det expansion
# using the explicit Huckel vector computed the right way.

# First compute Huckel vector by building psi_Huckel directly from
# the MO eigenvectors of the 1-electron H (more robust than the symbolic expansion):
h1_1e = np.array([[0, -1, 0], [-1, 0, -1], [0, -1, 0]], dtype=float)  # h=-1, s=0
mo_E, C_mo = np.linalg.eigh(h1_1e)  # MOs of 1-e H
# psi_1 = lowest energy (bonding), psi_2 = middle (nonbonding)
idx = np.argsort(mo_E)
C_mo = C_mo[:, idx]

# Expand |psi_1 alpha> |psi_1 beta> |psi_2 alpha> |psi_2 beta> in the 9-det vbt3 basis.
# For each det D with alpha-occ = {p_1, p_2} (sorted), beta-occ = {q_1, q_2} (sorted):
# coefficient in canonical (orbital-interleaved) basis is:
#   det([C_1(p_1), C_1(p_2); C_2(p_1), C_2(p_2)]) * det([C_1(q_1), C_1(q_2); C_2(q_1), C_2(q_2)])
# times the sign of the permutation from spin-major to canonical order.

def canon_idx(ds):
    out = []
    for c in ds:
        orb = 'abc'.index(c.lower())
        spin = 0 if c.islower() else 1
        out.append(2*orb + spin)
    return out

def vbt3_sign(ds):
    idx = canon_idx(ds); inv = 0; n = len(idx)
    for i in range(n):
        for j in range(i+1, n):
            if idx[i] > idx[j]: inv += 1
    return (-1)**inv

def sm_to_canonical_sign(alpha_occ, beta_occ):
    """Sign to go from c†_{p_1 α} c†_{p_2 α} c†_{q_1 β} c†_{q_2 β}|0>
    (α sorted, β sorted) to the canonical orbital-interleaved ordering."""
    # slot indices in spin-major order:
    idx_sm = [2*'abc'.index(x) for x in alpha_occ] + \
             [2*'abc'.index(x)+1 for x in beta_occ]
    # inversion count
    inv = 0
    n = len(idx_sm)
    for i in range(n):
        for j in range(i+1, n):
            if idx_sm[i] > idx_sm[j]: inv += 1
    return (-1)**inv

psi_vbt3 = np.zeros(9)
for i, ds in enumerate(det_strings):
    alpha_occ = sorted([c for c in ds if c.islower()])
    beta_occ  = sorted([c.lower() for c in ds if c.isupper()])
    p1, p2 = alpha_occ
    q1, q2 = beta_occ
    p1i, p2i = 'abc'.index(p1), 'abc'.index(p2)
    q1i, q2i = 'abc'.index(q1), 'abc'.index(q2)
    # 2x2 MO coef matrix [mu=1, mu=2] for (p_1, p_2)
    alpha_det = C_mo[p1i, 0]*C_mo[p2i, 1] - C_mo[p2i, 0]*C_mo[p1i, 1]
    beta_det  = C_mo[q1i, 0]*C_mo[q2i, 1] - C_mo[q2i, 0]*C_mo[q1i, 1]
    sign_sm = sm_to_canonical_sign(alpha_occ, beta_occ)
    coef_canon = sign_sm * alpha_det * beta_det
    # to vbt3 basis
    psi_vbt3[i] = vbt3_sign(ds) * coef_canon

print(f"\n  |psi_Huckel| = {np.linalg.norm(psi_vbt3):.6f} (should be 1)")

# Check at U=0
Hn = np.array(Hline_fn(0, 0), dtype=float); Hn = 0.5*(Hn+Hn.T)
E_expect = psi_vbt3 @ Hn @ psi_vbt3
print(f"  <psi_Huckel|H_0|psi_Huckel> = {E_expect:.6f}  "
      f"(should be -2*sqrt(2) = {-2*np.sqrt(2):.6f})")

# Now overlap with GS at various U=J
print(f"\n  Overlap of numerical GS with |psi_Huckel> along U=J axis:")
print(f"  {'U':>5s}  {'<GS|psi_H>':>12s}  {'|<GS|psi_H>|^2':>14s}")
for Uv in [0, 0.5, 1, 2, 4, 8, 16]:
    Hn = np.array(Hline_fn(Uv, Uv), dtype=float); Hn = 0.5*(Hn+Hn.T)
    ev, vec = np.linalg.eigh(Hn)
    c_gs = vec[:, 0]
    ovlp = psi_vbt3 @ c_gs
    print(f"  {Uv:>5.1f}  {ovlp:>12.8f}  {ovlp**2:>14.10f}")
