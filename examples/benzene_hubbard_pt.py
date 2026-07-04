"""
Worked example: Rayleigh-Schrodinger perturbation series for benzene with
on-site Hubbard repulsion (zero_ii=True, orthogonal AOs s=0, h=-1).

The series

    E_FCI/t = -8 + (3/2) u - (29/288) u^2 + 0 * u^3 - (2855 / 5 971 968) u^4
              + O(u^5)           where u = U/t

is derived symbolically end-to-end:

  1.  symvb builds the 400x400 symbolic H_1e, H_2e, and S over the full
      Sz=0 determinantal basis.
  2.  symvb.symmetry projects to the 38-dimensional D_6 A_1g block
      that carries the ground state.
  3.  The 38x38 matrix has integer eigenvalues at U=0; sympy.solve gives
      them exactly.
  4.  Eigenvectors are D-orthonormalised within each degenerate subspace
      so the generalised-metric PT sums are rigorous.
  5.  Rayleigh-Schrodinger up to 4th order yields the quoted rational
      coefficients.

The physical meaning of the E_2 = -29/288 denominator is unpacked inline
(it's the MP2 formula evaluated on the 6-site half-filled Hubbard ring).

Run from the repo root:  PYTHONPATH=. python3 examples/benzene_hubbard_pt.py
"""
import os
import sys
import pickle
import time

import numpy as np
import sympy as sp

# Allow running directly from the examples/ subdirectory without installing.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from symvb import Molecule, SlaterDet, symmetry


# --- 1. build H_1e, S, H_2e once and cache -------------------------------
# Molecule.ring(6) is exactly the benzene pi ring: nearest-neighbour edges
# ab,bc,cd,de,ef,af with resonance h, overlap s, and on-site Hubbard U
# (max_2e_centers=1).  The cache stays the separate (H1, S, H2) triple that
# the other benzene examples and the teaching notebooks consume.
CACHE = '/tmp/benzene_hubbard_matrices.pkl'

m = Molecule.ring(6)
m.generate_basis(3, 3, 6)

if os.path.exists(CACHE):
    with open(CACHE, 'rb') as f:
        H1, S, H2 = pickle.load(f)
else:
    print("Building 400x400 symbolic H, S, H2 (one-time, several minutes) ...")
    t0 = time.time()
    H1 = m.build_matrix(m.basis, op='H')
    S  = m.build_matrix(m.basis, op='S')
    H2 = m.o2_matrix(m.basis)
    print(f"  done in {time.time() - t0:.1f}s")
    with open(CACHE, 'wb') as f:
        pickle.dump((H1, S, H2), f)

det_strings = [fp.dets[0].det_string for fp in m.basis]


# --- 2. project to the 38-dim A_1g block ---------------------------------
def canon(ds):
    fp = SlaterDet(ds).get_sorted()
    return fp.dets[0].det_string, fp.coefs[0]

C6 = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'e', 'e': 'f', 'f': 'a'}
sig = {'a': 'a', 'b': 'f', 'c': 'e', 'd': 'd', 'e': 'c', 'f': 'b'}
perms = [symmetry.apply_orbital_permutation(om, det_strings, canon)[0]
         for om in (C6, sig)]
U_mat, orbits = symmetry.totally_symmetric_basis(perms, 400)
N = U_mat.shape[1]
print(f"A_1g block dimension: {N}")

U_sp = sp.zeros(400, N)
for col, orb in enumerate(orbits):
    for idx in orb:
        U_sp[idx, col] = 1
orbit_sizes = [len(o) for o in orbits]
D = sp.diag(*[sp.Integer(sz) for sz in orbit_sizes])

h, s, Usym = sp.symbols('h s U')
H_full = sp.Matrix(H1 + H2).subs({h: -1, s: 0})
H0_full = H_full.subs(Usym, 0)
V_full  = sp.Matrix(H_full - H0_full).subs(Usym, 1)

H0_red = sp.Matrix(U_sp.T * H0_full * U_sp)
V_red  = sp.Matrix(U_sp.T * V_full  * U_sp)

Dinv = sp.diag(*[sp.Rational(1, sz) for sz in orbit_sizes])
M = Dinv * H0_red                    # eigenvalues of M = those of (H0_red, D)

cp = M.charpoly(sp.Symbol('E'))
eigenvalues = sorted(set(sp.solve(cp.as_expr(), sp.Symbol('E'))),
                     key=lambda r: float(r))
print(f"Distinct eigenvalues of A_1g block: {eigenvalues}")


# --- 3. D-orthonormal eigenvectors ---------------------------------------
eig_data = []
for r in eigenvalues:
    ker = (M - r * sp.eye(N)).nullspace()
    orth = []
    for v in ker:
        w = v
        for u in orth:
            w = w - ((u.T * D * w)[0, 0]) * u
        norm2 = (w.T * D * w)[0, 0]
        if norm2 == 0:
            continue
        orth.append(w / sp.sqrt(norm2))
    for w in orth:
        eig_data.append((r, w))
print(f"Total A_1g eigenvectors: {len(eig_data)}")


# --- 4. perturbation-theory coefficients (all exact) ---------------------
E0 = eig_data[0][0]
Nd = len(eig_data)
V_mn = sp.zeros(Nd, Nd)
for mi, (_, vm) in enumerate(eig_data):
    for ni, (_, vn) in enumerate(eig_data):
        V_mn[mi, ni] = sp.simplify((vm.T * V_red * vn)[0, 0])

# Recursion:  |psi_n> = R [(V - E_1) |psi_{n-1}>  -  sum_{k=2}^{n-1} E_k |psi_{n-k}>]
#              E_{n+1} = <0 | V | psi_n>
# Work in the eigenbasis of H_0 so R is diagonal and the recursion is cheap.
E0_g = eig_data[0][0]
eig_E = [eig_data[i][0] for i in range(Nd)]
V_eig = V_mn   # already in the H_0 eigenbasis (eig_data is orthonormal in D-metric)

r_diag = [sp.Rational(0)] * Nd
for n in range(1, Nd):
    denom = E0_g - eig_E[n]
    if denom != 0:
        r_diag[n] = sp.Rational(1) / denom

n_max = 5          # |psi_5> lets us read off E_6
psi = [None] * (n_max + 1)
psi[0] = [sp.Rational(0)] * Nd
psi[0][0] = sp.Rational(1)

E_coef = [sp.Rational(0)] * (n_max + 2)
E_coef[0] = E0_g
E_coef[1] = V_eig[0, 0]

for n in range(1, n_max + 1):
    rhs = [sum(V_eig[m, l] * psi[n - 1][l] for l in range(Nd)) for m in range(Nd)]
    for m in range(Nd):
        rhs[m] -= E_coef[1] * psi[n - 1][m]
    for k in range(2, n):
        for m in range(Nd):
            rhs[m] -= E_coef[k] * psi[n - k][m]

    psi[n] = [sp.simplify(r_diag[m] * rhs[m]) if r_diag[m] != 0 else sp.Rational(0)
              for m in range(Nd)]
    E_coef[n + 1] = sp.simplify(sum(V_eig[0, m] * psi[n][m] for m in range(Nd)))

E1, E2, E3, E4 = E_coef[1], E_coef[2], E_coef[3], E_coef[4]
E5, E6 = E_coef[5], E_coef[6]

print(f"\nRayleigh-Schrodinger coefficients (h = -1, s = 0):")
print(f"  E_0 = {E0}")
print(f"  E_1 = {E1}                       # mean-field double occupancy")
print(f"  E_2 = {E2}          # MP2, rational")
print(f"  E_3 = {E3}                        # odd order vanishes (particle-hole symmetry)")
print(f"  E_4 = {E4}")
print(f"  E_5 = {E5}                        # odd order vanishes")
print(f"  E_6 = {E6}")


# --- 5. why is E_2 = -29/288 ? --------------------------------------------
print("\n=== Decoding E_2 = -29/288 from the Hueckel MO picture ===")
# Hueckel MOs: energies e_k = 2 beta cos(2 pi k / 6) with beta = h = -1.
#   k = 0       : e = -2    (bonding, occupied)
#   k = +/- 1   : e = -1    (bonding, occupied)
#   k = +/- 2   : e = +1    (antibonding, virtual)
#   k = 3       : e = +2    (antibonding, virtual)
# Hubbard U in MO basis:  (ab|cd) = U/N * delta(k_a + k_c == k_b + k_d mod N)
# MP2 for opposite-spin only (Hubbard):
#   E_2 / U^2 = sum_{i up, j dn, a up, b dn}  (1/N^2) * delta_momentum_conservation
#                        / (eps_i + eps_j - eps_a - eps_b)
occ = {-1: -1, 0: -2, 1: -1}       # momentum -> energy
vir = {-2: 1, 2: 1, 3: 2}

S_mp2 = sp.Rational(0)
count = 0
for i_k, i_e in occ.items():
    for j_k, j_e in occ.items():
        for a_k, a_e in vir.items():
            for b_k, b_e in vir.items():
                if (i_k + j_k - a_k - b_k) % 6 != 0:
                    continue
                count += 1
                S_mp2 += sp.Rational(1, i_e + j_e - a_e - b_e)

N_sites = 6
E2_decoded = S_mp2 / N_sites ** 2
print(f"  # momentum-conserving (i_up, j_dn -> a_up, b_dn) quadruples: {count}")
print(f"  sum of 1/Delta_eps                                        : {S_mp2}")
print(f"  E_2 / U^2  = sum / N_sites^2 = {S_mp2} / {N_sites**2} = {E2_decoded}")
print(f"  Matches symvb PT:  {E2_decoded == E2}")

# The full interpretation:
#   288 = 6^2 * 8
#     6^2 : two factors of 1/N from the on-site Hubbard matrix element in MO basis.
#     8   : common denominator from summing 1/(E_i + E_j - E_a - E_b)
#           over the 19 momentum-conserving quadruples.
#   29  : the integer numerator that emerges after clearing 8 into the sum.
print("\nSo 288 = 6^2 * 8 (N^2 from MO matrix elements; 8 from MO energy gap LCM)")
print("   29 is the integer combinatorial sum over the 19 allowed momentum channels.")
