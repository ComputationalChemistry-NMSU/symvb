"""Phase 0 recon for PPP Schrieffer-Wolff on benzene.

Goal: characterise the algebra of the cached PPP H2 (benzene_ujk_matrices.pkl)
to decide on a partition strategy for the SW expansion.

Reports for each two-electron piece V_X (X in {U, J, K, M}, extracted as
partial H2 / partial X at s = 0):

  - whether V_X is diagonal in the determinant basis
    (= whether it can sit in H_0 alongside V_U)
  - whether [V_X, N_d] = 0  (N_d-block-diagonality)
  - whether [V_X, S^2] = 0  (singlet sector preserved)
  - whether [V_X, eta^2] = 0  (eta-pairing remains a good QN)

Then decomposes the singlet-A_1g sub-block by N_d.  The output dictates
which partition (N_d, or finer) to use, and which subset of the 22-dim
singlet-A_1g block stays inside SW.
"""
import os
import sys
import time
import pickle

import numpy as np
import sympy as sp
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from symvb import Molecule, SlaterDet, symmetry
from symvb.spin import s_squared_matrix, eta_squared_matrix


CACHE = '/tmp/benzene_ujk_matrices.pkl'
ORBS = list('abcdef')
SITE_SIGNS = {'a': +1, 'b': -1, 'c': +1, 'd': -1, 'e': +1, 'f': -1}


def build_basis():
    m = Molecule(zero_ii=True,
                 subst={'s': ('S_ab','S_bc','S_cd','S_de','S_ef','S_af'),
                        'h': ('H_ab','H_bc','H_cd','H_de','H_ef','H_af')},
                 interacting_orbs=['ab','bc','cd','de','ef','af'])
    m.generate_basis(3, 3, 6)
    return m, [fp.dets[0].det_string for fp in m.basis]


def a1g_projector(det_strings):
    def canon(ds):
        fp = SlaterDet(ds).get_sorted()
        return fp.dets[0].det_string, fp.coefs[0]
    C6    = {'a':'b','b':'c','c':'d','d':'e','e':'f','f':'a'}
    sigma = {'a':'a','b':'f','c':'e','d':'d','e':'c','f':'b'}
    perms = [symmetry.apply_orbital_permutation(om, det_strings, canon)[0]
             for om in (C6, sigma)]
    U_a, _ = symmetry.totally_symmetric_basis(perms, len(det_strings))
    return U_a


def project_kernel(M, U, tol=1e-8):
    Mp = U.T @ M @ U
    Mp = 0.5 * (Mp + Mp.T)
    ev, vec = np.linalg.eigh(Mp)
    return U @ vec[:, np.abs(ev) < tol]


def double_occ(ds):
    occ = {}
    for c in ds:
        occ.setdefault(c.lower(), [False, False])
        if c.islower(): occ[c.lower()][0] = True
        else:           occ[c.lower()][1] = True
    return sum(1 for ab in occ.values() if ab[0] and ab[1])


# ----------------------------------------------------------------------
# Load cache and extract per-integral pieces at s = 0
# ----------------------------------------------------------------------
print(f"Loading {CACHE} ...")
with open(CACHE, 'rb') as f:
    H1_sym, S_sym, H2_sym = pickle.load(f)

h_s, s_s = sp.symbols('h s')
U_s, J_s, K_s, M_s = sp.symbols('U J K M')

# Sanity: at s=0 the metric is identity
S_at_s0 = np.array(sp.Matrix(S_sym).subs({s_s: 0}), dtype=float)
assert np.allclose(S_at_s0, np.eye(400), atol=1e-10), "S != I at s = 0"

# Extract per-integral pieces by setting all-but-one to zero, then dividing.
# (H2 is linear in U, J, K, M at s=0 by inspection of subst_2e structure;
#  cross-terms are eliminated.)
def piece(symbol):
    subs = {s_s: 0, U_s: 0, J_s: 0, K_s: 0, M_s: 0}
    subs[symbol] = 1
    M = np.array(sp.Matrix(H2_sym).subs(subs), dtype=float)
    return M

print("Extracting V_U, V_J, V_K, V_M at s = 0 ...")
t0 = time.time()
V_U = piece(U_s)
V_J = piece(J_s)
V_K = piece(K_s)
V_M = piece(M_s)
print(f"  done in {time.time() - t0:.1f}s")

# Also reconstruct: V_total at unit (U,J,K,M) and check linearity
V_total = np.array(sp.Matrix(H2_sym).subs({s_s: 0, U_s: 1, J_s: 1, K_s: 1, M_s: 1}),
                   dtype=float)
linear_resid = np.max(np.abs(V_total - (V_U + V_J + V_K + V_M)))
print(f"  linearity residual ||V_total - (V_U+V_J+V_K+V_M)||_inf = {linear_resid:.2e}")
assert linear_resid < 1e-10, "H2 is not linear in (U,J,K,M) at s=0!"

V_t = np.array(sp.Matrix(H1_sym).subs({h_s: 1, s_s: 0}), dtype=float)


# ----------------------------------------------------------------------
# Build basis and reference operators
# ----------------------------------------------------------------------
m, det_strings = build_basis()
N = len(det_strings)
print(f"\nFCI basis dim: {N}")

print("Building A_1g, S^2, eta^2 ...")
t0 = time.time()
U_a = a1g_projector(det_strings)
S2 = s_squared_matrix(det_strings)
E2 = eta_squared_matrix(det_strings, SITE_SIGNS, ORBS)
Nd = np.diag([double_occ(d) for d in det_strings]).astype(float)
print(f"  done in {time.time() - t0:.2f}s")

print(f"  dim A_1g = {U_a.shape[1]}")
U_s_ = project_kernel(S2, U_a)
print(f"  dim singlet-A_1g = {U_s_.shape[1]}")
U_e_ = project_kernel(E2, U_s_)
print(f"  dim singlet-A_1g, eta=0 = {U_e_.shape[1]}")


# ----------------------------------------------------------------------
# Algebra check: for each V in {V_t, V_U, V_J, V_K, V_M}, report:
#   - is V diagonal in det basis?           (off-diag norm)
#   - does V commute with N_d?              (||[V, N_d]||_inf)
#   - does V commute with S^2?              (||[V, S^2]||_inf)
#   - does V commute with eta^2?            (||[V, eta^2]||_inf)
# ----------------------------------------------------------------------
def diag_offdiag(M, tol=1e-12):
    d = np.diag(np.diag(M))
    off = M - d
    return np.linalg.norm(d, np.inf), np.max(np.abs(off))


def commutator_norm(A, B):
    return np.max(np.abs(A @ B - B @ A))


pieces = [('V_t', V_t), ('V_U', V_U), ('V_J', V_J),
          ('V_K', V_K), ('V_M', V_M)]

print("\n" + "=" * 80)
print(f"{'op':>5}   {'||diag||':>10}  {'||off-diag||':>14}    "
      f"{'[V,N_d]':>10}  {'[V,S^2]':>10}  {'[V,eta^2]':>12}")
print("=" * 80)
for name, V in pieces:
    nd, no = diag_offdiag(V)
    cn = commutator_norm(V, Nd)
    cs = commutator_norm(V, S2)
    ce = commutator_norm(V, E2)
    print(f"{name:>5}   {nd:>10.3f}  {no:>14.3f}    "
          f"{cn:>10.2e}  {cs:>10.2e}  {ce:>12.2e}")


# ----------------------------------------------------------------------
# N_d decomposition of A_1g, singlet-A_1g, singlet-A_1g eta=0
# (These are what's preserved by V_J / V_K / V_M -- depends on commutators
#  above; we report unconditionally for reference.)
# ----------------------------------------------------------------------
def Nd_dist(U):
    Mb = U.T @ Nd @ U
    ev = np.linalg.eigvalsh(0.5 * (Mb + Mb.T))
    ev_int = np.round(ev).astype(int)
    if not np.allclose(ev, ev_int, atol=1e-6):
        return None  # N_d not preserved
    return dict(sorted(Counter(ev_int.tolist()).items()))


print("\n" + "=" * 80)
print("N_d decomposition by symmetry block:")
print("=" * 80)
print(f"  full A_1g (38)            : {Nd_dist(U_a)}")
print(f"  singlet-A_1g (22)         : {Nd_dist(U_s_)}")
print(f"  singlet-A_1g, eta=0 (14)  : {Nd_dist(U_e_)}")


# ----------------------------------------------------------------------
# Suggest a partition strategy based on the algebra
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("Partition strategy suggestion")
print("=" * 80)


def is_zero(x, tol=1e-8): return abs(x) < tol


# Build an H_0 candidate from all pieces that (i) are diagonal in det basis,
# and (ii) commute with all symmetry projectors needed.
# Off-diagonal pieces become the perturbation V.
print("\nDiagonal pieces (candidates for H_0):")
for name, V in pieces:
    nd, no = diag_offdiag(V)
    diag_ok = is_zero(no)
    n_d_ok = is_zero(commutator_norm(V, Nd))
    s2_ok  = is_zero(commutator_norm(V, S2))
    e2_ok  = is_zero(commutator_norm(V, E2))
    flags = []
    flags.append("diag" if diag_ok else "off-diag")
    flags.append("[N_d]=0" if n_d_ok else "[N_d]!=0")
    flags.append("[S^2]=0" if s2_ok else "[S^2]!=0")
    flags.append("[eta^2]=0" if e2_ok else "[eta^2]!=0")
    print(f"  {name:>5}: {', '.join(flags)}")
