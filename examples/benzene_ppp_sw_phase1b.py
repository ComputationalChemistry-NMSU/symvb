"""Phase 1b: full PPP SW for benzene  (V_t + V_U + V_J + V_K + V_M).

Builds on Phase 1a: H_0 = V_U + V_J still acts as a scalar  k(U-J) + 15J
on each N_d = k subspace, even after V_K and V_M are added (they don't
appear in H_0).  But:

  - V_K is off-diagonal in det basis and breaks eta^2 .  We work in the
    22-dim singlet-A_1g block (no eta projection).  V_K commutes with
    N_d, so it acts inside each N_d block.  V_K | P_A_1g >  is in
    P_A_1g (intra-P), giving a *direct* contribution at order 1.

  - V_M breaks both eta^2 and N_d  but |Delta N_d| <= 1, so V_M slots
    next to V_t in the perturbation.

The Bloch effective Hamiltonian is then computed at orders 1, 2, 3, 4
numerically at several (h, K, M) sample points (with U-J = 1, t = h),
and the trace and determinant of the 2x2 P_A_1g block are fit as
polynomials in (h, K, M).  These are basis-invariants and give a
complete characterisation of the strong-coupling spin Hamiltonian on
P_A_1g.

Output: rational polynomial expressions for tr(H_eff^n|P_A_1g) and
det(H_eff^n|P_A_1g)  in units of  1 / (U-J)^(n-1).
"""
import os
import sys
import time
import pickle
import itertools

import numpy as np
import sympy as sp
from collections import Counter
from numpy.linalg import lstsq

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
# Load PPP cache, extract V_t, V_U, V_J, V_K, V_M  at s = 0
# ----------------------------------------------------------------------
print(f"Loading {CACHE} ...")
with open(CACHE, 'rb') as f:
    H1_sym, S_sym, H2_sym = pickle.load(f)

h_s, s_s = sp.symbols('h s')
U_s, J_s, K_s, M_s = sp.symbols('U J K M')

print("Extracting V_t, V_U, V_J, V_K, V_M at s = 0 ...")
t0 = time.time()
V_t = np.array(sp.Matrix(H1_sym).subs({h_s: 1, s_s: 0}), dtype=float)
def piece(sym):
    subs = {s_s: 0, U_s: 0, J_s: 0, K_s: 0, M_s: 0}; subs[sym] = 1
    return np.array(sp.Matrix(H2_sym).subs(subs), dtype=float)
V_U = piece(U_s); V_J = piece(J_s); V_K = piece(K_s); V_M = piece(M_s)
print(f"  {time.time()-t0:.1f}s")


# ----------------------------------------------------------------------
# 22-dim singlet-A_1g block (no eta projection: V_K, V_M break eta^2)
# ----------------------------------------------------------------------
print("\nBuilding singlet-A_1g block ...")
m, det_strings = build_basis()
U_a = a1g_projector(det_strings)
S2 = s_squared_matrix(det_strings)
Nd_diag = np.array([double_occ(d) for d in det_strings], dtype=float)
Nd_400 = np.diag(Nd_diag)
U_22 = project_kernel(S2, U_a)
print(f"  dim = {U_22.shape[1]}")

# Sort by N_d
Nd_in_22 = U_22.T @ Nd_400 @ U_22
ev, vec = np.linalg.eigh(0.5 * (Nd_in_22 + Nd_in_22.T))
order = np.argsort(np.round(ev).astype(int), kind='stable')
ev = ev[order]; vec = vec[:, order]
U_22 = U_22 @ vec
Nd_eigs = np.round(ev).astype(int)
slices = {k: np.where(Nd_eigs == k)[0] for k in range(4)}
P_idx = slices[0]
DIM = U_22.shape[1]
print(f"  N_d distribution: P={len(slices[0])}, Q_1={len(slices[1])}, "
      f"Q_2={len(slices[2])}, Q_3={len(slices[3])}")

# Project all the operator pieces
Vt22 = U_22.T @ V_t @ U_22
VK22 = U_22.T @ V_K @ U_22
VM22 = U_22.T @ V_M @ U_22

# Verify V_J is still scalar per N_d block here
VJ22 = U_22.T @ V_J @ U_22
ok = True
for k in range(4):
    blk = VJ22[np.ix_(slices[k], slices[k])]
    diff = np.max(np.abs(blk - (15-k)*np.eye(len(slices[k]))))
    ok = ok and diff < 1e-9
print(f"  V_J = (15-N_d)*J on each block: {'PASS' if ok else 'FAIL'}")

# Drop noise
for M_ in (Vt22, VK22, VM22):
    M_[np.abs(M_) < 1e-10] = 0.0


# ----------------------------------------------------------------------
# Resolvent (units U-J = 1):  G = -Q_k / k   on Q_k blocks
# ----------------------------------------------------------------------
G = np.zeros((DIM, DIM))
for i in range(DIM):
    if Nd_eigs[i] > 0:
        G[i, i] = -1.0 / Nd_eigs[i]

P_proj = np.diag([1.0 if Nd_eigs[i] == 0 else 0.0 for i in range(DIM)])


def H_eff_at(h_val, K_val, M_val):
    """Bloch H_eff at orders 1..4 for given (t=h, K, M); units U-J = 1."""
    V = h_val * Vt22 + K_val * VK22 + M_val * VM22
    H1 = P_proj @ V @ P_proj
    W1 = G @ V @ P_proj
    H2 = P_proj @ V @ W1
    W2 = G @ V @ W1
    H3 = P_proj @ V @ W2
    PVO1 = P_proj @ V @ W1
    W3 = G @ V @ W2 - G @ W1 @ PVO1
    H4 = P_proj @ V @ W3
    return H1, H2, H3, H4


# ----------------------------------------------------------------------
# Sample (h, K, M), fit each H_eff^n entry as a polynomial in (h, K, M)
# of degree exactly n.
# ----------------------------------------------------------------------
def monomials_of_degree(n):
    return [(a, b, c) for a in range(n+1)
            for b in range(n+1-a)
            for c in range(n+1-a-b)
            if a + b + c == n]


def fit_poly(samples, values, mons):
    A = np.array([[s[0]**a * s[1]**b * s[2]**c for (a,b,c) in mons]
                  for s in samples])
    coef, *_ = lstsq(A, values, rcond=None)
    return coef


def fmt_poly(coef, mons, syms=('h', 'K', 'M'), tol=1e-6):
    terms = []
    for c, (a, b, k) in zip(coef, mons):
        cr = sp.nsimplify(float(c), rational=True, tolerance=1e-6)
        if cr == 0:
            continue
        sym = sp.Mul(*[sp.Symbol(syms[i])**e
                       for i, e in enumerate((a, b, k)) if e > 0]) \
              if (a + b + k) > 0 else 1
        terms.append(cr * sym)
    return sp.Add(*terms) if terms else sp.S(0)


print("\nSampling and fitting H_eff^(1..4) on P_A_1g ...")
np.random.seed(42)
N_samples = 60
samples = np.random.uniform(-2, 2, (N_samples, 3))

# Storage: at each sample, the 2x2 H_eff^n on P_A_1g
H_eff_samples = {n: np.zeros((N_samples, len(P_idx), len(P_idx))) for n in (1,2,3,4)}
for s_idx, s in enumerate(samples):
    h_v, K_v, M_v = s
    H1, H2, H3, H4 = H_eff_at(h_v, K_v, M_v)
    for n, H in zip((1,2,3,4), (H1, H2, H3, H4)):
        Mb = H[np.ix_(P_idx, P_idx)]
        H_eff_samples[n][s_idx] = 0.5 * (Mb + Mb.T)

print("\n" + "=" * 76)
print("H_eff^(n)|P_A_1g  - matrix entries as polynomials in (h, K, M)")
print("       (units h^a K^b M^c / (U-J)^(n-1))")
print("=" * 76)

P_A1g_dim = len(P_idx)
for n in (1, 2, 3, 4):
    mons = monomials_of_degree(n)
    print(f"\n--- order {n}  ({len(mons)} monomials of degree {n}) ---")
    Mfit = sp.zeros(P_A1g_dim, P_A1g_dim)
    for i in range(P_A1g_dim):
        for j in range(i, P_A1g_dim):
            vals = H_eff_samples[n][:, i, j]
            coef = fit_poly(samples, vals, mons)
            poly = fmt_poly(coef, mons)
            Mfit[i, j] = poly
            Mfit[j, i] = poly
    sp.pprint(Mfit)

    tr = sp.simplify(Mfit.trace())
    det = sp.simplify(Mfit.det())
    print(f"  trace : {tr}")
    print(f"  det   : {det}")
