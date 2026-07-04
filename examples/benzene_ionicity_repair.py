"""Which ionicity class repairs the covalent-only wrong-sign artefact in benzene?

Manuscript claim (benzene ionicity-decomposition / covalent-only failure-mode
section): weakening a single ring edge, h_ab = lambda * h with lambda in [0, 1]
and all overlaps fixed at s = 0.2, the covalent-only five-Rumer model has a
spurious maximum in the ground-state energy E(lambda) (positive slope at
lambda = 0), whereas the exact FCI is monotone until the on-site repulsion is
strong. This script shows WHICH determinants repair the sign:

  * covalent-only (20 dets, n_d = 0): slope@0 = +0.087|h|, max at lambda ~ 0.32
    (wrong sign); U-independent, because the on-site (1111) integral vanishes
    identically on the covalent manifold.
  * add the singly-ionic class (200 dets, n_d <= 1): slope@0 flips to
    -0.210|h|, monotone, recovering 76% of the FCI slope -0.277|h|. The
    mono-ionic (single charge-transfer / superexchange) class ALONE fixes the
    qualitative failure.
  * add the doubly-ionic class (380 dets, n_d <= 2): slope@0 = -0.336|h|
    (overshoots FCI), and reproduces the FCI wrong-sign onset to three figures.

The wrong-sign onset at large U is a BOUNDARY (transcritical) transition: the
interior maximum grows continuously out of the lambda = 0 edge as the slope
there changes sign, so the onset condition is exactly rho_ab(0, U*) = 0, with
rho_ab(lambda) = <Psi| dH/dh_ab |Psi> the Hellmann-Feynman edge population.
Onsets: U*/|h| = 7.05 (n_d <= 1), 8.21 (n_d <= 2), 8.21 (FCI); the FCI value
matches the argmax-grid onset 8.25 of benzene_fci_vs_U_onset.py.

H(lambda, U) = A0 + lambda * dA + U * BU, where A0 = H1(H_ab = 0),
dA = H1(H_ab = h) - A0 (H1 the one-electron matrix, linear in H_ab), and
BU = H2(U = 1) the on-site (1111) two-electron matrix (linear in U; evaluating
at U = 1 gives the per-unit-U coefficient and avoids squaring U). Both 400x400
symbolic matrices are the cached ones shared with benzene_fci_vs_U_onset.py.

Run from the repo root:  PYTHONPATH=. python3 examples/benzene_ionicity_repair.py
Runtime ~1-2 min with the /tmp caches present (first cache build is minutes).
"""
import os
# single-threaded BLAS: thousands of small generalized eigenproblems are far
# faster (and do not thrash) without thread spawn overhead. Must precede numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import pickle
import itertools

import numpy as np
from scipy.linalg import eigh
import sympy as sp

from symvb import Molecule

h, s = sp.symbols('h s')
Hab = sp.Symbol('H_ab')
Usym = sp.Symbol('U')
H_VAL, S_VAL = -1.0, 0.2
SUBST = dict(
    zero_ii=True,
    subst={'s': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af'),
           'h': ('H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af')},   # H_ab kept free
    interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'])

C_H1 = '/tmp/benzene_full_aromaticity_HS.pkl'         # (H1, S), one-electron
C_H2 = '/tmp/benzene_aromaticity_H2_onsite.pkl'       # on-site U two-electron
if os.path.exists(C_H1):
    H1, S = pickle.load(open(C_H1, 'rb'))
else:
    print('building one-electron cache (minutes)...')
    mf = Molecule(**SUBST)
    mf.generate_basis(3, 3, 6)
    H1 = mf.build_matrix(mf.basis, op='H')
    S = mf.build_matrix(mf.basis, op='S')
    pickle.dump((H1, S), open(C_H1, 'wb'))
if os.path.exists(C_H2):
    H2 = pickle.load(open(C_H2, 'rb'))
else:
    print('building on-site U two-electron cache...')
    mu = Molecule(subst_2e={'U': ('1111',)}, max_2e_centers=1, **SUBST)
    mu.generate_basis(3, 3, 6)
    H2 = mu.o2_matrix(mu.basis)
    pickle.dump(H2, open(C_H2, 'wb'))

print('substituting numerical values into the cached 400x400 matrices...')
A0 = np.array(H1.subs({h: H_VAL, s: S_VAL, Hab: 0.0}).tolist(), float)
dA = np.array(H1.subs({h: H_VAL, s: S_VAL, Hab: H_VAL}).tolist(), float) - A0
BU = np.array(H2.subs({s: S_VAL, Usym: 1.0}).tolist(), float)     # per unit U
Sf = np.array(S.subs({s: S_VAL}).tolist(), float)
N = A0.shape[0]
assert N == 400
assert np.abs(BU).max() > 0, 'on-site U did not enter H2'

# ---- ionicity class of every determinant, in generate_det_strings order ----
# The cached matrices are built from mf.generate_basis(3, 3, 6), which iterates
# alpha 3-combinations (outer) x beta 3-combinations (inner) over orbitals a..f.
# n_d = number of orbitals occupied by BOTH an alpha and a beta electron.
labels = np.empty(N, dtype=int)
i = 0
for a in itertools.combinations('abcdef', 3):
    for b in itertools.combinations('abcdef', 3):
        labels[i] = len(set(a) & set(b))
        i += 1
assert i == N
counts = {k: int((labels == k).sum()) for k in range(4)}
assert counts == {0: 20, 1: 180, 2: 180, 3: 20}, counts
cov_idx = np.where(labels == 0)[0]
# the on-site U two-electron operator is identically zero on the covalent block
assert np.abs(BU[np.ix_(cov_idx, cov_idx)]).max() < 1e-12, \
    'on-site U leaks into the covalent (n_d = 0) block'
print(f'determinant classes n_d = 0..3: {counts}  (covalent block U-free: OK)')

IDX = {
    'cov-only(20)': cov_idx,
    'n_d<=1(200)':  np.where(labels <= 1)[0],
    'n_d<=2(380)':  np.where(labels <= 2)[0],
    'FCI(400)':     None,
}


def gs(idx, lam, U):
    """Ground-state (energy, S-normalized eigvec, restricted dA) on subset idx."""
    H = A0 + lam * dA + U * BU
    if idx is None:
        Hs, Ss, dAs = H, Sf, dA
    else:
        ix = np.ix_(idx, idx)
        Hs, Ss, dAs = H[ix], Sf[ix], dA[ix]
    w, v = eigh(Hs, Ss, subset_by_index=[0, 0])
    return w[0], v[:, 0], dAs


def rho0(idx, U):
    """Edge population rho_ab at lambda = 0: (1/h) <Psi| dA |Psi>."""
    _, c, dAs = gs(idx, 0.0, U)
    return (1.0 / H_VAL) * float(c @ dAs @ c)


def escan(idx, U, ng):
    lam = np.linspace(0.0, 1.0, ng)
    E = np.array([gs(idx, l, U)[0] for l in lam])
    return lam, E


def argmax_lambda(idx, U, ng):
    lam, E = escan(idx, U, ng)
    return lam[int(np.argmax(E))]


# ---- U = 0 table: slope@0 = -rho_ab(0), monotonicity, fraction of FCI ----
print('\n=== U = 0: sign of the single-edge response ===')
print(f'{"model":>13} | {"slope@0/|h|":>11} | {"monotone?":>9} | '
      f'{"rho_ab(0)":>9} | {"frac FCI":>8}')
fci_slope = -rho0(None, 0.0)
slope = {}
for k, idx in IDX.items():
    r = rho0(idx, 0.0)
    _, E = escan(idx, 0.0, 401)
    mono = bool(np.all(np.diff(E) < 0))
    slope[k] = (-r, mono, r)
    print(f'{k:>13} | {-r:>+11.4f} | {str(mono):>9} | {r:>+9.4f} | '
          f'{-r/fci_slope:>8.3f}')

# covalent-only wrong-sign maximum location
cov_lam_max = argmax_lambda(cov_idx, 0.0, 2001)
print(f'covalent-only wrong-sign maximum at lambda = {cov_lam_max:.3f}')

# ---- onset via the rho_ab(0, U) = 0 crossing (boundary criterion) ----
print('\n=== wrong-sign onset U* : rho_ab(0, U*) = 0 ===')


def onset(idx, Uhi=40.0):
    if rho0(idx, 0.0) < 0:            # already wrong sign at U = 0 (covalent)
        return 0.0
    assert rho0(idx, Uhi) < 0, 'no crossing below Uhi'
    lo, hi = 0.0, Uhi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if rho0(idx, mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


Ustar = {}
for k, idx in IDX.items():
    Ustar[k] = onset(idx)
    print(f'{k:>13}: U*/|h| = {Ustar[k]:.3f}')

# ---- boundary (transcritical) character: the max leaves the lambda=0 edge ----
print('\n=== onset is a boundary transition (max detaches from lambda = 0) ===')
for k in ('n_d<=1(200)', 'n_d<=2(380)', 'FCI(400)'):
    idx, U = IDX[k], Ustar[k]
    lam_below = argmax_lambda(idx, U - 0.5, 1201)   # still pinned at 0
    lam_above = argmax_lambda(idx, U + 0.5, 1201)   # small, nonzero
    print(f'{k:>13}: argmax lambda  {lam_below:.4f} (at U*-0.5)  ->  '
          f'{lam_above:.4f} (at U*+0.5)')
    assert lam_below == 0.0, f'{k}: max not at boundary just below onset'
    assert 0.0 < lam_above < 0.05, f'{k}: max did not detach continuously'

# ================= assertions on the headline numbers =================
tol = 3e-3
assert abs(slope['cov-only(20)'][0] - 0.087) < tol and not slope['cov-only(20)'][1]
assert abs(cov_lam_max - 0.32) < 0.02
assert abs(slope['n_d<=1(200)'][0] - (-0.210)) < tol and slope['n_d<=1(200)'][1]
assert abs(slope['n_d<=2(380)'][0] - (-0.336)) < tol and slope['n_d<=2(380)'][1]
assert abs(slope['FCI(400)'][0] - (-0.277)) < tol and slope['FCI(400)'][1]
assert abs(-slope['n_d<=1(200)'][0] / -slope['FCI(400)'][0] - 0.76) < 0.02
assert abs(Ustar['n_d<=1(200)'] - 7.05) < 0.2
assert abs(Ustar['n_d<=2(380)'] - 8.21) < 0.1
assert abs(Ustar['FCI(400)'] - 8.21) < 0.1
print('\nall assertions passed: mono-ionic class restores the sign (76% of FCI '
      'slope); onset is boundary-type at rho_ab(0,U*)=0, U* ~ 8.2 through n_d<=2.')
