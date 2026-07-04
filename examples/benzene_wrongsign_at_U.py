"""Does the section 7.2 covalent-only wrong-sign artefact survive at the
PPP correlation strength?

The manuscript demonstrates the covalent-only non-monotonicity at U = 0,
s = 0.2. Referee objection: at U = 0 the wavefunction is 93% ionic, where
nobody would use a covalent-only basis; at the PPP point (U/|h| ~ 4),
where covalent-only reasoning is actually deployed, does the artefact
persist?

Covalent side: the on-site U block vanishes IDENTICALLY on the covalent
manifold (no covalent determinant carries a doubly occupied site, so no
(aa|aa) integral survives the cofactor spin matching, at any overlap);
the script asserts this symbolically, making the covalent-only curve
exactly U-independent. The scan is therefore computed once.
FCI side: H(lambda, U) = H1(h_ab = lambda*h) + H2(U), combining the two
cached 400x400 symbolic matrices; the combination is validated at U = 0
against the published figure-5 endpoint energies before use.

Run from the repo root: PYTHONPATH=. python3 examples/benzene_wrongsign_at_U.py
(uses /tmp/benzene_full_aromaticity_HS.pkl and /tmp/benzene_hubbard_matrices.pkl)
"""
import pickle

import numpy as np
import sympy as sp
from scipy.linalg import eigh

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, FixedPsi

H_VAL, S_VAL = -1.0, 0.2
U_VALS = (0.0, 4.0, 8.0)
lambdas = np.linspace(0.0, 1.0, 41)

# ---------------- covalent five-structure block ----------------
m = Molecule(
    zero_ii=True,
    subst={'s': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af'),
           'h': ('H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af')},   # H_ab free
    interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'],
    subst_2e={'U': ('1111',)},
    max_2e_centers=1,
)
PARENT = 'aBcDeF'
rumer = [
    FixedPsi(PARENT, coupled_pairs=[(0, 1), (2, 3), (4, 5)]),
    FixedPsi(PARENT, coupled_pairs=[(0, 5), (1, 2), (3, 4)]),
    FixedPsi(PARENT, coupled_pairs=[(0, 1), (2, 5), (3, 4)]),
    FixedPsi(PARENT, coupled_pairs=[(0, 3), (1, 2), (4, 5)]),
    FixedPsi(PARENT, coupled_pairs=[(0, 5), (1, 4), (2, 3)])]

h, s, U = sp.symbols('h s U')
Hab = sp.Symbol('H_ab')
print('building covalent 5x5 blocks (1e symbolic, 2e once; ~1 min)...')
H1s = m.build_matrix(rumer, op='H')
Ss = m.build_matrix(rumer, op='S')
H2s = m.o2_matrix(rumer)
assert sp.Matrix(H2s).expand().is_zero_matrix, \
    'on-site U block should vanish identically on the covalent manifold'
print('covalent on-site-U block identically zero (symbolic): OK '
      '-> E_cov5 exactly U-independent')

def cov_scan(u):
    # u is irrelevant by the theorem above; kept for the uniform report
    Sn0 = np.array(Ss.subs({s: S_VAL}).tolist(), dtype=float)
    E = np.empty_like(lambdas)
    for k, lam in enumerate(lambdas):
        H1n = np.array(H1s.subs({h: H_VAL, s: S_VAL, Hab: lam * H_VAL}).tolist(),
                       dtype=float)
        E[k] = eigh(H1n, Sn0, eigvals_only=True)[0]
    return E

# ---------------- FCI from the two caches ----------------
print('loading cached 400x400 matrices...')
with open('/tmp/benzene_full_aromaticity_HS.pkl', 'rb') as fh:
    H_ar, S_ar = pickle.load(fh)
with open('/tmp/benzene_hubbard_matrices.pkl', 'rb') as fh:
    H1_hu, S_hu, H2_hu = pickle.load(fh)

# 1e part with H_ab free, evaluated once at the two ends (linear in H_ab)
sub0 = {h: H_VAL, s: S_VAL, Hab: 0.0}
sub1 = {h: H_VAL, s: S_VAL, Hab: H_VAL}
print('substituting (4 dense 400x400 substitutions; ~1-2 min)...')
H1_0 = np.array(H_ar.subs(sub0).tolist(), dtype=float)
H1_1 = np.array(H_ar.subs(sub1).tolist(), dtype=float)
dH = H1_1 - H1_0
S_n = np.array(S_ar.subs({s: S_VAL}).tolist(), dtype=float)
H2U = np.array(sp.diff(H2_hu, U).subs({s: S_VAL}).tolist(), dtype=float)

# cross-cache basis consistency: overlap matrices must agree entry-wise
S_hu_n = np.array(S_hu.subs({s: S_VAL}).tolist(), dtype=float)
assert np.allclose(S_n, S_hu_n, atol=1e-12), 'cache basis orderings differ!'
print('cache basis consistency: OK (overlap matrices identical)')

def fci_scan(u):
    H2n = u * H2U
    E = np.empty_like(lambdas)
    for k, lam in enumerate(lambdas):
        E[k] = eigh(H1_0 + lam * dH + H2n, S_n,
                    eigvals_only=True, subset_by_index=[0, 0])[0]
    return E

# validation hook: U = 0 endpoints must match the published figure-5 values
E_f0 = fci_scan(0.0)
assert abs(E_f0[0] - (-5.485)) < 2e-3 and abs(E_f0[-1] - (-6.190)) < 2e-3, \
    (E_f0[0], E_f0[-1])
print(f'U = 0 FCI endpoints reproduce figure-5 values: OK '
      f'({E_f0[0]:.3f}, {E_f0[-1]:.3f})')

# ---------------- verdict per U ----------------
for u in U_VALS:
    Ec = cov_scan(u)
    Ef = fci_scan(u)
    imax = int(np.argmax(Ec))
    interior = 0 < imax < len(lambdas) - 1 and Ec[imax] > Ec[0] + 1e-9
    mono = bool(np.all(np.diff(Ef) < 0))
    lam_star = lambdas[imax] if interior else float('nan')
    print(f'U/|h| = {u:>4}:  covalent-only interior max: '
          f'{"YES at lam = %.3f" % lam_star if interior else "no":<22} '
          f'rise over [0, lam*]: {Ec[imax]-Ec[0]:.4f}|h|   '
          f'FCI monotone: {mono}')
