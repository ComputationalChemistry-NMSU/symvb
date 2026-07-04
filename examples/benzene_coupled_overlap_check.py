"""Robustness check for manuscript section 7.2: scale s_ab together with h_ab.

The main-text bond-weakening coordinate (Eq. 16) scales h_ab = lam*h while
holding s_ab = s = 0.2 fixed. This script tests the coupled coordinate
h_ab = lam*h, s_ab = lam*s (all other edges at h = -1, s = 0.2) and checks

  (1) the covalent-only five-structure energy keeps its interior maximum
      (the wrong-sign artifact), and
  (2) the FCI energy remains monotone decreasing in lam.

FCI at U = 0 is taken from the 6x6 generalized Huckel problem (three lowest
MOs doubly occupied), exact for a one-electron Hamiltonian.

Run from the repo root: PYTHONPATH=. python3 examples/benzene_coupled_overlap_check.py
Expected: interior max at lam = 0.490; FCI monotone. (Fixed-s_ab reference:
max at lam = 0.32, manuscript Fig. 5c.)
"""
import os
import sys

import numpy as np
import sympy as sp
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, FixedPsi

m = Molecule(
    zero_ii=True,
    subst={'s': ('S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af'),   # S_ab free
           'h': ('H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af')},  # H_ab free
    interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'])
PARENT = 'aBcDeF'
rumer = [
    FixedPsi(PARENT, coupled_pairs=[(0, 1), (2, 3), (4, 5)]),
    FixedPsi(PARENT, coupled_pairs=[(0, 5), (1, 2), (3, 4)]),
    FixedPsi(PARENT, coupled_pairs=[(0, 1), (2, 5), (3, 4)]),
    FixedPsi(PARENT, coupled_pairs=[(0, 3), (1, 2), (4, 5)]),
    FixedPsi(PARENT, coupled_pairs=[(0, 5), (1, 4), (2, 3)])]

Hs = m.build_matrix(rumer, op='H')
Ss = m.build_matrix(rumer, op='S')
h, s = sp.symbols('h s')
Hab, Sab = sp.Symbol('H_ab'), sp.Symbol('S_ab')
H_VAL, S_VAL = -1.0, 0.2

lambdas = np.linspace(0.0, 1.0, 101)
E_cov = np.empty_like(lambdas)
E_fci = np.empty_like(lambdas)


def huckel_fci(lam):
    Hm = np.zeros((6, 6))
    Sm = np.eye(6)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    for (i, j) in edges:
        hij, sij = H_VAL, S_VAL
        if (i, j) == (0, 1):
            hij, sij = lam * H_VAL, lam * S_VAL
        Hm[i, j] = Hm[j, i] = hij
        Sm[i, j] = Sm[j, i] = sij
    ev = eigh(Hm, Sm, eigvals_only=True)
    return 2.0 * ev[:3].sum()


for k, lam in enumerate(lambdas):
    subs = {h: H_VAL, s: S_VAL, Hab: lam * H_VAL, Sab: lam * S_VAL}
    Hn = np.array(Hs.subs(subs).tolist(), dtype=float)
    Sn = np.array(Ss.subs(subs).tolist(), dtype=float)
    E_cov[k] = eigh(Hn, Sn, eigvals_only=True)[0]
    E_fci[k] = huckel_fci(lam)

imax = int(np.argmax(E_cov))
print(f'covalent-only: E(0) = {E_cov[0]:.5f}, interior max at lam = '
      f'{lambdas[imax]:.3f} (E = {E_cov[imax]:.5f}), E(1) = {E_cov[-1]:.5f}')
assert 0 < imax < len(lambdas) - 1 and E_cov[imax] > E_cov[0], \
    'wrong-sign artifact did NOT persist under coupled overlap'
dE = np.diff(E_fci)
print(f'FCI monotone decreasing along coupled coordinate: {np.all(dE < 0)}'
      f'  (max dE = {dE.max():.2e})')
print(f'FCI: E(0) = {E_fci[0]:.5f} -> E(1) = {E_fci[-1]:.5f}')
