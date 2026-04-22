"""
Independent cross-validation of vbt3 benzene Hubbard ground-state energies
against PySCF's FCI solver.

For a six-site Hubbard ring with nearest-neighbour hopping t and on-site U,
the Hamiltonian reduces to

    H = -t Σ_{<ij>, σ} (c†_iσ c_jσ + h.c.)  +  U Σ_i n_iα n_iβ

In the s = 0 (orthogonal-AO) limit vbt3 returns the exact 400×400 symbolic
matrix H(t, U); substituting numerical values and diagonalising is
equivalent to FCI in the minimal Sz = 0 basis of 6 orbitals / 6 electrons.

PySCF solves the same problem by feeding the one- and two-electron
integrals directly to `fci.direct_spin1.kernel` with no AO-to-MO
transformation.  The two solvers are algorithmically independent:
vbt3 works in a Slater-determinant VB basis with cofactor expansions;
PySCF uses a string-based determinant representation and Davidson
diagonalisation of the full CI Hamiltonian.  Agreement to machine
precision is therefore a genuine cross-validation.
"""
import os
import sys
import pickle
import numpy as np
import sympy as sp
from pyscf import fci, ao2mo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def pyscf_benzene_hubbard(t, U):
    """
    FCI ground-state energy for benzene Hubbard at (t, U) via PySCF.

    One-electron: nearest-neighbour ring with h_ij = -t for |i-j| = 1 (mod 6).
    Two-electron: purely on-site, U δ_{ijkl} δ_{ij} δ_{ik}.
    Returns the six-orbital / six-electron Sz = 0 ground-state energy.
    """
    L = 6
    h1 = np.zeros((L, L))
    for i in range(L):
        j = (i + 1) % L
        h1[i, j] = h1[j, i] = -t

    eri = np.zeros((L, L, L, L))
    for i in range(L):
        eri[i, i, i, i] = U

    na = nb = 3
    e, _ = fci.direct_spin1.kernel(h1, eri, L, (na, nb))
    return float(e)


def vbt3_benzene_hubbard(t, U, _cache={}):
    """
    Ground-state energy from the cached vbt3 symbolic matrices.
    The cache at /tmp/benzene_hubbard_matrices.pkl stores a 3-tuple
    (H1, S, H2) of 400×400 SymPy matrices in the full Sz = 0 basis,
    keeping h, s, U symbolic (built by examples/benzene_hubbard_pt.py).
    We set s = 0 (orthogonal AOs), h = -t and evaluate at the requested U.
    """
    if not _cache:
        cache = '/tmp/benzene_hubbard_matrices.pkl'
        with open(cache, 'rb') as f:
            H1, S, H2 = pickle.load(f)
        h_s, s_s, U_s = sp.symbols('h s U')
        # At s = 0 the overlap is the identity and H = H1 + H2.
        H1_0 = sp.Matrix(H1).subs(s_s, 0)
        H2_0 = sp.Matrix(H2).subs(s_s, 0)
        _cache['H1'] = np.array(H1_0.subs(h_s, 1).tolist(), dtype=float)
        # H1 is linear in h; H2 is linear in U at s = 0.
        _cache['H2_unit_U'] = np.array(H2_0.subs(U_s, 1).tolist(), dtype=float)

    H = (-t) * _cache['H1'] + U * _cache['H2_unit_U']

    from scipy.linalg import eigh
    return float(eigh(H, eigvals_only=True)[0])


def main():
    print(f"{'U/t':>6} {'E_vbt3/|t|':>15} {'E_PySCF/|t|':>15} "
          f"{'|Δ| (a.u.)':>13} {'rel. err.':>12}")
    print('-' * 68)
    t = 1.0
    for U_over_t in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        U = U_over_t * t
        E_vbt3 = vbt3_benzene_hubbard(-t, U)  # h = -t convention
        E_pyscf = pyscf_benzene_hubbard(t, U)
        diff = abs(E_vbt3 - E_pyscf)
        rel = diff / max(abs(E_pyscf), 1e-12)
        print(f"{U_over_t:6.2f} {E_vbt3:15.10f} {E_pyscf:15.10f} "
              f"{diff:13.2e} {rel:12.2e}")


if __name__ == '__main__':
    main()
