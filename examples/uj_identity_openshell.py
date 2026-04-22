"""Test Eq. (X2) of the manuscript on open-shell references.

The operator identity  H(U=J, K=M=0, s=0) = H_Huckel + C(N,2)*U*I  was
proved in §4.6.2.4 without reference to a particular electronic state --
it is a statement about the two-electron operator on any fixed-N sector.
Consequences should therefore hold for radical, biradical, and triplet
references as well.  We test three open-shell systems:

  (a) H2+ cation       : 2 orbitals, 1 electron           (N = 1)
  (b) allyl cation     : 3 orbitals, 3 electrons          (N = 3)
  (c) allyl triplet    : 3 orbitals, 4 electrons, Sz = 1  (N = 4)

For each we confirm:
  * Eq. (X1):  H_2e = C(N,2) * U * I  on the N-electron sector
  * Eq. (X2):  every Huckel eigenvector stays an eigenvector; every
               eigenvalue shifts by C(N,2)*U; level spacings preserved
  * Eq. (X3):  the first-order coefficients satisfy alpha_U + alpha_J = C(N,2)
               (but the individual values depend on the reference).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import sympy as sp

from vbt3 import Molecule
from vbt3.fixed_psi import generate_dets


def build_full(Na, Nb, Norb, interacting_orbs, h_pairs, zero_ii=True):
    """Build H1+H2, S for given fragment at s=0 with U=J, K=M=0.

    Returns numerical (H_1e_np, H_2e_np, H_at_UJ1_np), basis size, hucke_evals.
    """
    # substitute: assign all intra-pairs the same symbol h, overlap s
    subst_1e = {'h': tuple('H_' + p for p in h_pairs),
                's': tuple('S_' + p for p in h_pairs)}
    subst_2e = {'U': ('1111',), 'J': ('1212',),
                'K': ('1122',), 'M': ('1112', '1121', '1222')}
    m = Molecule(
        zero_ii=zero_ii,
        interacting_orbs=h_pairs,
        subst=subst_1e,
        subst_2e=subst_2e,
        max_2e_centers=2,
    )
    P = generate_dets(Na, Nb, Norb)
    Ndet = len(P)

    H1 = m.build_matrix(P, op='H')
    H2 = m.o2_matrix(P)
    S  = m.build_matrix(P, op='S')

    h, s, U, J, K, M = sp.symbols('h s U J K M')
    H1s = sp.Matrix(H1).subs({s: 0, h: -1})
    H2s = sp.Matrix(H2).subs({s: 0, K: 0, M: 0})     # set K, M to zero
    Ss  = sp.Matrix(S).subs({s: 0, h: -1})
    assert Ss == sp.eye(Ndet), "overlap should be identity at s=0"

    # H_2e at U=J=1: should equal C(N,2) * I on the N-sector (Eq. X1)
    H_2e_at_UJ1 = np.array(H2s.subs({U: 1, J: 1}), dtype=float)

    # First-order coefficients: <HF|V_U|HF>, <HF|V_J|HF> (see test c)
    # We compute them by evaluating <gs|H_2e|gs> at (U=1, J=0) and (U=0, J=1)
    H_2e_U = np.array(H2s.subs({U: 1, J: 0}), dtype=float)
    H_2e_J = np.array(H2s.subs({U: 0, J: 1}), dtype=float)

    H1_num = np.array(H1s, dtype=float)
    return H1_num, H_2e_U, H_2e_J, H_2e_at_UJ1, Ndet


def N_electron_count(Na, Nb):
    return Na + Nb


def Cn2(N):
    return N * (N - 1) // 2


# ---------------------------------------------------------------------
# Test (a):  H2+  (1 alpha, 0 beta, 2 orbs)
# ---------------------------------------------------------------------
print("=" * 70)
print("(a) H2+ cation  (N = 1, no 2e interaction possible)")
print("=" * 70)
H1, HU, HJ, HUJ1, N = build_full(1, 0, 2, ['ab'], ['ab'])
N_el = 1
print(f"  basis size = {N}")
print(f"  C(1, 2)    = {Cn2(N_el)}")
print(f"  max|H_2e(U=J=1)| = {np.max(np.abs(HUJ1)):.3e}  "
      f"(expect 0 -- no 2e interaction for 1 electron)")
evals_h = np.linalg.eigvalsh(H1)
print(f"  Huckel eigenvalues (h=-1): {evals_h}")
# Test that H(U=J) just shifts by C(N,2)*U = 0
shifted = H1 + HUJ1          # H at U=J=1
evals_full = np.linalg.eigvalsh(shifted)
print(f"  Eigenvalues at U=J=1    : {evals_full}  "
      f"(should equal Huckel + {Cn2(N_el)}*1 = shift by 0)")
assert np.allclose(evals_full, evals_h, atol=1e-12)
print("  Eq. (X2) holds trivially at N = 1.")


# ---------------------------------------------------------------------
# Test (b):  allyl cation  (2 alpha, 1 beta, 3 orbs, Sz = +1/2)
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("(b) Allyl cation  (N = 3, 3 electrons in 3 orbitals)")
print("=" * 70)
H1, HU, HJ, HUJ1, N = build_full(2, 1, 3, ['ab', 'bc'], ['ab', 'bc'])
N_el = 3
print(f"  basis size = {N}")
print(f"  C(3, 2)    = {Cn2(N_el)}")
# Check H_2e(U=J=1) = C(N,2) * I
residual = HUJ1 - Cn2(N_el) * np.eye(N)
print(f"  max|H_2e(U=J=1) - C(N,2)*I| = {np.max(np.abs(residual)):.3e}  "
      f"(Eq. X1)")
assert np.max(np.abs(residual)) < 1e-12
# Eigenvalues
evals_h = np.linalg.eigvalsh(H1)
evals_full = np.linalg.eigvalsh(H1 + HUJ1)
print(f"  shift = {evals_full - evals_h}   (all should equal {Cn2(N_el)})")
assert np.allclose(evals_full - evals_h, Cn2(N_el), atol=1e-12)
print("  Eq. (X2) holds on the 3-electron cation sector.")

# First-order coefficients on the Huckel cation GS (2 electrons in psi_1, 1 in psi_2)
evals_h, V_h = np.linalg.eigh(H1)
psi_gs = V_h[:, 0]  # lowest Huckel eigenvector IN THE DET BASIS -- but we need the HF reference det expansion
# Better: use the closed-shell Hartree-Fock-like reference |psi_1^2 psi_2^1> as a
# DET-basis vector. We construct it by using spin-orbital occupation numbers.
# For allyl cation: psi_1 = (a + sqrt(2) b + c)/2, psi_2 = (a - c)/sqrt(2).
# The HF det is |psi_1 alpha> |psi_1 beta> |psi_2 alpha>. In the AO basis this is
# a linear combination of the 9 cation dets (2 alpha, 1 beta in 3 orbs = C(3,2)*C(3,1)=9).
# To get alpha_U, alpha_J we can use the eigenvector of H1 with the lowest eigenvalue
# as the HF reference in det basis (for a one-electron Hamiltonian, that's exactly HF).
psi_HF = V_h[:, 0]
aU = psi_HF @ HU @ psi_HF
aJ = psi_HF @ HJ @ psi_HF
print(f"  alpha_U = <HF|V_U|HF> = {aU:.6f}")
print(f"  alpha_J = <HF|V_J|HF> = {aJ:.6f}")
print(f"  sum     = {aU + aJ:.6f}    (expect C(3,2) = 3)")
assert abs(aU + aJ - Cn2(N_el)) < 1e-10
print("  -> Eq. (X3) holds on the CATION Huckel reference (open-shell cation!)")


# ---------------------------------------------------------------------
# Test (c):  allyl triplet  (3 alpha, 1 beta, 3 orbs, Sz = +1)
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("(c) Allyl triplet Ms = +1  (N = 4, Sz = +1 sector)")
print("=" * 70)
H1, HU, HJ, HUJ1, N = build_full(3, 1, 3, ['ab', 'bc'], ['ab', 'bc'])
N_el = 4
print(f"  basis size = {N}   (C(3,3)*C(3,1) = 3)")
print(f"  C(4, 2)    = {Cn2(N_el)}")
residual = HUJ1 - Cn2(N_el) * np.eye(N)
print(f"  max|H_2e(U=J=1) - C(N,2)*I| = {np.max(np.abs(residual)):.3e}  "
      f"(Eq. X1)")
assert np.max(np.abs(residual)) < 1e-12
evals_h = np.linalg.eigvalsh(H1)
evals_full = np.linalg.eigvalsh(H1 + HUJ1)
print(f"  Huckel eigenvalues        : {evals_h}")
print(f"  at U=J=1 eigenvalues      : {evals_full}")
print(f"  shifts                    : {evals_full - evals_h}   (expect {Cn2(N_el)})")
assert np.allclose(evals_full - evals_h, Cn2(N_el), atol=1e-12)
print("  Eq. (X2) holds on the 4-electron Ms=+1 triplet sector.")

# First-order coefficients on the triplet Huckel GS
evals_h, V_h = np.linalg.eigh(H1)
psi_HF = V_h[:, 0]
aU = psi_HF @ HU @ psi_HF
aJ = psi_HF @ HJ @ psi_HF
print(f"  alpha_U = <T|V_U|T> = {aU:.6f}")
print(f"  alpha_J = <T|V_J|T> = {aJ:.6f}")
print(f"  sum     = {aU + aJ:.6f}    (expect C(4,2) = 6)")
assert abs(aU + aJ - Cn2(N_el)) < 1e-10
print("  -> Eq. (X3) holds on the TRIPLET reference (different alpha_U, alpha_J split)")


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("Summary: Eq. (X2) is reference-independent")
print("=" * 70)
print("""
  The operator identity  H(U=J, K=M=0, s=0) = H_Huckel + C(N,2)*U*I
  is an identity on the N-electron sector of Fock space, so it
  applies to every reference state (closed-shell, open-shell radical,
  biradical, triplet).  What changes between references is only the
  SPLIT between alpha_U and alpha_J in the first-order PT coefficient:

    H2+ cation     (N=1) : alpha_U = alpha_J = 0           (no 2e; C(1,2) = 0)
    allyl cation   (N=3) : alpha_U + alpha_J = 3           (C(3,2))
    allyl triplet  (N=4) : alpha_U + alpha_J = 6           (C(4,2))

  This generalises §4.6.2.4 to open-shell systems, and sharpens the
  original claim: the (U-J) reduction of §4.4 is shadow of the scalar
  identity on Fock space, not a feature of the closed-shell Huckel
  reference.  Open-shell radical, biradical, and triplet references
  inherit the same scalar shift; their effective-parameter
  combinations of (U, J, K, M) will differ only in the K- and M-
  dependent cross-terms that lie outside the identity's reach.
""")
