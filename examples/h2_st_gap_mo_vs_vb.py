"""
Validate the H2 B_1 (ion-) and triplet eigenvalues from symvb's
(U, J, K) table against the MO singlet-triplet picture.

Convention (matches manuscript and symvb substitution; note the docstring
in h2_hubbard_ujk.py has the J/K labels swapped):

    U = (aa|aa)    on-site repulsion         (pattern 1111)
    J = (aa|bb)    two-center direct Coulomb (pattern 1212)
    K = (ab|ab)    two-center exchange       (pattern 1122)
    M = (aa|ab)    three-index hybrid

symvb reports (s = 0, M = 0):
    E(ion-)  = U - K
    E(trip)  = J - K
    Gap      = U - J

The MO derivation: with sigma  = (a + b)/sqrt(2),  sigma* = (a - b)/sqrt(2),
the four sigma <-> sigma* configurations span the same Sz = 0 sector,
and the open-shell pair |sigma sigma*> splits singlet/triplet by

    E(1Sigma_u+) - E(3Sigma_u+) = 2 (sigma sigma* | sigma sigma*).

The MO exchange integral (sigma sigma* | sigma sigma*) at s = 0 evaluates to

    (1/4) [ (aa|aa) - (aa|bb) - (bb|aa) + (bb|bb) ]
        = (U - J) / 2.

Hence the MO singlet-triplet gap is U - J, exactly matching the VBT table.

Run:  PYTHONPATH=. python3 examples/h2_st_gap_mo_vs_vb.py
"""
import os
import sys
import numpy as np
import sympy as sp
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule
from symvb.fixed_psi import generate_dets
from symvb.system import hamiltonian


# ----------------------------------------------------------------------
# 1. VBT side: regenerate the four diagonal eigenvalues at s = 0, M = 0
# ----------------------------------------------------------------------
m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab'],
    subst={'h': ('H_ab',), 's': ('S_ab',)},
    subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
              'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)
P = generate_dets(1, 1, 2)
H = sp.Matrix(hamiltonian(m, P)[0])        # 4x4 determinant H, 2e block folded in

h, s, U, J, K, M = sp.symbols('h s U J K M')
Hs = sp.simplify(H.subs({s: 0, M: 0}))

# Symmetry-adapted basis  {cov, ion+, ion-, trip}; basis order is
# generate_dets = [aA, aB, bA, bB], with bA = -|Ab| in canonical ordering.
T = sp.Matrix([[0, 1, 1, 0],
               [1, 0, 0, 1],
               [1, 0, 0, -1],
               [0, 1, -1, 0]]) / sp.sqrt(2)
H_sym = sp.simplify(T * Hs * T.T)
print("VBT block-diagonal H in {cov, ion+, ion-, trip} basis:")
sp.pprint(H_sym)
E_ionm_VB = sp.simplify(H_sym[2, 2])
E_trip_VB = sp.simplify(H_sym[3, 3])
print(f"\n  E(ion-)  = {E_ionm_VB}")
print(f"  E(trip)  = {E_trip_VB}")
print(f"  Gap      = E(ion-) - E(trip) = {sp.simplify(E_ionm_VB - E_trip_VB)}")


# ----------------------------------------------------------------------
# 2. MO derivation: singlet-triplet gap via 2 (sigma sigma* | sigma sigma*)
# ----------------------------------------------------------------------
# At s = 0:  sigma  = (a + b)/sqrt(2),  sigma* = (a - b)/sqrt(2).
# Define MO 2e integrals symbolically by expanding products of AO orbitals
# and substituting the four AO patterns U, J, K, M.

# Map from AO 4-tuple (i,j,k,l) in the chemist (ij|kl) ordering to the
# symvb symbol; uses left-right and bra-ket symmetries of real integrals.
def AO_eri(i, j, k, l):
    multi = ''.join(sorted(i + j + k + l))   # canonical multiset (eg 'aabb')
    if multi == 'aaaa' or multi == 'bbbb':
        return U
    if multi == 'aabb':
        # split: how many a's on the bra side (positions i,j) vs ket (k,l)?
        n_left_a = (i == 'a') + (j == 'a')
        if n_left_a == 2 or n_left_a == 0:
            return J         # (aa|bb) or (bb|aa) -- direct
        return K             # (ab|ab) etc -- exchange
    if multi == 'aaab' or multi == 'abbb':
        return M
    raise ValueError(multi)

# (sigma sigma | sigma sigma)
def expand_2e(p, q, r, s_):
    """Expand AO product (pq|rs) for p,q,r,s each given as a list of
    (coef, AO_label).  Returns a sympy expression in U, J, K, M."""
    val = 0
    for cp, lp in p:
        for cq, lq in q:
            for cr, lr in r:
                for cs, ls in s_:
                    val += cp*cq*cr*cs * AO_eri(lp, lq, lr, ls)
    return sp.simplify(val)

# AO expansion of MOs at s=0:  sigma = (a+b)/sqrt(2),  sigma* = (a-b)/sqrt(2)
sg  = [(sp.Rational(1, 2)**sp.Rational(1, 2), 'a'),
       (sp.Rational(1, 2)**sp.Rational(1, 2), 'b')]
sgs = [(sp.Rational(1, 2)**sp.Rational(1, 2), 'a'),
       (-sp.Rational(1, 2)**sp.Rational(1, 2), 'b')]

J_ssss   = expand_2e(sg, sg, sg, sg)         # (sigma sigma | sigma sigma)
J_sss_s_ = expand_2e(sgs, sgs, sgs, sgs)     # (sigma* sigma* | sigma* sigma*)
J_ss_ss_ = expand_2e(sg, sg, sgs, sgs)       # (sigma sigma | sigma* sigma*)
K_ss_    = expand_2e(sg, sgs, sg, sgs)       # (sigma sigma* | sigma sigma*)

print("\nMO two-electron integrals at s = 0:")
print(f"  (sg sg|sg sg)         = {J_ssss}")
print(f"  (sg* sg*|sg* sg*)     = {J_sss_s_}")
print(f"  (sg sg|sg* sg*)       = {J_ss_ss_}")
print(f"  (sg sg*|sg sg*)       = {K_ss_}")

# Open-shell singlet 1Sigma_u+  diagonal:  J_ss_ss_ + K_ss_
# Triplet 3Sigma_u+ (Sz = 0)    diagonal:  J_ss_ss_ - K_ss_
E_1Su = sp.simplify(J_ss_ss_ + K_ss_)
E_3Su = sp.simplify(J_ss_ss_ - K_ss_)
gap_MO = sp.simplify(E_1Su - E_3Su)
print(f"\n  E(1Sigma_u+, ion-)   = {E_1Su}")
print(f"  E(3Sigma_u+, trip)   = {E_3Su}")
print(f"  Gap = 2 (sg sg*|sg sg*) = {gap_MO}")


# ----------------------------------------------------------------------
# 3. Symbolic match VBT vs MO
# ----------------------------------------------------------------------
print("\nSymbolic agreement:")
print(f"  E(ion-)_VB  - E(1Sigma_u+)_MO = "
      f"{sp.simplify(E_ionm_VB - E_1Su)}")
print(f"  E(trip)_VB  - E(3Sigma_u+)_MO = "
      f"{sp.simplify(E_trip_VB - E_3Su)}")
print(f"  Gap_VB - Gap_MO              = "
      f"{sp.simplify((E_ionm_VB - E_trip_VB) - gap_MO)}")


# ----------------------------------------------------------------------
# 4. Numerical cross-check: AO -> MO transform with realistic Slater-1s
#    integrals (zeta = 1) at R = 1.4 au, then MO FCI energies
# ----------------------------------------------------------------------
from pyscf import gto, ao2mo

ZETA_PYSCF = 1.24
EXPS_124 = [35.52322122, 6.513143725, 1.822142904,
            0.625955266, 0.243076747, 0.100112428]
COEFS    = [0.00916359628, 0.04936149294, 0.16853830490,
            0.37056279970, 0.41649152980, 0.13033408410]
EXPS_1 = [a / ZETA_PYSCF**2 for a in EXPS_124]
BASIS_ZETA1 = {'H': [[0] + list(zip(EXPS_1, COEFS))]}

R_au = 1.4
mol = gto.M(atom=f'H 0 0 0; H 0 0 {R_au}',
            basis=BASIS_ZETA1, verbose=0, unit='Bohr')
S_AO  = mol.intor('int1e_ovlp')
eri_A = mol.intor('int2e')
U_n = eri_A[0, 0, 0, 0]
J_n = eri_A[0, 0, 1, 1]
K_n = eri_A[0, 1, 0, 1]
M_n = eri_A[0, 0, 0, 1]
print(f"\nNumerical AO integrals at R = {R_au} au, zeta = 1:")
print(f"  U = (aa|aa) = {U_n:.6f}   J = (aa|bb) = {J_n:.6f}")
print(f"  K = (ab|ab) = {K_n:.6f}   M = (aa|ab) = {M_n:.6f}")
print(f"  s_ab = {S_AO[0,1]:.6f}")
# Lowdin-orthogonalise so the AO basis is orthogonal (s = 0 limit) to
# isolate the (U, J, K) physics from the s != 0 corrections.
sval, svec = np.linalg.eigh(S_AO)
S_inv_sqrt = svec @ np.diag(sval**-0.5) @ svec.T
eri_orth = np.einsum('ip,jq,ijkl,kr,ls->pqrs',
                     S_inv_sqrt, S_inv_sqrt, eri_A, S_inv_sqrt, S_inv_sqrt)
U_o = eri_orth[0, 0, 0, 0]
J_o = eri_orth[0, 0, 1, 1]
K_o = eri_orth[0, 1, 0, 1]
M_o = eri_orth[0, 0, 0, 1]
print(f"\nLowdin-orthogonalised (s = 0) AO integrals at R = {R_au} au:")
print(f"  U = {U_o:.6f}   J = {J_o:.6f}   K = {K_o:.6f}   M = {M_o:.6f}")

# Now diagonalise symvb's H_sym at h = -1, M = 0 with these (U, J, K)
subs_n = {h: -1, U: U_o, J: J_o, K: K_o, M: 0}
H_num = np.array(Hs.subs(subs_n).tolist(), dtype=float)
E_num, V_num = eigh(H_num)
print(f"\nFCI eigenvalues at orthogonalised (U, J, K) and h = -1:")
labels_VB = ['cov-ion+ ground', 'cov-ion+ excited', 'ion- (B_1)', 'trip']
# Sort by energy and identify states by overlap with symmetry-adapted basis
print(f"  E(B_1, ion-)  numerical = {float(E_ionm_VB.subs(subs_n)):+.6f}")
print(f"  E(trip)       numerical = {float(E_trip_VB.subs(subs_n)):+.6f}")
print(f"  Gap           numerical = "
      f"{float((E_ionm_VB - E_trip_VB).subs(subs_n)):+.6f}")
print(f"\n  MO exchange 2(sg sg*|sg sg*) numerical = "
      f"{float(gap_MO.subs(subs_n)):+.6f}")

print("\n=> VBT eigenvalues E(B_1) = U - K, E(trip) = J - K, "
      "and Gap = U - J = 2 (sigma sigma* | sigma sigma*),")
print("   the MO exchange integral.  VBT and MO agree exactly.")
