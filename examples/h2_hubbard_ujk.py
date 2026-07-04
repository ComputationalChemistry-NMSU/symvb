"""
H2 with full PPP-type two-electron integrals (U, J, K) beyond on-site U.

Convention (matches symvb's subst_2e integral-pattern naming and
the standard MO-theory convention used throughout the manuscript):
    U = (aa|aa)          on-site repulsion         (pattern 1111)
    J = (aa|bb)          two-center direct Coulomb (pattern 1212)
    K = (ab|ab)          two-center exchange       (pattern 1122)
    M = (aa|ab), etc.    three-index integrals, set to 0 in PPP/ZDO

(The pattern strings count the four orbital indices in physicist
order, so pattern 1212 = <12|12> = (aa|bb) chemist = direct, and
pattern 1122 = <11|22> = (ab|ab) chemist = exchange.)

The 4x4 A_1 x A_1 x B_1 x triplet block-diagonalisation of the
Sz = 0 subspace now reads, at s = 0 and M = 0:

    |cov>           (A_1, singlet)  diagonal  J + K
    |ion_+>         (A_1, singlet)  diagonal  U + K
    |ion_->         (B_1, singlet)  diagonal  U - K
    |trip>          (triplet, S=1)  diagonal  J - K

with |cov>-|ion_+> coupled by 2h, and the other two blocks trivial.

Closed-form eigenvalues:

    E(A_1, ground)   = (U + J + 2K)/2  -  sqrt( (U - J)^2 / 4  +  4 h^2 )
    E(A_1, excited)  = (U + J + 2K)/2  +  sqrt( (U - J)^2 / 4  +  4 h^2 )
    E(ion_-)         = U - K
    E(trip)          = J - K

Ground state identification:
    - Conventional H2 chemistry (small K):  A_1 ground is lowest.
    - Singlet-triplet crossing:  E(trip) = E(A_1 ground) when
          K * (U + 2K - J) = 2 h^2
      (for h^2 = t^2).  Beyond this, the triplet dominates -- a
      Hund's-coupling / ferromagnetic instability that the full
      3-parameter model captures naturally.

Run:  PYTHONPATH=. python3 examples/h2_hubbard_ujk.py
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


m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab'],
    subst={'h': ('H_ab',), 's': ('S_ab',)},
    subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
              'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)

P = generate_dets(1, 1, 2)
print("Basis:", [p.dets[0].det_string for p in P])

# one call folds the two-electron block into H (S is unused here: the
# analysis lives on the s = 0, M = 0 block where S = I)
H = sp.Matrix(hamiltonian(m, P)[0])

h, s, U, J, K, M = sp.symbols('h s U J K M')

# Orthogonal-AO limit with M = 0 (PPP / zero differential overlap)
Hs = sp.simplify(H.subs({s: 0, M: 0}))
print("\nFull 4x4 H at s=0, M=0, general (U,J,K):")
sp.pprint(Hs)

# Symmetry-adapted basis  {cov, ion+, ion-, trip}
T = sp.Matrix([[0, 1, 1, 0],       # |cov>
               [1, 0, 0, 1],       # |ion+>
               [1, 0, 0, -1],      # |ion->
               [0, 1, -1, 0]]) / sp.sqrt(2)
H_sym = sp.simplify(T * Hs * T.T)
print("\nH in {|cov>, |ion+>, |ion->, |trip>} basis:")
sp.pprint(H_sym)

# Closed-form eigenvalues
E_gs = (U + J + 2 * K) / 2 - sp.sqrt(((U - J) / 2) ** 2 + 4 * h ** 2)
E_ex = (U + J + 2 * K) / 2 + sp.sqrt(((U - J) / 2) ** 2 + 4 * h ** 2)
E_im = U - K
E_t  = J - K

print("\nClosed-form eigenvalues:")
print(f"  E(A_1, ground)  = {E_gs}")
print(f"  E(A_1, excited) = {E_ex}")
print(f"  E(ion_-)        = {E_im}")
print(f"  E(trip)         = {E_t}")

# Numerical verification
print("\nVerification against direct diagonalisation  (the GROUND state is")
print("the minimum of the four closed-form branches above):")
print(f"{'U':>5} {'J':>5} {'K':>5}  {'E_num':>10}  "
      f"{'E_gs':>10}  {'E_ion-':>10}  {'E_trip':>10}  ground")
for Uv, Jv, Kv in [(0, 0, 0), (1, 0, 0), (2, 1, 0.3), (5, 2, 1), (10, 1, 5)]:
    subs = {h: -1, U: Uv, J: Jv, K: Kv}
    Hn = np.array(Hs.subs(subs).tolist(), dtype=float)
    E_num = eigh(Hn)[0][0]
    Es = [float(e.subs(subs)) for e in [E_gs, E_im, E_t]]
    tag = ['A_1_gs', 'ion-', 'triplet'][int(np.argmin(Es))]
    print(f"{Uv:5} {Jv:5} {Kv:5}  {E_num:+10.4f}  "
          f"{Es[0]:+10.4f}  {Es[1]:+10.4f}  {Es[2]:+10.4f}  {tag}")

print("\nSinglet-triplet crossing condition:  K (U + 2K - J) = 2 h^2")
print("At h = -1:                             K (U + 2K - J) = 2")
