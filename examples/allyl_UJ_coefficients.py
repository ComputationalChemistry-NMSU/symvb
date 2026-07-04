"""Derive the first-order PT coefficient alpha_M for the allyl Huckel
closed-shell reference |psi_1^2 psi_2^2> symbolically.

alpha_M = <Psi_Huckel | dH/dM | Psi_Huckel>

Build |Psi_Huckel> as an exact sympy vector in the 9-dim det basis, then
contract with the symbolic H linearly.
"""
import os, sys
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule
from symvb.fixed_psi import generate_dets
from symvb.huckel import solve
from symvb.mo_projection import mo_determinant_in_ao

m = Molecule(
    zero_ii=True, interacting_orbs=['ab', 'bc'],
    subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
    subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
              'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)
P = generate_dets(2, 2, 3)
det_strings = [p.dets[0].det_string for p in P]
H1 = m.build_matrix(P, op='H')
H2 = m.o2_matrix(P)
H = sp.Matrix(H1 + H2)
h, s, U, J, K, M = sp.symbols('h s U J K M')
H = H.subs({s: 0, h: -1})

# Build |Psi_Huckel> = |psi_1^2 psi_2^2> in the 9-det AO basis. mo_determinant_in_ao
# expands the closed-shell MO determinant (doubly-occupied MOs 0 and 1 of the
# 3-chain Huckel solution) into AO determinants with the correct fermion signs.
r2 = sp.sqrt(2)
hr = solve(sp.Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), site_labels='abc')
psi_H = mo_determinant_in_ao(hr.coefficients, ([0, 1], [0, 1]),
                             det_strings, site_labels='abc')
psi_H = psi_H / sp.sqrt((psi_H.T * psi_H)[0, 0])   # normalise to <Psi_H|Psi_H> = 1

# Sanity: norm
norm2 = sp.simplify((psi_H.T * psi_H)[0, 0])
print(f"<Psi_H | Psi_H> = {norm2}")

# Expectation value E(U, J, K, M, h=-1, s=0)
E_exp = sp.simplify((psi_H.T * H * psi_H)[0, 0])
print(f"\n<Psi_H | H | Psi_H> = {E_exp}")

# Coefficients of U, J, K, M, and constant
E_poly = sp.Poly(sp.expand(E_exp), U, J, K, M)
const_part = E_poly.nth(0, 0, 0, 0)
alpha_U = E_poly.nth(1, 0, 0, 0)
alpha_J = E_poly.nth(0, 1, 0, 0)
alpha_K = E_poly.nth(0, 0, 1, 0)
alpha_M = E_poly.nth(0, 0, 0, 1)
print(f"\n  const  = {sp.simplify(const_part)}    (expected -2*sqrt(2))")
print(f"  alpha_U = {sp.simplify(alpha_U)}   (expected 11/8)")
print(f"  alpha_J = {sp.simplify(alpha_J)}   (expected 37/8)")
print(f"  alpha_K = {sp.simplify(alpha_K)}   (expected -3/4)")
print(f"  alpha_M = {sp.simplify(alpha_M)}   <<-- deriving this")
print(f"  alpha_M numerical: {float(sp.simplify(alpha_M))}")
print(f"  alpha_M / sqrt(2): {sp.simplify(alpha_M / r2)}")

# Consistency check X3
print(f"\n  alpha_U + alpha_J = {sp.simplify(alpha_U + alpha_J)}  "
      f"(should be C(4,2) = 6)")
