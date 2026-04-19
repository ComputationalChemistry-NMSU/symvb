"""Derive the first-order PT coefficient alpha_M for the allyl Huckel
closed-shell reference |psi_1^2 psi_2^2> symbolically.

alpha_M = <Psi_Huckel | dH/dM | Psi_Huckel>

Build |Psi_Huckel> as an exact sympy vector in the 9-dim det basis, then
contract with the symbolic H linearly.
"""
import os, sys
import sympy as sp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from vbt3 import Molecule, SlaterDet, symmetry
from vbt3.fixed_psi import generate_dets

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

# Build |Psi_Huckel> in the 9-dim basis symbolically
# Use the canonical-basis Huckel expansion and vbt3 conversion signs.
def canon_idx(ds):
    out = []
    for c in ds:
        orb = 'abc'.index(c.lower())
        spin = 0 if c.islower() else 1
        out.append(2*orb + spin)
    return out
def vbt3_sign(ds):
    idx = canon_idx(ds); inv = 0; n = len(idx)
    for i in range(n):
        for j in range(i+1, n):
            if idx[i] > idx[j]: inv += 1
    return (-1)**inv
def sm_to_canon(alpha_occ, beta_occ):
    idx = [2*'abc'.index(x) for x in alpha_occ] + \
          [2*'abc'.index(x)+1 for x in beta_occ]
    inv = 0; n = len(idx)
    for i in range(n):
        for j in range(i+1, n):
            if idx[i] > idx[j]: inv += 1
    return (-1)**inv

# Huckel MOs (symbolic)
# psi_1 = (1/2) a + (sqrt(2)/2) b + (1/2) c
# psi_2 = (1/sqrt(2)) a + 0 b + (-1/sqrt(2)) c
r2 = sp.sqrt(2)
C1 = {'a': sp.Rational(1, 2), 'b': r2/2, 'c': sp.Rational(1, 2)}
C2 = {'a': 1/r2, 'b': sp.Integer(0), 'c': -1/r2}

psi_H = sp.zeros(9, 1)
for i, ds in enumerate(det_strings):
    a_occ = sorted([c for c in ds if c.islower()])
    b_occ = sorted([c.lower() for c in ds if c.isupper()])
    p1, p2 = a_occ; q1, q2 = b_occ
    a_det = C1[p1]*C2[p2] - C1[p2]*C2[p1]
    b_det = C1[q1]*C2[q2] - C1[q2]*C2[q1]
    sign_sm = sm_to_canon(a_occ, b_occ)
    psi_H[i, 0] = sp.simplify(vbt3_sign(ds) * sign_sm * a_det * b_det)

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
