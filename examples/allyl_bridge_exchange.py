"""Overlap-only exchange through a strictly closed-shell bridge (3c4e).

Constrain the central atom b of the allyl frame to strict double occupancy.
The covalent manifold that remains is two-dimensional: the long-bond
singlet Phi_ac = [a.c]_s b^2 (manuscript Eq. 8) and its triplet partner.
Spin symmetry splits it into two one-dimensional blocks, so both energies
are exact Rayleigh quotients, and the singlet-triplet gap is an exact
closed form (no perturbation theory):

    E_S = (U + 8 h s^3 - 4 h s) / (2 s^4 - 2 s^2 + 1)
    E_T = (U - 4 h s) / (1 - 2 s^2)
    E_T - E_S = 2 s^3 [U s + 4|h|(1 - s^2)]
                / [(1 - 2 s^2)(1 - 2 s^2 + 2 s^4)]      (h = -|h|)
              = 8|h| s^3 + 2 U s^4 + O(s^5)

At s = 0 the gap vanishes identically at every U: superexchange through a
strictly doubly occupied bridge requires the charge-transfer structures
that the constraint excludes (Anderson's kinetic mechanism). At s != 0 an
antiferromagnetic coupling appears anyway, carried entirely by the overlap
cofactors (leading order 8|h| s^3 via the composite s_ab * s_bc path), with
no 1/U anywhere; U enters only at O(s^4), with positive sign (a metric
effect). The metric singularity at s^2 = 1/2 bounds the trust region.

Setup: allyl chain a-b-c, h and s on the two edges only (s_ac = 0),
on-site Hubbard U, 9-determinant S_z = 0 basis.

Run from the repo root: PYTHONPATH=. python3 examples/allyl_bridge_exchange.py
"""
import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, hamiltonian
from symvb.fixed_psi import generate_dets
from symvb.spin import s_squared_matrix

# ------------------------------------------------------------------ build
m = Molecule(zero_ii=True, interacting_orbs=['ab', 'bc'],
             subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
             subst_2e={'U': ('1111',)}, max_2e_centers=1)
P = generate_dets(2, 2, 3)
dets = [p.dets[0].det_string for p in P]
H, S = hamiltonian(m, P)          # (H includes the on-site 2e block; S is the overlap)
H, S = sp.Matrix(H), sp.Matrix(S)
syms = {str(x): x for x in (H.free_symbols | S.free_symbols)}
h, s, U = syms['h'], syms['s'], syms['U']

# the two strict-b^2 covalent determinants (b doubly, a/c singly occupied)
i1 = dets.index('aBbC')      # a(alpha) b^2 c(beta)
i2 = dets.index('bAcB')      # a(beta)  b^2 c(alpha)

# ------------------------------------------- spin identification at s = 0
S2 = np.array(s_squared_matrix(dets, orbs='abc'), dtype=float)
for sign, expect in ((+1, 0.0), (-1, 2.0)):
    v = np.zeros(len(dets)); v[i1] = 1; v[i2] = sign
    val = v @ S2 @ v / (v @ v)
    assert abs(val - expect) < 1e-12, (sign, val)
print('spin assignment: D1+D2 singlet (<S^2>=0), D1-D2 triplet (<S^2>=2): OK')

# ------------------------------------------------- exact Rayleigh quotients
energies = {}
for sign, name in ((+1, 'S'), (-1, 'T')):
    v = sp.zeros(len(dets), 1); v[i1] = 1; v[i2] = sign
    energies[name] = sp.simplify(sp.cancel((v.T * H * v)[0] / (v.T * S * v)[0]))
E_S, E_T = energies['S'], energies['T']
print(f'E_S = {E_S}')
print(f'E_T = {E_T}')

gap = sp.simplify(sp.factor(sp.together(E_T - E_S)))
gap_ref = 2*s**3*(U*s + 4*(-h)*(1 - s**2)) / ((1 - 2*s**2)*(1 - 2*s**2 + 2*s**4))
assert sp.simplify(gap - gap_ref.subs(-h, sp.Symbol('absh'))
                   .subs(sp.Symbol('absh'), -h)) == 0
print(f'E_T - E_S = {gap}')

# limits and series
assert gap.subs(s, 0) == 0
print('s = 0: gap vanishes identically at every U (no CT, no superexchange): OK')
t = sp.Symbol('t', positive=True)
ser = sp.series(gap.subs(h, -t), s, 0, 6).removeO().expand()
assert ser.coeff(s, 3) == 8*t and ser.coeff(s, 4) == 2*U
print(f'series (h = -t): {ser}   [leading 8|h|s^3; U enters at +2Us^4]: OK')

# ------------------------------------------------------- numerical check
vals = {h: -1.0, s: 0.25, U: 3.0}
Hn = np.array(H.subs(vals), dtype=float)
Sn = np.array(S.subs(vals), dtype=float)
v1 = np.zeros(9); v1[i1] = 1; v1[i2] = 1
v2 = np.zeros(9); v2[i1] = 1; v2[i2] = -1
num_gap = (v2@Hn@v2)/(v2@Sn@v2) - (v1@Hn@v1)/(v1@Sn@v1)
assert abs(num_gap - float(gap.subs(vals))) < 1e-12
print(f'numeric check at (h, s, U) = (-1, 0.25, 3): gap = {num_gap:.8f}: OK')
