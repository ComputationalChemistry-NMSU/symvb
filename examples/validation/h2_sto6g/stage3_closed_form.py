"""Stage 3: closed-form validation of the symbolic solution path (2c2e).

Stages 1-2 validate the matrix construction: integrals are substituted
into the symbolic matrices and the eigenproblem is solved numerically.
This stage validates the other half of the workflow, the closed-form
route: the 2x2 secular equation over {covalent, symmetric-ionic} is
solved SYMBOLICALLY, with all seven integrals kept symbolic (e = h_aa,
h = h_ab, s, and the four two-electron integrals U, CB, EX, M), giving
the ground-state energy and Chirgwin-Coulson weights as explicit
closed-form expressions. The integral values are then substituted into
those expressions and compared against the FCI reference and the XMVB
weights.

The symmetric-ionic combination aA + bB spans, together with the
covalent structure, the A1 block that contains the ground state; its
Chirgwin-Coulson weight splits equally over the two ionic structures of
the 3-structure basis, so w_ion_sym/2 compares against each XMVB ionic
weight.

Run from this directory: PYTHONPATH=../../.. python3 stage3_closed_form.py
"""
import json
import re

import sympy as sp

from symvb import FixedPsi, Molecule, System

ints = json.load(open('integrals.json'))

m = Molecule(
    zero_ii=False,
    interacting_orbs=['ab'],
    subst={'e': ('H_aa', 'H_bb'), 'h': ('H_ab',), 's': ('S_ab',)},
    subst_2e={'U': ('1111',), 'CB': ('1212',), 'EX': ('1122',),
              'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)
cov = FixedPsi('aB'); cov.add_str_det('bA', coef=1)
ion = FixedPsi('aA'); ion.add_str_det('bB', coef=1)
H, S = System.from_structures(m, [cov, ion]).hamiltonian()
H, S = sp.Matrix(H), sp.Matrix(S)

# ---- closed-form solution, all integrals symbolic ----
E = sp.Symbol('E')
roots = sp.solve(sp.expand((H - E * S).det()), E)
assert len(roots) == 2

sub = {sp.Symbol('e'): ints['h_aa'], sp.Symbol('h'): ints['h_ab'],
       sp.Symbol('s'): ints['s_ab'], sp.Symbol('U'): ints['U'],
       sp.Symbol('CB'): ints['coul'], sp.Symbol('EX'): ints['exch'],
       sp.Symbol('M'): ints['M']}
E0 = min(roots, key=lambda r: float(r.subs(sub)))      # ground root, still symbolic
n_ops = sp.count_ops(E0)

# closed-form eigenvector ratio and Chirgwin-Coulson weights
r = sp.simplify(-(H[0, 1] - E0 * S[0, 1]) / (H[1, 1] - E0 * S[1, 1]))
c = sp.Matrix([1, 1 / r]) if False else sp.Matrix([r, 1])   # c = (c_cov, c_ion)
c = sp.Matrix([-(H[0, 1] - E0 * S[0, 1]), H[0, 0] - E0 * S[0, 0]])
norm = (c.T * S * c)[0]
w = [(c[i] * sum(S[i, j] * c[j] for j in range(2)) / norm) for i in range(2)]

e_total = float(E0.subs(sub)) + ints['e_nuc']
w_num = [float(wi.subs(sub)) for wi in w]

xmo = open('job.xmo').read()
w_xmvb = [float(x) for x in re.findall(
    r'^\s+\d+\s+(-?\d\.\d+)', xmo.split('WEIGHTS OF STRUCTURES')[1], re.M)[:3]]

print(f'closed-form ground-state energy: {n_ops} sympy operations in 7 symbols')
print(f'E(closed form + integrals) = {e_total:.8f} Ha   '
      f'diff vs FCI = {e_total - ints["e_fci"]:+.2e}')
print(f'w_cov(closed form)     = {w_num[0]:.8f}   XMVB: {w_xmvb[0]:.8f}')
print(f'w_ion_sym/2 (closed)   = {w_num[1] / 2:.8f}   XMVB: {w_xmvb[1]:.8f}')

assert abs(e_total - ints['e_fci']) < 1e-9
assert abs(w_num[0] - w_xmvb[0]) < 5e-7
assert abs(w_num[1] / 2 - w_xmvb[1]) < 5e-7
assert abs(sum(w_num) - 1) < 1e-10
print('ALL CHECKS PASSED: substitution into the closed-form expressions '
      'reproduces the FCI energy and the XMVB weights')
