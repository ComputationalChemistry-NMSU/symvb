"""The one-electron bond (H2+), and why it is half the two-electron bond.

Textbook statement (Shaik, S.; Hiberty, P. C. A Chemist's Guide to Valence
Bond Theory; Wiley: Hoboken, NJ, 2008, Chapter 3, Basic VB Theory):

    One electron shared by two atomic orbitals a and b gives a bonding and an
    antibonding level,

        E_+ = (h_aa + h_ab) / (1 + s)      (bonding)
        E_- = (h_aa - h_ab) / (1 - s)      (antibonding)

    The one-electron bond energy, relative to the electron on a single atom
    (energy h_aa), is

        E_+ - h_aa = (h_ab - s h_aa) / (1 + s) = beta / (1 + s),

    with the reduced resonance integral beta = h_ab - s h_aa. Placing two
    electrons in the same bonding orbital doubles this, so at the independent-
    electron level the one-electron bond is exactly half the two-electron
    bond. (In reality electron repulsion, absent here, makes it somewhat more
    than half.)

symvb construction: one electron in two orbitals at the one-electron
(Hueckel) level (two_electron=False). The 2x2 generalized eigenproblem is
solved in closed form; the site energy h_aa is kept symbolic (zero_ii=False,
unified to eps) so beta = h_ab - s h_aa emerges on its own.

Run from the repo root:
    PYTHONPATH=. python3 examples/qualitative_vb/02_one_electron_bond.py
"""
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from symvb import Molecule, hamiltonian
from symvb.fixed_psi import generate_dets

# ------------------------------------------------------------------ build
m = Molecule(zero_ii=False, interacting_orbs=['ab'],
             subst={'h': ('H_ab',), 'eps': ('H_aa', 'H_bb'), 's': ('S_ab',)})
P = generate_dets(1, 0, 2)                      # one electron in two orbitals: 'a', 'b'
dets = [p.dets[0].det_string for p in P]
H, S = hamiltonian(m, P, two_electron=False)
H, S = sp.Matrix(H), sp.Matrix(S)
syms = {str(x): x for x in (H.free_symbols | S.free_symbols)}
h, s, eps = syms['h'], syms['s'], syms['eps']
beta = h - eps * s
print('build: 2-det one-electron H(h, eps), S(s) via symvb  OK')
print('  dets:', dets)
assert H == sp.Matrix([[eps, h], [h, eps]])
assert S == sp.Matrix([[1, s], [s, 1]])

# ------------------------------------------------------- exact 2x2 roots
E = sp.Symbol('E')
roots = sp.solve((H - E * S).det(), E)
E_bond = min(roots, key=lambda r: float(r.subs({h: -1, eps: 0, s: sp.Rational(1, 5)})))
E_anti = max(roots, key=lambda r: float(r.subs({h: -1, eps: 0, s: sp.Rational(1, 5)})))
assert sp.simplify(E_bond - (eps + h) / (1 + s)) == 0
assert sp.simplify(E_anti - (eps - h) / (1 - s)) == 0
print('E_+ = (h_aa + h_ab)/(1 + s)  bonding;  E_- = (h_aa - h_ab)/(1 - s)  antibonding  OK')

# ------------------------------------------------- one-electron bond energy
bond_1e = sp.simplify(E_bond - eps)             # relative to the electron on one atom
assert sp.simplify(bond_1e - beta / (1 + s)) == 0
anti_1e = sp.simplify(E_anti - eps)
assert sp.simplify(anti_1e - (-beta / (1 - s))) == 0
print('one-electron bond energy E_+ - h_aa = beta/(1 + s),  beta = h_ab - s h_aa  OK')

# ---------------------------------------- one-electron bond is half the 2e bond
# Two electrons in the same bonding orbital, at this independent-electron level,
# give 2 E_+; relative to the two separated atoms (2 h_aa) that is 2 beta/(1 + s).
bond_2e_mo = sp.simplify(2 * E_bond - 2 * eps)
assert sp.simplify(bond_2e_mo - 2 * beta / (1 + s)) == 0
assert sp.simplify(bond_1e / bond_2e_mo - sp.Rational(1, 2)) == 0
print('two electrons in the bonding MO give 2 beta/(1 + s): the 1e bond is exactly half  OK')

# ------------------------------------------------------- numeric spot table
# h = -1 (|h| units), h_aa = 0 so beta = -1. Compare the one-electron bond with
# the independent-electron two-electron bond (twice as deep at every s).
b1 = sp.lambdify(s, -bond_1e.subs({h: -1, eps: 0}), 'numpy')       # binding energies (positive)
b2 = sp.lambdify(s, -bond_2e_mo.subs({h: -1, eps: 0}), 'numpy')
print('\n  h = -1, h_aa = 0  (beta = -1):')
print('    {:>5} | {:>12} | {:>12} | {:>7}'.format('s', 'D(1e)', 'D(2e, MO)', 'ratio'))
print('    ' + '-' * 44)
for sv in (0.0, 0.1, 0.2, 0.3, 0.4):
    d1, d2 = float(b1(sv)), float(b2(sv))
    print('    {:>5.2f} | {:>12.4f} | {:>12.4f} | {:>7.3f}'.format(sv, d1, d2, d1 / d2))
    assert abs(d1 / d2 - 0.5) < 1e-12

print('\nall assertions passed')
