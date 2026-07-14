"""The two-center three-electron bond and its destabilization by overlap.

Textbook statement (Shaik, S.; Hiberty, P. C. A Chemist's Guide to Valence
Bond Theory; Wiley: Hoboken, NJ, 2008, Chapter 3, Basic VB Theory):

    Three electrons in two orbitals (as in He2+ or F2-) are described by the
    two resonance structures A(:)B(.) and A(.)B(:), a doubly occupied orbital
    next to a singly occupied one. Their in-phase combination is the bonding
    three-electron bond. Because the odd electron shares space with a closed
    shell, the bond is weakened by overlap and turns repulsive once the
    overlap is large; it is not a full electron-pair bond.

symvb construction: three electrons in two orbitals at the one-electron
(Hueckel) level (two_electron=False). The S_z = 1/2 space is exactly
two-dimensional (the determinants aAb and aBb), so every energy below is an
exact Rayleigh quotient, no perturbation theory. The site energy h_aa is kept
symbolic (zero_ii=False, unified to eps), so the reduced resonance integral
beta = h_ab - s h_aa emerges on its own. Both determinants are pure doublets
(<S^2> = 3/4); which combination is bonding is decided by energy, not assumed.

What is proved (all relative to the separated fragments, energy 3 h_aa; the
closed forms are exact in beta = h_ab - s h_aa):

    single structure A(:)B(.):   E_struct - 3 h_aa = -2 beta s / (1 - s^2)
    bonding (in-phase):          E_+     - 3 h_aa =  beta (1 - 3 s) / (1 - s^2)
    antibonding (out-of-phase):  E_-     - 3 h_aa = -beta (1 + 3 s) / (1 - s^2)
    resonance stabilization:     E_struct - E_+   = -beta / (1 + s)

With a bonding resonance integral (beta < 0): the single localized structure
is already repulsive (Pauli, the -2 beta s/(1 - s^2) term, the same form as
the Heitler-London triplet). Resonance lowers the in-phase combination by
-beta/(1 + s). The net three-electron bond energy beta(1 - 3 s)/(1 - s^2) is
deepest at s = 0 (value beta = -|h|, matching the one-electron bond there),
weakens monotonically as s grows, vanishes at s = 1/3, and is repulsive for
s > 1/3. There is no interior optimal overlap: overlap only ever weakens the
bond.

Run from the repo root:
    PYTHONPATH=. python3 examples/qualitative_vb/03_three_electron_bond.py
"""
import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from symvb import Molecule, hamiltonian
from symvb.fixed_psi import generate_dets
from symvb.spin import s_squared_matrix

# ------------------------------------------------------------------ build
m = Molecule(zero_ii=False, interacting_orbs=['ab'],
             subst={'h': ('H_ab',), 'eps': ('H_aa', 'H_bb'), 's': ('S_ab',)})
P = generate_dets(2, 1, 2)                      # S_z = 1/2, three electrons in two orbitals
dets = [p.dets[0].det_string for p in P]        # ['aAb', 'aBb'] : A(:)B(.) and A(.)B(:)
H, S = hamiltonian(m, P, two_electron=False)
H, S = sp.Matrix(H), sp.Matrix(S)
syms = {str(x): x for x in (H.free_symbols | S.free_symbols)}
h, s, eps = syms['h'], syms['s'], syms['eps']
beta = h - eps * s
print('build: 2-det one-electron H(h, s, eps), S(s) via symvb  OK')
print('  dets:', dets, ' (A(:)B(.) and A(.)B(:))')

# ---------------------------------------------------- both structures are doublets
S2 = s_squared_matrix(dets, orbs='ab')
assert abs(S2[0, 0] - 0.75) < 1e-12 and abs(S2[1, 1] - 0.75) < 1e-12
for sign in (+1, -1):
    v = np.zeros(2); v[0] = 1; v[1] = sign
    assert abs(float(v @ S2 @ v / (v @ v)) - 0.75) < 1e-12
print('both structures, and both combinations, are pure doublets (<S^2> = 3/4)  OK')

# ------------------------------------------------------- exact energies
ref = 3 * eps                                   # separated fragments: A(:) + B(.) = 3 h_aa
E_struct = sp.simplify(H[0, 0] / S[0, 0])       # a single localized structure


def rayleigh(sign):
    v = sp.zeros(2, 1); v[0] = 1; v[1] = sign
    return sp.simplify((v.T * H * v)[0] / (v.T * S * v)[0])


# choose the bonding (lower) combination by energy at a bonding reference point
probe = {h: -1, eps: 0, s: sp.Rational(1, 5)}
E_plus, E_minus = rayleigh(+1), rayleigh(-1)
if float(E_plus.subs(probe)) > float(E_minus.subs(probe)):
    E_plus, E_minus = E_minus, E_plus
print('  bonding = in-phase (aAb + aBb); antibonding = out-of-phase (aAb - aBb)  OK')

# ------------------------------------------------------- closed forms in beta
assert sp.simplify((E_struct - ref) - (-2 * beta * s / (1 - s ** 2))) == 0
assert sp.simplify((E_plus - ref) - beta * (1 - 3 * s) / (1 - s ** 2)) == 0
assert sp.simplify((E_minus - ref) - (-beta * (1 + 3 * s) / (1 - s ** 2))) == 0
print('E_struct - 3 h_aa = -2 beta s/(1 - s^2)   (Pauli, same form as the HL triplet)  OK')
print('E_bond  - 3 h_aa =  beta (1 - 3 s)/(1 - s^2);  E_anti - 3 h_aa = -beta (1 + 3 s)/(1 - s^2)  OK')

R = sp.simplify(E_struct - E_plus)              # resonance stabilization
assert sp.simplify(R - (-beta / (1 + s))) == 0
print('resonance stabilization E_struct - E_bond = -beta/(1 + s)  OK')

# ------------------------------------------------------- trend of the net bond
bond = sp.simplify(E_plus - ref)                # net three-electron bond energy
assert sp.simplify(bond.subs(s, 0) - beta.subs(s, 0)) == 0     # deepest at s = 0: beta = -|h|
assert sp.simplify(bond.subs(s, sp.Rational(1, 3))) == 0       # vanishes at s = 1/3
# strictly weakening: d/ds of (1 - 3 s)/(1 - s^2) is negative on 0 <= s < 1
shape = (1 - 3 * s) / (1 - s ** 2)
dshape = sp.simplify(sp.diff(shape, s))
assert sp.simplify(dshape * (1 - s ** 2) ** 2 - (-3 * s ** 2 + 2 * s - 3)) == 0
assert sp.Poly(-3 * s ** 2 + 2 * s - 3, s).discriminant() < 0   # numerator never zero -> one sign
assert float(dshape.subs(s, sp.Rational(1, 5))) < 0            # and that sign is negative
print('net bond beta(1 - 3 s)/(1 - s^2): deepest at s = 0 (= beta), zero at s = 1/3, '
      'repulsive beyond; monotone in s  OK')

# ------------------------------------------------------- numeric spot table
# h = -1 (|h| units), h_aa = 0 so beta = -1. Positive = binding, negative = repulsive.
bnd = sp.lambdify(s, -bond.subs({h: -1, eps: 0}), 'numpy')      # binding energy of the 3e bond
rep = sp.lambdify(s, (E_struct - ref).subs({h: -1, eps: 0}), 'numpy')   # single-structure Pauli
res = sp.lambdify(s, R.subs({h: -1, eps: 0}), 'numpy')          # resonance stabilization
print('\n  h = -1, h_aa = 0  (beta = -1):')
print('    {:>5} | {:>12} | {:>14} | {:>12}'.format('s', 'D(3e bond)', 'struct(Pauli)', 'resonance'))
print('    ' + '-' * 51)
for sv in (0.0, 0.1, 0.2, 1.0 / 3.0, 0.4, 0.5):
    print('    {:>5.3f} | {:>12.4f} | {:>14.4f} | {:>12.4f}'.format(
        sv, float(bnd(sv)), float(rep(sv)), float(res(sv))))
assert float(bnd(0.2)) > 0                       # binding below s = 1/3
assert abs(float(bnd(1.0 / 3.0))) < 1e-9         # exactly zero at s = 1/3
assert float(bnd(0.4)) < 0                       # repulsive above s = 1/3

print('\nall assertions passed')
