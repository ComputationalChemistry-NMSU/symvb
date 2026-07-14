"""The Heitler-London covalent bond and Pauli triplet repulsion (1e level).

Textbook statement (Shaik, S.; Hiberty, P. C. A Chemist's Guide to Valence
Bond Theory; Wiley: Hoboken, NJ, 2008, Chapter 3, Basic VB Theory):

    Two singly occupied atomic orbitals a and b give a bonding covalent
    singlet and a repulsive triplet. Relative to the two separated atoms
    (energy 2 h_aa), the singlet is stabilized and the triplet is
    destabilized by an amount controlled by the reduced resonance integral
    beta = h_ab - s h_aa:

        singlet stabilization    =  2 beta s / (1 + s^2)
        triplet destabilization  = -2 beta s / (1 - s^2)

    With a bonding resonance integral (h_ab < 0) beta < 0, so the singlet
    drops (a bond) and the triplet rises (Pauli repulsion). At s = 0 the two
    coincide: at this one-electron level the singlet-triplet split is carried
    entirely by the atomic-orbital overlap.

symvb construction: a two-orbital, two-electron problem at the one-electron
(effective-Hamiltonian, Hueckel) level. We build H over the four S_z = 0
determinants with two_electron=False, so H is the bare sum of one-electron
integrals and the closed forms above are matched exactly. The site energy
h_aa is kept symbolic (zero_ii=False, unified to eps) so the reduced
resonance integral beta = h_ab - s h_aa emerges on its own. The covalent
singlet and the S_z = 0 triplet are read off by combining the two covalent
determinants aB and bA; which combination is which is decided by <S^2>, not
assumed.

Run from the repo root:
    PYTHONPATH=. python3 examples/qualitative_vb/01_hl_bond_and_triplet.py
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
# zero_ii=False turns the diagonal site energies on; subst unifies both site
# energies to eps and the single edge to (h, s). two_electron=False keeps the
# model at the one-electron (Hueckel) level: H is the sum of one-electron
# integrals only, with no electron-electron term.
m = Molecule(zero_ii=False, interacting_orbs=['ab'],
             subst={'h': ('H_ab',), 'eps': ('H_aa', 'H_bb'), 's': ('S_ab',)})
P = generate_dets(1, 1, 2)                      # S_z = 0: aA, aB, bA, bB
dets = [p.dets[0].det_string for p in P]
H, S = hamiltonian(m, P, two_electron=False)
H, S = sp.Matrix(H), sp.Matrix(S)
syms = {str(x): x for x in (H.free_symbols | S.free_symbols)}
h, s, eps = syms['h'], syms['s'], syms['eps']
beta = h - eps * s                              # reduced resonance integral
assert set(map(str, H.free_symbols)) == {'h', 's', 'eps'}
print('build: 4-det one-electron H(h, s, eps), S(s) via symvb  OK')
print('  dets:', dets)

# ---------------------------------------------------- spin identification
# The two covalent determinants (one electron per atom). Their symmetric and
# antisymmetric combinations are the covalent singlet and the S_z = 0 triplet;
# <S^2> decides which is which (0 -> singlet, 2 -> triplet), we do not assume.
i_aB, i_bA = dets.index('aB'), dets.index('bA')
S2 = s_squared_matrix(dets, orbs='ab')
label = {}
for sign in (+1, -1):
    v = np.zeros(len(dets)); v[i_aB] = 1; v[i_bA] = sign
    val = float(v @ S2 @ v / (v @ v))
    if abs(val) < 1e-12:
        label[sign] = 'singlet'
    elif abs(val - 2.0) < 1e-12:
        label[sign] = 'triplet'
    else:
        raise AssertionError('covalent combo is not a spin eigenstate')
assert label[+1] == 'singlet' and label[-1] == 'triplet'
print('spin: aB + bA is the covalent singlet (<S^2>=0); aB - bA is the triplet (<S^2>=2)  OK')


def rayleigh(sign):
    """Exact Rayleigh quotient of the covalent combination aB + sign*bA."""
    v = sp.zeros(len(dets), 1); v[i_aB] = 1; v[i_bA] = sign
    return sp.simplify((v.T * H * v)[0] / (v.T * S * v)[0])


E_S = rayleigh(+1)      # covalent singlet
E_T = rayleigh(-1)      # covalent triplet (S_z = 0 component)

# ------------------------------------------------------- closed forms
assert sp.simplify(E_S - 2 * (eps + h * s) / (1 + s ** 2)) == 0
assert sp.simplify(E_T - 2 * (eps - h * s) / (1 - s ** 2)) == 0
print('E_singlet = 2(h_aa + h_ab s)/(1 + s^2),  E_triplet = 2(h_aa - h_ab s)/(1 - s^2)  OK')

# the S_z = 1 determinant |ab> is an uncontaminated triplet: same energy as E_T
Pt = generate_dets(2, 0, 2)                     # S_z = 1: the single det 'ab'
Ht, St = hamiltonian(m, Pt, two_electron=False)
E_T_sz1 = sp.simplify(sp.Matrix(Ht)[0, 0] / sp.Matrix(St)[0, 0])
assert sp.simplify(E_T_sz1 - E_T) == 0
print('S_z = 1 determinant |ab> reproduces the triplet energy E_triplet  OK')

# stabilization / destabilization relative to the separated atoms (2 h_aa),
# expressed through the reduced resonance integral beta = h_ab - s h_aa
stab = sp.simplify(E_S - 2 * eps)
dest = sp.simplify(E_T - 2 * eps)
assert sp.simplify(stab - 2 * beta * s / (1 + s ** 2)) == 0
assert sp.simplify(dest - (-2 * beta * s / (1 - s ** 2))) == 0
print('singlet stabilization = 2 beta s/(1 + s^2);  triplet = -2 beta s/(1 - s^2),  '
      'beta = h_ab - s h_aa  OK')

# at zero overlap the split closes: both structures sit at 2 h_aa
assert stab.subs(s, 0) == 0 and dest.subs(s, 0) == 0
assert sp.simplify(E_S.subs(s, 0) - 2 * eps) == 0 and sp.simplify(E_T.subs(s, 0) - 2 * eps) == 0
print('s = 0: singlet and triplet are degenerate at 2 h_aa (overlap-driven split)  OK')

# ------------------------------------------------------- numeric spot table
# bonding resonance h = -1 (|h| units), h_aa = 0 so beta = h = -1.
D_cov = sp.lambdify(s, -stab.subs({h: -1, eps: 0}), 'numpy')     # binding energy = -stabilization
rep_T = sp.lambdify(s, dest.subs({h: -1, eps: 0}), 'numpy')      # triplet repulsion
print('\n  h = -1, h_aa = 0  (beta = -1):')
print('    {:>5} | {:>10} | {:>12}'.format('s', 'D_cov', 'E_triplet'))
print('    ' + '-' * 33)
for sv in (0.0, 0.1, 0.2, 0.3, 0.4):
    print('    {:>5.2f} | {:>10.4f} | {:>12.4f}'.format(sv, float(D_cov(sv)), float(rep_T(sv))))
assert abs(float(D_cov(0.0))) < 1e-12                       # no 1e binding at s = 0
assert float(D_cov(0.3)) > 0 and float(rep_T(0.3)) > 0     # singlet binds, triplet repels

print('\nall assertions passed')
