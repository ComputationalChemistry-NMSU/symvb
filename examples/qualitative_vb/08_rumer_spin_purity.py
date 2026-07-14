"""Spin purity of every Rumer structure, and the mis-signed-coupling pitfall.

Textbook statement (Shaik, S.; Hiberty, P. C. A Chemist's Guide to Valence
Bond Theory; Wiley: Hoboken, NJ, 2008, Chapter 3, Basic VB Theory):

    A Rumer (bond-diagram) structure is a spin eigenfunction: each singlet-
    coupled pair contributes S = 0, and an unpaired electron contributes its
    doublet spin. This is what makes VB structures physically meaningful. The
    coupling sign matters: the antisymmetric combination of the same two
    determinants is a triplet, not a singlet, so a mis-signed pair coupling
    produces a spin-contaminated structure that is not an eigenfunction at all.

symvb note (a documented pitfall of this package): under symvb's canonical
determinant ordering the Heitler-London singlet sign can flip relative to the
naive |a-up b-down> - |a-down b-up| formula, so one must never assume which
combination is the singlet. The check here is the safeguard behind scripts
05-07: at s = 0 (orthogonal atomic orbitals, where S^2 is defined) we build the
S^2 matrix over the determinant span of each structure and confirm it is an
eigenstate with the expected eigenvalue S(S+1). We then flip one coupling sign
and show the structure stops being a spin eigenstate.

Run from the repo root:
    PYTHONPATH=. python3 examples/qualitative_vb/08_rumer_spin_purity.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from symvb import FixedPsi
from symvb import operators as op
from symvb.system import structure_vector
from symvb.operators import canonicalize
from symvb.spin import s_squared_matrix


def spin(structure, orbs):
    """Return (<S^2>, residual) for a FixedPsi structure.

    The determinant basis is the canonical span of the structure together with
    the determinants that S^2 produces from it (S^2|psi> via the operator), so
    it is complete enough to represent any spin leakage. residual == 0 (to
    numerical precision) iff the structure is an S^2 eigenstate; a non-zero
    residual means S^2 sends part of the state outside its own span, i.e. it is
    spin-contaminated.
    """
    image = op.s_squared(list(orbs)).apply(structure)     # S^2 |structure>
    basis, seen = [], set()
    for src in (structure, image):
        for d in src.dets:
            ck = canonicalize(d.det_string)[0]
            if ck not in seen:
                seen.add(ck)
                basis.append(ck)
    S2 = s_squared_matrix(basis, orbs=orbs)
    v = np.array(structure_vector(structure, basis), dtype=float).ravel()
    val = float(v @ S2 @ v / (v @ v))
    resid = float(np.linalg.norm(S2 @ v - val * v))
    return val, resid


def mis_signed(structure):
    """A copy of `structure` with the sign of its second determinant flipped
    (breaks the singlet/doublet coupling)."""
    bad = FixedPsi()
    for i, (d, c) in enumerate(structure):
        bad.add_str_det(d.det_string, coef=(-c if i == 1 else c))
    return bad


# ------------------------------------------------------------------ cases
# (label, structure, orbitals, expected S(S+1))  --  the structures used in 05-07.
CASES = [
    ('allyl   R1  (a-b)|c',       FixedPsi('aBc', coupled_pairs=[(0, 1)]),          'abc',    0.75),
    ('allyl   R2  a|(b-c)',       FixedPsi('aBc', coupled_pairs=[(1, 2)]),          'abc',    0.75),
    ('butadiene K (a-b)(c-d)',    FixedPsi('aBcD', coupled_pairs=[(0, 1), (2, 3)]), 'abcd',   0.0),
    ('butadiene D (a-d)(b-c)',    FixedPsi('aBcD', coupled_pairs=[(0, 3), (1, 2)]), 'abcd',   0.0),
    ('benzene Kek1',              FixedPsi('aBcDeF', coupled_pairs=[(0, 1), (2, 3), (4, 5)]), 'abcdef', 0.0),
    ('benzene Kek2',              FixedPsi('aBcDeF', coupled_pairs=[(0, 5), (1, 2), (3, 4)]), 'abcdef', 0.0),
    ('benzene Dew1',              FixedPsi('aBcDeF', coupled_pairs=[(0, 1), (2, 5), (3, 4)]), 'abcdef', 0.0),
    ('benzene Dew2',              FixedPsi('aBcDeF', coupled_pairs=[(0, 3), (1, 2), (4, 5)]), 'abcdef', 0.0),
    ('benzene Dew3',              FixedPsi('aBcDeF', coupled_pairs=[(0, 5), (1, 4), (2, 3)]), 'abcdef', 0.0),
]

print('Rumer / Kekule / Dewar structures: S^2 eigenvalue check (s = 0)')
print('  {:<26} {:>7} {:>9} {:>10}  {}'.format(
    'structure', '<S^2>', 'expected', 'residual', 'eigenstate?'))
print('  ' + '-' * 72)
for label, st, orbs, expected in CASES:
    val, resid = spin(st, orbs)
    ok = (abs(val - expected) < 1e-10) and (resid < 1e-10)
    assert ok, (label, val, resid, expected)
    print('  {:<26} {:>7.4f} {:>9.4f} {:>10.1e}  {}'.format(
        label, val, expected, resid, 'yes' if ok else 'NO'))

print('\nMis-signed couplings (one sign flipped): must NOT be spin eigenstates')
print('  {:<26} {:>7} {:>10}  {}'.format('structure', '<S^2>', 'residual', 'eigenstate?'))
print('  ' + '-' * 60)
for label, st, orbs, expected in (CASES[0], CASES[2], CASES[4]):   # allyl, butadiene, benzene
    bad = mis_signed(st)
    val, resid = spin(bad, orbs)
    is_eig = resid < 1e-10
    assert not is_eig, (label, val, resid)             # contamination -> not an eigenstate
    print('  {:<26} {:>7.4f} {:>10.2e}  {}'.format(
        label.split()[0] + ' mis-signed', val, resid, 'yes' if is_eig else 'NO'))

print('\nEvery correctly signed structure is a spin eigenstate with the expected '
      'S(S+1); every mis-signed one is contaminated.')
print('all assertions passed')
