"""Allyl radical: two Rumer structures, equal mixing, covalent resonance (1e level).

Textbook statement (Shaik, S.; Hiberty, P. C. A Chemist's Guide to Valence
Bond Theory; Wiley: Hoboken, NJ, 2008, Chapter 3, Basic VB Theory):

    The pi system of the allyl radical (three orbitals a-b-c, three electrons)
    is described by two covalent Rumer structures,

        R1 = (a-b singlet-paired) with the odd electron on c,
        R2 = (b-c singlet-paired) with the odd electron on a.

    They are equivalent by symmetry, so they enter the ground state with equal
    weight, and their spin-coupling overlap is <R1|R2> = 1/2. The delocalized
    (resonating) ground state is their symmetric combination.

symvb construction: a three-orbital, three-electron chain at the one-electron
(Hueckel) level. Each Rumer structure is a FixedPsi built by singlet-coupling
one adjacent pair with couple_orbitals; which combination is the doublet is
decided by <S^2>, not assumed. H and S over the two structures are built with
two_electron=False (bare one-electron integrals), so H is the sum of resonance
integrals and the closed forms below are exact. Overlap s is kept symbolic.

Result proved here: at the one-electron level the covalent H is proportional
to s, so the resonance is carried entirely by the atomic-orbital overlap
(it vanishes for orthogonal orbitals). The structure overlap is 1/2, the two
structures mix exactly 50:50 at every s, and

    E(one Rumer) = 2 h s / (s^2 + 2),   E(resonating) = 2 h s / (s^2 + 1),
    RE = E(one Rumer) - E(resonating)  = -2 h s / ((s^2+1)(s^2+2)).

A numeric comparison to the full one-electron FCI shows this covalent-only
model reproduces the qualitative 50:50 picture but only a small fraction of
the Hueckel resonance energy; the remainder is ionic and delocalization.

Run from the repo root:
    PYTHONPATH=. python3 examples/qualitative_vb/05_allyl_radical_resonance.py
"""
import os
import sys

import numpy as np
import sympy as sp
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from symvb import Molecule, FixedPsi, hamiltonian
from symvb.system import ground_state, chirgwin_coulson, structure_vector
from symvb.fixed_psi import generate_dets
from symvb.spin import s_squared_matrix

# ------------------------------------------------------------------ build
# Linear chain a-b-c at the one-electron level (hubbard=False -> no U;
# zero_ii=True -> site energies dropped, so h and s are the only symbols).
m = Molecule.chain(3, hubbard=False)
R1 = FixedPsi('aBc', coupled_pairs=[(0, 1)])    # (a-b) singlet, odd electron on c
R2 = FixedPsi('aBc', coupled_pairs=[(1, 2)])    # (b-c) singlet, odd electron on a
H, S = hamiltonian(m, [R1, R2], two_electron=False)
H, S = sp.Matrix(H), sp.Matrix(S)
syms = {str(x): x for x in (H.free_symbols | S.free_symbols)}
h, s = syms['h'], syms['s']
print('build: allyl 2-structure one-electron H(h, s), S(s) via symvb  OK')

# --------------------------------------------------- spin identification
# Verify each Rumer structure is a pure doublet (<S^2> = 3/4) before using it.
allyl_dets = [p.dets[0].det_string for p in generate_dets(2, 1, 3)]
S2 = s_squared_matrix(allyl_dets, orbs='abc')
for name, R in (('R1', R1), ('R2', R2)):
    v = np.array(structure_vector(R, allyl_dets), dtype=float).ravel()
    val = float(v @ S2 @ v / (v @ v))
    resid = float(np.linalg.norm(S2 @ v - val * v))
    assert abs(val - 0.75) < 1e-12 and resid < 1e-12, (name, val, resid)
print('spin: R1 and R2 are pure doublets, <S^2> = 3/4 (resid 0)  OK')

# ------------------------------------------------------- closed-form H, S
H_ref = sp.Matrix([[2 * h * s, 4 * h * s], [4 * h * s, 2 * h * s]])
S_ref = sp.Matrix([[s**2 + 2, 2 * s**2 + 1], [2 * s**2 + 1, s**2 + 2]])
assert sp.simplify(H - H_ref) == sp.zeros(2)
assert sp.simplify(S - S_ref) == sp.zeros(2)
print('H = [[2hs, 4hs], [4hs, 2hs]],  S = [[s^2+2, 2s^2+1], [2s^2+1, s^2+2]]  OK')

# structure overlap: normalized <R1|R2>, and its s = 0 (spin-only) value
ovlp = sp.simplify(S[0, 1] / S[0, 0])            # equal diagonals, so this is normalized
assert sp.simplify(ovlp - (2 * s**2 + 1) / (s**2 + 2)) == 0
assert ovlp.subs(s, 0) == sp.Rational(1, 2)
print('structure overlap <R1|R2> = (2s^2+1)/(s^2+2) -> 1/2 at s = 0  OK')

# overlap-driven: the one-electron covalent H vanishes at s = 0
assert H.subs(s, 0) == sp.zeros(2)
print('H(s=0) = 0: covalent-covalent resonance is overlap-driven at the 1e level  OK')

# ---------------------------------------------------- energies and weights
# Pick the ground root with a nonzero-overlap reference (at s = 0 both roots
# collapse to 0 and the selection would be ambiguous).
REF = {'h': -1, 's': sp.Rational(1, 5)}
E_res, c = ground_state(H, S, ref=REF)
E_res = sp.simplify(E_res)
E_one = sp.simplify(H[0, 0] / S[0, 0])
RE = sp.simplify(E_one - E_res)
assert sp.simplify(E_res - 2 * h * s / (s**2 + 1)) == 0
assert sp.simplify(E_one - 2 * h * s / (s**2 + 2)) == 0
assert sp.simplify(RE + 2 * h * s / ((s**2 + 1) * (s**2 + 2))) == 0
print('E(resonating) = 2hs/(s^2+1),  E(one Rumer) = 2hs/(s^2+2)  OK')
print('RE = E(one) - E(resonating) = -2hs/((s^2+1)(s^2+2))  OK')

# the resonating state is the symmetric combination; weights are exactly 50:50
assert [sp.simplify(x) for x in c] == [c[0], c[0]]         # c proportional to (1, 1)
w = chirgwin_coulson(c, S, simplify=True)
assert list(w) == [sp.Rational(1, 2), sp.Rational(1, 2)]
print('Chirgwin-Coulson weights (R1, R2) = (1/2, 1/2), exact at every s  OK')

# leading resonance per electron (coefficient of s at h = -1), used by script 06
RE_lead = sp.simplify(sp.series(RE.subs(h, -1), s, 0, 2).removeO())
assert RE_lead == s
print('leading resonance RE = |h| s + O(s^3);  per electron |h| s / 3  OK')

# -------------------------------------------- numeric vs full 1e FCI
# Full one-electron ground state over the 9 S_z = +1/2 determinants (covalent
# and ionic), compared with the covalent-only 2-structure model.
Pf = generate_dets(2, 1, 3)
Hf, Sf = hamiltonian(m, Pf, two_electron=False)
Hf, Sf = sp.Matrix(Hf), sp.Matrix(Sf)
print('\n  h = -1:  covalent 2-structure model vs full one-electron FCI')
print('    {:>5} | {:>9} | {:>9} | {:>9} | {:>12}'.format(
    's', 'E_1Rumer', 'E_2cov', 'E_FCI', 'RE captured'))
print('    ' + '-' * 56)
for sv in (0.0, 0.1, 0.2, 0.3):
    sub = {h: -1.0, s: sv}
    Hn = np.array(Hf.subs(sub), dtype=float)
    Sn = np.array(Sf.subs(sub), dtype=float)
    E_fci = float(eigh(Hn, Sn, subset_by_index=[0, 0])[0][0])
    E1 = float(E_one.subs(sub))
    E2 = float(E_res.subs(sub))
    frac = (E1 - E2) / (E1 - E_fci) if abs(E1 - E_fci) > 1e-12 else 0.0
    print('    {:>5.2f} | {:>9.4f} | {:>9.4f} | {:>9.4f} | {:>11.1%}'.format(
        sv, E1, E2, E_fci, frac))
    # covalent model never overshoots the FCI, and gives no resonance at s = 0
    assert E2 >= E_fci - 1e-9
    if sv == 0.0:
        assert abs(E1 - E2) < 1e-12
print('covalent-only captures the 50:50 picture but a small share of the '
      'resonance energy (rest is ionic + delocalization)')

print('\nall assertions passed')
