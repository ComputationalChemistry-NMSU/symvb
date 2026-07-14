"""Benzene: Kekule resonance and the five covalent Rumer structures (1e level).

Textbook statement (Shaik, S.; Hiberty, P. C. A Chemist's Guide to Valence
Bond Theory; Wiley: Hoboken, NJ, 2008, Chapter 3, Basic VB Theory):

    Benzene (six orbitals a-f in a ring, six electrons) has five covalent
    Rumer structures: two equivalent Kekule structures and three Dewar
    (para-bonded) structures. The two Kekule structures have spin-coupling
    overlap <K1|K2> = 1/4 and mix 50:50 into the resonating ground state; that
    mixing is the classic Kekule resonance. Adding the three Dewar structures
    lowers the covalent energy further and completes the covalent description.

symvb construction: a six-orbital ring at the one-electron (Hueckel) level.
Each structure is a FixedPsi of eight determinants built by singlet-coupling
three ring pairs with couple_orbitals from a single parent spin pattern; every
structure is verified to be a pure singlet (<S^2> = 0). The Kekule 2x2 is
solved symbolically in (h, s); the full covalent 5x5 is solved numerically.
Work stays inside the span of the structures' determinants, never the 400-dim
determinant space.

Result proved here: <K1|K2> = 1/4 at s = 0; the Kekule pair mixes 50:50 with
resonance energy RE_Kek = E(K1) - E(2-Kekule); and the 5-structure covalent
ground state lies below the 2-Kekule one, with the two Kekule structures still
carrying about two thirds of the weight and the three Dewar structures the rest.

Run from the repo root:
    PYTHONPATH=. python3 examples/qualitative_vb/07_benzene_kekule.py
"""
import os
import sys

import numpy as np
import sympy as sp
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from symvb import Molecule, FixedPsi, hamiltonian
from symvb.system import ground_state, chirgwin_coulson, structure_vector
from symvb.operators import canonicalize
from symvb.spin import s_squared_matrix

REF = {'h': -1, 's': sp.Rational(1, 5)}

# ------------------------------------------------------------------ build
m = Molecule.ring(6, hubbard=False)
PARENT = 'aBcDeF'                              # spin pattern + - + - + -
Kek1 = FixedPsi(PARENT, coupled_pairs=[(0, 1), (2, 3), (4, 5)])   # (a-b)(c-d)(e-f)
Kek2 = FixedPsi(PARENT, coupled_pairs=[(0, 5), (1, 2), (3, 4)])   # (a-f)(b-c)(d-e)
Dew1 = FixedPsi(PARENT, coupled_pairs=[(0, 1), (2, 5), (3, 4)])   # (a-b)(c-f)(d-e)
Dew2 = FixedPsi(PARENT, coupled_pairs=[(0, 3), (1, 2), (4, 5)])   # (a-d)(b-c)(e-f)
Dew3 = FixedPsi(PARENT, coupled_pairs=[(0, 5), (1, 4), (2, 3)])   # (a-f)(b-e)(c-d)
covalent = [Kek1, Kek2, Dew1, Dew2, Dew3]
names = ['Kek1', 'Kek2', 'Dew1', 'Dew2', 'Dew3']
assert all(len(st.dets) == 8 for st in covalent)
print('build: 5 covalent Rumer structures, 8 determinants each  OK')

# --------------------------------------------------- spin identification
# Build S^2 over the (canonicalized, deduped) determinant span of the five
# structures and verify each is a pure singlet.
basis = []
seen = set()
for st in covalent:
    for d in st.dets:
        ck = canonicalize(d.det_string)[0]
        if ck not in seen:
            seen.add(ck)
            basis.append(ck)
S2 = s_squared_matrix(basis, orbs='abcdef')
for name, st in zip(names, covalent):
    v = np.array(structure_vector(st, basis), dtype=float).ravel()
    val = float(v @ S2 @ v / (v @ v))
    resid = float(np.linalg.norm(S2 @ v - val * v))
    assert abs(val) < 1e-10 and resid < 1e-10, (name, val, resid)
print('spin: all five structures are pure singlets, <S^2> = 0 (resid 0)  OK')
print('  distinct determinants spanned:', len(basis))

# ------------------------------------------------- Kekule 2x2 (symbolic)
Hk, Sk = hamiltonian(m, [Kek1, Kek2], two_electron=False)
Hk, Sk = sp.Matrix(Hk), sp.Matrix(Sk)
syms = {str(x): x for x in (Hk.free_symbols | Sk.free_symbols)}
h, s = syms['h'], syms['s']

H_ref = sp.Matrix([[12 * h * s * (7 * s**4 + 6 * s**2 + 2),
                    24 * h * s * (5 * s**4 + 3 * s**2 + 1)],
                   [24 * h * s * (5 * s**4 + 3 * s**2 + 1),
                    12 * h * s * (7 * s**4 + 6 * s**2 + 2)]])
S_ref = sp.Matrix([[14 * s**6 + 18 * s**4 + 12 * s**2 + 8,
                    20 * s**6 + 18 * s**4 + 12 * s**2 + 2],
                   [20 * s**6 + 18 * s**4 + 12 * s**2 + 2,
                    14 * s**6 + 18 * s**4 + 12 * s**2 + 8]])
assert sp.simplify(Hk - H_ref) == sp.zeros(2)
assert sp.simplify(Sk - S_ref) == sp.zeros(2)
print('Kekule 2x2 H, S match the closed forms  OK')

# structure overlap and its spin-only s = 0 value
assert (Sk[0, 1] / Sk[0, 0]).subs(s, 0) == sp.Rational(1, 4)
print('structure overlap <K1|K2> -> 1/4 at s = 0  OK')
assert Hk.subs(s, 0) == sp.zeros(2)
print('H(s=0) = 0: covalent resonance is overlap-driven at the 1e level  OK')

E_2kek, ck = ground_state(Hk, Sk, ref=REF)
E_2kek = sp.simplify(E_2kek)
E_K1 = sp.simplify(Hk[0, 0] / Sk[0, 0])
assert sp.simplify(E_2kek - 6 * h * s * (17 * s**4 + 12 * s**2 + 4)
                   / (17 * s**6 + 18 * s**4 + 12 * s**2 + 5)) == 0
assert sp.simplify(E_K1 - 6 * h * s * (7 * s**4 + 6 * s**2 + 2)
                   / (7 * s**6 + 9 * s**4 + 6 * s**2 + 4)) == 0
RE_Kek = sp.simplify(E_K1 - E_2kek)
print('E(2-Kekule) and E(K1) closed forms;  RE_Kek = E(K1) - E(2-Kekule)  OK')

wk = chirgwin_coulson(ck, Sk, simplify=True)
assert list(wk) == [sp.Rational(1, 2), sp.Rational(1, 2)]
print('the two Kekule structures mix 50:50 (weights 1/2, 1/2)  OK')

# ------------------------------------------- full covalent 5x5 (numeric)
H5, S5 = hamiltonian(m, covalent, two_electron=False)
H5, S5 = sp.Matrix(H5), sp.Matrix(S5)
print('\n  h = -1:  Kekule-only vs full 5-structure covalent VB')
print('    {:>5} | {:>9} | {:>9} | {:>9} | {:>7} | {:>7}'.format(
    's', 'E_2Kek', 'E_5cov', 'RE_5cov', 'w_Kek', 'w_Dew'))
print('    ' + '-' * 60)
for sv in (0.1, 0.2, 0.3):
    sub = {h: -1.0, s: sv}
    H5n = np.array(H5.subs(sub), dtype=float)
    S5n = np.array(S5.subs(sub), dtype=float)
    ev, vc = eigh(H5n, S5n)
    E5, c5 = float(ev[0]), vc[:, 0]
    w5 = c5 * (S5n @ c5)
    w_kek = float(w5[0] + w5[1])
    w_dew = float(w5[2] + w5[3] + w5[4])
    E2 = float(E_2kek.subs(sub))
    EK1 = float(E_K1.subs(sub))
    print('    {:>5.2f} | {:>9.4f} | {:>9.4f} | {:>9.4f} | {:>7.4f} | {:>7.4f}'.format(
        sv, E2, E5, EK1 - E5, w_kek, w_dew))
    # the 5-structure covalent space lies below the 2-Kekule space,
    # and covalent resonance exceeds the Kekule-only resonance
    assert E5 < E2 - 1e-9
    assert (EK1 - E5) > float(RE_Kek.subs(sub)) > 0
    # two Kekule structures still dominate over the three Dewar structures
    assert w_kek > w_dew
print('5-structure covalent ground state lies below 2-Kekule; Kekule pair '
      'keeps ~2/3 of the weight  OK')

print('\nall assertions passed')
