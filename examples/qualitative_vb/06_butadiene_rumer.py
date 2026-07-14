"""Butadiene: the short-bond Kekule structure dominates the long-bond Dewar (1e level).

Textbook statement (Shaik, S.; Hiberty, P. C. A Chemist's Guide to Valence
Bond Theory; Wiley: Hoboken, NJ, 2008, Chapter 3, Basic VB Theory):

    The pi system of butadiene (four orbitals a-b-c-d, four electrons) has two
    covalent Rumer structures,

        K = (a-b)(c-d)   the Kekule (two short double bonds),
        D = (a-d)(b-c)   the Dewar-type long-bond structure.

    Their spin-coupling overlap is <K|D> = 1/2. Because the long bond (a-d)
    spans the whole chain, D mixes only weakly: the ground state is dominated
    by K, and the per-electron resonance stabilization is far smaller than in
    allyl. Butadiene is, to a good approximation, a single Kekule structure.

symvb construction: a four-orbital, four-electron chain at the one-electron
(Hueckel) level. K and D are FixedPsi structures built by singlet-coupling the
two pairs with couple_orbitals; spin purity (<S^2> = 0) is verified, not
assumed. H and S over {K, D} are built with two_electron=False, s symbolic.

Result proved here: <K|D> = 1/2 at s = 0; the covalent H is again proportional
to s; the Chirgwin-Coulson weights approach (K, D) = (1/2 + sqrt(3)/6,
1/2 - sqrt(3)/6) ~ (0.79, 0.21) as s -> 0, so K carries almost four fifths of
the wavefunction; and the leading resonance per electron is (2 sqrt(3) - 3)|h|s/4
~ 0.12|h|s, about a third of allyl's |h|s/3, confirming the weak long-bond
mixing.

Run from the repo root:
    PYTHONPATH=. python3 examples/qualitative_vb/06_butadiene_rumer.py
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

REF = {'h': -1, 's': sp.Rational(1, 5)}

# ------------------------------------------------------------------ build
m = Molecule.chain(4, hubbard=False)
K = FixedPsi('aBcD', coupled_pairs=[(0, 1), (2, 3)])    # (a-b)(c-d) Kekule
D = FixedPsi('aBcD', coupled_pairs=[(0, 3), (1, 2)])    # (a-d)(b-c) long-bond Dewar
H, S = hamiltonian(m, [K, D], two_electron=False)
H, S = sp.Matrix(H), sp.Matrix(S)
syms = {str(x): x for x in (H.free_symbols | S.free_symbols)}
h, s = syms['h'], syms['s']
print('build: butadiene 2-structure one-electron H(h, s), S(s) via symvb  OK')

# --------------------------------------------------- spin identification
buta_dets = [p.dets[0].det_string for p in generate_dets(2, 2, 4)]
S2 = s_squared_matrix(buta_dets, orbs='abcd')
for name, R in (('K', K), ('D', D)):
    v = np.array(structure_vector(R, buta_dets), dtype=float).ravel()
    val = float(v @ S2 @ v / (v @ v))
    resid = float(np.linalg.norm(S2 @ v - val * v))
    assert abs(val) < 1e-12 and resid < 1e-12, (name, val, resid)
print('spin: K and D are pure singlets, <S^2> = 0 (resid 0)  OK')

# ------------------------------------------------------- closed-form H, S
H_ref = sp.Matrix([[4 * h * s * (4 * s**2 + 3), 4 * h * s * (2 * s**2 + 3)],
                   [4 * h * s * (2 * s**2 + 3), 16 * h * s**3]])
S_ref = sp.Matrix([[4 * s**4 + 6 * s**2 + 4, 2 * s**4 + 6 * s**2 + 2],
                   [2 * s**4 + 6 * s**2 + 2, 4 * s**4 + 4]])
assert sp.simplify(H - H_ref) == sp.zeros(2)
assert sp.simplify(S - S_ref) == sp.zeros(2)
print('H, S match the closed forms (K and D have unequal norms: S00 != S11)  OK')

# structure overlap: normalized <K|D>, spin-only value at s = 0
ovlp0 = sp.simplify((S[0, 1] / sp.sqrt(S[0, 0] * S[1, 1])).subs(s, 0))
assert ovlp0 == sp.Rational(1, 2)
print('structure overlap <K|D> -> 1/2 at s = 0  OK')

# overlap-driven at the one-electron level
assert H.subs(s, 0) == sp.zeros(2)
print('H(s=0) = 0: covalent resonance is overlap-driven at the 1e level  OK')

# --------------------------------------------------- isolated-structure energies
E_K = sp.simplify(H[0, 0] / S[0, 0])
E_D = sp.simplify(H[1, 1] / S[1, 1])
assert sp.simplify(E_K - 2 * h * s * (4 * s**2 + 3) / (2 * s**4 + 3 * s**2 + 2)) == 0
assert sp.simplify(E_D - 4 * h * s**3 / (s**4 + 1)) == 0
print('E_K = 2hs(4s^2+3)/(2s^4+3s^2+2),  E_D = 4hs^3/(s^4+1)  OK')

# --------------------------------------------------- resonating ground state
E_res, c = ground_state(H, S, ref=REF)
E_res = sp.simplify(E_res)
# E_res is an eigenvalue of the 2x2 generalized problem, and the lower root
assert sp.simplify((H - E_res * S).det()) == 0
for sv in (0.1, 0.2, 0.3):
    sub = {h: -1.0, s: sv}
    lo = float(eigh(np.array(H.subs(sub), float), np.array(S.subs(sub), float),
                    subset_by_index=[0, 0])[0][0])
    assert abs(lo - float(E_res.subs(sub))) < 1e-9
RE = sp.simplify(E_K - E_res)
print('E(resonating) is the lower root of det(H - E S) = 0  OK')

# --------------------------------------------------- weights: K dominates
w = chirgwin_coulson(c, S)                       # symbolic; simplify via limit below
wK0 = sp.nsimplify(sp.limit(w[0], s, 0))
wD0 = sp.nsimplify(sp.limit(w[1], s, 0))
assert sp.simplify(wK0 - (sp.Rational(1, 2) + sp.sqrt(3) / 6)) == 0
assert sp.simplify(wD0 - (sp.Rational(1, 2) - sp.sqrt(3) / 6)) == 0
assert wK0 > wD0
print('Chirgwin-Coulson weights -> (K, D) = (1/2 + sqrt(3)/6, 1/2 - sqrt(3)/6) '
      '~ (0.79, 0.21) as s -> 0: K dominates  OK')

# --------------------------------------------------- per-electron RE vs allyl
RE_lead = sp.simplify(sp.series(RE.subs(h, -1), s, 0, 2).removeO())
assert sp.simplify(RE_lead - (2 * sp.sqrt(3) - 3) * s) == 0
buta_per_e = sp.simplify(RE_lead / 4)

# allyl, rebuilt here so the comparison is proved, not quoted
ma = Molecule.chain(3, hubbard=False)
A1 = FixedPsi('aBc', coupled_pairs=[(0, 1)])
A2 = FixedPsi('aBc', coupled_pairs=[(1, 2)])
Ha, Sa = hamiltonian(ma, [A1, A2], two_electron=False)
Ha, Sa = sp.Matrix(Ha), sp.Matrix(Sa)
Ea, _ = ground_state(Ha, Sa, ref=REF)
RE_allyl = sp.simplify(Ha[0, 0] / Sa[0, 0] - Ea)
allyl_lead = sp.simplify(sp.series(RE_allyl.subs(h, -1), s, 0, 2).removeO())
allyl_per_e = sp.simplify(allyl_lead / 3)
assert allyl_per_e == s / 3
# butadiene per-electron resonance is well below allyl's
assert float(buta_per_e.subs(s, 1)) < float(allyl_per_e.subs(s, 1))
print('per-electron leading RE:  butadiene {:.4f}|h|s  <  allyl {:.4f}|h|s  OK'.format(
    float(buta_per_e.subs(s, 1)), float(allyl_per_e.subs(s, 1))))

# --------------------------------------------------- numeric spot values
print('\n  h = -1:  butadiene covalent {K, D} model')
print('    {:>5} | {:>8} | {:>8} | {:>8} | {:>7} | {:>7}'.format(
    's', 'E_K', 'E_res', 'RE', 'w_K', 'w_D'))
print('    ' + '-' * 54)
for sv in (0.1, 0.2, 0.3):
    sub = {h: -1.0, s: sv}
    ww = chirgwin_coulson(np.array([float(x.subs(sub)) for x in c]),
                          np.array(S.subs(sub), float))
    print('    {:>5.2f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>7.4f} | {:>7.4f}'.format(
        sv, float(E_K.subs(sub)), float(E_res.subs(sub)),
        float(RE.subs(sub)), float(ww[0]), float(ww[1])))

print('\nall assertions passed')
