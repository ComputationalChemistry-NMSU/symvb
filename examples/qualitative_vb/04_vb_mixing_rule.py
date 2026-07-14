"""The non-orthogonal VB mixing rule, verified on the H2 covalent-ionic 2x2.

Textbook statement (Shaik, S.; Hiberty, P. C. A Chemist's Guide to Valence
Bond Theory; Wiley: Hoboken, NJ, 2008, Chapter 3, Basic VB Theory):

    For two normalized VB structures with energies E1 < E2, overlap S12 and
    interaction matrix element H12, the second-order stabilization of the
    lower structure is

        Delta E = (H12 - E1 S12)^2 / (E1 - E2).

    Because E1 < E2 the denominator is negative and the numerator is a square,
    so Delta E < 0: mixing always stabilizes the lower structure. The
    reduced coupling (H12 - E1 S12) is what matters, not H12 alone; at S12 = 0
    it reduces to the familiar H12^2 / (E1 - E2).

symvb construction, in two steps.

  Lemma. A general 2x2 generalized eigenproblem with the off-diagonal
  elements scaled by a formal bookkeeping parameter x,

        [[E1, x H12], [x H12, E2]] c = E [[1, x S12], [x S12, 1]] c,

  is solved for its exact lower root, which is then expanded to O(x^2). The
  constant term is E1, the O(x) term vanishes, and the O(x^2) term is exactly
  the mixing-rule expression. This is proved symbolically for arbitrary
  E1 < E2, H12, S12.

  Application. The covalent (aB + bA) and ionic (aA + bB) structures of H2 are
  fed to the facade with the on-site Hubbard U and the atomic-orbital overlap
  s kept symbolic (subst_2e U pattern, exactly as in examples/h2_charge_shift.py,
  two-electron block ON). The normalized 2x2 reproduces manuscript eq (5). Its
  covalent structure is the lower one for U > 0, so the rule gives its exact
  second-order charge-shift stabilization,

        Delta E_mix = -4 h_ab^2 (1 - s^2)^2 / (U (1 + s^2)^3),

  the resonance energy of the charge-shift bond. A numeric spot check compares
  the exact ground state with the E1 + Delta E_mix estimate.

Run from the repo root:
    PYTHONPATH=. python3 examples/qualitative_vb/04_vb_mixing_rule.py
"""
import os
import sys

import numpy as np
import sympy as sp
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from symvb import Molecule, FixedPsi, System

x = sp.Symbol('x')                              # formal scaling on the off-diagonals

# ============================================================== Lemma
# General two-structure GHEP with E1 < E2 (gap D = E2 - E1 > 0), overlap S12,
# coupling H12. Off-diagonals carry the bookkeeping factor x.
E1 = sp.Symbol('E1', real=True)
D = sp.Symbol('D', positive=True)               # E2 = E1 + D, enforces E1 < E2
H12 = sp.Symbol('H12', real=True)
S12 = sp.Symbol('S12', real=True)
a, b = E1, E1 + D

# det(H - E S) = 0 is the quadratic  A E^2 + B E + C = 0 with
A = 1 - x ** 2 * S12 ** 2
B = -(a + b - 2 * x ** 2 * H12 * S12)
C = a * b - x ** 2 * H12 ** 2
lower = (-B - sp.sqrt(B ** 2 - 4 * A * C)) / (2 * A)     # minus branch = lower root (a < b)

ser = sp.series(lower, x, 0, 3).removeO()
c0 = sp.simplify(ser.subs(x, 0))
c1 = sp.simplify(ser.coeff(x, 1))
c2 = sp.simplify(ser.coeff(x, 2))
mix = sp.simplify((H12 - E1 * S12) ** 2 / (E1 - (E1 + D)))       # (H12 - E1 S12)^2 / (E1 - E2)

assert sp.simplify(c0 - E1) == 0
assert c1 == 0
assert sp.simplify(c2 - mix) == 0
print('lemma: lowest root = E1 + 0*x + [(H12 - E1 S12)^2/(E1 - E2)] x^2 + O(x^3)  OK')

# ============================================================== H2 application
h, s, U = sp.symbols('h s U')
m = Molecule(zero_ii=True, interacting_orbs=['ab'], subst={'h': ('H_ab',), 's': ('S_ab',)},
             subst_2e={'U': ('1111',)}, max_2e_centers=1)
cov = FixedPsi('aB'); cov.add_str_det('bA', coef=1)     # covalent: one electron per atom
ion = FixedPsi('aA'); ion.add_str_det('bB', coef=1)     # ionic: doubly occupied atom
Hs, Ss = System.from_structures(m, [cov, ion]).hamiltonian()
Hs, Ss = sp.simplify(Hs), sp.simplify(Ss)

# normalize the two structures, then read off (E1, E2, H12, S12)
n0, n1 = sp.sqrt(Ss[0, 0]), sp.sqrt(Ss[1, 1])
E1h = sp.simplify(Hs[0, 0] / Ss[0, 0])          # covalent structure energy
E2h = sp.simplify(Hs[1, 1] / Ss[1, 1])          # ionic structure energy
H12h = sp.simplify(Hs[0, 1] / (n0 * n1))
S12h = sp.simplify(Ss[0, 1] / (n0 * n1))
assert E1h == 2 * h * s / (1 + s ** 2)
assert E2h == (U + 2 * h * s) / (1 + s ** 2)
assert H12h == 2 * h / (1 + s ** 2)
assert S12h == 2 * s / (1 + s ** 2)
print('H2 cov/ion 2x2 normalized: E_cov, E_ion, H12, S12 match manuscript eq (5)  OK')

# covalent is the lower structure for U > 0 (the mixing rule stabilizes it)
assert sp.simplify(E2h - E1h) == U / (1 + s ** 2)
print('  E_ion - E_cov = U/(1 + s^2) > 0: the covalent structure is the lower one  OK')

# the rule, instantiated on H2 (fully symbolic in h, s, U)
mix_h = sp.simplify((H12h - E1h * S12h) ** 2 / (E1h - E2h))
target = -4 * h ** 2 * (1 - s ** 2) ** 2 / (U * (1 + s ** 2) ** 3)
assert sp.simplify(mix_h - target) == 0
print('Delta E_mix = -4 h_ab^2 (1 - s^2)^2 / (U (1 + s^2)^3)  (charge-shift resonance energy)  OK')

# direct confirmation: expand the exact lower root of the H2 2x2 in x, at h = -1,
# with U, s symbolic. The lone radical is sqrt(U^2/(s^2 + 1)^2) = U/(s^2 + 1); with
# U > 0 and s real it denests through sqrt(factor(.)), which closes the O(x^2) term.
Up, sr = sp.Symbol('U', positive=True), sp.Symbol('s', real=True)
sub = {h: -1, U: Up, s: sr}
E1n, E2n = E1h.subs(sub), E2h.subs(sub)
H12n, S12n = H12h.subs(sub), S12h.subs(sub)
An = 1 - x ** 2 * S12n ** 2
Bn = -(E1n + E2n - 2 * x ** 2 * H12n * S12n)
Cn = E1n * E2n - x ** 2 * H12n ** 2
lowern = (-Bn - sp.sqrt(Bn ** 2 - 4 * An * Cn)) / (2 * An)
sern = sp.series(lowern, x, 0, 3).removeO()


def denest(e):
    e = sp.expand(e)
    for p in e.atoms(sp.Pow):
        if p.exp == sp.Rational(1, 2):
            e = e.subs(p, sp.sqrt(sp.factor(p.base)))
    return sp.simplify(e)


c0n = denest(sern.subs(x, 0))
c2n = denest(sern.coeff(x, 2))
assert sp.simplify(c0n - E1n) == 0
assert sp.simplify(c2n - mix_h.subs(sub)) == 0
print('direct H2 root expansion (h = -1, U/s symbolic): O(1) = E_cov, O(x^2) = Delta E_mix  OK')

# ------------------------------------------------------- numeric spot check
# exact ground state of the (un-normalized) 2x2 vs the E_cov + Delta E_mix estimate.
# The rule is a second-order estimate: for a two-level repulsion it slightly
# overshoots the true lowering, and the residual shrinks as U (the gap) grows.
Hf = sp.lambdify((s, U), Hs.subs({h: -1}), 'numpy')
Sf = sp.lambdify((s, U), Ss.subs({h: -1}), 'numpy')
Ecov_f = sp.lambdify((s, U), E1h.subs({h: -1}), 'numpy')
mix_f = sp.lambdify((s, U), mix_h.subs({h: -1}), 'numpy')
print('\n  h = -1 (|h| units), s = 0.2:  exact ground state vs mixing-rule estimate')
print('    {:>5} | {:>10} | {:>10} | {:>10} | {:>10}'.format('U', 'E_exact', 'E_cov+dE', 'dE_mix', 'residual'))
print('    ' + '-' * 58)
sv = 0.2
errs = []
for Uv in (4.0, 8.0, 16.0, 32.0):
    Hn = np.array(Hf(sv, Uv), float)
    Sn = np.array(Sf(sv, Uv), float)
    E_exact = float(eigh(Hn, Sn, subset_by_index=[0, 0])[0][0])
    E_cov = float(Ecov_f(sv, Uv))
    est = E_cov + float(mix_f(sv, Uv))
    errs.append(abs(est - E_exact))
    print('    {:>5.1f} | {:>10.5f} | {:>10.5f} | {:>10.5f} | {:>10.6f}'.format(
        Uv, E_exact, est, float(mix_f(sv, Uv)), est - E_exact))
    assert est < E_cov                          # mixing lowers the covalent structure
    assert est <= E_exact + 1e-9                # second order overshoots the true lowering
assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))   # converges as U grows
assert errs[-1] < 0.01                          # near-exact at the largest gap
print('    residual shrinks monotonically with U: the rule is exact to second order  OK')

print('\nall assertions passed')
