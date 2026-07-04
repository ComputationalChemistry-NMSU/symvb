"""Site-energy asymmetry on the bridge of the allyl 3c4e chain.

Put a one-electron site energy eps on the central orbital b (alpha_a =
alpha_c = 0, alpha_b = eps), keep the Hubbard on-site U, and ask what the
offset does to the long-bond Rumer structure Phi_ac = [a . c]_s b^2, the VB
signature of biradical character. This backs two manuscript claims in the
three-centre four-electron section:

  * Heteroatom offset paragraph. Because Phi_ac double-occupies the bridge,
    its site-energy content is 2*eps while the two Kekule structures
    (b singly occupied) carry only eps. So the covalent block is NOT
    eps-invariant: at s = 0 its diagonal is (U+eps, U+eps, U+2eps). A
    bridge more electronegative than the terminals (eps < 0) stabilises the
    doubly-occupied-bridge long bond more than the Kekule structures and
    enhances the biradical weight; an electron-poor bridge (eps > 0)
    suppresses it. The dependence is strong, not a small correction: at the
    carbon-pi point U/|h| = 4.4 the long-bond weight is 0.49 / 0.31 / 0.18
    at eps/|h| = -1 / 0 / +1. The closed forms are compact in

        q^2 = (1 - eps / sqrt(eps^2 + 8 h^2)) / 2      (0 <= q^2 <= 1)

    with the U = 0 (uncorrelated closed shell) weight  w_ac = q^4 / 2  and
    the U -> infinity (covalent-block) weight  w_ac = q^2. At eps = 0 these
    are 1/8 and 1/2, recovering the symmetric allyl result.

  * The (2U + 4 eps) s^4 sentence. The strict-b^2 bridge-exchange gap
    (examples/allyl_bridge_exchange.py, manuscript eq 11) picks up eps only
    through the overlap metric. Both determinants are b^2, so eps cancels in
    the leading 8|h| s^3 superexchange term; it first enters at s^4, and the
    on-site-like O(s^4) coefficient is 2U + 4*eps, i.e. the metric leaks the
    full doubly-occupied-bridge content 2*eps in lockstep with U (the same
    U + 2*eps that sits on the Phi_ac diagonal).

Every matrix element is built by symvb (site energies via zero_ii=False,
alpha_b unified to the symbol eps through subst; alpha_a = alpha_c pinned to
zero afterwards). Nothing is hand-derived. All results are asserted.

Run from the repo root: PYTHONPATH=. python3 examples/allyl_site_asymmetry.py
"""
import os
import sys
import time

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import FixedPsi, Molecule
from symvb.fixed_psi import generate_dets
from symvb.spin import s_squared_matrix

t0 = time.time()
h, s, U, eps = sp.symbols('h s U eps')
H_aa, H_cc = sp.symbols('H_aa H_cc')

# ------------------------------------------------------------------ build
# zero_ii=False turns the diagonal site energies on; subst unifies the central
# site energy H_bb to eps and both edge parameters to h, s. The terminal site
# energies H_aa, H_cc are pinned to zero after the build (alpha_a = alpha_c = 0).
m = Molecule(zero_ii=False, interacting_orbs=['ab', 'bc'],
             subst={'h': ('H_ab', 'H_bc'), 'eps': ('H_bb',), 's': ('S_ab', 'S_bc')},
             subst_2e={'U': ('1111',)}, max_2e_centers=1)
P = generate_dets(2, 2, 3)
dets = [p.dets[0].det_string for p in P]
H = sp.Matrix(m.build_matrix(P, op='H') + m.o2_matrix(P)).subs({H_aa: 0, H_cc: 0})
S = sp.Matrix(m.build_matrix(P, op='S'))
assert set(map(str, H.free_symbols)) == {'h', 's', 'U', 'eps'}
print('build: 9-det H(h, s, U, eps), S(s) via symvb  OK')

# ------------------------------------------------------------ Rumer structures
# Covalent structures (template: examples/allyl_long_bond_vb.py). couple_pairs
# spin-pairs the chosen creation slots; canonicalize expands into the 9-det
# basis. Phi_ac = [a . c]_s b^2 is the long bond.
Phi_ab = FixedPsi('aBcC', coupled_pairs=[(0, 1)])   # [a.b]_s c^2
Phi_bc = FixedPsi('aAbC', coupled_pairs=[(2, 3)])   # [b.c]_s a^2
Phi_ac = FixedPsi('abBC', coupled_pairs=[(0, 3)])   # [a.c]_s b^2  (long bond)
for fp in (Phi_ab, Phi_bc, Phi_ac):
    fp.canonicalize()
idx = {d: i for i, d in enumerate(dets)}


def to_standard(ds):
    """(standard uLuL string, sign) for a raw det string with uuLL spin order."""
    orbs = 'abcdefghij'
    so = [2 * orbs.index(c.lower()) + (0 if c.islower() else 1) for c in ds]
    al = sorted(c for c in ds if c.islower())
    be = sorted(c for c in ds if c.isupper())
    std = ''.join(al[i] + be[i] for i in range(min(len(al), len(be))))
    std += ''.join(al[len(be):]) + ''.join(be[len(al):])
    stdso = [2 * orbs.index(c.lower()) + (0 if c.islower() else 1) for c in std]
    inv = lambda l: sum(1 for i in range(len(l)) for j in range(i + 1, len(l)) if l[i] > l[j])
    return std, (1 if abs(inv(so) - inv(stdso)) % 2 == 0 else -1)


def vb_vector(fp):
    v = sp.zeros(9, 1)
    for d, c in fp:
        std, sgn = to_standard(d.det_string)
        v[idx[std]] += sgn * c
    return v


V = sp.Matrix.hstack(vb_vector(Phi_ab), vb_vector(Phi_bc), vb_vector(Phi_ac))
assert sp.simplify(V.T * S.subs({s: 0}) * V - 2 * sp.eye(3)) == sp.zeros(3, 3)
V_np = np.array(V, float)
V_hat = V_np / np.sqrt(np.diag(V_np.T @ V_np))[None, :]   # orthonormal at s = 0
print('Rumer structures Phi_ab, Phi_bc, Phi_ac orthonormal at s=0  OK')

# ------------------------------------------ (1) covalent 3x3 diagonal carries 2*eps
H_vb = sp.simplify(V.T * H.subs({s: 0}) * V / 2)          # orthonormal Rumer basis
assert H_vb == sp.Matrix([[U + eps, 0, -h],
                          [0, U + eps, -h],
                          [-h, -h, U + 2 * eps]])
print('covalent 3x3 diagonal = (U+eps, U+eps, U+2eps): long bond carries 2*eps  OK')

# ---------------------------------------------------- FCI long-bond weight machinery
H_fn = sp.lambdify((U, eps), H.subs({s: 0, h: -1}), 'numpy')   # h = -1 (|h| units)
S2_9 = s_squared_matrix(dets, orbs='abc')


def fci_gs(Uv, epsv):
    """Ground-state vector of the 9-det FCI at s = 0, h = -1."""
    Hn = np.array(H_fn(Uv, epsv), float)
    Hn = 0.5 * (Hn + Hn.T)
    return np.linalg.eigh(Hn)[1][:, 0]


def w_ac_fci(Uv, epsv):
    """Chirgwin-Coulson (= Lowdin, s=0) weight of Phi_ac in the 9-det FCI GS."""
    return float((V_hat[:, 2] @ fci_gs(Uv, epsv)) ** 2)


# closed forms (h = -1): q^2 in [0, 1]; U=0 -> q^4/2, U->inf -> q^2
q2 = (1 - eps / sp.sqrt(eps ** 2 + 8)) / 2
w_U0 = q2 ** 2 / 2
w_inf = q2

# ---------------------------------------- (2) U = 0 weight equals q^4/2 (exact + FCI)
assert sp.nsimplify(w_U0.subs(eps, 0)) == sp.Rational(1, 8)
assert sp.nsimplify(w_U0.subs(eps, -1)) == sp.Rational(2, 9)
assert sp.nsimplify(w_U0.subs(eps, 1)) == sp.Rational(1, 18)
print('w_ac(U=0) closed form q^4/2:  eps=0 -> 1/8, eps=-1 -> 2/9, eps=+1 -> 1/18  OK')
for epsv in [-2, -1, -0.5, 0, 0.5, 1, 2]:
    assert abs(w_ac_fci(0.0, epsv) - float(w_U0.subs(eps, epsv))) < 1e-9
print('w_ac(U=0) q^4/2 matches 9-det FCI over eps in [-2, 2] (< 1e-9)  OK')

# ------------------------------------------ (3) large-U FCI weight approaches q^2
for epsv in [-2, -1, 0, 1, 2]:
    assert abs(w_ac_fci(1e5, epsv) - float(w_inf.subs(eps, epsv))) < 1e-3
assert sp.nsimplify(w_inf.subs(eps, 0)) == sp.Rational(1, 2)
assert sp.nsimplify(w_inf.subs(eps, -1)) == sp.Rational(2, 3)
assert sp.nsimplify(w_inf.subs(eps, 1)) == sp.Rational(1, 3)
print('w_ac(U->inf) -> q^2:  eps=0 -> 1/2, eps=-1 -> 2/3, eps=+1 -> 1/3 (FCI at U=1e5)  OK')

# ------------------------------------------ (4) carbon-pi point U/|h| = 4.4 headline
ppp = {epsv: w_ac_fci(4.4, epsv) for epsv in (-1, 0, 1)}
assert abs(ppp[-1] - 0.494) < 1e-3
assert abs(ppp[0] - 0.311) < 1e-3
assert abs(ppp[1] - 0.180) < 1e-3
print(f'PPP U/|h|=4.4:  w_ac(eps=-1,0,+1) = '
      f'{ppp[-1]:.3f} / {ppp[0]:.3f} / {ppp[1]:.3f}  OK')

# ------------------------------------------ (5) mechanism: per-det site content
# The eps-coefficient of each determinant's s=0 diagonal is exactly the bridge
# (b) occupation, so only the b^2 structures (long bond + b^2 ionics) feel 2*eps.
Hdiag = H.subs({s: 0})
for i, ds in enumerate(dets):
    b_occ = sum(1 for c in ds if c.lower() == 'b')
    assert sp.simplify(sp.diff(Hdiag[i, i], eps) - b_occ) == 0
print('per-det: d(diag)/d(eps) = bridge occupation (only b^2 dets carry 2*eps)  OK')
# directional: raising eps drains the long bond into the b-empty a^2c^2 ionic det
i_ac2 = dets.index('aAcC')           # ionic a^2 c^2, bridge empty
w_ac2 = lambda ev: float(fci_gs(4.4, ev)[i_ac2] ** 2)
assert w_ac_fci(4.4, 1) < w_ac_fci(4.4, 0) < w_ac_fci(4.4, -1)   # long bond falls with eps
assert w_ac2(-1) < w_ac2(0) < w_ac2(1)                           # a^2c^2 ionic rises
print('eps>0 drains long bond, feeds the bridge-empty a^2c^2 ionic determinant  OK')

# ------------------------------------------ (6) strict-b^2 bridge exchange with eps
i1 = dets.index('aBbC')      # a(alpha) b^2 c(beta)
i2 = dets.index('bAcB')      # a(beta)  b^2 c(alpha)
S2f = np.array(s_squared_matrix(dets, orbs='abc'), float)
for sign, expect in ((+1, 0.0), (-1, 2.0)):
    v = np.zeros(9); v[i1] = 1; v[i2] = sign
    assert abs(v @ S2f @ v / (v @ v) - expect) < 1e-12
print('strict-b^2: D1+D2 singlet, D1-D2 triplet  OK')

energies = {}
for sign, name in ((+1, 'S'), (-1, 'T')):
    v = sp.zeros(9, 1); v[i1] = 1; v[i2] = sign
    energies[name] = sp.cancel((v.T * H * v)[0] / (v.T * S * v)[0])
gap = sp.simplify(sp.together(energies['T'] - energies['S']))

gap_ref = (2 * s ** 3 * (-U * s + 2 * eps * s ** 3 - 2 * eps * s - 4 * h * s ** 2 + 4 * h)
           / (4 * s ** 6 - 6 * s ** 4 + 4 * s ** 2 - 1))
assert sp.simplify(gap - gap_ref) == 0
print('exact gap E_T - E_S(eps) matches the closed form  OK')

# reduces to manuscript eq (11) at eps = 0
eq11 = (2 * s ** 3 * (U * s + 4 * (-h) * (1 - s ** 2))
        / ((1 - 2 * s ** 2) * (1 - 2 * s ** 2 + 2 * s ** 4)))
assert sp.simplify(gap.subs(eps, 0) - eq11) == 0
assert gap.subs(s, 0) == 0
print('eps=0 -> eq (11); gap(s=0) = 0 at every U, eps  OK')

# series in s (h = -t, t = |h|): eps-free 8|h|s^3, then (2U + 4 eps) s^4
t = sp.Symbol('t', positive=True)
ser = sp.series(gap.subs(h, -t), s, 0, 5).removeO().expand()
assert ser.coeff(s, 3) == 8 * t
assert sp.expand(ser.coeff(s, 4)) == 2 * U + 4 * eps
print('series: coeff(s^3) = 8|h| (eps-free); coeff(s^4) = 2U + 4 eps  OK')

# the eps-only part of the gap is carried entirely at O(s^4) and up
gap_eps_part = sp.simplify(gap - gap.subs(eps, 0))
eps_ref = 4 * eps * s ** 4 * (s ** 2 - 1) / (4 * s ** 6 - 6 * s ** 4 + 4 * s ** 2 - 1)
assert sp.simplify(gap_eps_part - eps_ref) == 0
print('eps-part of the gap = 4 eps s^4 (s^2 - 1)/(4s^6 - 6s^4 + 4s^2 - 1)  OK')

# numeric cross-check of the symbolic gap at a few (h, s, U, eps)
gap_fn = sp.lambdify((h, s, U, eps), gap, 'numpy')
Hn_fn = sp.lambdify((h, s, U, eps), H, 'numpy')
Sn_fn = sp.lambdify((h, s, U, eps), S, 'numpy')
for (hv, sv, Uv, ev) in [(-1, 0.25, 3.0, 0.7), (-1.0, 0.3, 5.0, -1.5), (-0.8, 0.15, 2.0, 2.0)]:
    Hn = np.array(Hn_fn(hv, sv, Uv, ev), float)
    Sn = np.array(Sn_fn(hv, sv, Uv, ev), float)
    vS = np.zeros(9); vS[i1] = 1; vS[i2] = 1
    vT = np.zeros(9); vT[i1] = 1; vT[i2] = -1
    num = (vT @ Hn @ vT) / (vT @ Sn @ vT) - (vS @ Hn @ vS) / (vS @ Sn @ vS)
    assert abs(num - float(gap_fn(hv, sv, Uv, ev))) < 1e-10
print('symbolic gap matches Rayleigh-quotient numerics at s != 0  OK')

print(f'\nall assertions passed in {time.time() - t0:.1f}s')
