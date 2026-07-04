"""(H2)2+ disphenoid: a four-structure covalent+ionic model for k_crit(U).

The one-electron disphenoid ground state is the 2x2 resonance of two
charge-localised covalent structures (manuscript Eq. 12). At finite on-site
U the exact ground state leaks out of that 2x2 into intra-fragment ionic
configurations. This script shows that almost all of that physics, and the
finite-U Robin-Day threshold k_crit(U), is recovered by a *four*-structure
space:

  C1 = sigma_1^2 sigma_2   covalent, hole on unit 2   (unit 1 intact H2)
  C2 = sigma_1 sigma_2^2   covalent, hole on unit 1
  I1 = sigma_1*^2 sigma_2  spectator unit 1 promoted to its antibonding
                           (ionic) pair, hole on unit 2
  I2 = sigma_2*^2 sigma_1  mirror image

with sigma_k=(.+.)/sqrt2 the bonding and sigma_k*=(.-.)/sqrt2 the
antibonding combination of unit k. The dropped fifth/sixth structures
(sigma_k*^2 sigma_k, all three electrons on one unit: H2^- ... H2^2+) are
full charge separation and contribute ~1% to k_crit.

Under the pair-swap (unit 1 <-> unit 2) the symmetric point block-diagonalises
into 2x2 (odd, holding the ground state) + 2x2 (even); the asymmetric stretch
couples them into the full 4x4.

Run from the repo root:  PYTHONPATH=. python3 examples/h2h2_plus_4structure.py
"""
import os
import pickle
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, hamiltonian, structure_vector
from symvb.fixed_psi import FixedPsi, generate_dets

CACHE = '/tmp/disphenoid_4structure_Hfci.pkl'
ORBS = 'abcd'
hs1_, hs2_, hl_, s_, U_ = sp.symbols('h_s_1 h_s_2 h_l s U')


def make_molecule():
    return Molecule(
        zero_ii=True,
        interacting_orbs=['ab', 'cd', 'ac', 'ad', 'bc', 'bd'],
        subst={'h_s_1': ('H_ab',), 'h_s_2': ('H_cd',),
               'h_l': ('H_ac', 'H_ad', 'H_bc', 'H_bd'),
               's': ('S_ab', 'S_cd', 'S_ac', 'S_ad', 'S_bc', 'S_bd')},
        subst_2e={'U': ('1111',)}, max_2e_centers=1)


# ---- FCI Hamiltonian over the 24 Sz=1/2 determinants (s = 0), cached ----
P = generate_dets(2, 1, 4)
det_strings = [p.dets[0].det_string for p in P]
if os.path.exists(CACHE):
    Hf = pickle.load(open(CACHE, 'rb'))
else:
    m = make_molecule()
    Hf = hamiltonian(m, P)[0].subs(s_, 0)   # 2e block folded in, then s -> 0
    pickle.dump(Hf, open(CACHE, 'wb'))


def det3(cA, cB, cC):
    p = FixedPsi()
    for ii, oi in enumerate(ORBS):
        for oj in ORBS[ii + 1:]:
            cf2 = cA.get(oi, 0) * cB.get(oj, 0) - cA.get(oj, 0) * cB.get(oi, 0)
            if cf2 == 0:
                continue
            for ok in ORBS:
                cf = cf2 * cC.get(ok, 0)
                if cf != 0:
                    p.add_str_det(oi + oj + ok.upper(), coef=cf)
    return p


def vec(fp):
    # structure_vector folds in the fermion sign of the canonical reordering
    v = structure_vector(fp, det_strings)
    return v / sp.sqrt((v.T * v)[0])           # exact normalisation (s = 0)


s1 = {'a': 1, 'b': 1}
s1s = {'a': 1, 'b': -1}
s2 = {'c': 1, 'd': 1}
s2s = {'c': 1, 'd': -1}
C1 = vec(det3(s2, s1, s1))
C2 = vec(det3(s1, s2, s2))
I1 = vec(det3(s2, s1s, s1s))
I2 = vec(det3(s1, s2s, s2s))
Vb = sp.Matrix.hstack(C1, C2, I1, I2)

S4 = sp.simplify(Vb.T * Vb)
assert S4 == sp.eye(4), S4
H4 = sp.simplify(Vb.T * Hf * Vb)
hs = (hs1_ + hs2_) / 2
print("Symbolic 4x4 in basis {C1, C2, I1, I2} (s = 0):")
sp.pprint(H4)

# exact structure of the symbolic 4x4 (verified):
assert sp.simplify(H4[0, 0] - (2 * hs1_ + hs2_ + U_ / 2)) == 0   # covalent C1 = Eq.(12) diag + U/2
assert sp.simplify(H4[2, 2] - (-2 * hs1_ + hs2_ + U_ / 2)) == 0  # ionic I1
assert sp.simplify(H4[0, 1] - (-2 * hl_)) == 0                   # C-C hole transfer (one-electron)
assert sp.simplify(H4[0, 2] - U_ / 2) == 0                       # C-I coupling: purely U/2, zero at U=0
assert H4[0, 3] == 0 and H4[2, 3] == 0                           # no cross terms
print("\n  covalent 2x2 block = Eq.(12) matrix shifted by U/2;  C-I coupling = U/2")
print("  (a two-electron coupling, zero at U=0 -> the covalent 2x2 is exact there).")

# symmetric point: pair-swap block-diagonalisation into 2x2 (odd, GS) + 2x2 (even)
T = sp.sqrt(sp.Rational(1, 2)) * sp.Matrix([[1, 0, 1, 0], [1, 0, -1, 0],
                                            [0, 1, 0, 1], [0, 1, 0, -1]])
Hb = sp.simplify(T.T * H4.subs({hs2_: hs1_}) * T)   # columns: even_C, even_I, odd_C, odd_I
assert Hb[0, 2] == 0 and Hb[0, 3] == 0 and Hb[1, 2] == 0 and Hb[1, 3] == 0
print("\n  at eta = 0, even/odd (pair-swap) basis {even_C, even_I | odd_C, odd_I}")
print("  block-diagonalises into 2x2 + 2x2; the ground state is the lower root of the")
print("  odd block (bottom-right):")
sp.pprint(Hb)

# closed-form symmetric-point ground-state energy = lower root of the odd block
odd = Hb[2:, 2:]
p, q, cc = odd[0, 0], odd[1, 1], odd[0, 1]
E0U = sp.simplify((p + q) / 2 - sp.sqrt((p - q) ** 2 / 4 + cc ** 2))
E0_closed = hs1_ + hl_ + U_ / 2 - sp.sqrt((2 * hs1_ + hl_) ** 2 + (U_ / 2) ** 2)
assert sp.simplify(E0U - E0_closed) == 0
# at U=0 (eta=0), reduces to Eq.(12)|_{eta=0} = 3 h_s + 2 h_l (sign-resolved numerically)
assert abs(float(E0_closed.subs({U_: 0, hs1_: -1, hl_: sp.Rational(-3, 10)})) - (-3.6)) < 1e-12
# manuscript eq-15 R-form: E_0 = h_s + h_l + (U - R)/2 with R = sqrt((4h_s+2h_l)^2 + U^2)
R_e0 = sp.sqrt((4 * hs1_ + 2 * hl_) ** 2 + U_ ** 2)
assert sp.simplify(E0_closed - (hs1_ + hl_ + (U_ - R_e0) / 2)) == 0
print("\nsymbolic symmetric-point ground-state energy:")
print("  E0(U) = h_s + h_l + U/2 - sqrt((2 h_s + h_l)^2 + (U/2)^2)")
print("        = covalent-ionic avoided crossing; -> 3 h_s + 2 h_l = Eq.(12)|_{eta=0} at U=0")

# ---- numeric: k_crit from the 4x4 vs full FCI vs covalent-only 2x2 ----
syms = [hs1_, hs2_, hl_, U_]
H0 = np.array(Hf.subs({x: 0 for x in syms}).tolist(), float)
Ms = [np.array(sp.diff(Hf, x).tolist(), float) for x in syms]
H4n = np.array(H4.tolist(), object)
H4_0 = np.array(sp.Matrix(H4).subs({x: 0 for x in syms}).tolist(), float)
H4_M = [np.array(sp.diff(H4, x).tolist(), float) for x in syms]


def Hfull(h1, h2, hl, U):
    A = H0 + h1 * Ms[0] + h2 * Ms[1] + hl * Ms[2] + U * Ms[3]
    return (A + A.T) / 2


def H4num(h1, h2, hl, U):
    A = H4_0 + h1 * H4_M[0] + h2 * H4_M[1] + hl * H4_M[2] + U * H4_M[3]
    return (A + A.T) / 2


def curv(f, d=1e-4):
    return (f(d) + f(-d) - 2 * f(0.0)) / d ** 2


def slope(f, d=1e-4):
    return (f(d) - f(-d)) / (2 * d)


hl = -0.3
print(f"\nRobin-Day threshold k_crit at h_s = -1, h_l = {hl}:")
print(f"{'U/|h_s|':>8} {'4-structure':>12} {'full FCI':>10} {'covalent 2x2':>13} {'4-st error':>11}")
k_cov = 1.0 / (8.0 * abs(hl))
for U in [0.0, 1.0, 2.0, 4.0, 8.0]:
    E4 = lambda e: np.linalg.eigvalsh(H4num(-1 + e / 2, -1 - e / 2, hl, U))[0]
    assert abs(slope(E4)) < 1e-6, (U, slope(E4))     # eta = 0 is stationary (no linear term)
    k4 = -curv(E4)
    kF = -curv(lambda e: np.linalg.eigvalsh(Hfull(-1 + e / 2, -1 - e / 2, hl, U))[0])
    print(f"{U:>8.1f} {k4:>12.5f} {kF:>10.5f} {k_cov:>13.5f} {abs(k4-kF)/kF*100:>10.2f}%")
    assert abs(k4 - kF) / kF < 0.012, (U, k4, kF)
print("\nfour-structure k_crit within 1.2% of FCI across U; covalent-only 2x2"
      " is frozen at 1/(8|h_l|) and misses the entire correlation effect.")

# --- the threshold k_crit(U) is itself a closed form (4-state, exact for the model) ---
# k_crit = -d2 E_GS/d eta^2|_0 via 2nd-order PT (H4 linear in eta = h_s_1 - h_s_2).
# Fully analytic but unwieldy, which is why the manuscript quotes evaluated values.
W4 = (sp.diff(H4, hs1_) - sp.diff(H4, hs2_)) / 2
Wb = sp.simplify(T.T * W4.subs({hs2_: hs1_}) * T)          # even/odd basis at eta=0
Oblk, Eblk, Xc = Hb[2:, 2:], Hb[:2, :2], Wb[2:, :2]        # odd(GS), even, odd<->even coupling
o = sorted(Oblk.eigenvects(),
           key=lambda t: float(t[0].subs({hs1_: -1, hl_: sp.Rational(-3, 10), U_: 1})))
l0, v0 = o[0][0], o[0][2][0]
v0 = v0 / sp.sqrt((v0.T * v0)[0])
kcrit_sym = sum(2 * (v0.T * Xc * (wv[0] / sp.sqrt((wv[0].T * wv[0])[0])))[0] ** 2 / (mu - l0)
                for mu, m, wv in Eblk.eigenvects())
f_kc = sp.lambdify((hs1_, hl_, U_), kcrit_sym, 'numpy')
for U in [1.0, 4.0, 8.0]:
    k_num = -curv(lambda e: np.linalg.eigvalsh(H4num(-1 + e / 2, -1 - e / 2, hl, U))[0])
    assert abs(float(f_kc(-1, hl, U)) - k_num) < 1e-6, (U, float(f_kc(-1, hl, U)), k_num)
# the manuscript writes it as a single radical R = sqrt((4h_s+2h_l)^2 + U^2):
Rc = sp.sqrt((4 * hs1_ + 2 * hl_) ** 2 + U_ ** 2)
kcrit_compact = (Rc * (36 * (hl_ ** 2 - 4 * hs1_ ** 2) - U_ ** 2)
                 + 4 * U_ ** 2 * (2 * hl_ - 5 * hs1_)
                 + 72 * (hl_ + 2 * hs1_) ** 2 * (hl_ - 2 * hs1_)) / (4 * U_ ** 2 * hl_ * Rc)
f_compact = sp.lambdify((hs1_, hl_, U_), kcrit_compact, 'numpy')
# manifestly-finite rationalized form (manuscript): radical rationalized, no U^2 denominator
R0 = sp.sqrt((4 * hs1_ + 2 * hl_) ** 2)     # = |4h_s+2h_l| = R at U=0
kcrit_rat = (36 * (hl_ ** 2 - 4 * hs1_ ** 2) / (Rc + R0)
             + 4 * (2 * hl_ - 5 * hs1_) - Rc) / (4 * hl_ * Rc)
f_rat = sp.lambdify((hs1_, hl_, U_), kcrit_rat, 'numpy')
for hsv, hlv, Uv in [(-1, -0.3, 2), (-1.2, -0.5, 5), (-0.7, -0.4, 3)]:   # generic params
    assert abs(float(f_kc(hsv, hlv, Uv)) - float(f_compact(hsv, hlv, Uv))) < 1e-9
    assert abs(float(f_kc(hsv, hlv, Uv)) - float(f_rat(hsv, hlv, Uv))) < 1e-9
# manuscript eq-17 second line: small-U expansion in magnitudes |h_s|,|h_l|
#   k_crit = 1/(8|h_l|) - (14|h_s| - 11|h_l|)/(16|h_l| R0^3) U^2 + O(U^4),  R0 = |4h_s+2h_l| = 2(2|h_s|+|h_l|)
ahs, ahl = sp.symbols('a_hs a_hl', positive=True)
ser = sp.series(kcrit_compact.subs({hs1_: -ahs, hl_: -ahl}), U_, 0, 4)
assert sp.simplify(ser.coeff(U_, 0) - 1 / (8 * ahl)) == 0
assert sp.simplify(ser.coeff(U_, 2) + (14 * ahs - 11 * ahl) / (128 * ahl * (2 * ahs + ahl) ** 3)) == 0
print("k_crit(U) closed form: single radical R = sqrt((4 h_s + 2 h_l)^2 + U^2); even-sector gap")
print("  cancels between channels. Rationalized (manuscript) form has no U^2 in the denominator,")
print("  so U->0 -> 1/(8|h_l|) is manifest. Verified == block-PT == numeric curvature.")
