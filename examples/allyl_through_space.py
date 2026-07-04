"""Allyl anion (3c4e): through-bond vs through-space two-electron integrals.

Extends examples/allyl_hubbard_ujk.py.  The existing script unifies
adjacent and 1,3 two-electron integrals through pattern-based substitution
(subst_2e={'J': '1212', 'K': '1122'}) -- ALL (ab|ab)-pattern integrals
share one symbol J, and ALL (aa|bb)-pattern integrals share K.  This
hides the physical distinction between NEAREST-NEIGHBOR (through-bond)
and 1,3 (through-space) integrals.

In a linear 3-site chain the through-space integrals are qualitatively
different from through-bond:

    J_13 = (ac|ac)    two-center exchange between terminal p_pi orbitals
                      ~ 0   (decays exponentially with 1,3 distance)
    K_13 = (aa|cc)    direct 1,3 Coulomb repulsion
                      ~ 1/R_ac (substantial at typical C-C-C geometry)

This script builds the 2e Hamiltonian with raw T_xxxx symbols (no
subst_2e pattern), then relabels by physical role:

    U      = T_aaaa = T_bbbb = T_cccc          on-site
    K_adj  = T_aabb = T_bbcc                   through-bond Coulomb
    K_13   = T_aacc                            through-space Coulomb
    J_adj  = T_abab = T_bcbc                   through-bond exchange
    J_13   = T_acac                            through-space exchange
    M      = three-index (aa|ab) etc.          dropped (ZDO)

and derives the first- and second-order Rayleigh-Schroedinger coefficients
as polynomials in all five parameters.  Two non-obvious results emerge:

(a) THE (U - J) REDUCTION FAILS when J_adj and J_13 are distinguished.

    The E_2 = (sqrt(2)/512) [-21(U-J)^2 - 28 K(U-J) - 52 K^2] result of
    the existing script relies on forcing J_adj = J_13 = J AND K_adj =
    K_13 = K via pattern substitution.  Once the two J integrals are
    decoupled, no linear shift  U -> U_eff + alpha J_adj + beta J_13
    makes E_2 independent of (J_adj, J_13).  The "U - J" combination is
    thus an artefact of the orbital-pattern simplification, not a
    rigorous feature of the 3c4e Hamiltonian.

(b) THROUGH-BOND AND THROUGH-SPACE COULOMB COUPLE TO U WITH OPPOSITE SIGN.

    Isolating each Coulomb channel with all exchange integrals set to zero:

        E_2(U, K_adj, K_13=0) = -(sqrt(2)/512) [21 U^2 + 64 U K_adj + 96 K_adj^2]
        E_2(U, K_13, K_adj=0) = -(sqrt(2)/512) [21 U^2 - 36 U K_13  + 148 K_13^2]

    The U . K_adj cross-term coefficient is NEGATIVE (through-bond
    Coulomb reinforces correlation stabilisation), while U . K_13 is
    POSITIVE (through-space Coulomb weakens it).  The K^2 coefficient
    is also larger in magnitude for K_13 than K_adj (148 vs 96 out of
    512), so per-unit 1,3 Coulomb is a more potent correlation channel
    than nearest-neighbor Coulomb.

Both results are specific to the allyl closed-shell reference density
rho = (3/2, 1, 3/2); they are NOT a generic consequence of distinguishing
through-bond from through-space in an arbitrary system.
"""
import os
import sys
import time

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, SlaterDet, symmetry
from symvb.fixed_psi import generate_dets


# ------------------------------------------------------------------------
# 1. Build raw matrices with no 2e pattern substitution
# ------------------------------------------------------------------------
m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab', 'bc'],      # 1e: nearest-neighbor only
    subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
    subst_2e={},                        # no 2e pattern substitution
    max_2e_centers=2,
)
P = generate_dets(2, 2, 3)
det_strings = [p.dets[0].det_string for p in P]

t0 = time.time()
H1 = m.build_matrix(P, op='H')
S  = m.build_matrix(P, op='S')
H2 = m.o2_matrix(P)
print(f"Symbolic matrix build: {time.time() - t0:.1f}s")

H = sp.Matrix(H1 + H2)
S_mat = sp.Matrix(S)
h, s = sp.symbols('h s')


# ------------------------------------------------------------------------
# 2. Relabel raw T_xxxx symbols to physically-named parameters
# ------------------------------------------------------------------------
U  = sp.Symbol('U')
Ka = sp.Symbol('K_adj')
K3 = sp.Symbol('K_13')
Ja = sp.Symbol('J_adj')
J3 = sp.Symbol('J_13')

relabel = {}
two_e_names = sorted([str(x) for x in H.free_symbols if str(x).startswith('T_')])
for name in two_e_names:
    idx = name[2:]
    letters = list(idx)
    unique = sorted(set(letters))
    if len(unique) == 1:
        relabel[sp.Symbol(name)] = U
    elif len(unique) == 2:
        counts = {c: letters.count(c) for c in unique}
        x, y = unique
        is_13 = (x == 'a' and y == 'c')
        pattern = ''.join(['1' if c == x else '2' for c in letters])
        if counts[x] == 2 and counts[y] == 2:
            if pattern == '1122':
                relabel[sp.Symbol(name)] = K3 if is_13 else Ka
            elif pattern in ('1212', '1221', '2112', '2121'):
                relabel[sp.Symbol(name)] = J3 if is_13 else Ja
            else:
                raise RuntimeError(f"Unexpected 2-2 pattern {pattern} for {name}")
        else:
            # 3-index two-center integrals (M-type) -- drop via ZDO
            relabel[sp.Symbol(name)] = sp.Integer(0)
    else:
        # Genuinely three-center 2e integrals -- already dropped by max_2e_centers=2
        relabel[sp.Symbol(name)] = sp.Integer(0)

H_named = H.subs(relabel)
H_s0 = H_named.subs({s: 0, h: -1})
assert S_mat.subs({s: 0}) == sp.eye(9)


# ------------------------------------------------------------------------
# 3. Project to sigma = +1 reflection subspace
# ------------------------------------------------------------------------
def canon(ds):
    fp = SlaterDet(ds).get_sorted()
    return fp.dets[0].det_string, fp.coefs[0]

sig = {'a': 'c', 'b': 'b', 'c': 'a'}
perm, signs = symmetry.apply_orbital_permutation(sig, det_strings, canon)

U_plus = []
seen = [False] * 9
for i in range(9):
    if seen[i]:
        continue
    j = perm[i]
    sj = signs[i]
    if j == i:
        seen[i] = True
        if sj == 1:
            v = sp.zeros(9, 1); v[i] = 1
            U_plus.append(v)
    else:
        seen[i] = seen[j] = True
        v = sp.zeros(9, 1); v[i] = 1; v[j] = sj
        U_plus.append(v / sp.sqrt(2))

Up = sp.Matrix.hstack(*U_plus)
H_red = sp.simplify(Up.T * H_s0 * Up)
Nd = H_red.shape[0]


# ------------------------------------------------------------------------
# 4. RS perturbation coefficients as polynomials in (U, K_adj, K_13, J_adj, J_13)
# ------------------------------------------------------------------------
H0 = H_red.subs({U: 0, Ka: 0, K3: 0, Ja: 0, J3: 0})
V_red = sp.simplify(H_red - H0)

E_sym = sp.Symbol('E')
roots_all = sorted(set(sp.solve(H0.charpoly(E_sym).as_expr(), E_sym)),
                   key=lambda r: float(r))

eig = []
for r in roots_all:
    ker = (H0 - r * sp.eye(Nd)).nullspace()
    orth = []
    for v in ker:
        w = v
        for u in orth:
            w = w - (u.T * w)[0, 0] * u
        n2 = (w.T * w)[0, 0]
        if n2 == 0:
            continue
        orth.append(w / sp.sqrt(n2))
    for w in orth:
        eig.append((r, w))

E0, c0 = eig[0]
E_1 = sp.simplify((c0.T * V_red * c0)[0, 0])

V_mn = sp.zeros(Nd, Nd)
for mi, (_, vm) in enumerate(eig):
    for ni, (_, vn) in enumerate(eig):
        V_mn[mi, ni] = sp.simplify((vm.T * V_red * vn)[0, 0])

E_2 = sp.Rational(0)
for ni in range(1, Nd):
    E_2 += V_mn[0, ni] ** 2 / (E0 - eig[ni][0])
E_2 = sp.expand(sp.simplify(E_2))

print(f"\nE_1 = {E_1}")
print(f"\nE_2 = {E_2}")


# ------------------------------------------------------------------------
# 5. (U - J) reduction: does it survive distinguishing J_adj from J_13?
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("(U - J) reduction test: can U -> U_eff + alpha J_adj + beta J_13")
print("eliminate both J_adj and J_13 from E_2?")
print("=" * 70)
alpha, beta, Ueff = sp.symbols('alpha beta U_eff')
E2_shift = sp.expand(E_2.subs(U, Ueff + alpha * Ja + beta * J3))
poly = sp.Poly(E2_shift, Ja, J3)
# All nonzero-J monomials must vanish for the reduction to hold
J_terms = [sp.simplify(coef) for monom, coef in
           zip(poly.monoms(), poly.coeffs()) if monom != (0, 0)]
sol = sp.solve(J_terms, [alpha, beta], dict=True)
if sol:
    print(f"  Reduction holds with (alpha, beta) = {sol}")
else:
    print("  NO (alpha, beta) eliminates all J-terms -- the (U-J) reduction")
    print("  of arXiv/Eq.(13) relies on pattern-unification J_adj = J_13.")

# Verify that pattern unification J_adj = J_13 = J, K_adj = K_13 = K
# does collapse to the published (U-J) form.
Js, Ks = sp.symbols('J K')
E2_unified = sp.simplify(E_2.subs({Ja: Js, J3: Js, Ka: Ks, K3: Ks}))
published = sp.sqrt(2) * (-21 * (U - Js)**2 - 28 * Ks * (U - Js) - 52 * Ks**2) / 512
check = sp.simplify(E2_unified - published)
print(f"\n  Cross-check against existing allyl_hubbard_ujk.py closed form:")
print(f"    diff = {check}   (should be 0)")


# ------------------------------------------------------------------------
# 6. Through-bond vs through-space cross-sections
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Through-bond vs through-space cross-sections of E_2")
print("=" * 70)
scenarios = [
    ("pure U (baseline)",                     {Ka: 0, K3: 0, Ja: 0, J3: 0}),
    ("pure through-bond Coulomb (U, K_adj)",  {K3: 0, Ja: 0, J3: 0}),
    ("pure through-space Coulomb (U, K_13)",  {Ka: 0, Ja: 0, J3: 0}),
    ("physical PPP (J_13 = 0)",               {J3: 0}),
    ("adjacent-only PPP (J_13 = K_13 = 0)",   {J3: 0, K3: 0}),
]
for label, subs in scenarios:
    e = sp.expand(sp.simplify(E_2.subs(subs)))
    print(f"\n  {label}")
    print(f"    E_2 = {e}")


# ------------------------------------------------------------------------
# 7. Print the two headline closed forms
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Sign-of-cross-coupling: U . K flips sign between through-bond and through-space")
print("=" * 70)
E2_Kadj = sp.expand(E_2.subs({K3: 0, Ja: 0, J3: 0}))
E2_K13  = sp.expand(E_2.subs({Ka: 0, Ja: 0, J3: 0}))
print(f"\n  E_2(U, K_adj) = {E2_Kadj}")
print(f"  -> U . K_adj coefficient: {sp.simplify(E2_Kadj.coeff(U).coeff(Ka))}  (negative)")
print(f"\n  E_2(U, K_13)  = {E2_K13}")
print(f"  -> U . K_13 coefficient : {sp.simplify(E2_K13.coeff(U).coeff(K3))}  (positive)")
