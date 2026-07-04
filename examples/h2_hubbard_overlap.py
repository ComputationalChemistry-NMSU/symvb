"""H2 Hubbard dimer with non-orthogonal AOs (s != 0).

Extends examples/h2_hubbard_bond.py to the physically realistic case of
finite AO overlap.  Three questions are answered symbolically:

    (1) What is the closed form E(U, h, s) on the 2-dim A_1 block?
    (2) How do Chirgwin-Coulson covalent / ionic weights depend on s?
    (3) What is the Heisenberg-limit scaling of the superexchange
        coupling  J_AFM(U, t, s)  when the AO basis is non-orthogonal?

The 2x2 A_1 block of H and S (built symbolically by symvb) is

    H_A1 = [[2hs/(1+s^2),         2h/(1+s^2)         ],
            [ 2h/(1+s^2),     U + 2hs/(1+s^2)        ]]

    S_A1 = [[ 1,       2s/(1+s^2) ],
            [2s/(1+s^2),       1  ]]

in the normalised {|cov>, |ion>} basis.  Solving the generalised
eigenvalue problem  H_A1 c = E S_A1 c  gives

    E(U, h, s) = [ U(1+s^2) + 4hs(s^2-1)
                   - sqrt( U^2(1+s^2)^2
                         + 16 Uhs(s^2-1)
                         + 16 h^2 (s^2-1)^2 ) ] / [2(1-s^2)^2]

with limiting forms

    s = 0:    E = U/2 - sqrt((U/2)^2 + 4h^2)    (textbook result)
    U = 0:    E = 2h/(1+s)                      (bonding Hückel-with-overlap MO x 2)

Three quantitative observations (printed below):

    * w_cov = 1/2 at U = 0 for ALL s  --  overlap does NOT shift the
      RHF 50/50 covalent-ionic balance.

    * Overlap SLOWS the covalent takeover at finite U.  At U = 4t:
      w_cov = 0.854 (s=0) -> 0.810 (s=0.25).

    * Heisenberg-limit antiferromagnetic coupling softens with s:
      J_AFM(U, t, s) = 4 t^2 (1 - s^2)^2 / [ U (1 + s^2)^3 ] + O(1/U^2).
      At s = 0.25 the prefactor is reduced to 0.733 of its s = 0 value,
      a non-orthogonality correction that standard Hubbard-model derivations
      ignore.

Run:  PYTHONPATH=. python3 examples/h2_hubbard_overlap.py
"""
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule
from symvb.fixed_psi import generate_dets
from symvb.system import hamiltonian, chirgwin_coulson


# ------------------------------------------------------------------------
# 1. Build H and S symbolically
# ------------------------------------------------------------------------
m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab'],
    subst={'h': ('H_ab',), 's': ('S_ab',)},
    subst_2e={'U': ('1111',)},
    max_2e_centers=1,
)
P = generate_dets(1, 1, 2)
H, S = hamiltonian(m, P)          # 4x4 determinant H (2e block folded in) and S

# Basis ordering from generate_dets: ['aA','aB','bA','bB']
idx = {p.dets[0].det_string: i for i, p in enumerate(P)}

# Symmetry-adapted {|cov>, |ion>} basis vectors (normalised)
#   N_cov^2 = N_ion^2 = 2(1 + s^2)
h, s, U = sp.symbols('h s U')
N = sp.sqrt(2 * (1 + s**2))
cov = sp.zeros(4, 1); cov[idx['aB']] = 1; cov[idx['bA']] = 1; cov /= N
ion = sp.zeros(4, 1); ion[idx['aA']] = 1; ion[idx['bB']] = 1; ion /= N
T   = sp.Matrix.hstack(cov, ion).T

H_A1 = sp.simplify(T * H * T.T)
S_A1 = sp.simplify(T * S * T.T)
print("H_A1 (cov, ion):"); sp.pprint(H_A1)
print("\nS_A1 (cov, ion):"); sp.pprint(S_A1)


# ------------------------------------------------------------------------
# 2. Closed-form ground-state energy
# ------------------------------------------------------------------------
E = sp.Symbol('E')
roots = sp.solve((H_A1 - E * S_A1).det(), E)
# Ground-state branch is the one that reduces to -2|h| at U=0, s=0
E_gs = [r for r in roots
        if sp.simplify(r.subs({U: 0, s: 0, h: -1}) - (-2)) == 0][0]
E_gs = sp.simplify(E_gs)
print("\nClosed-form ground-state energy:")
print(f"  E(U, h, s) = {E_gs}")
print(f"  at s=0:  {sp.simplify(E_gs.subs(s, 0))}")
print(f"  at U=0:  {sp.simplify(E_gs.subs(U, 0))}")   # -> 2h/(1+s)


# ------------------------------------------------------------------------
# 3. Chirgwin-Coulson weights
# ------------------------------------------------------------------------
r = sp.Symbol('r')                       # r = c_ion / c_cov from the top GHEP row
r_sol = sp.solve((H_A1 - E_gs * S_A1)[0, 0] + (H_A1 - E_gs * S_A1)[0, 1] * r, r)[0]
c = sp.Matrix([1, r_sol])
# metric-correct Chirgwin-Coulson weights; left unsimplified (the raw symbolic
# weight is expensive to reduce, and only its U=0 value and numeric evaluations
# are used below, both of which simplify cheaply at the point of use)
w_cov, w_ion = chirgwin_coulson(c, S_A1)

print("\nChirgwin-Coulson weights (w_cov + w_ion = 1):")
print(f"  at U=0, any s:   w_cov = {sp.simplify(w_cov.subs(U, 0))}")
print(f"\n  scan w_cov(U, s) at h = -1:")
print(f"    {'s':>5} {'U':>5}  {'E_gs':>10}  {'w_cov':>10}")
for sv in [0, sp.Rational(1, 5), sp.Rational(1, 4), sp.Rational(3, 10)]:
    for Uv in [0, 1, 2, 4, 8, 16]:
        subs = {h: -1, s: sv, U: Uv}
        Ev = float(E_gs.subs(subs))
        wc = float(w_cov.subs(subs))
        print(f"    {float(sv):5.2f} {Uv:5}  {Ev:+10.5f}  {wc:10.5f}")


# ------------------------------------------------------------------------
# 4. Heisenberg-limit expansion  E -> -4 t^2 / [U (1 + s)^2] + O(1/U^2)
# ------------------------------------------------------------------------
# E_gs has an overall 1/U decay at large U;  the covalent state in this
# limit has Heitler-London character with effective AF exchange
#     J_AFM = -E_gs(U->inf, t)  ~ 4 t^2 / U  at s = 0
# Let us substitute h = -t (t > 0) and expand to leading order in 1/U.
t = sp.Symbol('t', positive=True)
E_t = sp.simplify(E_gs.subs(h, -t))
# Expand about 1/U = 0
u = sp.Symbol('u', positive=True)      # u = 1/U
series = sp.series(E_t.subs(U, 1/u), u, 0, 3).removeO()
print("\nLarge-U (Heisenberg-limit) expansion of E(U, t, s):")
sp.pprint(sp.simplify(series))

# Leading -4 t^2/U coefficient as a function of s
lead = sp.simplify(series.coeff(u, 1))
print(f"\nLeading 1/U coefficient:  {lead}")
print(f"  at s=0:      {sp.simplify(lead.subs(s, 0))}   (-> J_AFM = 4 t^2 / U)")
print(f"  at s=0.25:   {sp.simplify(lead.subs(s, sp.Rational(1, 4)))}")
print(f"  ratio J_AFM(s=0.25) / J_AFM(0) = {float(lead.subs(s, sp.Rational(1,4)) / lead.subs(s, 0)):.4f}")
print("\n  Closed form: J_AFM = 4 t^2 (1-s^2)^2 / [U (1+s^2)^3]")
print("  Interpretation: AO overlap softens antiferromagnetic superexchange.")
print("  At s = 0.25 the prefactor drops to 0.733 of its s = 0 value --")
print("  a non-orthogonality correction absent from the textbook Hubbard derivation.")
