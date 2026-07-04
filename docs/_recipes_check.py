"""Executable check for every snippet in docs/recipes.md.

Run:  PYTHONPATH=. python3 docs/_recipes_check.py
Each recipe below is the verbatim body of a recipes.md section; keep them in
sync. The asserts pin the documented outputs.
"""
import numpy as np
import sympy as sp
from scipy.linalg import eigh

import symvb
from symvb import (Molecule, FixedPsi, System, huckel, spin, operators, mo_projection)
from symvb.fixed_psi import generate_dets
from symvb.system import hamiltonian, ground_state, chirgwin_coulson, structure_vector


def recipe(name):
    print("=" * 70)
    print("RECIPE:", name)
    print("=" * 70)


# ---------------------------------------------------------------------------
recipe("1. Build and solve a model in three lines (System facade)")
# H2 covalent/ionic bond
m = Molecule(zero_ii=True, interacting_orbs=['ab'],
             subst={'h': ('H_ab',), 's': ('S_ab',)},
             subst_2e={'U': ('1111',)}, max_2e_centers=1)
cov = FixedPsi('aB'); cov.add_str_det('bA', coef=1)
ion = FixedPsi('aA'); ion.add_str_det('bB', coef=1)
bond = System.from_structures(m, [cov, ion])
h, s, U = sp.symbols('h s U')
E, c = bond.ground_state()
w_cov, w_ion = bond.weights()
assert sp.simplify(E.subs(s, 0) - (U/2 - sp.sqrt((U/2)**2 + 4*h**2))) == 0
assert sp.simplify(w_cov.subs({s: 0, U: 0})) == sp.Rational(1, 2)
print("E_gs(s=0) =", sp.simplify(E.subs(s, 0)))
print("w_cov(U=0) =", sp.simplify(w_cov.subs({s: 0, U: 0})))

# topology constructor: a ring fills in every edge + on-site U for you
benzene = System.ring(6)
assert benzene.m.interacting_orbs == ['ab', 'bc', 'cd', 'de', 'ef', 'af']
assert len(benzene.det_strings) == 400
print("System.ring(6): ", len(benzene.det_strings), "determinants, edges",
      benzene.m.interacting_orbs)


# ---------------------------------------------------------------------------
recipe("2. Operator matrices without hand-rolling (operators DSL)")
# Spin and eta-pairing matrices over a determinant basis (s = 0).
dets = [p.dets[0].det_string for p in generate_dets(2, 2, 3)]   # allyl, 9 dets
S2_dsl = operators.s_squared(['a', 'b', 'c']).matrix(dets)
S2_ref = sp.Matrix(spin.s_squared_matrix(dets, orbs='abc'))
assert sp.simplify(S2_dsl - S2_ref).is_zero_matrix
print("operators.s_squared(...).matrix(dets) == spin.s_squared_matrix(...) : OK")

# a Heisenberg coupling S_i . S_j, a hopping, and a doubly-occupied projector
SdotS = operators.s_dot('a', 'b').matrix(dets)
N_hop = operators.hop('a', 'b').matrix(dets)
assert SdotS.shape == (9, 9) and N_hop.shape == (9, 9)

# a bond singlet built straight from the vacuum (no coupled_pairs bookkeeping)
hl_ab = operators.bond_singlet_creator('a', 'b').apply('')   # -> FixedPsi
print("bond_singlet_creator('a','b')|0> =", hl_ab)


# ---------------------------------------------------------------------------
recipe("3. Huckel MOs without writing the C matrix by hand")
res = huckel.solve_ring(3)                       # cyclopropenyl C3 ring
print("eigenvalues:", res.eigenvalues)
print("MO energies eps_k(h,s):", res.energies)
# closed-shell anion = MOs 0 and 1 doubly occupied
E_huckel = sp.simplify(res.energy_of_occupation([2, 2, 0]).subs(res.s_symbol, 0))
print("closed-shell energy at s=0:", E_huckel)
# coefficients are the MO-over-AO rows you would otherwise transcribe
print("coefficient matrix (rows = MOs):")
sp.pprint(res.coefficients)


# ---------------------------------------------------------------------------
recipe("4. Spin and eta projection (reduce an FCI block)")
# Build the 9-det allyl FCI at s=0 and project onto the singlet sector.
mA = Molecule(zero_ii=True, interacting_orbs=['ab', 'bc'],
              subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
              subst_2e={'U': ('1111',)}, max_2e_centers=1)
Hsym, Ssym = hamiltonian(mA, generate_dets(2, 2, 3))
Hn = np.array(Hsym.subs({sp.Symbol('h'): -1, sp.Symbol('s'): 0,
                         sp.Symbol('U'): 2}).tolist(), float)
S2 = np.array(spin.s_squared_matrix(dets, orbs='abc'), float)
H_sing, U_sing = spin.project_onto_S(Hn, S2, target_S=0)
print("9-det FCI -> singlet block:", Hn.shape[0], "->", H_sing.shape[0])
E_full = float(np.linalg.eigvalsh(Hn)[0])
E_sing = float(np.linalg.eigvalsh(0.5*(H_sing + H_sing.T))[0])
assert abs(E_full - E_sing) < 1e-9      # ground state is a singlet here
print("ground energy from full and singlet block agree:", round(E_sing, 6))


# ---------------------------------------------------------------------------
recipe("5. Symmetry projection (trivial-irrep / Reynolds projector)")
# The C_3 cyclic + reflection generators of the allyl frame, as orbital maps.
rot = {'a': 'b', 'b': 'c', 'c': 'a'}            # 3-cycle
refl = {'a': 'c', 'c': 'a'}                     # swap ends
P_sym = operators.reynolds_projector([rot, refl]).matrix(dets)
# it is a projector: P^2 = P
assert sp.simplify(P_sym * P_sym - P_sym).is_zero_matrix
rank = sp.Matrix(P_sym).rank()
print("totally-symmetric projector rank (allyl, 9 dets):", rank)


# ---------------------------------------------------------------------------
recipe("6. MO determinant in the AO basis + verify an eigenpair")
# Expand the closed-shell Huckel determinant (MOs 0,1 doubly occupied) in AO dets.
C = res.coefficients                            # 3 x 3 (rows = MOs)
psi = mo_projection.mo_determinant_in_ao(C, ([0, 1], [0, 1]), dets,
                                         site_labels=['a', 'b', 'c'])
print("Hueckel closed-shell determinant has",
      sum(1 for x in psi if x != 0), "AO-determinant components")
# verify a known eigenpair of a 2x2 symbolically
Hs = sp.Matrix([[0, 2*h], [2*h, U]])
Ss = sp.eye(2)
E2 = (U - sp.sqrt(U**2 + 16*h**2)) / 2
v2 = sp.Matrix([2*h, E2])
assert mo_projection.verify_eigenpair(Hs, Ss, v2, E2)
print("verify_eigenpair: (H - E S) v == 0  -> OK")


# ---------------------------------------------------------------------------
recipe("7. Fast numeric evaluation of a big symbolic matrix")
# Reduce a symbolic H(h, s, U) to numpy anchors via its linearity, then solve
# only the lowest eigenpair with scipy -- the pattern that keeps 400-dim scans
# fast (lambdify of a large symbolic matrix is pathologically slow).
hsym, ssym, Usym = sp.symbols('h s U')
H0 = np.array(Hsym.subs({hsym: -1, ssym: 0, Usym: 0}).tolist(), float)   # U-free part
HU = np.array(Hsym.subs({hsym: -1, ssym: 0, Usym: 1}).tolist(), float) - H0  # dH/dU
Sn = np.eye(H0.shape[0])
for Uval in (0.0, 2.0, 8.0):
    Hn = H0 + Uval * HU
    e = eigh(Hn, Sn, eigvals_only=True, subset_by_index=[0, 0])[0]   # lowest only
    print(f"  U={Uval:>4}:  E_gs = {e:+.6f}")


# ---------------------------------------------------------------------------
recipe("8. Site energies (heteroatom models)")
# a-b-c chain with the central site offset by eps (alpha_a = alpha_c = 0)
m8 = Molecule(zero_ii=False, interacting_orbs=['ab', 'bc'],
              subst={'h': ('H_ab', 'H_bc'), 'eps': ('H_bb',), 's': ('S_ab', 'S_bc')},
              subst_2e={'U': ('1111',)}, max_2e_centers=1)
H8, S8 = hamiltonian(m8, generate_dets(2, 2, 3))
H8 = H8.subs({sp.Symbol('H_aa'): 0, sp.Symbol('H_cc'): 0})
eps = sp.Symbol('eps')
assert eps in H8.free_symbols
# at eps = 0 the model reduces exactly to the zero_ii=True chain of recipe 4
assert sp.simplify(H8.subs(eps, 0) - Hsym) == sp.zeros(9, 9)
assert sp.simplify(sp.Matrix(S8) - sp.Matrix(Ssym)) == sp.zeros(9, 9)
print("eps on the diagonal; eps = 0 reduces to the zero_ii=True model : OK")


print("\nALL RECIPES OK")
