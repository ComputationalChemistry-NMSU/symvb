# symvb recipes

Task-oriented snippets for the things people most often hand-roll. Each recipe
is a few lines, says what it replaces, and is checked end-to-end by
[`_recipes_check.py`](_recipes_check.py) (run
`PYTHONPATH=. python3 docs/_recipes_check.py`). For the full surface see
[`api.md`](api.md); for a guided tour of the operator algebra see
[`operators_tutorial.md`](operators_tutorial.md).

A reminder on conventions: determinant strings use **case for spin** (lowercase
= α, uppercase = β), e.g. `'aB'` is α on `a`, β on `b`. The determinant basis is
**orthonormal only at zero AO overlap** (`s = 0`); the operator-DSL `matrix()`
and the spin/symmetry helpers assume that, while `Molecule.build_matrix` carries
`s` symbolically. The two-electron integrals are **already inside** the matrices
that carry them, so never scale `o2_matrix` by `U` yourself.

---

## 1. Build and solve a model in three lines

**Problem.** Get a Hamiltonian, its ground state, and structure weights without
the build/transform/charpoly/weight boilerplate.

```python
from symvb import Molecule, FixedPsi, System

m = Molecule(zero_ii=True, interacting_orbs=['ab'],
             subst={'h': ('H_ab',), 's': ('S_ab',)},
             subst_2e={'U': ('1111',)}, max_2e_centers=1)
cov = FixedPsi('aB'); cov.add_str_det('bA', coef=1)   # Heitler-London singlet
ion = FixedPsi('aA'); ion.add_str_det('bB', coef=1)   # symmetric ionic

bond = System.from_structures(m, [cov, ion])
E, c         = bond.ground_state()      # picks the bonding root; E is symbolic
w_cov, w_ion = bond.weights()           # metric-correct Chirgwin-Coulson weights
```

Topology constructors fill in every edge and the on-site `U` for you:

```python
benzene = System.ring(6)                # interacting_orbs ['ab','bc','cd','de','ef','af'], + U
chain   = System.chain(3)               # linear a-b-c
H, S    = benzene.hamiltonian()         # 400x400 SymPy matrices, 2e block folded in
```

The 400-determinant symbolic build takes about a minute (the checker asserts
the ring topology and determinant count rather than re-building it); recipe 7
shows how to scan a matrix that size numerically.

*Replaces:* `sp.Matrix(build_matrix('H')) + sp.Matrix(o2_matrix())`, a hand
basis transform, `charpoly`/`solve` with manual root-picking, and by-hand weight
algebra — and removes the `U·o2` double-counting trap.

---

## 2. Build operator matrices without hand-rolling

**Problem.** You need an `S²`, `η²`, `Sᵢ·Sⱼ`, hopping, or bond-singlet operator
in a determinant basis.

```python
from symvb import operators, spin
from symvb.fixed_psi import generate_dets

dets = [p.dets[0].det_string for p in generate_dets(2, 2, 3)]   # allyl, 9 dets

S2    = operators.s_squared(['a', 'b', 'c']).matrix(dets)       # == spin.s_squared_matrix(dets, 'abc')
SiSj  = operators.s_dot('a', 'b').matrix(dets)                  # Heisenberg coupling
tij   = operators.hop('a', 'b').matrix(dets)                    # c†_a c_b + h.c.
hl_ab = operators.bond_singlet_creator('a', 'b').apply('')      # (|aB⟩+|bA⟩)/√2  as a FixedPsi
```

`operators.matrix(basis)` works at `s = 0`; `.apply(state)` returns a `FixedPsi`;
`.expectation(state)` gives `⟨ψ|O|ψ⟩`. η-pairing: `operators.eta_squared({'a':+1,'b':-1,'c':+1})`.

*Replaces:* hand-coded second-quantization, and writing singlet structures with
`coupled_pairs` bookkeeping when a creator from the vacuum is clearer.

---

## 3. Hückel MOs without writing the C matrix by hand

**Problem.** You want the molecular orbitals, their energies, and a closed-shell
energy for a ring or graph — not a transcribed coefficient matrix.

```python
from symvb import huckel

res = huckel.solve_ring(3)              # cyclopropenyl C3 ring (solve_ring(6) = benzene)
res.eigenvalues                          # (2, -1, -1)
res.energies                             # (2h/(2s+1), -h/(1-s), -h/(1-s))
res.coefficients                         # 3x3 SymPy matrix, rows = MOs over AOs
res.energy_of_occupation([2, 2, 0])      # closed-shell C3H3- anion: psi1^2 psi2^2
```

For an arbitrary graph, including open chains such as allyl:
`huckel.solve(adjacency, site_labels=...)`.

*Replaces:* hand-written MO coefficient matrices like the allyl
`C_mo = [[0.5, 1/√2, 0.5], …]` that show up across the example scripts.

---

## 4. Reduce an FCI block by spin (or η-pairing)

**Problem.** Project a numeric Hamiltonian onto its singlet (or any total-spin)
sector before diagonalizing.

```python
import numpy as np, sympy as sp
from symvb import Molecule, spin
from symvb.system import hamiltonian
from symvb.fixed_psi import generate_dets

m = Molecule(zero_ii=True, interacting_orbs=['ab', 'bc'],
             subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
             subst_2e={'U': ('1111',)}, max_2e_centers=1)
dets = [p.dets[0].det_string for p in generate_dets(2, 2, 3)]
H, S = hamiltonian(m, generate_dets(2, 2, 3))
Hn = np.array(H.subs({sp.Symbol('h'): -1, sp.Symbol('s'): 0, sp.Symbol('U'): 2}).tolist(), float)

S2 = np.array(spin.s_squared_matrix(dets, orbs='abc'), float)
H_singlet, U_singlet = spin.project_onto_S(Hn, S2, target_S=0)   # 9x9 -> 6x6
```

η-pairing on an alternant ring uses `spin.eta_squared_matrix(dets, site_signs, orbs=...)`
the same way (valid only at `s = 0`).

*Replaces:* hand-built `S²` projectors and the eigenvalue bookkeeping to pick out
a spin sector.

---

## 5. Project onto the totally symmetric subspace

**Problem.** You want the trivial-irrep (A₁/A₁g) projector of a point group,
with fermion signs handled (so it works above half filling too).

```python
from symvb import operators

rot  = {'a': 'b', 'b': 'c', 'c': 'a'}          # C_3 cycle of the allyl frame
refl = {'a': 'c', 'c': 'a'}                     # end-swap mirror
P = operators.reynolds_projector([rot, refl]).matrix(dets)   # group closure is automatic
assert (P*P - P).is_zero_matrix                 # it is a projector
rank = P.rank()                                 # dimension of the symmetric subspace
```

For exact-arithmetic symmetry-adapted bases see
`symvb.symmetry.totally_symmetric_basis` / `signed_totally_symmetric_basis`, and
`symvb.symmetry.detect_permutation_group(H, S)` to discover the group from a
built matrix.

*Replaces:* hand-rolled Reynolds sums and orbit construction (which silently drop
fermion signs above half filling).

---

## 6. Expand an MO determinant in AO dets, and verify an eigenpair

**Problem.** Project a closed-shell (or CI) MO determinant into the AO
determinant basis, and prove a claimed eigenpair is exact.

```python
from symvb import huckel, mo_projection

C = huckel.solve_ring(3).coefficients
# MOs 0 and 1 doubly occupied: occupation = (alpha_mo_indices, beta_mo_indices);
# dets is the 9-determinant basis from recipe 2
psi = mo_projection.mo_determinant_in_ao(C, ([0, 1], [0, 1]), dets, site_labels=['a','b','c'])

# verify (H - E S) v == 0 as a polynomial identity
import sympy as sp
h, U = sp.symbols('h U')
H = sp.Matrix([[0, 2*h], [2*h, U]]); S = sp.eye(2)
E = (U - sp.sqrt(U**2 + 16*h**2)) / 2
mo_projection.verify_eigenpair(H, S, sp.Matrix([2*h, E]), E)     # True, or raises
```

`mo_projection.linear_combination_in_ao(C, terms, dets)` does the same for a sum
of MO determinants (`terms` = `(coef, (alpha_idx, beta_idx))` pairs).

*Replaces:* manual MO→AO Slater-determinant expansions and ad-hoc residual checks.

---

## 7. Evaluate a large symbolic matrix fast

**Problem.** A 400-dimensional symbolic `H(h, s, U)` must be scanned numerically.
`lambdify` of a matrix that size compiles for ~a minute and evaluates slowly;
naïve per-point `.subs` is also slow.

```python
import numpy as np, sympy as sp
from scipy.linalg import eigh

h, s, U = sp.symbols('h s U')
# H is first-order in U, so two substitutions give numpy anchor matrices:
H0 = np.array(H.subs({h: -1, s: 0, U: 0}).tolist(), float)            # U-free part
HU = np.array(H.subs({h: -1, s: 0, U: 1}).tolist(), float) - H0       # dH/dU
Sn = np.eye(H0.shape[0])

for Uval in (0.0, 2.0, 8.0):
    Hn = H0 + Uval * HU
    E0 = eigh(Hn, Sn, eigvals_only=True, subset_by_index=[0, 0])[0]   # lowest only
```

The same trick handles linearity in any one-electron edge integral (`h_ab`), and
`subset_by_index=[0, 0]` asks LAPACK for just the ground state. See the benzene
notebook (`notebooks/04_benzene_covalent_only.ipynb`) for the full pattern.

*Replaces:* `lambdify`-ing a 400×400 matrix (slow compile and slow eval) and
re-substituting symbols inside a tight scan loop.

---

## 8. Site energies (heteroatom models)

**Problem.** Your centers are not equivalent (heteroatoms, formal charges): you
need diagonal one-electron site energies, not just couplings.

```python
import sympy as sp
from symvb import Molecule
from symvb.system import hamiltonian
from symvb.fixed_psi import generate_dets

# a-b-c chain with the central site offset by eps (alpha_a = alpha_c = 0)
m = Molecule(zero_ii=False, interacting_orbs=['ab', 'bc'],
             subst={'h': ('H_ab', 'H_bc'), 'eps': ('H_bb',), 's': ('S_ab', 'S_bc')},
             subst_2e={'U': ('1111',)}, max_2e_centers=1)
H, S = hamiltonian(m, generate_dets(2, 2, 3))
H = H.subs({sp.Symbol('H_aa'): 0, sp.Symbol('H_cc'): 0})
```

`zero_ii=False` keeps the diagonal integrals `H_ii`; `subst` unifies any of them
to a named symbol like any other integral, and the ones you want at zero are
substituted away after the build. At `eps = 0` this reduces exactly to the
`zero_ii=True` model of recipe 4. A worked application (the 3c4e long-bond
weight versus a bridge offset) is `examples/allyl_site_asymmetry.py`.

*Replaces:* faking site energies by hand-editing matrix diagonals after the
build.
