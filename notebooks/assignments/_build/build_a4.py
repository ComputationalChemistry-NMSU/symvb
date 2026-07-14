"""Build assignment A4 -- "Spin: singlet or triplet?"

Emits two notebooks from one set of cell definitions:
  notebooks/assignments/A4_spin_singlet_triplet.ipynb            (student)
  notebooks/assignments/solutions/A4_spin_singlet_triplet_solutions.ipynb

Run from anywhere:  python3 notebooks/assignments/_build/build_a4.py
"""
import os
import nbformat as nbf

SLUG = 'A4_spin_singlet_triplet'


def make_cells(solution):
    """Return the list of cells. `solution` selects the filled-in exercise cells."""
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

    def code(src):
        cells.append(nbf.v4.new_code_cell(src.strip("\n")))

    def SOLUTION(student_src, solution_src):
        """One exercise cell with two variants; the build flag picks one."""
        code(solution_src if solution else student_src)

    # =================================================================
    md(r"""
# Assignment A4 — Spin: singlet or triplet?

**Goal.** Use the `symvb` operator language to work with electron spin
directly on Slater determinants. You will apply the total-spin operator
$\hat{S}^2$ to determinants, diagonalize it in the four-determinant basis of
H₂, build the singlet and the triplet as explicit wave functions, confirm
their spin quantum numbers, and finish by seeing a physical consequence: on
the H₂ Hamiltonian, the triplet energy does not depend on the electron
repulsion $U$, while the singlet is pulled below it.

**Prerequisites.** Assignments A1–A3. From A1 you know that `symvb` writes a
determinant as a string with lowercase for α electrons and uppercase for β
(so `'aB'` is one α electron on orbital `a` and one β electron on `b`). From
A3 you know the H₂ covalent/ionic resonance model and how to get its
ground-state energy with the `System` facade.

**Estimated time.** 45–60 minutes.

**How to run.** Start Jupyter from the repository root so that `symvb` is
importable, or select the `Python 3.11 (symvb)` kernel:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. jupyter lab
```

Each exercise is followed by a **checkpoint** cell. A checkpoint that runs
without error and prints `Checkpoint N passed.` means your answer is correct.
The checkpoints in the student notebook will fail until you replace the `...`
placeholders with working code.
""")

    # =================================================================
    md(r"""
## 1. Setup

We need `sympy` for the symbolic algebra, the operator module
`symvb.operators`, and a few determinant tools. The `operators` module is the
second-quantized, orthogonal-orbital path through `symvb`: it builds spin,
number, and hopping operators and lets them act on determinants. It assumes
the determinant basis is orthonormal, which for our purposes means the atomic
orbital overlap $s = 0$.
""")

    code(r"""
import sympy as sp
from sympy import init_printing
init_printing()

from symvb import operators as op
from symvb import Molecule, System, FixedPsi
from symvb.fixed_psi import generate_dets


def det_dict(psi):
    "Turn a FixedPsi (a linear combination of determinants) into a plain dict."
    return {d.det_string: c for d, c in psi}
""")

    # =================================================================
    md(r"""
## 2. Warm-up: reading occupations and spin off a determinant

Before spin, two simpler operators show how the machinery reads a
determinant. `number(orb, spin)` counts electrons of a given spin on an
orbital; with `spin=None` it counts both spins. Applying it to a state
returns that state scaled by the count (or the empty state if the orbital is
unoccupied). We look at `'aB'`: one α on `a`, one β on `b`.
""")

    code(r"""
# 'aB' has an alpha electron on a, and a beta electron on b.
print("number('a', 'alpha') on 'aB':", det_dict(op.number('a', 'alpha').apply('aB')))   # a-alpha occupied
print("number('b', 'alpha') on 'aB':", det_dict(op.number('b', 'alpha').apply('aB')))   # no alpha on b -> empty
print("number('b')          on 'aB':", det_dict(op.number('b').apply('aB')))             # both spins: b-beta occupied
""")

    md(r"""
The projection of total spin on the $z$ axis is
$\hat{S}_z = \tfrac12 \sum_i (\hat n_{i\alpha} - \hat n_{i\beta})$, built by
`s_z(orbs)`. On `'aB'` the one α and one β cancel, so $\hat{S}_z = 0$ and the
operator annihilates the state. On `'ab'` (both electrons α) it returns
$+1 \cdot |ab\rangle$.
""")

    code(r"""
print("s_z(['a','b']) on 'aB':", det_dict(op.s_z(['a', 'b']).apply('aB')))   # Sz = 0 -> empty
print("s_z(['a','b']) on 'ab':", det_dict(op.s_z(['a', 'b']).apply('ab')))   # both alpha -> Sz = +1
""")

    # =================================================================
    md(r"""
## 3. Exercise 1 — Apply $\hat{S}^2$ to a determinant

The total-spin-squared operator is `s_squared(orbs)`. Unlike $\hat{S}_z$, it
does not leave a single determinant alone: acting on `'aB'` it returns a
*combination* of determinants. Compute it and read off the result.

You should find

$$
\hat{S}^2\,|aB\rangle = |aB\rangle - |bA\rangle .
$$

This is the key fact that a single determinant with one α and one β electron
is **not** a spin eigenstate: it is a mixture of a singlet and a triplet, and
$\hat{S}^2$ exposes the mixture.
""")

    SOLUTION(
        student_src=r"""
# EXERCISE 1
# Apply s_squared over the two orbitals ['a','b'] to the determinant 'aB'.
# Store the resulting FixedPsi in `s2_on_aB`, and its dict form in `s2_dict`.
s2_on_aB = ...
s2_dict = ...
print("S^2 |aB> =", s2_dict)
""",
        solution_src=r"""
# EXERCISE 1 -- solution
s2_on_aB = op.s_squared(['a', 'b']).apply('aB')
s2_dict = det_dict(s2_on_aB)
print("S^2 |aB> =", s2_dict)
""",
    )

    code(r"""
# CHECKPOINT 1
assert isinstance(s2_dict, dict), "s2_dict should be a dict of {det_string: coefficient}"
assert s2_dict == {'aB': 1, 'bA': -1}, \
    f"expected S^2|aB> = |aB> - |bA>, i.e. {{'aB': 1, 'bA': -1}}, got {s2_dict}"
print('Checkpoint 1 passed.')
""")

    # =================================================================
    md(r"""
## 4. Exercise 2 — Diagonalize $\hat{S}^2$ in the four-determinant basis

The H₂ $S_z = 0$ space (one α and one β electron in two orbitals) has four
determinants: `'aB'`, `'bA'`, `'aA'`, and `'bB'`. The `.matrix(basis)` method
builds the matrix of an operator in any basis you give it. Build the
$\hat{S}^2$ matrix in this basis and find its eigenvalues.

An operator's eigenvalues of $\hat{S}^2$ are $S(S+1)$: a **singlet** ($S=0$)
gives $0$, a **triplet** ($S=1$) gives $2$. You should find three eigenvalues
equal to $0$ and one equal to $2$: three singlets (the two doubly occupied
"ionic" determinants are both singlets, plus the covalent singlet) and one
triplet component.
""")

    SOLUTION(
        student_src=r"""
# EXERCISE 2
# Build the S^2 matrix over the basis ['aB','bA','aA','bB'] and collect its
# eigenvalues *with multiplicity* into a sorted list `eigs`.
basis = ['aB', 'bA', 'aA', 'bB']
S2 = ...
# S2.eigenvals() returns {eigenvalue: multiplicity}; expand it to a flat list.
eigs = ...
print("S^2 matrix:")
print(S2)
print("eigenvalues (with multiplicity):", eigs)
""",
        solution_src=r"""
# EXERCISE 2 -- solution
basis = ['aB', 'bA', 'aA', 'bB']
S2 = op.s_squared(['a', 'b']).matrix(basis)
eigs = []
for value, mult in S2.eigenvals().items():
    eigs += [value] * mult
eigs = sorted(eigs)
print("S^2 matrix:")
print(S2)
print("eigenvalues (with multiplicity):", eigs)
""",
    )

    code(r"""
# CHECKPOINT 2
assert S2.shape == (4, 4), "S2 should be a 4x4 matrix"
assert sorted(eigs) == [0, 0, 0, 2], \
    f"expected eigenvalues [0, 0, 0, 2] (three singlets, one triplet), got {sorted(eigs)}"
print('Checkpoint 2 passed.')
""")

    # =================================================================
    md(r"""
## 5. Exercise 3 — Build the singlet and the triplet, and confirm their spin

The two combinations that *are* spin eigenstates are

$$
|\text{singlet}\rangle = \tfrac{1}{\sqrt2}\bigl(|aB\rangle + |bA\rangle\bigr),
\qquad
|\text{triplet}\rangle = \tfrac{1}{\sqrt2}\bigl(|aB\rangle - |bA\rangle\bigr).
$$

We build them as `FixedPsi` objects and check their spin with
`op.expectation`. One convention matters here. `expectation(state)` returns
the *unnormalized* $\langle\psi|\hat O|\psi\rangle$, not the average per
particle. If you feed it $|aB\rangle - |bA\rangle$ without the $1/\sqrt2$,
then $\langle\psi|\psi\rangle = 2$ and the raw expectation is doubled. The
next cell shows this explicitly.
""")

    code(r"""
# The unnormalized combinations have <psi|psi> = 2, so expectation() is doubled.
singlet_unnorm = FixedPsi('aB') + FixedPsi('bA')
triplet_unnorm = FixedPsi('aB') - FixedPsi('bA')

S2op = op.s_squared(['a', 'b'])
norm2 = op.identity().expectation(singlet_unnorm)          # <psi|psi> = 2
print("<psi|psi> for the unnormalized state:", norm2)
print("raw   <S^2> on unnormalized singlet:", S2op.expectation(singlet_unnorm))   # 0
print("raw   <S^2> on unnormalized triplet:", S2op.expectation(triplet_unnorm))   # 4 = 2 x 2
print("per-particle (divide by <psi|psi>):",
      S2op.expectation(triplet_unnorm) / norm2)             # 2
""")

    md(r"""
Now build the **normalized** states so the expectation values come out
directly as $S(S+1) = 0$ and $2$. A `FixedPsi` does not multiply by a scalar,
so build each one from an empty `FixedPsi()` and add each determinant with the
coefficient $1/\sqrt2$ using `add_str_det(det_string, coef=...)`.
""")

    SOLUTION(
        student_src=r"""
# EXERCISE 3
# Build the normalized singlet and triplet as FixedPsi objects, each with the
# coefficient 1/sqrt(2) on both determinants. Then set exp_singlet and
# exp_triplet to their S^2 expectation values.
inv = 1 / sp.sqrt(2)

singlet = FixedPsi()
# ... add 'aB' and 'bA', each with coef inv ...
...

triplet = FixedPsi()
# ... add 'aB' with coef inv and 'bA' with coef -inv ...
...

exp_singlet = ...
exp_triplet = ...
print("normalized singlet:", det_dict(singlet))
print("<S^2> singlet:", sp.simplify(exp_singlet))
print("<S^2> triplet:", sp.simplify(exp_triplet))
""",
        solution_src=r"""
# EXERCISE 3 -- solution
inv = 1 / sp.sqrt(2)

singlet = FixedPsi()
singlet.add_str_det('aB', coef=inv)
singlet.add_str_det('bA', coef=inv)

triplet = FixedPsi()
triplet.add_str_det('aB', coef=inv)
triplet.add_str_det('bA', coef=-inv)

exp_singlet = S2op.expectation(singlet)
exp_triplet = S2op.expectation(triplet)
print("normalized singlet:", det_dict(singlet))
print("<S^2> singlet:", sp.simplify(exp_singlet))
print("<S^2> triplet:", sp.simplify(exp_triplet))
""",
    )

    code(r"""
# CHECKPOINT 3
assert sp.simplify(exp_singlet) == 0, \
    f"the normalized singlet should give <S^2> = 0, got {sp.simplify(exp_singlet)}"
assert sp.simplify(exp_triplet) == 2, \
    f"the normalized triplet should give <S^2> = 2, got {sp.simplify(exp_triplet)}"
print('Checkpoint 3 passed.')
""")

    # =================================================================
    md(r"""
## 6. Exercise 4 — Singlet and triplet on the H₂ Hamiltonian

Spin is not just bookkeeping: the singlet and the triplet have different
energies, and they respond differently to the electron repulsion $U$. We
build the same H₂ Hubbard model from A3, now written explicitly over the
four-determinant basis, keeping the resonance integral $h$, the atomic
overlap $s$, and the on-site repulsion $U$ symbolic.

The triplet is the antisymmetric covalent combination
$|aB\rangle - |bA\rangle$. Because both determinants keep the two electrons
on different atoms, no doubly occupied ("ionic") determinant enters, and the
triplet never pays the on-site repulsion $U$. The singlet, in contrast, mixes
in the ionic determinants and is stabilized by that mixing.

The next cell builds `H` and `S` and the singlet ground-state energy
`E_sing` with the `System` facade (as in A3). Your task is to build the
triplet energy as a Rayleigh quotient and confirm two statements.
""")

    code(r"""
h, s, U = sp.symbols('h s U')

# The H2 Hubbard molecule from A3: one edge a-b, on-site repulsion U only.
m = Molecule(zero_ii=True, interacting_orbs=['ab'],
             subst={'h': ('H_ab',), 's': ('S_ab',)},
             subst_2e={'U': ('1111',)}, max_2e_centers=1)
P = generate_dets(1, 1, 2)
dets = [p.dets[0].det_string for p in P]
print("four-determinant basis:", dets)                       # ['aA', 'aB', 'bA', 'bB']

H = sp.Matrix(m.build_matrix(P, op='H')) + sp.Matrix(m.o2_matrix(P))
Smat = sp.Matrix(m.build_matrix(P, op='S'))

# Singlet ground state, via the facade over the covalent and ionic structures.
cov = FixedPsi('aB'); cov.add_str_det('bA', coef=1)
ion = FixedPsi('aA'); ion.add_str_det('bB', coef=1)
E_sing, _ = System.from_structures(m, [cov, ion]).ground_state()
print("E_singlet(h, s, U) =", sp.simplify(E_sing))
""")

    md(r"""
Build the triplet as a coefficient vector over the four determinants (order
`['aA', 'aB', 'bA', 'bB']`), then form its energy as the Rayleigh quotient

$$
E_{\text{triplet}} = \frac{\mathbf{t}^{\top} H\, \mathbf{t}}
                          {\mathbf{t}^{\top} S\, \mathbf{t}} .
$$
""")

    SOLUTION(
        student_src=r"""
# EXERCISE 4
# Build the triplet coefficient vector t over the basis ['aA','aB','bA','bB'],
# i.e. |aB> - |bA> (zero on the two ionic determinants). Then compute the
# Rayleigh-quotient energy E_trip and simplify it.
idx = {d: i for i, d in enumerate(dets)}
t = sp.zeros(4, 1)
# ... set the aB and bA entries to +1 and -1 ...
...

E_trip = ...
E_trip = sp.simplify(E_trip)
print("E_triplet(h, s, U) =", E_trip)
""",
        solution_src=r"""
# EXERCISE 4 -- solution
idx = {d: i for i, d in enumerate(dets)}
t = sp.zeros(4, 1)
t[idx['aB']] = 1
t[idx['bA']] = -1

E_trip = (t.T * H * t)[0] / (t.T * Smat * t)[0]
E_trip = sp.simplify(E_trip)
print("E_triplet(h, s, U) =", E_trip)
""",
    )

    code(r"""
# CHECKPOINT 4
# (a) The triplet energy does not depend on U at all.
assert U not in E_trip.free_symbols, \
    f"E_trip should be independent of U, but it contains U: {E_trip}"
# (b) At orthogonal atomic orbitals (s = 0) the triplet sits exactly at zero.
assert sp.simplify(E_trip.subs(s, 0)) == 0, \
    f"E_trip should be 0 at s = 0, got {sp.simplify(E_trip.subs(s, 0))}"
# (c) The singlet is stabilized below the triplet for every U (at s = 0, h = -1).
for Uv in [0, 1, 2, 4, 8, 20]:
    e_s = float(E_sing.subs({h: -1, s: 0, U: Uv}))
    e_t = float(E_trip.subs({h: -1, s: 0, U: Uv}))
    assert e_s < e_t, f"singlet should lie below triplet at U={Uv}: {e_s} vs {e_t}"
print('Checkpoint 4 passed.')
""")

    md(r"""
Read the result back as chemistry. At $s = 0$ the triplet energy is exactly
zero for **every** value of $U$: with the two electrons forced onto different
atoms it feels no on-site repulsion, so raising or lowering $U$ does nothing
to it. The singlet, at $s = 0$ and $h = -1$, is

$$
E_{\text{singlet}} = \frac{U}{2} - \sqrt{\Bigl(\frac{U}{2}\Bigr)^2 + 4} ,
$$

which is negative for every finite $U$: it always lies below the triplet, and
the singlet–triplet gap $\sqrt{(U/2)^2 + 4} - U/2$ shrinks as $U$ grows but
never closes. The singlet wins because it can borrow stability from the ionic
determinants; the triplet cannot.
""")

    # =================================================================
    md(r"""
## 7. Wrap-up

You worked with electron spin directly on Slater determinants:

1. Saw that a single one-α-one-β determinant is not a spin eigenstate:
   $\hat{S}^2|aB\rangle = |aB\rangle - |bA\rangle$.
2. Diagonalized $\hat{S}^2$ in the four-determinant H₂ basis and found three
   singlets and one triplet ($S(S+1) = 0$ three times, $2$ once).
3. Built the normalized singlet and triplet as `FixedPsi` states and
   confirmed their spin through `expectation`, learning that `expectation`
   returns the unnormalized $\langle\psi|\hat O|\psi\rangle$.
4. Found on the H₂ Hamiltonian that the triplet energy is independent of the
   on-site repulsion $U$ and sits at zero when the orbitals are orthogonal,
   while the singlet is stabilized below it for every $U$.

### Where to go next

Notebook `01_h2_2c2e.ipynb` in the teaching-notebook tier takes this same
four-determinant system and derives the full closed-form ground-state energy
$E(U, h, s)$, the covalent/ionic weights, and the singlet–triplet gap
including the atomic-overlap corrections. Assignment A5 moves from two centers
to three: the allyl anion and its long bond.
""")

    return cells


def write(cells, path):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    }
    nb.cells = cells
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nbf.write(nb, path)
    print(f"Wrote {path}  ({len(cells)} cells)")


here = os.path.dirname(__file__)
asg = os.path.normpath(os.path.join(here, '..'))

write(make_cells(solution=False), os.path.join(asg, f'{SLUG}.ipynb'))
write(make_cells(solution=True), os.path.join(asg, 'solutions', f'{SLUG}_solutions.ipynb'))
