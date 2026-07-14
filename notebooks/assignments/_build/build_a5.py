"""Build assignment A5 -- "Three centers: the allyl anion and the long bond"

Emits two notebooks from one set of cell definitions:
  notebooks/assignments/A5_allyl_long_bond.ipynb                     (student)
  notebooks/assignments/solutions/A5_allyl_long_bond_solutions.ipynb

Run from anywhere:  python3 notebooks/assignments/_build/build_a5.py
"""
import os
import nbformat as nbf

SLUG = 'A5_allyl_long_bond'


def make_cells(solution):
    """Return the list of cells. `solution` selects the filled-in exercise cells."""
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

    def code(src):
        cells.append(nbf.v4.new_code_cell(src.strip("\n")))

    def SOLUTION(student_src, solution_src):
        code(solution_src if solution else student_src)

    # =================================================================
    md(r"""
# Assignment A5 — Three centers: the allyl anion and the long bond

**Goal.** Move from a two-center bond to a three-center π system, the allyl
anion (three p orbitals `a`, `b`, `c` in a row, holding four electrons). You
will enumerate its determinant basis, build the three valence-bond (Rumer)
structures with a bond-creating operator, single out the **long-bond**
structure that pairs the two end atoms across the middle one, and measure its
weight in the ground state. That weight is the algebraic fingerprint of
**biradical character**: it grows from $\tfrac18$ to $\tfrac12$ as electron
correlation strengthens.

**Prerequisites.** Assignment A4. You should be comfortable with determinant
strings (lowercase α, uppercase β), the `symvb` operator language, and
building a Hamiltonian with `Molecule` and the `System` facade.

**Estimated time.** 45–60 minutes.

**How to run.** Start Jupyter from the repository root, or use the
`Python 3.11 (symvb)` kernel:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. jupyter lab
```

Each exercise is followed by a **checkpoint**. A checkpoint that prints
`Checkpoint N passed.` means your answer is correct; the placeholders (`...`)
make the checkpoints fail until you fill them in.
""")

    # =================================================================
    md(r"""
## 1. Setup

Alongside `sympy` and `numpy` we import the operator module, the `Molecule`
matrix-element engine, the `System` facade, and three helpers from the
facade: `ground_state` (lowest root and its eigenvector), `chirgwin_coulson`
(structure weights under a metric), and `structure_vector` (a valence-bond
structure written out as a column over the determinant basis).
""")

    code(r"""
import sympy as sp
from sympy import init_printing
init_printing()
import numpy as np

from symvb import operators as op
from symvb import Molecule, System, FixedPsi
from symvb.fixed_psi import generate_dets
from symvb.system import ground_state, chirgwin_coulson, structure_vector

h, s, U = sp.symbols('h s U')


def det_dict(psi):
    "Turn a FixedPsi into a plain dict {det_string: coefficient}."
    return {d.det_string: c for d, c in psi}
""")

    # =================================================================
    md(r"""
## 2. Exercise 1 — The nine-determinant basis

The allyl anion is four π electrons in three orbitals. In the $S_z = 0$
sector (two α, two β), the number of determinants is
$\binom{3}{2}\binom{3}{2} = 9$. `generate_dets(n_alpha, n_beta, n_orb)`
returns the basis, one determinant per entry.
""")

    SOLUTION(
        student_src=r"""
# EXERCISE 1
# Generate the two-alpha, two-beta, three-orbital basis. Store the list of
# FixedPsi objects in `P` and the list of determinant strings in `dets`.
P = ...
dets = ...
print(len(dets), "determinants:")
print(dets)
""",
        solution_src=r"""
# EXERCISE 1 -- solution
P = generate_dets(2, 2, 3)
dets = [p.dets[0].det_string for p in P]
print(len(dets), "determinants:")
print(dets)
""",
    )

    code(r"""
# CHECKPOINT 1
assert len(dets) == 9, f"expected 9 determinants, got {len(dets)}"
assert set(dets) == {'aAbB', 'aAbC', 'aBbC', 'aAcB', 'aAcC', 'aBcC', 'bAcB', 'bAcC', 'bBcC'}, \
    f"unexpected determinant set: {sorted(dets)}"
print('Checkpoint 1 passed.')
""")

    # =================================================================
    md(r"""
## 3. Building valence-bond structures with a bond creator

A valence-bond **structure** is a specific pattern of bonds and lone pairs.
The friendly way to write one in `symvb` is `bond_singlet_creator(i, j)`: an
operator that adds a spin-paired singlet bond on orbitals `i` and `j`. Acting
on the empty determinant `''` (the vacuum) it creates the two-electron
Heitler–London singlet; acting on a determinant that already holds a lone
pair, it lays the new bond on top of the spectators.
""")

    code(r"""
B_ab = op.bond_singlet_creator('a', 'b')
print("bond(a,b) on the vacuum:", det_dict(B_ab.apply('')))     # (|aB> + |bA>)/sqrt(2)
print("bond(a,b) on lone pair c^2:", det_dict(B_ab.apply('cC')))  # the same bond, times c^2
""")

    md(r"""
Each structure below is one bond plus one doubly occupied "lone pair" orbital,
written `xX` for a pair on orbital `x`. There are three ways to pair three
atoms two at a time:

- $\Phi_{ab}$: a bond on the adjacent pair $(a,b)$, lone pair on $c$;
- $\Phi_{bc}$: a bond on the adjacent pair $(b,c)$, lone pair on $a$;
- $\Phi_{ac}$: a bond on the **end** atoms $(a,c)$, lone pair on the middle
  atom $b$. This is the **long bond**, because it pairs two electrons a full
  bond length apart, across the occupied center. Chemists care about it
  because a large long-bond weight means the molecule carries unpaired
  electron density on its ends, that is, biradical character.

(An equivalent route builds these structures with
`FixedPsi(parent, coupled_pairs=[...])`; the bond creator is used here because
it reads more directly.)
""")

    SOLUTION(
        student_src=r"""
# EXERCISE 2
# Build the three Rumer structures with bond_singlet_creator. Each is a bond
# on one pair of atoms applied to the lone pair on the third atom:
#   Phi_ab = bond(a,b) on lone pair c^2   -> apply to 'cC'
#   Phi_bc = bond(b,c) on lone pair a^2   -> apply to 'aA'
#   Phi_ac = bond(a,c) on lone pair b^2   -> apply to 'bB'   (the long bond)
Phi_ab = ...
Phi_bc = ...
Phi_ac = ...
structs = [Phi_ab, Phi_bc, Phi_ac]
for name, fp in zip(['Phi_ab', 'Phi_bc', 'Phi_ac'], structs):
    print(name, det_dict(fp))
""",
        solution_src=r"""
# EXERCISE 2 -- solution
Phi_ab = op.bond_singlet_creator('a', 'b').apply('cC')
Phi_bc = op.bond_singlet_creator('b', 'c').apply('aA')
Phi_ac = op.bond_singlet_creator('a', 'c').apply('bB')
structs = [Phi_ab, Phi_bc, Phi_ac]
for name, fp in zip(['Phi_ab', 'Phi_bc', 'Phi_ac'], structs):
    print(name, det_dict(fp))
""",
    )

    code(r"""
# CHECKPOINT 2
assert set(det_dict(Phi_ab)) == {'aBcC', 'bAcC'}, "Phi_ab should be the (a,b) bond with a c^2 lone pair"
assert set(det_dict(Phi_bc)) == {'aAbC', 'aAcB'}, "Phi_bc should be the (b,c) bond with an a^2 lone pair"
assert set(det_dict(Phi_ac)) == {'aBbC', 'bAcB'}, "Phi_ac should be the (a,c) long bond with a b^2 lone pair"
for fp in structs:                                    # each structure is normalized
    assert sp.simplify(sum(c**2 for c in det_dict(fp).values()) - 1) == 0
print('Checkpoint 2 passed.')
""")

    # =================================================================
    md(r"""
## 4. From structures to a covalent Hamiltonian

To work with the three structures as a basis we write each as a column over
the nine determinants, using `structure_vector`. There is one subtlety it
handles for us: `symvb` stores determinants in a fixed canonical spin-orbital
order, and rewriting a structure into that order can flip a fermion sign;
`structure_vector` tracks that sign. We apply `nsimplify` so the coefficients
are exact surds rather than floating-point.
""")

    code(r"""
V = sp.Matrix.hstack(*[structure_vector(st, dets) for st in structs]).applyfunc(sp.nsimplify)
print("structure columns V (9 x 3):")
print(V.T)                                  # transpose just to print compactly
# each column is a unit vector:
print("column norms^2:", [sp.simplify((V[:, i].T * V[:, i])[0]) for i in range(3)])
""")

    md(r"""
The three-center chain itself comes from `Molecule.chain(3)`: it fills in the
`a`–`b` and `b`–`c` edges, each with resonance integral $h$ and overlap $s$,
and the on-site repulsion $U$, exactly the allyl Hubbard model. A note on a
neighboring shortcut: `System.chain(3)` would build a system too, but its
default filling is two electrons, not the four of the anion, so we pair the
`Molecule.chain(3)` topology with our own four-electron basis rather than
calling `System.chain(3)`.

We hand the three structures to `System.from_structures`, which builds the
$3\times3$ covalent Hamiltonian and overlap directly in the structure basis.
Setting the atomic overlap $s = 0$ gives the clean orthogonal-orbital form.
""")

    code(r"""
mc = Molecule.chain(3)
cov = System.from_structures(mc, structs)
H_cov_full, S_cov_full = cov.hamiltonian()

H_cov = H_cov_full.subs(s, 0).applyfunc(sp.nsimplify)
S_cov = S_cov_full.subs(s, 0).applyfunc(sp.nsimplify)
print("H_cov =")
print(H_cov)                                 # [[U, 0, -h], [0, U, -h], [-h, -h, U]]
print("S_cov = identity?", S_cov == sp.eye(3))
""")

    md(r"""
Read off the structure of $H_{\text{cov}}$: every structure carries the same
$U$ on the diagonal, the two adjacent-bond structures do **not** couple to
each other (the $(1,2)$ entry is zero), and each couples to the long bond
$\Phi_{ac}$ through $-h$. The long bond is the hinge that links the two
Kekulé-like structures.
""")

    # =================================================================
    md(r"""
## 5. Exercise 3 — Covalent-sector weights $(\tfrac14, \tfrac14, \tfrac12)$

Solve the $3\times3$ covalent problem and read off the structure weights.
`ground_state(H, S)` returns the lowest energy and its eigenvector (at the
default reference $h=-1$, $s=0$); `chirgwin_coulson(c, S)` turns that
eigenvector into weights. You should find $(\tfrac14, \tfrac14, \tfrac12)$:
within the covalent sector, the long bond carries half the weight.
""")

    SOLUTION(
        student_src=r"""
# EXERCISE 3
# Solve the covalent 3x3 and compute its Chirgwin-Coulson weights.
# Use ground_state(H_cov, S_cov) -> (energy, eigenvector), then
# chirgwin_coulson(eigenvector, S_cov, simplify=True) -> list of 3 weights.
E_cov, c_cov = ...
w_cov = ...
w_cov = [sp.nsimplify(x) for x in w_cov]
print("covalent ground-state energy:", sp.simplify(E_cov))
print("covalent weights (w_ab, w_bc, w_ac):", w_cov)
""",
        solution_src=r"""
# EXERCISE 3 -- solution
E_cov, c_cov = ground_state(H_cov, S_cov)
w_cov = chirgwin_coulson(c_cov, S_cov, simplify=True)
w_cov = [sp.nsimplify(x) for x in w_cov]
print("covalent ground-state energy:", sp.simplify(E_cov))
print("covalent weights (w_ab, w_bc, w_ac):", w_cov)
""",
    )

    code(r"""
# CHECKPOINT 3
assert w_cov == [sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 2)], \
    f"expected covalent weights (1/4, 1/4, 1/2), got {w_cov}"
print('Checkpoint 3 passed.')
""")

    # =================================================================
    md(r"""
## 6. Two meanings of "long-bond weight"

The covalent $\tfrac12$ above is the long bond's share **within the three
covalent structures**. The full ground state also contains ionic
determinants (both electrons of a pair on one atom), so the long bond's share
of the **whole** wave function is smaller. These are two different, both
legitimate, numbers, and `symvb` can report either:

- **Over the nine-determinant basis** (the route we use next): project the
  full ground state onto the normalized long-bond structure and square the
  overlap. Because the three covalent structures do not exhaust the wave
  function, these weights sum to *less* than one; the remainder is ionic.
  This answers "what fraction of the entire wave function is the long bond?"
- **`System.weights(structures=...)`** renormalizes over the structure space,
  so its weights sum to one. It answers "of the covalent part, what fraction
  is the long bond?" At $U = 0$ this route reports $0.2$, while the
  nine-determinant route reports $\tfrac18$; the two differ only by that
  renormalization.

We use the nine-determinant route so that "long-bond weight" always means a
fraction of the full wave function.
""")

    md(r"""
## 7. Exercise 4 — The long-bond weight in the full ground state at $U = 0$

At $U = 0$ the Hamiltonian is purely one-electron, so the ground state is the
closed-shell (uncorrelated) determinant. Build the one-electron Hamiltonian
at $h = -1$, $s = 0$ (an ordinary number matrix), find its lowest eigenvalue
and the matching eigenvector, normalize it, and project onto the long-bond
column `V[:, 2]`. The squared overlap is the long-bond weight; you should get
exactly $\tfrac18$.
""")

    SOLUTION(
        student_src=r"""
# EXERCISE 4
# H0 is the one-electron Hamiltonian at U = 0 (h = -1, s = 0), already built.
H0 = sp.Matrix(mc.build_matrix(P, op='H')).subs({s: 0, h: -1})

# (a) lowest eigenvalue of H0 (use key=lambda e: float(e) to compare exact roots)
E_gs = ...
# (b) the matching eigenvector, from the nullspace of (H0 - E_gs * I), normalized
psi = ...
psi = psi / sp.sqrt((psi.T * psi)[0])
# (c) long-bond weight = squared overlap of psi with the long-bond column V[:, 2]
w_ac_0 = ...
w_ac_0 = sp.simplify(w_ac_0)
print("ground-state energy at U = 0:", E_gs)
print("long-bond weight w_ac(U=0):", sp.nsimplify(w_ac_0))
""",
        solution_src=r"""
# EXERCISE 4 -- solution
H0 = sp.Matrix(mc.build_matrix(P, op='H')).subs({s: 0, h: -1})

E_gs = min(H0.eigenvals(), key=lambda e: float(e))
psi = (H0 - E_gs * sp.eye(9)).nullspace()[0]
psi = psi / sp.sqrt((psi.T * psi)[0])
w_ac_0 = (V[:, 2].T * psi)[0] ** 2
w_ac_0 = sp.simplify(w_ac_0)
print("ground-state energy at U = 0:", E_gs)
print("long-bond weight w_ac(U=0):", sp.nsimplify(w_ac_0))
""",
    )

    code(r"""
# CHECKPOINT 4
assert sp.simplify(w_ac_0 - sp.Rational(1, 8)) == 0, \
    f"expected w_ac(U=0) = 1/8, got {sp.nsimplify(w_ac_0)}"
print('Checkpoint 4 passed.')
""")

    md(r"""
For comparison, the renormalized route of the previous section reports the
covalent *composition* instead. Note that it gives $0.2$, not $\tfrac18$:
""")

    code(r"""
w_renorm = System(mc, P).weights(structures=structs, subs={U: 0, h: -1, s: 0})
print("weights(structures=...) at U = 0 (renormalized):", np.round(w_renorm, 4))
""")

    # =================================================================
    md(r"""
## 8. Exercise 5 — Correlation grows the long bond toward $\tfrac12$

Now turn on the on-site repulsion $U$. As $U$ grows, the ionic determinants
become costly and the wave function shifts weight into the covalent
structures, and specifically into the long bond. We scan the full
nine-determinant ground state numerically (the `subs=` path uses a fast
numerical eigensolver) and track the long-bond weight.

Fill in the scan: at each $U$ on the grid, solve the ground state, project onto
the long-bond column, and store the squared overlap. You should see the weight
climb monotonically from $\tfrac18$, heading toward the covalent-sector value
$\tfrac12$ that you found in Exercise 3.
""")

    code(r"""
# The full nine-determinant system and a numeric long-bond-weight function.
sys_fci = System(mc, P)
V_num = np.array(V, float)                    # normalized structure columns as floats

def long_bond_weight(Uv):
    "Long-bond weight in the FCI ground state at (U=Uv, h=-1, s=0)."
    _, psi = sys_fci.ground_state(subs={U: Uv, h: -1, s: 0})
    return float((V_num[:, 2] @ psi) ** 2)
""")

    SOLUTION(
        student_src=r"""
# EXERCISE 5
# Evaluate long_bond_weight on the grid below and collect the values into the
# numpy array `vals`.
grid = np.linspace(0, 40, 41)
vals = ...
for Uv, w in list(zip(grid, vals))[::8]:
    print(f"U/|h| = {Uv:5.1f}   w_ac = {w:.4f}")
""",
        solution_src=r"""
# EXERCISE 5 -- solution
grid = np.linspace(0, 40, 41)
vals = np.array([long_bond_weight(Uv) for Uv in grid])
for Uv, w in list(zip(grid, vals))[::8]:
    print(f"U/|h| = {Uv:5.1f}   w_ac = {w:.4f}")
""",
    )

    code(r"""
# CHECKPOINT 5
vals = np.asarray(vals, float)
assert vals.shape == (41,), "vals should have one entry per grid point"
assert abs(vals[0] - 1/8) < 1e-9, f"the scan should start at 1/8, got {vals[0]}"
assert (np.diff(vals) > 0).all(), "the long-bond weight should increase monotonically with U"
# the covalent-sector limit (U -> infinity) is exactly 1/2, from Exercise 3
assert w_cov[2] == sp.Rational(1, 2), "the covalent-sector long-bond weight should be exactly 1/2"
assert vals[-1] < 0.5, "at finite U the weight stays below the 1/2 covalent limit"
print('Checkpoint 5 passed.')
""")

    md(r"""
The long-bond weight rises from $\tfrac18$ at $U = 0$ toward $\tfrac12$ as
$U \to \infty$, without ever reaching it at finite $U$. Physically, the more
the electrons avoid sharing an atom (large $U$), the more the ground state
looks like two electrons pinned on the end atoms with the middle one holding
a lone pair, the long-bond, biradical, picture. The weight is a continuous
dial for how biradical the π system is.
""")

    # =================================================================
    md(r"""
## 9. Wrap-up

You built a three-center π system from the determinants up and measured its
biradical character:

1. Enumerated the nine-determinant allyl-anion basis.
2. Wrote the three Rumer structures with `bond_singlet_creator`, and named the
   long bond $\Phi_{ac}$ that pairs the two ends across the middle atom.
3. Solved the covalent $3\times3$ block and found the weights
   $(\tfrac14, \tfrac14, \tfrac12)$.
4. Measured the long-bond weight in the full ground state: exactly $\tfrac18$
   at $U = 0$, distinguishing it from the renormalized covalent composition.
5. Watched correlation grow the long-bond weight monotonically from $\tfrac18$
   toward the covalent-sector value $\tfrac12$.

### Where to go next

Notebook `02_allyl_long_bond.ipynb` in the teaching-notebook tier develops
this system much further: the closed-form long-bond energy gain
$\Delta_{\text{lb}} = -\sqrt2\,|h|$, the tie between the long-bond weight and
an independent natural-orbital biradical index, an overlap-only superexchange
through the closed-shell bridge, and what happens when a heteroatom breaks the
symmetry of the center. The same physics is the allyl section of the manuscript
*Symbolic Valence Bond Theory for Chemists*.
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
