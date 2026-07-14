"""Build assignment A3 (student + solutions) from one set of cell definitions.

Run from anywhere:

    python3 notebooks/assignments/_build/build_a3.py

Emits two notebooks:
    notebooks/assignments/A3_h2_resonance.ipynb            (student)
    notebooks/assignments/solutions/A3_h2_resonance_solutions.ipynb
"""
import os
import nbformat as nbf

SLUG = 'A3_h2_resonance'


def make_cells(mode):
    """Return the cell list for ``mode`` in {'student', 'solution'}."""
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

    def code(src):
        cells.append(nbf.v4.new_code_cell(src.strip("\n")))

    def SOLUTION(src_student, src_solution):
        code(src_solution if mode == 'solution' else src_student)

    # -----------------------------------------------------------------
    md(r"""
# Assignment A3 — The chemical bond as resonance: H$_2$

**Goal.** Build the H$_2$ bond as a resonance between two valence-bond
structures, a **covalent** one (one electron on each atom) and an **ionic** one
(both electrons on the same atom), and watch the covalent structure take over as
electron repulsion grows. You will assemble the two structures as `FixedPsi`
objects, combine them with `System.from_structures`, read off the closed-form
ground-state energy

$$
E_{\text{gs}}(U, h)\big|_{s=0} = \frac{U}{2} - \sqrt{\left(\frac{U}{2}\right)^2 + 4h^2},
$$

and compute the Chirgwin-Coulson **weights** of the two structures as a function
of the on-site repulsion $U$.

**Prerequisites.** Assignments A1 (determinant strings) and A2 (the
one-electron secular problem and the `Molecule` / `hamiltonian` /
`ground_state` facade). The parameters here are the same $h$ and $s$ as A2, plus
the on-site Coulomb repulsion $U$.

**Estimated time.** 45 to 60 minutes.

**How to run.** From the repository root:

```
PYTHONPATH=. jupyter notebook notebooks/assignments/
```

or a VS Code kernel rooted at the repository directory. **Exercise** cells hold
`...` placeholders; the **Checkpoint** after each raises until correct, then
prints a confirmation. Some cells build symbolic expressions and take a few
seconds.
""")

    # -----------------------------------------------------------------
    md(r"""
## 1. Setup

We keep the symbols $h$ (resonance integral), $s$ (overlap), and $U$ (on-site
repulsion). The new imports are `FixedPsi`, which represents a single VB
structure, and `System`, the facade that assembles and solves a set of
structures.
""")

    code(r"""
import sympy as sp
sp.init_printing()

from symvb import Molecule, System, FixedPsi

h, s, U = sp.symbols('h s U')
""")

    # -----------------------------------------------------------------
    md(r"""
## 2. The Hubbard molecule and the two structures

For two electrons we need the electron-electron repulsion. We keep the smallest
possible piece: the on-site Coulomb integral $U = \langle aa|aa\rangle$, the
energy cost of putting *both* electrons on the *same* atom. In `Molecule` this
is `subst_2e={'U': ('1111',)}` (rename the on-site two-electron integral to the
symbol `U`) together with `max_2e_centers=1` (keep only on-site repulsion, drop
longer-range two-electron integrals). That combination is exactly the Hubbard
model of H$_2$.
""")

    code(r"""
# The Hubbard molecule: one-electron h and s (as in A2) plus on-site U.
m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab'],
    subst={'h': ('H_ab',), 's': ('S_ab',)},
    subst_2e={'U': ('1111',)},
    max_2e_centers=1,
)
""")

    md(r"""
A **`FixedPsi`** is a labelled linear combination of determinants: exactly one
valence-bond structure. We build two.

- The **covalent** (Heitler-London) structure keeps one electron on each atom,
  spin-paired: `aB + bA`. Start from `FixedPsi('aB')` and add the `'bA'`
  determinant.
- The **ionic** structure is the symmetric combination of the two
  doubly-occupied configurations: `aA + bB`.

The worked cell assembles both and prints them.
""")

    code(r"""
# Worked example: the covalent and ionic structures as FixedPsi objects.
cov = FixedPsi('aB'); cov.add_str_det('bA', coef=1)   # Heitler-London covalent
ion = FixedPsi('aA'); ion.add_str_det('bB', coef=1)   # symmetric ionic
print("covalent structure:", cov)
print("ionic structure:   ", ion)
""")

    # -----------------------------------------------------------------
    md(r"""
## 3. The bond, and its closed-form energy

`System.from_structures(molecule, [structure, ...])` bundles the molecule with a
list of VB structures; its `.ground_state()` returns the symbolic ground-state
energy `E` and coefficient vector `c`, having assembled the $2 \times 2$
Hamiltonian (with the on-site $U$ folded in) and picked the bonding root for
you. This is the two-electron analogue of the `ground_state(H, S)` you called in
A2.
""")

    md(r"""
### Exercise 1 — Assemble and solve the bond

Build `bond = System.from_structures(m, [cov, ion])`, then call
`bond.ground_state()` and unpack it into `E_gs, c_gs`. The checkpoint sets the
overlap to zero and confirms your energy is the textbook orthogonal-atom result
$U/2 - \sqrt{(U/2)^2 + 4h^2}$.
""")

    SOLUTION(
        r"""
# --- EXERCISE 1 ---
bond = ...                                # System.from_structures(m, [cov, ion])
E_gs, c_gs = ...                          # bond.ground_state()
""",
        r"""
bond = System.from_structures(m, [cov, ion])
E_gs, c_gs = bond.ground_state()
""")

    code(r"""
# Checkpoint 1
E_s0 = sp.simplify(E_gs.subs(s, 0))
target = U / 2 - sp.sqrt((U / 2)**2 + 4 * h**2)
assert sp.simplify(E_s0 - target) == 0, \
    "E_gs at s=0 should be U/2 - sqrt((U/2)^2 + 4 h^2)"
print("E_gs(s=0) =", E_s0)
print('Checkpoint 1 passed.')
""")

    md(r"""
Two quick sanity checks on that formula, no work required, just read them:

- at $U = 0$ (no repulsion) it becomes $-2|h|$, twice the bonding-orbital energy
  of A2 at $s = 0$: the closed-shell molecular-orbital picture;
- as $U \to \infty$ the ionic structure is priced out and the energy tends to
  $-4h^2/U \to 0^-$: the Heitler-London limit, two electrons that avoid each
  other.
""")

    code(r"""
print("E_gs at U=0, s=0:", sp.simplify(E_gs.subs({U: 0, s: 0})))   # -2*sqrt(h**2) = -2|h|
""")

    # -----------------------------------------------------------------
    md(r"""
## 4. How much covalent, how much ionic?

The ground state is a mixture $c_{\text{cov}}|\text{cov}\rangle +
c_{\text{ion}}|\text{ion}\rangle$. The **Chirgwin-Coulson weights** turn those
coefficients into a covalent share and an ionic share that sum to one (even when
the structures overlap). `bond.weights()` returns them symbolically as a
length-2 vector, `(w_cov, w_ion)`.

At $U = 0$ and $s = 0$ the bond is the closed-shell MO $\sigma_g^2$, which is an
*exact* fifty-fifty covalent-ionic mixture, a fact worth pinning down exactly.
""")

    md(r"""
### Exercise 2 — The fifty-fifty point

Compute the symbolic weights `w = bond.weights()`, take the covalent weight
`w_cov = w[0]`, and evaluate it at $U = 0$, $s = 0$, simplifying to an exact
fraction as `w_cov_0`. The checkpoint confirms it is exactly $\tfrac12$.
""")

    SOLUTION(
        r"""
# --- EXERCISE 2 ---
w = ...                                   # bond.weights()  (symbolic length-2 vector)
w_cov = ...                               # the covalent weight, w[0]
w_cov_0 = ...                             # sp.simplify(w_cov) at U=0, s=0
""",
        r"""
w = bond.weights()
w_cov = w[0]
w_cov_0 = sp.simplify(w_cov.subs({U: 0, s: 0}))
""")

    code(r"""
# Checkpoint 2
assert w_cov_0 == sp.Rational(1, 2), \
    "at U=0, s=0 the covalent weight is exactly 1/2 (the closed-shell MO)"
print("w_cov(U=0, s=0) =", w_cov_0)
print('Checkpoint 2 passed.')
""")

    # -----------------------------------------------------------------
    md(r"""
## 5. The covalent takeover

Now let the repulsion grow. As $U/|h|$ increases, double occupancy costs more,
the ionic structure is suppressed, and the covalent weight rises from
$\tfrac12$ toward $1$. We scan $U/|h| \in \{0, 2, 4, 8, 16\}$ at $s = 0$.

For a numerical value you *could* substitute into the symbolic `w_cov`, but the
facade offers a faster route: `bond.weights(subs={...})` takes a dictionary of
numeric substitutions and returns the weights as plain floats through a
numerical solver, the same path you would use for a large system where the
symbolic form is unwieldy. The worked cell shows one point.
""")

    code(r"""
# Worked example: the numeric (subs=) weights path at a single point.
w_at_4 = bond.weights(subs={U: 4, h: -1, s: 0})
print("weights at U/|h| = 4:  w_cov = %.4f,  w_ion = %.4f" % (w_at_4[0], w_at_4[1]))
""")

    md(r"""
### Exercise 3 — Scan the covalent weight

Fill the loop so `w_scan` collects the covalent weight
`float(bond.weights(subs={U: Uv, h: -1, s: 0})[0])` at each `Uv` in the list.
The checkpoint confirms the weight climbs monotonically and exceeds $0.9$ by
$U/|h| = 16$.
""")

    SOLUTION(
        r"""
# --- EXERCISE 3 ---
U_over_h = [0, 2, 4, 8, 16]
w_scan = []
for Uv in U_over_h:
    wc = ...                              # covalent weight at U=Uv, h=-1, s=0 (a float)
    w_scan.append(wc)

for Uv, wc in zip(U_over_h, w_scan):
    print(f"  U/|h| = {Uv:2d}   w_cov = {wc:.4f}")
""",
        r"""
U_over_h = [0, 2, 4, 8, 16]
w_scan = []
for Uv in U_over_h:
    wc = float(bond.weights(subs={U: Uv, h: -1, s: 0})[0])
    w_scan.append(wc)

for Uv, wc in zip(U_over_h, w_scan):
    print(f"  U/|h| = {Uv:2d}   w_cov = {wc:.4f}")
""")

    code(r"""
# Checkpoint 3
assert abs(w_scan[0] - 0.5) < 1e-9, "at U=0 the weight is exactly 1/2"
assert all(b > a for a, b in zip(w_scan, w_scan[1:])), \
    "the covalent weight should increase monotonically with U"
assert w_scan[-1] > 0.9, "by U/|h|=16 the bond is predominantly covalent"
assert abs(w_scan[-1] - 0.9851) < 1e-3, \
    "the U/|h|=16 covalent weight is about 0.985"
print('Checkpoint 3 passed.')
""")

    # -----------------------------------------------------------------
    md(r"""
## 6. What you have built, and where it leads

From two hand-written structures you assembled the H$_2$ bond, derived its
ground-state energy in closed form, and traced its covalent/ionic composition
from the fifty-fifty molecular-orbital limit at $U = 0$ to a nearly pure
covalent bond at strong repulsion. That crossover is the valence-bond account of
electron correlation in a chemical bond: the "left-right" correlation that keeps
the two electrons on opposite atoms.

Notebook `01_h2_2c2e.ipynb` continues the same $2 \times 2$ to its other
consequences: the singlet-triplet gap split into a direct-overlap term and an
overlap-suppressed superexchange term, and the covalent/resonance split that
defines a **charge-shift** bond. From here the series moves to more atoms:
the allyl three-center system, the mixed-valence disphenoid, and aromatic
benzene.
""")

    return cells


def write(mode, path):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    }
    nb.cells = make_cells(mode)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nbf.write(nb, path)
    print(f"Wrote {path}  ({len(nb.cells)} cells)")
    return len(nb.cells)


if __name__ == '__main__':
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
    write('student', os.path.join(base, f'{SLUG}.ipynb'))
    write('solution', os.path.join(base, 'solutions', f'{SLUG}_solutions.ipynb'))
