"""Build assignment A2 (student + solutions) from one set of cell definitions.

Run from anywhere:

    python3 notebooks/assignments/_build/build_a2.py

Emits two notebooks:
    notebooks/assignments/A2_first_secular_problem.ipynb            (student)
    notebooks/assignments/solutions/A2_first_secular_problem_solutions.ipynb
"""
import os
import nbformat as nbf

SLUG = 'A2_first_secular_problem'


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
# Assignment A2 — One electron, two atoms: your first secular problem

**Goal.** Build the simplest non-trivial valence-bond object, a single electron
shared between two atomic orbitals (the H$_2^+$ cation), and solve it
symbolically. You will assemble the $2 \times 2$ Hamiltonian $H$ and overlap
$S$ with the `symvb` facade, solve the secular equation
$\det(H - E\,S) = 0$ with `sympy`, and read the two roots as the bonding and
antibonding levels

$$
E_{\text{bond}} = \frac{h}{1+s}, \qquad E_{\text{anti}} = -\frac{h}{1-s},
$$

where $h = \langle a|\hat h|b\rangle$ is the resonance integral and
$s = \langle a|b\rangle$ the atomic-orbital overlap. Along the way you will see
why a non-zero overlap pushes the antibonding level *up* more than it pulls the
bonding level *down*.

**Prerequisites.** Assignment A1 (determinant strings and `generate_dets`). A
first course's worth of linear algebra: eigenvalues and $2 \times 2$
determinants. No prior `sympy` needed; the pieces are introduced as used.

**Estimated time.** 45 to 60 minutes.

**How to run.** From the repository root:

```
PYTHONPATH=. jupyter notebook notebooks/assignments/
```

or open in VS Code with a kernel rooted at the repository directory so
`import symvb` resolves. **Exercise** cells have `...` placeholders to fill in;
the **Checkpoint** after each one raises until your answer is right, then prints
a confirmation.
""")

    # -----------------------------------------------------------------
    md(r"""
## 1. Setup

We import `sympy`, the `Molecule` parameter container, the high-level
`hamiltonian` / `ground_state` helpers, and `generate_dets` from A1.
`sp.init_printing()` turns on pretty math rendering for `sympy` output.
""")

    code(r"""
import sympy as sp
sp.init_printing()

from symvb import Molecule, System
from symvb.system import hamiltonian, ground_state
from symvb.fixed_psi import generate_dets

h, s, E = sp.symbols('h s E')
""")

    # -----------------------------------------------------------------
    md(r"""
## 2. Building the one-electron problem

A `Molecule` is a parameter container: it tells `symvb` how to turn
spin-orbital indices into physical integrals when it computes matrix elements.
Three arguments set up our two-site problem:

- **`interacting_orbs=['ab']`** — sites `a` and `b` are coupled (a single
  edge). In a bigger molecule you would list every bonded pair.
- **`subst={'h': ('H_ab',), 's': ('S_ab',)}`** — rename the internal
  one-electron coupling `H_ab` to the symbol `h` and the overlap `S_ab` to `s`,
  so the output matrices are written in the symbols we want to solve for.
- **`zero_ii=True`** — set the on-site energies $H_{aa}, H_{bb}$ to zero. That
  just fixes the energy origin at the isolated-atom level, so the two roots come
  out symmetric about zero.

The basis is one alpha electron in two orbitals: `generate_dets(1, 0, 2)`,
which is the two determinants `['a', 'b']`. The facade call
`hamiltonian(m, basis)` returns `(H, S)`. It takes a `two_electron=` flag that
folds in electron-electron repulsion; with a single electron there is none, so
we pass `two_electron=False` (this also skips the two-electron build, which is
the expensive step on larger systems).
""")

    code(r"""
# Worked example: assemble H and S for one electron on two sites.
m = Molecule(zero_ii=True, interacting_orbs=['ab'],
             subst={'h': ('H_ab',), 's': ('S_ab',)})
basis = generate_dets(1, 0, 2)
print("basis:", [p.dets[0].det_string for p in basis])

H, S = hamiltonian(m, basis, two_electron=False)
""")

    code(r"""
H     # the 2x2 Hamiltonian: zero on the diagonal, h off-diagonal
""")

    code(r"""
S     # the 2x2 overlap: 1 on the diagonal, s off-diagonal
""")

    md(r"""
So `H` is $\begin{pmatrix} 0 & h \\ h & 0 \end{pmatrix}$ and `S` is
$\begin{pmatrix} 1 & s \\ s & 1 \end{pmatrix}$. The off-diagonal of `H` is the
resonance integral $h$; the off-diagonal of `S` is the overlap $s$ that makes
this a *generalised* eigenvalue problem rather than an ordinary one.
""")

    # -----------------------------------------------------------------
    md(r"""
## 3. Solving the secular equation

The allowed energies are the roots of the secular determinant
$\det(H - E\,S) = 0$. Two `sympy` calls do this: `M.det()` gives the symbolic
determinant of a matrix `M`, and `sp.solve(expr, E)` returns the list of `E`
that make `expr` zero. The worked cell demonstrates both on a throwaway
matrix before you use them for real.
""")

    code(r"""
# Worked example: .det() and sp.solve on a small symbolic matrix.
x = sp.Symbol('x')
M_toy = sp.Matrix([[x, 1], [1, x]])
print("det(M_toy) =", M_toy.det())          # x**2 - 1
print("roots      =", sp.solve(M_toy.det(), x))   # [-1, 1]
""")

    md(r"""
### Exercise 1 — Solve $\det(H - E\,S) = 0$

Form the secular determinant of `H - E*S`, solve it for `E`, and simplify each
root. Assign the simplified list to `levels`. The checkpoint confirms the two
roots are exactly $h/(1+s)$ and $-h/(1-s)$.
""")

    SOLUTION(
        r"""
# --- EXERCISE 1 ---
# 1) build the secular determinant of (H - E*S)
# 2) solve it for E
# 3) simplify each root
secular = ...                       # det(H - E*S)
levels  = ...                       # list of simplified roots
""",
        r"""
secular = (H - E * S).det()
levels  = [sp.simplify(r) for r in sp.solve(secular, E)]
""")

    code(r"""
# Checkpoint 1
got = {sp.simplify(r) for r in levels}
want = {sp.simplify(h / (1 + s)), sp.simplify(-h / (1 - s))}
assert got == want, f"expected {{h/(1+s), -h/(1-s)}}, got {levels}"
print("levels:", levels)
print('Checkpoint 1 passed.')
""")

    # -----------------------------------------------------------------
    md(r"""
## 4. Bonding, antibonding, and the overlap asymmetry

With the Hückel sign convention $h < 0$, the bonding level is the lower one,
$E_{\text{bond}} = h/(1+s)$, and the antibonding level is
$E_{\text{anti}} = -h/(1-s)$. At orthogonal orbitals ($s = 0$) they sit
symmetrically at $\pm h$: the bond is stabilised by exactly as much as the
antibond is destabilised.

Turn on the overlap and that symmetry breaks. Measured from the isolated-atom
level (energy $0$), the bonding level is *stabilised* by $|E_{\text{bond}}|$ and
the antibonding level is *destabilised* by $|E_{\text{anti}}|$. Their algebraic
sum $E_{\text{bond}} + E_{\text{anti}}$ tells you which effect wins: a positive
sum means the antibonding level is pushed up more than the bonding level is
pulled down. This asymmetry is why two closed-shell atoms repel (both
antibonding and bonding levels filled gives a net destabilisation), the
overlap-driven origin of Pauli repulsion.

The next worked cell shows how to substitute numbers into a `sympy` expression
and turn the result into a float, which you will need for the exercise.
""")

    code(r"""
# Worked example: numerical evaluation of a symbolic expression.
expr = h / (1 + s)
value = float(expr.subs({h: -1, s: sp.Rational(1, 4)}))   # -1 / (1.25)
print("h/(1+s) at h=-1, s=1/4 =", value)
""")

    md(r"""
### Exercise 2 — The destabilisation asymmetry

Define the two levels as `E_bond` and `E_anti`, form their algebraic sum
`asymmetry = sp.simplify(E_bond + E_anti)`, and evaluate that sum numerically at
$h = -1$, $s = \tfrac14$ as `asymmetry_val` (a float). The checkpoint verifies
`asymmetry` equals $-2hs/(1-s^2)$ and that its value is positive, confirming the
antibonding level moves more.
""")

    SOLUTION(
        r"""
# --- EXERCISE 2 ---
E_bond = ...                              # the bonding level  h/(1+s)
E_anti = ...                              # the antibonding level  -h/(1-s)
asymmetry = ...                           # sp.simplify(E_bond + E_anti)
asymmetry_val = ...                       # float value at h=-1, s=1/4
""",
        r"""
E_bond = h / (1 + s)
E_anti = -h / (1 - s)
asymmetry = sp.simplify(E_bond + E_anti)
asymmetry_val = float(asymmetry.subs({h: -1, s: sp.Rational(1, 4)}))
""")

    code(r"""
# Checkpoint 2
assert sp.simplify(asymmetry - (-2 * h * s / (1 - s**2))) == 0, \
    "E_bond + E_anti should simplify to -2*h*s/(1 - s**2)"
assert asymmetry_val > 0, \
    "at h=-1, s=1/4 the sum is positive: antibonding is destabilised more"
print("asymmetry =", asymmetry, "  value at h=-1, s=1/4 =", round(asymmetry_val, 4))
print('Checkpoint 2 passed.')
""")

    # -----------------------------------------------------------------
    md(r"""
## 5. The same answer from the facade

You solved the secular equation by hand, which is the point of this assignment.
For everyday use `symvb` ships `ground_state(H, S)`, which does the solve and
picks the *lowest* root for you (evaluated at a default reference point,
$h = -1$, $s = 0$, to break the tie). It returns `(E, c)`: the ground-state
energy and its (un-normalised) coefficient vector.
""")

    md(r"""
### Exercise 3 — Confirm the bonding root

Call `ground_state(H, S)` and unpack it into `E_gs, c_gs`. The checkpoint
confirms `E_gs` is exactly the bonding level $h/(1+s)$ you derived above.
""")

    SOLUTION(
        r"""
# --- EXERCISE 3 ---
E_gs, c_gs = ...                          # ground_state(H, S)
""",
        r"""
E_gs, c_gs = ground_state(H, S)
""")

    code(r"""
# Checkpoint 3
assert sp.simplify(E_gs - h / (1 + s)) == 0, \
    "ground_state should return the bonding root h/(1+s)"
print("E_gs =", sp.simplify(E_gs))
print('Checkpoint 3 passed.')
""")

    # -----------------------------------------------------------------
    md(r"""
## 6. Where this goes next

You built and solved a $2 \times 2$ generalised eigenvalue problem entirely in
symbols, and saw the overlap $s$ do something a purely orthogonal model cannot:
split the bonding stabilisation from the antibonding destabilisation.

Assignment **A3** adds the second electron. The two sites `a` and `b` become
the H$_2$ molecule, the covalent structure `aB + bA` competes with the ionic
structure `aA + bB`, and the same machinery, `System`, `ground_state`, and now
`weights`, delivers the bond's energy and its covalent/ionic composition in
closed form. The full treatment, including the singlet-triplet gap and
charge-shift bonding, is Notebook `01_h2_2c2e.ipynb`.
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
