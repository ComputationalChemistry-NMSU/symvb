"""Build assignment A1 (student + solutions) from one set of cell definitions.

Run from anywhere:

    python3 notebooks/assignments/_build/build_a1.py

Emits two notebooks:
    notebooks/assignments/A1_determinants_as_strings.ipynb            (student)
    notebooks/assignments/solutions/A1_determinants_as_strings_solutions.ipynb
"""
import os
import nbformat as nbf

SLUG = 'A1_determinants_as_strings'


def make_cells(mode):
    """Return the cell list for ``mode`` in {'student', 'solution'}.

    Cells are defined once. Exercise cells carry two variants via SOLUTION();
    the flag selects which is emitted. Checkpoint cells are identical in both.
    """
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

    def code(src):
        cells.append(nbf.v4.new_code_cell(src.strip("\n")))

    def SOLUTION(src_student, src_solution):
        code(src_solution if mode == 'solution' else src_student)

    # -----------------------------------------------------------------
    md(r"""
# Assignment A1 — Determinants as strings

**Goal.** Learn how `symvb` names a Slater determinant: a short text string in
which the *case* of each letter carries the electron's spin. By the end you
will read and write these strings fluently, compute the total spin projection
$S_z$ of a determinant, generate a whole determinant basis with one call, and
classify each determinant as *covalent* or *ionic* from the string alone. No
matrix algebra yet: this assignment is entirely about the language the rest of
the package speaks.

**Prerequisites.** Slater determinants at the level of one quantum-chemistry
course, and enough Python to write a short function. This is the first
assignment in the series, so nothing else is assumed. It leads into A2, where
these determinants become the rows and columns of your first secular matrix.

**Estimated time.** 45 to 60 minutes.

**How to run.** You need `symvb` on the import path. From the repository root:

```
PYTHONPATH=. jupyter notebook notebooks/assignments/
```

Or open the notebook in VS Code and pick a kernel whose working directory is
the repository root, so that `import symvb` resolves. Work top to bottom. Cells
marked **Exercise** contain `...` placeholders for you to fill in; the
**Checkpoint** cell right after each one will raise an `AssertionError` until
your answer is correct, and print a confirmation once it is.
""")

    # -----------------------------------------------------------------
    md(r"""
## 1. Setup

The only imports you need are `sympy` (for exact fractions when we get to
$S_z$) and one helper from `symvb` that generates determinant bases. Run the
next cell.
""")

    code(r"""
import sympy as sp
from collections import Counter
from symvb.fixed_psi import generate_dets
""")

    # -----------------------------------------------------------------
    md(r"""
## 2. The case convention

A Slater determinant in `symvb` is written as a string of characters, one per
occupied spin-orbital, in creation (left-to-right) order:

- a **lowercase** letter is an electron with **alpha** ($\uparrow$) spin;
- an **UPPERCASE** letter is an electron with **beta** ($\downarrow$) spin;
- the letter itself names the atomic orbital (site) `a`, `b`, `c`, ...

So `'aB'` is one alpha electron on site `a` and one beta electron on site `b`.
Three strings worth comparing at once:

| string | reading | comment |
|---|---|---|
| `'aB'` | $a_\alpha,\, b_\beta$ | one electron on each site |
| `'bA'` | $b_\alpha,\, a_\beta$ | also one on each site, but the spins are swapped: a *different* determinant |
| `'aA'` | $a_\alpha,\, a_\beta$ | *both* electrons on site `a` |

`'aB'` and `'bA'` are not the same object: which atom carries the spin-up
electron is part of the determinant's identity. The next cell reads a few
strings by pulling out the alpha sites (lowercase) and the beta sites
(uppercased so they compare as plain site labels).
""")

    code(r"""
# Worked example: read a determinant string into its alpha and beta sites.
for ds in ['aB', 'bA', 'aA']:
    alpha_sites = [ch for ch in ds if ch.islower()]
    beta_sites  = [ch.lower() for ch in ds if ch.isupper()]
    print(f"{ds!r:>6}:  alpha on {alpha_sites},  beta on {beta_sites}")
""")

    # -----------------------------------------------------------------
    md(r"""
## 3. Exercise 1 — Write determinant strings

Using the case convention, write the string for each occupation described
below. Assign each to the given variable as a Python string.

1. `d_ionic` — both electrons on site `a` (an ionic determinant).
2. `d_covA` — one alpha electron on `a`, one beta electron on `b`.
3. `d_covB` — one alpha electron on `b`, one beta electron on `a`
   (the spin-swapped partner of `d_covA`).
4. `d_three` — three electrons: alpha on `a`, alpha on `b`, beta on `c`.

The checkpoint compares the *content* of your strings (which sites carry alpha,
which carry beta), so the order in which you list the characters does not
matter, only the spins and sites.
""")

    SOLUTION(
        r"""
# --- EXERCISE 1: fill in each string ---
d_ionic = ...
d_covA  = ...
d_covB  = ...
d_three = ...
""",
        r"""
d_ionic = 'aA'
d_covA  = 'aB'
d_covB  = 'bA'
d_three = 'abC'
""")

    code(r"""
# Checkpoint 1
def _content(ds):
    a = tuple(sorted(ch for ch in ds if ch.islower()))
    b = tuple(sorted(ch.lower() for ch in ds if ch.isupper()))
    return a, b

assert _content(d_ionic) == (('a',), ('a',)), \
    "d_ionic should place both electrons (alpha and beta) on site a"
assert _content(d_covA) == (('a',), ('b',)), \
    "d_covA should be alpha on a, beta on b"
assert _content(d_covB) == (('b',), ('a',)), \
    "d_covB should be alpha on b, beta on a (the swap of d_covA)"
assert _content(d_covA) != _content(d_covB), \
    "d_covA and d_covB are different determinants"
assert _content(d_three) == (('a', 'b'), ('c',)), \
    "d_three should be alpha on a and b, beta on c"
print('Checkpoint 1 passed.')
""")

    # -----------------------------------------------------------------
    md(r"""
## 4. Exercise 2 — The spin projection $S_z$

The total spin projection of a determinant is

$$
S_z = \tfrac{1}{2}\,(N_\alpha - N_\beta),
$$

the number of alpha electrons minus the number of beta electrons, times one
half. Reading the string, $N_\alpha$ is the count of lowercase letters and
$N_\beta$ the count of uppercase letters.

Write the function `sz(ds)` that returns $S_z$ as an exact fraction. Use
`sp.Rational(numerator, denominator)` so that a half-integer stays exact rather
than turning into a float. (For example, `sp.Rational(1, 2)` is $\tfrac12$.)
""")

    SOLUTION(
        r"""
# --- EXERCISE 2: complete this function ---
def sz(ds):
    n_alpha = ...   # count the lowercase characters
    n_beta  = ...   # count the uppercase characters
    return ...      # (n_alpha - n_beta) / 2, as an exact sp.Rational
""",
        r"""
def sz(ds):
    n_alpha = sum(ch.islower() for ch in ds)
    n_beta  = sum(ch.isupper() for ch in ds)
    return sp.Rational(n_alpha - n_beta, 2)
""")

    code(r"""
# Checkpoint 2
assert sz('aB') == 0, "aB has one alpha and one beta: Sz = 0"
assert sz('ab') == 1, "ab has two alpha electrons: Sz = 1"
assert sz('AB') == -1, "AB has two beta electrons: Sz = -1"
assert sz('abC') == sp.Rational(1, 2), "abC has two alpha, one beta: Sz = 1/2"
assert sz('aAbB') == 0, "aAbB is the closed-shell (Sz = 0) determinant"
assert all(isinstance(sz(d), sp.Rational) for d in ['aB', 'abC']), \
    "sz should return an exact sp.Rational, not a float"
print('Checkpoint 2 passed.')
""")

    # -----------------------------------------------------------------
    md(r"""
## 5. Generating a determinant basis

You rarely write out a whole basis by hand. `generate_dets(Nela, Nelb, Norb)`
returns every determinant with `Nela` alpha electrons and `Nelb` beta electrons
distributed over `Norb` orbitals, in the $S_z$ sector fixed by those counts.
Each entry is a `FixedPsi` object (a labelled combination of determinants; here
each holds exactly one). Its single determinant's string is
`p.dets[0].det_string`.

The worked cell below lists the four determinants of the two-electron,
two-orbital problem, `generate_dets(1, 1, 2)`.
""")

    code(r"""
# Worked example: enumerate a basis and read off the strings.
basis_112 = generate_dets(1, 1, 2)
strings_112 = [p.dets[0].det_string for p in basis_112]
print("generate_dets(1, 1, 2) ->", len(basis_112), "determinants:", strings_112)
""")

    md(r"""
The count follows from simple combinatorics: choosing which `Nela` of the
`Norb` orbitals hold an alpha electron is $\binom{\text{Norb}}{\text{Nela}}$
ways, and independently $\binom{\text{Norb}}{\text{Nelb}}$ for the beta
electrons, so the basis size is the product

$$
\binom{\text{Norb}}{\text{Nela}} \times \binom{\text{Norb}}{\text{Nelb}}.
$$

For `(1, 1, 2)` that is $\binom{2}{1}\binom{2}{1} = 2 \times 2 = 4$, matching
the cell above.
""")

    # -----------------------------------------------------------------
    md(r"""
## 6. Exercise 3 — Predict, then check, the basis sizes

Using the product-of-binomials rule, predict the number of determinants in each
basis below, then let `generate_dets` confirm it. Assign your *predicted*
integers; the checkpoint compares them both to the formula and to the actual
basis lengths.

- `n_112` for `generate_dets(1, 1, 2)`
- `n_224` for `generate_dets(2, 2, 4)`
- `n_223` for `generate_dets(2, 2, 3)`

(You can compute $\binom{n}{k}$ with `sp.binomial(n, k)`, or just work it out by
hand.)
""")

    SOLUTION(
        r"""
# --- EXERCISE 3: predict each basis size (an integer) ---
n_112 = ...
n_224 = ...
n_223 = ...
""",
        r"""
n_112 = 4       # C(2,1) * C(2,1)
n_224 = 36      # C(4,2) * C(4,2) = 6 * 6
n_223 = 9       # C(3,2) * C(3,2) = 3 * 3
""")

    code(r"""
# Checkpoint 3
assert n_112 == len(generate_dets(1, 1, 2)) == 4
assert n_224 == len(generate_dets(2, 2, 4)) == 36
assert n_223 == len(generate_dets(2, 2, 3)) == 9
print('Checkpoint 3 passed.')
""")

    # -----------------------------------------------------------------
    md(r"""
## 7. Covalent versus ionic, from the string

A determinant is **ionic** when some site carries *both* an alpha and a beta
electron (a doubly occupied site), and **covalent** when every occupied site
holds exactly one electron. Chemically, ionic determinants are the
charge-separated structures (a lone pair sitting on one atom); covalent
determinants spread one electron per atom.

We measure this with the **ionicity** $n_d$: the number of doubly occupied
sites. A site is doubly occupied when the same letter appears once in lowercase
(alpha) and once in uppercase (beta). Reading `'aA'`, the site `a` appears as
both `a` and `A`, so $n_d = 1$: it is ionic. Reading `'aB'`, no site repeats
across the cases, so $n_d = 0$: it is covalent.

The alpha-set / beta-set extraction from Section 2 is exactly what you need:
the doubly occupied sites are the ones in *both* sets.
""")

    md(r"""
### Exercise 4 — Write the ionicity function

Complete `ionicity(ds)` so it returns $n_d$, the number of sites that carry
both spins. Build the set of alpha sites and the set of beta sites (as plain
lowercase labels), and count how many sites are in both.
""")

    SOLUTION(
        r"""
# --- EXERCISE 4: complete this function ---
def ionicity(ds):
    alpha = ...   # set of lowercase (alpha) sites
    beta  = ...   # set of sites carrying a beta electron, as lowercase labels
    return ...    # how many sites appear in BOTH sets
""",
        r"""
def ionicity(ds):
    alpha = {ch for ch in ds if ch.islower()}
    beta  = {ch.lower() for ch in ds if ch.isupper()}
    return len(alpha & beta)
""")

    code(r"""
# Checkpoint 4a — spot values
assert ionicity('aB') == 0, "aB is covalent: one electron per site"
assert ionicity('bA') == 0, "bA is covalent"
assert ionicity('aA') == 1, "aA is ionic: site a is doubly occupied"
assert ionicity('aAbB') == 2, "aAbB has two doubly occupied sites"
assert ionicity('abC') == 0, "abC spreads three electrons over three sites"
print('Checkpoint 4a passed.')
""")

    md(r"""
Now apply it across a whole basis. In `generate_dets(2, 2, 3)` you place four
electrons (two alpha, two beta) on three sites. With more electrons than sites,
*at least one* site must be doubly occupied, so no determinant here can be
covalent. The histogram of ionicities makes this concrete: how many of the nine
determinants have $n_d = 0$, $n_d = 1$, $n_d = 2$?
""")

    code(r"""
# Checkpoint 4b — ionicity histogram of the (2,2,3) basis
strings_223 = [p.dets[0].det_string for p in generate_dets(2, 2, 3)]
hist = Counter(ionicity(ds) for ds in strings_223)
print("(2,2,3) strings:", strings_223)
print("ionicity histogram (n_d: count):", dict(sorted(hist.items())))

assert hist[0] == 0, "no covalent determinant fits: 4 electrons, 3 sites"
assert hist[1] == 6
assert hist[2] == 3
assert sum(hist.values()) == 9

# By contrast, the (1,1,2) basis (2 electrons, 2 sites) has covalent members:
cov_112 = sum(ionicity(p.dets[0].det_string) == 0 for p in generate_dets(1, 1, 2))
assert cov_112 == 2, "aB and bA are the two covalent determinants of (1,1,2)"
print('Checkpoint 4b passed.')
""")

    # -----------------------------------------------------------------
    md(r"""
## 8. What you can now read

You can now read and write `symvb` determinant strings, compute their $S_z$,
generate a full basis in one call, and sort determinants into covalent and
ionic by inspecting the string. That covalent/ionic split is the backbone of
valence-bond theory: the H$_2$ bond, for instance, is a competition between the
covalent structure `aB + bA` and the ionic structure `aA + bB`, and their
relative weight is what distinguishes an ordinary bond from a charge-shift one.

Notebook `01_h2_2c2e.ipynb` builds exactly that competition and derives the
bond's closed-form energy. Before you get there, assignment **A2** takes the
one-electron version of this two-site problem and turns it into your first
secular determinant, solved symbolically.
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
