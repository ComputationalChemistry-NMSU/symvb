# symvb assignments

A ladder of short, self-checking assignments that introduce valence-bond
theory and the `symvb` package from the ground up. They are the entry tier: a
beginning graduate student who knows bra–ket notation, the general molecular-
orbital picture, and a little Python, but who is new to valence bond, SymPy,
and this package, should be able to work straight through them.

Each assignment is one focused idea, with every new `symvb` call shown in a
small worked example before you are asked to use it. The tier above these, the
teaching notebooks in [`../`](..) (`01_h2_2c2e.ipynb` and its siblings), takes
the same model systems much further and assumes you have done this set first.

## The ladder

Work through them in order; each builds on the last.

| # | File | You will learn to |
|---|------|-------------------|
| A1 | `A1_determinants_as_strings.ipynb` | Write and read Slater determinants as `symvb` strings (case for spin), compute a determinant's spin projection $S_z$, generate a basis, and tell covalent from ionic. |
| A2 | `A2_first_secular_problem.ipynb` | Build the $2\times2$ Hamiltonian and overlap for one electron shared between two orbitals (H₂⁺) and solve $\det(H - E S) = 0$ for the bonding and antibonding levels, overlap kept symbolic. |
| A3 | `A3_h2_resonance.ipynb` | Build the H₂ bond as a covalent/ionic resonance with `System.from_structures`, read off the closed-form ground-state energy, and track the Chirgwin–Coulson weights as the repulsion $U$ grows. |
| A4 | `A4_spin_singlet_triplet.ipynb` | Apply $\hat S^2$ to determinants, diagonalize it, build the singlet and triplet, and see why the H₂ triplet energy is independent of the on-site repulsion. |
| A5 | `A5_allyl_long_bond.ipynb` | Extend to the three-center allyl anion: build the Rumer structures, find the long-bond weight $\tfrac18$, and watch it grow toward $\tfrac12$ as a biradical signature. |

**Prerequisites.** A1 assumes only the background above. A2 and A3 build on A1;
A4 requires A1–A3; A5 requires A4. Each notebook restates what it needs from
the earlier ones in its header.

## How the checkpoints work

Every exercise cell has `...` placeholders for you to replace, followed by a
**checkpoint** cell. A checkpoint runs assertions on your variables and, if
they hold, prints `Checkpoint N passed.`

**A checkpoint that passes means your answer is right.** The checks are exact
wherever exactness is available (exact integers, rationals, or symbolic
equality), so passing is a real confirmation, not an approximate one. Until you
fill in an exercise, its checkpoint will fail; that is expected. If a checkpoint
fails after you have written an answer, read its message, it says what value was
expected.

## How to run

`symvb` is not pip-installed in this tree, so it must be on the Python path.
Start Jupyter **from the repository root**:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. jupyter lab
```

Then open a notebook from `notebooks/assignments/`. If you use the project's
virtual environment, select the **`Python 3.11 (symvb)`** kernel instead of
setting `PYTHONPATH`. Every assignment runs start to finish in a few seconds.

## Solutions

Fully worked solutions live in [`solutions/`](solutions), one per assignment
(`A<N>_<slug>_solutions.ipynb`). They ship in the working tree so you can check
your work.

> **Instructor note.** Because `solutions/` is present in the working tree, it
> will be committed along with everything else. If you plan to hand these out
> for graded work, add `notebooks/assignments/solutions/` to `.gitignore`
> before pushing, or distribute only the student notebooks.

## Rebuilding the notebooks

Each assignment is generated from a script under `_build/`, which emits both
the student notebook and its solution from one set of cell definitions:

```bash
python3 notebooks/assignments/_build/build_a4.py        # writes A4 (student + solution)
PYTHONPATH=. python3 notebooks/_build/_verify_nb.py assignments/solutions/A4_spin_singlet_triplet_solutions.ipynb
```

The verifier executes every code cell of a solution notebook and reports the
run time; a clean run means all checkpoints in that solution pass.
