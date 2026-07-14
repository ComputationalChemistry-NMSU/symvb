# symvb

Symbolic valence-bond theory in Python. `symvb` builds Slater
determinants over user-supplied orbital sets, computes one- and
two-electron matrix elements via Löwdin's cofactor expansion, and
returns the resulting Hamiltonian and overlap as `sympy` matrices —
everything stays symbolic in the bond, overlap, and Coulomb
parameters until you choose to substitute.

The package targets pedagogical and small-system research use:
H₂, allyl, benzene, (H₂)ₙ⁺ chains, cyclic polyenes, and similar
model systems where closed-form analysis is preferable to
purely numerical FCI.

## Install

From [PyPI](https://pypi.org/project/symvb/):

```bash
pip install symvb
```

Or install the development version straight from GitHub:

```bash
pip install git+https://github.com/ComputationalChemistry-NMSU/symvb.git
```

Or clone and run without installing:

```bash
git clone https://github.com/ComputationalChemistry-NMSU/symvb.git
cd symvb
PYTHONPATH=. python3 examples/h2_hubbard_bond.py
```

Python ≥ 3.8 (tested on 3.8 and 3.11). The symbolic core needs only
`sympy`; the numeric helpers (`ground_state(subs=...)`, the scan
patterns) and most examples also use `numpy` and `scipy`. For the
teaching notebooks add `jupyter` and `matplotlib`.

## Quick start

```python
from symvb import Molecule, FixedPsi, System

m = Molecule(zero_ii=True, interacting_orbs=['ab'],
             subst={'h': ('H_ab',), 's': ('S_ab',)},
             subst_2e={'U': ('1111',)}, max_2e_centers=1)
cov = FixedPsi('aB'); cov.add_str_det('bA', coef=1)   # Heitler-London singlet
ion = FixedPsi('aA'); ion.add_str_det('bB', coef=1)   # symmetric ionic

bond = System.from_structures(m, [cov, ion])
E, c         = bond.ground_state()      # symbolic ground-state energy + vector
w_cov, w_ion = bond.weights()           # Chirgwin-Coulson weights

benzene = System.ring(6)                # topology fills in every edge + on-site U
H, S    = benzene.hamiltonian()         # 400x400 sympy matrices, 2e block folded in (~1 min build)
```

The underlying symbolic matrices stay fully accessible
(`m.build_matrix(P, op='H')`, `m.o2_matrix(P)`). See
[docs/recipes.md](docs/recipes.md) for common tasks and
[docs/api.md](docs/api.md) for the full surface.

## Documentation

- [docs/api.md](docs/api.md) — API reference: modules, signatures, conventions.
- [docs/recipes.md](docs/recipes.md) — task-oriented cookbook; every snippet is
  verified by `docs/_recipes_check.py`.
- [docs/operators_tutorial.md](docs/operators_tutorial.md) — the second-quantized
  operator algebra, in depth.

[docs/README.md](docs/README.md) is the index.

## Teaching notebooks

`notebooks/` derives valence-bond theory from the ground up with `symvb`; every
result is *derived*, not quoted. The four main notebooks are companions to the
manuscript's four model systems:

1. **H₂, the two-center two-electron bond** — covalent/ionic weights versus
   correlation, the singlet–triplet gap, and charge-shift bonding.
2. **The allyl anion (3c4e)** — a long-bond Rumer structure as a biradical
   signature.
3. **The (H₂)₂⁺ disphenoid (4c3e)** — the Robin–Day Class II/III crossover.
4. **Benzene** — a covalent-only model inverts the sign of the energy response.

`notebooks/additional/` keeps further topics (the U=J operator identity, the
Hubbard→Heisenberg mapping, symmetry projection). See
[notebooks/README.md](notebooks/README.md), and open in Jupyter with:

```bash
PYTHONPATH=. jupyter notebook notebooks/
```

## Examples

`examples/` collects 88 stand-alone scripts, most of them cited from
the manuscript's source-data records. Worked examples cover H₂, H₂⁺,
allyl (anion / cation / triplet), benzene, (H₂)ₙ⁺ chains for
$n = 2, 3, 4$, cyclobutadiene dianion, cyclopentadienyl anion, F₂,
and benzene + O₃ aromaticity loss. Run any script from the repo root
with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python3 examples/benzene_heisenberg_mapping.py
```

A few scripts cache large symbolic matrices under `/tmp` on first run
(subsequent runs are fast), and the plotting scripts write their PNGs
into `figures/`.

`examples/qualitative_vb/` is a textbook companion: eight all-assert
scripts that prove the classic results of qualitative VB theory (the
two-, one-, and three-electron bonds, the VB mixing rule, and Rumer
resonance in allyl, butadiene, and benzene) as exact symbolic
identities. See its [README](examples/qualitative_vb/README.md).

## Tests

```bash
PYTHONPATH=. python3 -m pytest symvb -q     # 201 tests
PYTHONPATH=. python3 docs/_recipes_check.py # every documented recipe, executed
```

## License

MIT.

## Funding

Research reported in this repository was supported by an
Institutional Development Award (IDeA) from the National Institute
of General Medical Sciences of the National Institutes of Health
under grant number P20GM103451.
