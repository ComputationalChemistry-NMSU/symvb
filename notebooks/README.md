# symvb notebooks

A teaching tour of valence-bond theory built from the ground up with the
`symvb` symbolic toolkit. Every result is *derived* in front of you with
sympy, not quoted: from a list of Slater determinants to closed-form energies
and Chirgwin–Coulson structure weights.

The **four main notebooks** are companions to the four model systems of the
manuscript *"Symbolic Valence Bond Theory for Chemists: The symvb Package"*;
each reproduces that system's key closed forms and figures. The
**`additional/`** notebooks extend the toolkit with results outside the
manuscript's scope.

## Audience and prerequisites

- Comfortable with linear algebra (eigenvalue problems, basis changes)
- Familiar with Slater determinants at the level of a one-semester quantum
  chemistry course
- Some Python and Jupyter experience; sympy is introduced as needed

## Install

`symvb` depends only on `sympy`; the notebooks also use `numpy`,
`scipy`, and `matplotlib`.

```bash
pip install git+https://github.com/ComputationalChemistry-NMSU/symvb.git
pip install jupyter numpy scipy matplotlib
```

Or clone and run without installing:

```bash
git clone https://github.com/ComputationalChemistry-NMSU/symvb.git
cd symvb
PYTHONPATH=. jupyter notebook notebooks/
```

## The four main notebooks

Read them in order; each builds on the last.

| # | Notebook | Model system | What you derive | Manuscript |
|---|---|---|---|---|
| 1 | `01_h2_2c2e.ipynb` | H₂, two-center two-electron bond | The covalent/ionic 2×2, closed-form $E(U,h,s)$, the $50/50$ weight invariance at $U=0$, the singlet–triplet gap split into direct-overlap and superexchange terms, and the covalent/resonance **charge-shift** split. | Eqs 5–7, Fig 1 |
| 2 | `02_allyl_long_bond.ipynb` | Allyl anion, three-center four-electron π | The three Rumer structures, the covalent 3×3 block, the long-bond gain $\Delta_{\rm lb}=-\sqrt2|h|$ and weights $(\tfrac14,\tfrac14,\tfrac12)$, the FCI long-bond weight rising $1/8\to1/2$ as a **biradical** signature, and an overlap-only bridge superexchange. | Eqs 8–11, Fig 2 |
| 3 | `03_h2h2_plus_disphenoid.ipynb` | $(\mathrm{H}_2)_2^{\bullet+}$ disphenoid, four-center three-electron mixed valence | The four-structure model and its $4\times4$ Hamiltonian, the symmetric-point $E_0(U)$, the Marcus–Hush critical coupling $|h_l|_{\rm crit}=\lambda/4$, and correlation lowering it (the **Robin–Day** Class II/III boundary). | Eqs 12–16, Fig 3 |
| 4 | `04_benzene_covalent_only.ipynb` | Benzene, six-center aromatic ring | The 400-determinant FCI, the ionicity decomposition $(5,31,31,5)/72$, the covalent share shrinking with ring size, and a **covalent-only** model giving the *wrong sign* of the energy response to weakening one bond. | Eqs 17–18, Figs 4–5 |

## `additional/`

Teaching notebooks on `symvb` capabilities outside the manuscript's four
systems.

| Notebook | What you'll build |
|---|---|
| `additional/allyl_uj_identity.ipynb` | The Coulomb = exchange operator identity $V_U + V_J = \binom{N}{2}\hat S$ at orthogonal AOs, and how turning on overlap *breaks* it: an apparent algebra theorem is really a non-orthogonality statement. |
| `additional/benzene_hubbard_to_heisenberg.ipynb` | Benzene's rational perturbation series in $U/t$; the large-$U$ limit *derives* the Heisenberg coupling $J = 4t^2/U$ from the full FCI, including the Schrieffer–Wolff energy shift textbooks usually drop. |
| `additional/symmetry_as_a_tool.ipynb` | The $400$-dimensional benzene FCI collapsing to $38$ ($D_6$), $22$ ($\hat S^2$), then $14$ ($\eta$-pairing) by symbolic projection, and which symmetries survive AO overlap. |

## Rebuilding the notebooks

Each notebook is generated from a script under `_build/` (main set) or
`additional/_build/`, then checked by executing every code cell:

```bash
python3 notebooks/_build/build_nb1.py                 # writes 01_h2_2c2e.ipynb
PYTHONPATH=. python3 notebooks/_build/_verify_nb.py 01_h2_2c2e.ipynb
```

## License

MIT (same as the parent package).
