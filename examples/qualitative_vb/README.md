# Qualitative valence bond theory, verified symbolically

A companion to Chapter 3 ("Basic Valence Bond Theory") of S. Shaik and
P. C. Hiberty, *A Chemist's Guide to Valence Bond Theory*, Wiley (2008).
Each script states a classic qualitative-VB result and proves it as an
exact symbolic identity with symvb: every claimed formula ends in a
sympy `assert`, followed by a short numeric table. All scripts run in
seconds.

Run from the repo root (or from anywhere; the scripts locate the package
themselves):

```bash
PYTHONPATH=. python3 examples/qualitative_vb/01_hl_bond_and_triplet.py
```

## The scripts and what they prove

| script | textbook topic | exact result proved |
|---|---|---|
| `01_hl_bond_and_triplet.py` | the two-electron bond; Pauli repulsion | singlet stabilization `2βs/(1+s²)`, triplet destabilization `−2βs/(1−s²)`, with the reduced resonance integral `β = h_ab − s·h_aa` emerging symbolically |
| `02_one_electron_bond.py` | the one-electron bond (H₂⁺ type) | `E± = (h_aa ± h_ab)/(1 ± s)`; bond energy `β/(1+s)`, exactly half the two-electron MO bond at every `s` |
| `03_three_electron_bond.py` | the three-electron bond (He₂⁺ type) | bond energy `β(1−3s)/(1−s²)`: deepest at `s = 0`, zero at `s = 1/3`, repulsive beyond; resonance share `−β/(1+s)` |
| `04_vb_mixing_rule.py` | nonorthogonal VB mixing rule | `ΔE = (H₁₂ − E₁S₁₂)²/(E₁ − E₂)` proved as the exact second-order term; applied to H₂ it gives the charge-shift resonance energy `−4h²(1−s²)²/(U(1+s²)³)` |
| `05_allyl_radical_resonance.py` | Rumer structures; allyl resonance | `⟨R1\|R2⟩ = (2s²+1)/(s²+2) → 1/2` at `s = 0`; `RE = −2hs/((s²+1)(s²+2))`; weights exactly 1/2 : 1/2 |
| `06_butadiene_rumer.py` | butadiene: one dominant structure | `⟨K\|D⟩ → 1/2`; weights `→ (1/2+√3/6, 1/2−√3/6) ≈ (0.79, 0.21)`; per-electron RE `0.116\|h\|s` vs allyl `0.333\|h\|s` |
| `07_benzene_kekule.py` | Kekulé resonance | `⟨K1\|K2⟩ → 1/4` at `s = 0`; Kekulé pair mixes 50 : 50; the 5-structure (Kekulé + Dewar) covalent state lies lower, with the Kekulé pair keeping ~2/3 of the weight |
| `08_rumer_spin_purity.py` | spin eigenfunctions | every Rumer/Kekulé/Dewar structure above is an `S²` eigenstate with eigenvalue `S(S+1)`; flipping one pair-coupling sign destroys spin purity |

## Conventions

- Det strings: lowercase = α, uppercase = β (`'aB'` is a(α) b(β)).
- Chapter 3 works at the one-electron (effective-Hamiltonian) level; the
  scripts use `two_electron=False` accordingly, except script 04 where the
  on-site repulsion `U` enters symbolically. At this level the qualitative
  textbook formulas are *exact*, which is what the asserts check.
- `β = h_ab − s·h_aa` is the reduced resonance integral; scripts 01-03 keep
  the site energy `h_aa` symbolic so β emerges rather than being assumed.
- A recurring exact statement: with `h_aa = 0`, purely covalent resonance is
  overlap-driven; the covalent-covalent coupling vanishes identically at
  `s = 0` (scripts 05-07 assert `H(s=0) = 0`).
- Which sign couples a pair of determinants into a singlet depends on
  symvb's canonical ordering; the scripts never assume it, they verify it
  with `S²` (script 08 shows what goes wrong otherwise). `S²` machinery is
  valid at `s = 0` only.

## Where to go next in this repo

- Covalent-ionic mixing and charge-shift bonding: `examples/h2_charge_shift.py`
- The 3-centre 4-electron long bond (allyl anion): `examples/allyl_long_bond_vb.py`
- Full benzene ionicity and aromaticity analysis: `examples/benzene_aromaticity_loss.py`,
  `examples/make_fig_benzene_ionicity.py`
- Teaching notebooks with exercises: `notebooks/` and `notebooks/assignments/`
