# Operators on Slater determinants — a `symvb` tutorial

This tutorial walks through how second-quantized operators act on Slater
determinants in `symvb`, using the `symvb.operators` module. Every example
runs on a small system (H₂, allyl, benzene) so you can paste it into a
Python session and check the output yourself.

The audience is chemists who already write Hartree–Fock and CI by hand
and want a concrete picture of how the spin-statistics machinery
combines with the bookkeeping of fermion signs in `symvb`.

> **Scope.** `symvb.operators` is the *second-quantized*, *orthogonal-AO*
> path: spin operators, number / double-occupancy, hopping, η-pairing,
> orbital-permutation symmetries, and the projectors / VB structures
> built from them, all under the assumption that the AOs are orthogonal
> (s = 0). For non-orthogonal `H, S` matrix construction (the s ≠ 0
> path with one- and two-electron integrals via Löwdin cofactor) keep
> using `symvb.Molecule.build_matrix`. The two paths are complementary;
> they share `FixedPsi` as the wavefunction representation.

---

## Contents

1. [Determinants as strings](#1-determinants-as-strings)
2. [Primitives: c and c†](#2-primitives-c-and-c)
3. [Number operators: nᵢ, nᵢσ, double occupancy](#3-number-operators)
4. [Hopping](#4-hopping)
5. [Spin: Sᵢ_z, S±, S²](#5-spin-operators)
6. [Local spin correlations: Sᵢ·Sⱼ](#6-local-spin-correlations)
7. [Singlet projector and the bond singlet creator](#7-singlet-projector-and-bond-creator)
8. [η-pairing](#8-η-pairing)
9. [Symmetry: orbital permutations and Reynolds projectors](#9-symmetry-orbital-permutations-and-reynolds-projectors)
10. [Algebra and matrix construction](#10-algebra-and-matrix-construction)

---

## 1. Determinants as strings

`symvb` writes a Slater determinant as a string in which **lowercase
letters mark α electrons** and **uppercase letters mark β electrons**.
Creation order equals string order. So `aB` is

|aB⟩ = c†_{a,α} c†_{b,β} |0⟩

— one α electron on orbital `a`, one β electron on orbital `b`.

| String | α-set | β-set | Picture                       |
|--------|-------|-------|-------------------------------|
| `aB`   | {a}   | {b}   | one bond electron of each spin |
| `bA`   | {b}   | {a}   | same orbitals, swapped spins  |
| `aA`   | {a}   | {a}   | doubly-occupied orbital `a`   |
| `aAbB` | {a,b} | {a,b} | half-filled, both occupied    |

`symvb.operators` uses the **interleaved canonical form** — the same
form produced by `symvb.functions.generate_det_strings` and used by
`symvb.spin._canon_det`. Sorted α and β letters are paired up in lex
order (lowercase α first within each pair), with extras of the longer
block appended. For the four dets above the canonical form is `aB`,
`bA`, `aA`, `aAbB`. Strings the user passes can be in any order;
`canonicalize(s)` returns the canonical form together with the ±1 sign
that relates them.

```python
from symvb import operators as op

op.canonicalize('aAbB')   # ('aAbB', +1)  — already canonical
op.canonicalize('abAB')   # ('aAbB', -1)  — alpha-block-first → interleaved
op.canonicalize('Ba')     # ('aB',  -1)   — creation order reversed
```

The −1 in the second and third cases is Jordan–Wigner bookkeeping: a
creation operator had to commute past another to put the dets in
interleaved-canonical order. Jordan–Wigner ordering of spin-orbitals
is `(a,α) < (a,β) < (b,α) < (b,β) < …`.

---

## 2. Primitives: c and c†

Every operator in this module is built from elementary annihilators
`c(orb, spin)` and creators `cdag(orb, spin)`. Spins accept the labels
`'alpha'`, `'a'`, `0` (or `'beta'`, `'b'`, `1`).

```python
op.c('a', 'alpha')        # c_{a, α}
op.cdag('b', 'beta')      # c†_{b, β}
```

The action of an operator on a det is returned as a `symvb.FixedPsi` —
the same linear-combination-of-determinants object you build by hand
with `+`, `-`, and scalar `*`. Its `repr` lists each det with its
coefficient:

```python
op.c('a', 'alpha').apply('aB')      # |B|        — α on a removed (FixedPsi with one det)
op.cdag('a', 'beta').apply('B')     # |AB|       — β added on a (one β to its right in JW order)
```

The second result is a det with two β electrons (one on `a`, one on
`b`) — written canonical as `'AB'`. The Jordan–Wigner sign is +1 here
because the new mode `(a, β)` lands ahead of an empty `(b, α)` slot:
no occupied modes to its left.

To convert a `FixedPsi` to a plain dict for inspection:

```python
psi = op.s_squared(['a', 'b']).apply('aB')
{d.det_string: c for d, c in psi}    # {'aB': 1, 'bA': -1}
```

(Make sure you can rederive that on paper before moving on — every
sign in this module comes from the JW rule above.)

---

## 3. Number operators

`number(orb, spin)` is the spin-resolved occupation `n̂_{i,σ} = c†_{i,σ} c_{i,σ}`.
With `spin=None`, it returns the total `n̂_i = n̂_{i,α} + n̂_{i,β}`.
`double_occ(orb)` is the on-site double-occupancy operator
`n̂_{i,α} n̂_{i,β}` — the operator behind the Hubbard `U`.

```python
op.number('a', 'alpha').apply('aB')   # |aB|     — α on a is occupied
op.number('b', 'alpha').apply('aB')   # (empty)  — no α on b
op.number('a').apply('aB')            # |aB|     — total: α only
op.number('b').apply('aB')            # |aB|     — total: β only
op.double_occ('a').apply('aA')        # |aA|     — 'a' is doubly occupied
op.double_occ('a').apply('aB')        # (empty)  — 'a' is only singly occupied
```

These are the operators you'd use to compute ionicity, sketch a
Mulliken-style population, or evaluate the diagonal Hubbard term:

```python
# Hubbard double-occupancy on each of 4 ring sites:
D = sum(op.double_occ(s) for s in ['a', 'b', 'c', 'd'])
```

---

## 4. Hopping

`hop(i, j, spin=None, hermitian=True)` returns

  ĥ_{ij} = c†_{i,σ} c_{j,σ}    + h.c. (if `hermitian=True`)

with `spin=None` summing both spin channels. This is the operator
behind a Hückel/PPP `t` integral.

```python
op.hop('a', 'b', 'alpha', hermitian=False).apply('bA')
#  c†_{a,α} c_{b,α} |bA>  =  |aA>     — α moves from b to a
#  -> FixedPsi: |aA|
```

The hermitian version splits into two terms; symbolically:

```python
H = sp.Symbol('t')
T = -H * op.hop('a', 'b')                # -t (c†_a c_b + c†_b c_a) summed over spin
```

Notice that `hop('a', 'a')` is just the on-site number `n̂_a`. The code
handles `i == j` by skipping the duplicate hermitian conjugate.

---

## 5. Spin operators

Three layers, all built from `c`/`c†`:

| Layer | Operator     | Definition                         |
|-------|--------------|------------------------------------|
| 1     | `s_z(orbs)`  | (½) Σᵢ (n̂_{i,α} − n̂_{i,β})        |
| 2     | `s_plus(orbs)` / `s_minus(orbs)` | Σᵢ c†_{i,α} c_{i,β}  /  Σᵢ c†_{i,β} c_{i,α} |
| 3     | `s_squared(orbs)` | S₊S₋ + S_z² − S_z              |

Action on a det is just the rule that lowercase = α, uppercase = β:
S_z counts case differences, S₊ flips one upper-case letter to lower,
S₋ does the reverse, with JW signs along the way.

```python
op.s_z(['a', 'b']).apply('aB')          # (empty) — Sz = 0 on a one-α-one-β state
op.s_z(['a', 'b']).apply('aBcD')        # (empty) — Sz still 0 on this 4-electron state
op.s_plus(['a', 'b']).apply('aB')       # |ab|     — both electrons now α (M_S = +1 triplet)
op.s_squared(['a', 'b']).apply('aB')    # |aB|-|bA|
```

The last line says

  Ŝ² |aB⟩ = |aB⟩ − |bA⟩

In this two-orbital basis the singlet and the triplet T₀ in symvb
sign convention are

  |S⟩  = (|aB⟩ + |bA⟩)/√2     ⟶ S²|S⟩ = 0
  |T₀⟩ = (|aB⟩ − |bA⟩)/√2     ⟶ S²|T₀⟩ = 2 |T₀⟩

so |aB⟩ = (|S⟩ + |T₀⟩)/√2 and indeed

  Ŝ²|aB⟩ = 0·|S⟩/√2 + 2·|T₀⟩/√2 = √2 |T₀⟩ = |aB⟩ − |bA⟩.   ✓

The matrix form is the cleanest way to read the eigenstructure:

```python
S2 = op.s_squared(['a', 'b']).matrix(['aB', 'bA', 'aA', 'bB'])
# Matrix([[1, -1, 0, 0],
#         [-1, 1, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0]])
# eigvals: 0, 0, 0, 2 — three singlets (covalent + 2 ionic) and one T₀
```

This reproduces `symvb.spin.s_squared_matrix` byte-for-byte (see
`test_operators.py`); the difference is purely architectural — the new
form is composable.

---

## 6. Local spin correlations

`s_dot(i, j)` builds Sᵢ·Sⱼ = Sᵢ_z Sⱼ_z + ½(Sᵢ₊ Sⱼ₋ + Sᵢ₋ Sⱼ₊).
This is the operator that appears in the Heisenberg model, in the
strong-coupling limit of the Hubbard model (J₁ = 4t²/U; see
`notebooks/additional/benzene_hubbard_to_heisenberg.ipynb`), and in any
"local spin" analysis of a wavefunction.

```python
M = op.s_dot('a', 'b').matrix(['aB', 'bA'])     # 2×2 in the singly-occ sector
# eigenvalues: -3/4 (singlet) and +1/4 (triplet)  — exactly (S²−¾−¾)/2
```

The spectrum is the "spin-correlator" relation
Sᵢ·Sⱼ = (S² − Sᵢ² − Sⱼ²)/2 with Sᵢ² = ¾ on each singly-occupied site:

  singlet: (0 − ¾ − ¾)/2 = −¾
  triplet: (2 − ¾ − ¾)/2 = +¼

A worked use of this operator is the J₁ = 4t²/U extraction for benzene
(`examples/benzene_heisenberg_mapping.py`): tr(Sᵢ·Sⱼ_eff) over the
22-state singlet-A_{1g} block scales as `4t²/U · (combinatorial factor)`,
and the proportionality is exactly what `s_dot(i, j).matrix(basis)` would
produce numerically if you projected onto that block.

---

## 7. Singlet projector and bond creator

Two related but distinct operators are useful in VB.

**Singlet projector** `singlet_proj(i, j) = ¼ − Sᵢ·Sⱼ`. In the sector
where (i, j) is **singly occupied** (one electron each), this operator
has eigenvalue 1 on the singlet and 0 on the triplet:

```python
P = op.singlet_proj('a', 'b').matrix(['aB', 'bA'])
# eigvals 0 and 1 — extracts the singlet bonding combination
```

⚠️ Outside the (1, 1) sector this operator is *not* a projector. On a
closed shell `|aA⟩` it returns ¼·|aA⟩ (because Sᵃ·Sᵇ = 0 on that state,
not because the closed shell is "¼ singlet"). Restrict its
interpretation to the singly-occupied sector.

**Bond singlet creator**
`bond_singlet_creator(i, j) = (1/√2)(c†_{i,α} c†_{j,β} − c†_{i,β} c†_{j,α})`.
Acting on a state where i and j are unoccupied, this appends a
two-electron singlet bond on (i, j). Building VB structures from
bond creators is one route to the Rumer basis.

```python
B_ab = op.bond_singlet_creator('a', 'b')
B_ab.apply('cD')
# √2/2 |aBcD| + √2/2 |bAcD|
#
# That's exactly the (a, b) singlet `(|aB> + |bA>)/√2` (in symvb sign
# convention) tensored with the spectator |cD>.  Chain another
# `bond_singlet_creator(...)` to build a full Rumer structure with
# multiple singlet bonds.
```

---

## 8. η-pairing

The pseudospin (η-pairing) algebra is the charge analogue of spin
(Yang, Zhang 1989). On a bipartite lattice with sublattice signs
sᵢ = ±1,

  η₊ = Σᵢ sᵢ c†_{i,α} c†_{i,β}     (creates a doubly-occupied site,
                                    sublattice-signed)
  η₋ = (η₊)†                       (annihilates a pair)
  η_z = (N̂ − L)/2                  (charge minus half-filling)

`symvb` provides `eta_plus(site_signs)`, `eta_minus(site_signs)`,
`eta_z(orbs)`, and `eta_squared(site_signs)`. They mirror the spin
operators term-for-term:

```python
site_signs = {'a': 1, 'b': -1, 'c': 1, 'd': -1}     # bipartite H4 ring
op.eta_plus(site_signs).apply('aA')
# ±1·|aAbB|     — places a (α, β) pair on the only empty site (b),
#                 carrying the sublattice sign and a JW sign

from symvb.functions import generate_det_strings
op.eta_z(['a','b','c','d']).matrix(generate_det_strings(2,2,4))
# zero matrix at half-filling, η_z = 0
```

`η²` is built from the same identity as S²: `η_+ η_- + η_z² − η_z`.
`symvb.operators.eta_squared` reproduces `symvb.spin.eta_squared_matrix`
exactly. The caveat from the manuscript still applies: η₊ is a *true*
symmetry only at s = 0 (orthogonal AOs); at s ≠ 0 the operators don't
form SU(2) and `[H, η²] ≠ 0`. See `examples/benzene_eta_pairing.py`
for the diagnostic sweep.

---

## 9. Symmetry: orbital permutations and Reynolds projectors

Spatial symmetry acts on dets by permuting the orbital labels. We
provide

* `transposition(i, j)` — swap two orbital labels.
* `orbital_perm({old: new, ...})` — generic relabelling.
* `reynolds_projector(generators)` — averages over the group generated
  by a list of orbital-permutation dictionaries; this is the projector
  onto the trivial (totally-symmetric) irrep.

The fermion sign carried by each group element is tracked
automatically, so this works on bases with any filling (including the
over-half-filled, where the unsigned orbit-sum projector misses the
sign — see the cyclobutadiene-dianion case,
`examples/c4h4_dianion_closed_form.py`).

```python
op.transposition('a', 'b').apply('aB')
# |bA|     — swap labels (the fermion sign happens to be +1 here;
#            in interleaved JW the sign correction from the canonical
#            interpretation cancels the JW swap sign)

# C_4 rotation on H4 ring, induced from the orbital cycle a→b→c→d→a:
C4 = op.orbital_perm({'a': 'b', 'b': 'c', 'c': 'd', 'd': 'a'})

# Trivial-irrep (A_1g) projector for the dihedral group D_2 generated
# by C_2 = (a b)(c d) and σ = (a c)(b d):
R = op.reynolds_projector([
    {'a': 'b', 'b': 'a', 'c': 'd', 'd': 'c'},     # C_2
    {'a': 'c', 'c': 'a', 'b': 'd', 'd': 'b'},     # σ
])
R.apply('aBcD')   # FixedPsi: average over the 4-element group acting on |aBcD>
```

Reynolds projectors are idempotent (`R @ R == R`) and self-adjoint, as
expected; the test suite verifies this on a 2-site Z₂ example.

For benzene D₆ at half-filling, the same machinery produces the 38-dim
A_1g block from the 400-dim Sz=0 sector — see `symvb.symmetry`'s
`totally_symmetric_basis` for the specialized fast path that skips the
operator-tree machinery in favor of orbit sums. The two agree on the
half-filled-or-below cases; only the over-half-filled cases (e.g.
C₄H₄²⁻) need the signed Reynolds projector here.

---

## 10. Algebra and matrix construction

Everything is composable. The arithmetic operators have their natural
meaning:

| Expression                   | Meaning                              |
|------------------------------|--------------------------------------|
| `A + B`, `A - B`             | linear combination                   |
| `c * A`, `A * c`             | scalar multiplication (`c` a Python number or sympy expr) |
| `A @ B`  (or `A * B`)        | composition: B applied first, then A |
| `-A`                         | negation                             |

Build a one-band Hubbard Hamiltonian on H₄:

```python
import sympy as sp
t, U = sp.symbols('t U')
sites = ['a', 'b', 'c', 'd']
bonds = [('a','b'), ('b','c'), ('c','d'), ('d','a')]   # ring
T = -t * sum(op.hop(i, j) for i, j in bonds)
V = U * sum(op.double_occ(s) for s in sites)
H = T + V
```

Get the matrix in the Sz=0 sector:

```python
from symvb.functions import generate_det_strings
basis = generate_det_strings(2, 2, 4)
H_mat = H.matrix(basis)        # 36 × 36 sympy Matrix in t, U
```

…or just the diagonal `D̂` operator if all you want is double
occupation per state:

```python
D = sum(op.double_occ(s) for s in sites)
D_diag = D.matrix(basis)       # diagonal counts of doubly-occupied sites
```

Compute an expectation value against a symvb wavefunction. The H₂ singlet
is `(|aB⟩ + |bA⟩)/√2`, and `Ŝ²` should give 0 on it; the triplet T₀ is
`(|aB⟩ − |bA⟩)/√2` and gives 2:

```python
import symvb
singlet = symvb.FixedPsi('aB') + symvb.FixedPsi('bA')   # un-normalized; ⟨ψ|ψ⟩ = 2
triplet = symvb.FixedPsi('aB') - symvb.FixedPsi('bA')

S2 = op.s_squared(['a', 'b'])
# expectation returns ⟨ψ|S²|ψ⟩; divide by ⟨ψ|ψ⟩ if you want S(S+1).
S2.expectation(singlet)          # 0  → S(S+1) = 0  ✓
S2.expectation(triplet) / 2      # 2  → S(S+1) = 2  ✓
```

`apply`, `matrix`, and `expectation` all accept det strings, `SlaterDet`s,
or `FixedPsi`s as inputs, and `apply` always returns a `FixedPsi` keyed
by canonical (interleaved) strings. So you can pipe results back into
the rest of the codebase that already speaks `FixedPsi`.

---

## Summary

The `symvb.operators` module exposes a single object — `Operator` —
with three capabilities you actually use day to day:

1. `op.apply(state)` to see how an operator acts on a state.  Returns
   a `FixedPsi` (with dets keyed by canonical interleaved strings).
2. `op.matrix(basis)` to build the matrix in any user basis (signs are
   auto-tracked; basis elements may be strings, `SlaterDet`s, or
   `FixedPsi`s).
3. `op.expectation(state)` to evaluate ⟨ψ|Ô|ψ⟩.

…and a small library of pre-built physical operators — `number`,
`double_occ`, `hop`, `s_z`, `s_plus`, `s_minus`, `s_squared`, `s_dot`,
`eta_plus`, `eta_minus`, `eta_squared`, `transposition`,
`orbital_perm`, `reynolds_projector`, `singlet_proj`,
`bond_singlet_creator` — built by composing `c` and `c†` with the
algebra above.

The canonical det form used internally is the **interleaved** form
(e.g. `aAbBcC`) that matches `symvb.functions.generate_det_strings` and
`symvb.spin._canon_det`. `canonicalize(s)` translates from any symvb
string form, and `matrix(basis)` does this automatically.

`symvb.spin.s_squared_matrix` and `symvb.spin.eta_squared_matrix` are
now thin numpy-returning wrappers around `op.s_squared(...).matrix(...)`
and `op.eta_squared(...).matrix(...)`, so all three entry points
agree byte-for-byte.
