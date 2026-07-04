# symvb API reference

Symbolic valence-bond theory in Python. This is a reference for the public
surface; for worked usage see [`recipes.md`](recipes.md) and
[`operators_tutorial.md`](operators_tutorial.md). Full worked derivations,
one per model system of the paper, are the teaching notebooks in
[`../notebooks/`](../notebooks).

## Conventions

- **Determinant strings** encode spin by case: lowercase = α, uppercase = β.
  Creation order is string order. `'aB'` is α on `a`, β on `b`; `'aAbBcC'` is the
  half-filled closed-shell benzene determinant.
- **Canonical spin-orbital order** (used internally) is interleaved and
  orbital-alphabetical: `(orb_1,α), (orb_1,β), (orb_2,α), …`. `FixedPsi` and the
  operator DSL work in the user basis and convert as needed.
- **AO overlap.** `Molecule.build_matrix` carries the overlap `s` symbolically.
  The operator-DSL `matrix()`, the spin/η helpers, and the symmetry projectors
  assume the determinant basis is **orthonormal**, i.e. `s = 0`.
- **Integral names.** One-electron integrals are `H_<ij>` and `S_<ij>`
  (alphabetical pair); the on-site Hubbard integral is the chemist tuple
  `'1111'`. The `subst` / `subst_2e` dicts map a short symbol (`'h'`, `'s'`, `'U'`)
  onto those names. A matrix built with `subst_2e` already contains the
  two-electron integral; do not multiply it by `U`.
- **Site energies.** `zero_ii=True` (the default) zeros the diagonal
  one-electron integrals `H_ii`. With `zero_ii=False` they are kept and can be
  unified through `subst` like any other integral (e.g. `'eps': ('H_bb',)`),
  with the ones you want at zero substituted away after the build (recipe 8).

## Top-level imports (`from symvb import …`)

`Molecule`, `FixedPsi`, `SlaterDet`, `System`, `hamiltonian`, `ground_state`,
`chirgwin_coulson`, `structure_vector`, `verify_eigenpair`,
`EigenpairResidualError`, and the submodules `symmetry`, `spin`, `huckel`,
`mo_projection`, `operators`, `system`.

---

## `Molecule` — symbolic matrix elements

```python
Molecule(symm_offdiagonal=True, normalized_basis_orbs=True,
         interacting_orbs=None, subst=None, zero_ii=True,
         subst_2e=None, max_2e_centers=4, o2_method='blocked',
         orbitals=None)
```
Container that turns spin-orbital indices into physical integrals.
`interacting_orbs` is a list of alphabetical 2-char edge strings (`['ab','bc']`);
`subst` maps `{'h': ('H_ab', …), 's': ('S_ab', …)}`; `subst_2e` maps
`{'U': ('1111',)}`; `zero_ii=True` zeros the site energies; `max_2e_centers`
caps the two-electron integral spread (`1` = on-site only); `o2_method` is
`'blocked'` (default, faster) or `'direct'`.

Topology constructors (build the dicts for you):

```python
Molecule.ring(L, h='h', s='s', U='U', hubbard=True, zero_ii=True, **kw)   # cyclic L-ring
Molecule.chain(n, h='h', s='s', U='U', hubbard=True, zero_ii=True, **kw)  # linear n-chain
```
`Molecule.ring(6)` reproduces the benzene configuration exactly.

Key methods:

| call | returns |
|---|---|
| `build_matrix(basis, op='H')` | SymPy matrix of `⟨bᵢ\|op\|bⱼ⟩`, `op` in `'H'`/`'S'`; `basis` is a list of `FixedPsi`/`SlaterDet`/strings. Each entry is standardized into canonical creation order first, so hand-built structures written in a natural order (e.g. the αα ββ long bond from `coupled_pairs`) give the correct matrix |
| `o2_matrix(basis)` | SymPy matrix of the two-electron block (carries its integral name); same internal canonicalization as `build_matrix` |
| `energy(P, o2=False)` | Rayleigh quotient `⟨P\|H\|P⟩/⟨P\|P⟩` of a single `FixedPsi` |
| `getH(L, R)`, `getS(L, R)` | a single matrix element between two states |
| `generate_basis(Na, Nb, Norbs)` | precompute half-determinant blocks (speeds repeated `build_matrix`) |

> Prefer `symvb.hamiltonian(m, basis)` / `System` (below) over calling
> `build_matrix` + `o2_matrix` and combining by hand.

## Determinants — `FixedPsi`, `SlaterDet`, `generate_dets`

```python
generate_dets(Nela, Nelb, Norb) -> list[FixedPsi]      # the Sz basis, one det each
FixedPsi(det_string)                                    # a labelled linear combination of dets
FixedPsi(parent_string, coupled_pairs=[(i, j), …])      # spin-couple orbital positions into HL singlets
```
`FixedPsi` methods: `add_str_det(det_string, coef=+1)`, `add_det(slaterdet, coef)`,
`add_fixedpsi(other, coef)`, `couple_orbitals(o1, o2)`, `canonicalize()` (sort
orbital labels within each spin block), `standardize()` (rewrite into canonical
interleaved creation order, folding the fermion sign into the coefficient).
Iterating a `FixedPsi` yields `(SlaterDet, coef)` pairs; `d.det_string` is the string.

---

## `symvb.system` — the high-level facade

Standalone helpers (compose with any `Molecule`/matrices):

| call | returns / effect |
|---|---|
| `hamiltonian(molecule, basis, two_electron=True)` | `(H, S)`; always folds the two-electron block into `H` (under the `subst_2e` names when declared, otherwise the default `T_<abcd>` names). Pass `two_electron=False` for an intentionally one-electron model; this also skips the 2e build, the expensive step on large bases |
| `ground_state(H, S, ref=None, subs=None)` | `(E, c)`. With `subs=None`: symbolic, for **small blocks only** (2×2, 3×3 — the characteristic-polynomial solve does not scale); picks the lowest root at the reference point `ref` (default `h=-1, s=0`, repulsion `=1`); `E` simplified, `c` un-normalized. With `subs=<numeric dict>`: numeric path via `scipy.linalg.eigh`, returns `(float, ndarray)` — use this for FCI-sized bases |
| `chirgwin_coulson(c, S, groups=None, simplify=False)` | weights `c_i(Sc)_i/(c^T S c)`; SymPy or NumPy; `groups` sums index lists; `simplify` off by default |
| `structure_vector(structure, basis_dets)` | a `FixedPsi` structure as a column over `basis_dets`, with the fermion sign of the canonical reordering folded in (used to project a structure onto a determinant basis, e.g. an FCI ground state for weights) |

`System` bundles a molecule with a basis:

```python
System(molecule, basis, two_electron=True)
System.from_structures(molecule, structures, two_electron=True)
System.ring(L, n_alpha=None, n_beta=None, hubbard=True, two_electron=True, **kw)
System.chain(n, n_alpha=None, n_beta=None, hubbard=True, two_electron=True, **kw)
```
`two_electron=False` makes `.hamiltonian()` return the one-electron `H`
(see `hamiltonian` above).
Methods: `.hamiltonian() -> (H, S)`, `.H`, `.S`, `.ground_state(ref=None, subs=None)`,
`.weights(structures=None, groups=None, ref=None, subs=None)`,
`.structure_vector(structure)`. Both take `subs=` for the numeric path (see
`ground_state` above); `weights` then returns a NumPy array.
With `structures=`, `weights` projects a determinant-basis ground state onto a
(possibly non-orthogonal) VB-structure space. **Note the normalization:** the
returned weights are normalized over that structure space — they describe the
composition of the part of the wavefunction the structures span, not each
structure's share of the full wavefunction (for the latter, use
`structure_vector` + `chirgwin_coulson` over the determinant basis).

---

## `symvb.operators` — second-quantized operator algebra

Build operators, then `.matrix(basis)` (SymPy matrix at `s = 0`),
`.apply(state)` (→ `FixedPsi`), or `.expectation(state)` (scalar). Operators
support `+`, `-`, scalar `*`, and `@` (composition).

| constructor | operator |
|---|---|
| `c(orb, spin='alpha')`, `cdag(orb, spin='alpha')` | annihilation / creation |
| `number(orb, spin=None)`, `double_occ(orb)` | occupation; on-site double occupancy |
| `hop(i, j, spin=None, hermitian=True)` | `c†_i c_j` (+ h.c.) |
| `s_z(orbs)`, `s_plus(orbs)`, `s_minus(orbs)`, `s_squared(orbs)` | total-spin operators |
| `s_dot(i, j)` | `Sᵢ·Sⱼ` (Heisenberg coupling) |
| `eta_plus/eta_minus(site_signs)`, `eta_z(orbs)`, `eta_squared(site_signs)` | η-pairing (`site_signs` = `{orb: ±1}`) |
| `singlet_proj(i, j)`, `bond_singlet_creator(i, j)` | singlet projector `¼ − Sᵢ·Sⱼ`; bond-singlet creator |
| `transposition(i, j)`, `orbital_perm(map)`, `reynolds_projector(generators)` | orbital permutations; trivial-irrep projector (group closure automatic, fermion signs tracked) |
| `identity()`, `canonicalize(det_string)` | identity operator; canonical interleaved string + its ±1 sign |

See [`operators_tutorial.md`](operators_tutorial.md) for a guided tour.

## `symvb.huckel` — one-electron solver

```python
huckel.solve_ring(L, h='h', s='s', basis='real', site_labels=None) -> HuckelResult
huckel.solve(adjacency, site_labels=None, h='h', s='s', overlap=None) -> HuckelResult
```
`HuckelResult` fields: `site_labels`, `eigenvalues`, `energies` (`εₖ(h,s)`),
`coefficients` (n_mo × n_sites SymPy matrix, rows = MOs over AOs), `h_symbol`,
`s_symbol`, `point_group`, `irrep_labels`. Method:
`energy_of_occupation(occupation)` where `occupation` is per-MO counts
(`[2,2,0]`) or `(mo_index, n)` pairs.

## `symvb.spin` — spin and η-pairing in a det basis

```python
spin.s_squared_matrix(det_strings, orbs=None)                 -> SymPy matrix
spin.eta_squared_matrix(det_strings, site_signs, orbs=None)   -> SymPy matrix  (s = 0 only)
spin.project_onto_S(H, S2, target_S, tol=1e-8)                -> (H_proj, U_proj)
```
`project_onto_S` returns the Hamiltonian restricted to the `target_S` eigenspace
of `S2` and the isometry `U_proj` back to the full basis.

## `symvb.symmetry` — point-group reduction

```python
symmetry.detect_permutation_group(H_sym, S_sym=None)          # discover the group from a matrix
symmetry.totally_symmetric_basis(generators, N)               # exact-arithmetic A1 basis
symmetry.signed_totally_symmetric_basis(signed_generators, N, tol=1e-8)   # over-half-filled rings
symmetry.signed_totally_symmetric_basis_exact(signed_generators, N)        # same subspace, exact arithmetic (sympy)
symmetry.degenerate_block_basis(H_num, S_num=None, tol=1e-8)  # split numeric degenerate blocks
symmetry.generate_group(generators, N=None)                   # close a set of permutations into the full group
```
For the trivial-irrep projector as an operator, use
`operators.reynolds_projector(generators)` (recipe 5).

## `symvb.mo_projection` — MO ↔ AO and verification

```python
mo_projection.mo_determinant_in_ao(mo_coeffs, occupation, basis_dets, site_labels=None)
mo_projection.linear_combination_in_ao(mo_coeffs, terms, basis_dets, site_labels=None)
mo_projection.verify_eigenpair(H, S, v, E, simplify=None)     # True, or raises EigenpairResidualError
```
`occupation` is `(alpha_mo_indices, beta_mo_indices)`; `terms` is an iterable of
`(coefficient, occupation)` pairs. `verify_eigenpair` proves `(H − E S)v = 0` as a
polynomial identity (default per-entry simplifier `sympy.cancel`; pass
`sympy.simplify` for surds).

## `symvb.functions` — string and matrix utilities

`standardize_det(s) -> (standard_string, n_flips)` (fermion sign `(-1)**n_flips`);
`generate_det_strings(Na, Nb, Norbs)`; `canonical_chemist_iv(iv)`;
`simplify_matrix(mtx, factor=False)`.
