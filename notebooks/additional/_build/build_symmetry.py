"""Build notebooks/additional/symmetry_as_a_tool.ipynb."""
import os
import nbformat as nbf

NB = nbf.v4.new_notebook()
NB.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
}

cells = []
def md(text):  cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))
def code(src): cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# =====================================================================
md(r"""
# Additional notebook — Symmetry as a tool: $400 \to 38 \to 22 \to 14$

**The cascade.** Benzene's $S_{z}=0$ FCI is 400-dimensional. Three
successive symmetry projections collapse it to 14:

$$
\underbrace{400}_{S_{z}=0\text{ FCI}}
\;\xrightarrow{D_{6}\;\text{spatial}}\;
\underbrace{38}_{A_{1g}}
\;\xrightarrow{\hat S^{2}=0}\;
\underbrace{22}_{\text{singlet }A_{1g}}
\;\xrightarrow{\hat\eta^{2}=0}\;
\underbrace{14}_{\eta=0\text{ singlet }A_{1g}}
$$

The 14-dimensional final block contains the FCI ground state (and many
of its low-lying excitations). Each projection enforces a conservation
law that the physical Hamiltonian respects.

**The catch.** The $\eta$-pairing projection only works in the
*orthogonal-AO limit* ($s = 0$). At $s \ne 0$ the $\eta$ operators no
longer form a closed SU(2) algebra and $[\hat H, \hat\eta^{2}] \ne 0$.
You will see this break-down quantitatively.

**This notebook.** We unpack each layer of the cascade. The
Hubbard-to-Heisenberg notebook used the $D_{6}$ projection as a black box;
here we open the box, then push further into spin and $\eta$.

**Prerequisites.** The Hubbard-to-Heisenberg notebook (same benzene Hubbard
matrices, same $A_{1g}$ projector). Familiarity with elementary group theory
(generators, orbits) helps but is not strictly required.

**Runtime.** ~1–2 min end-to-end with the cache present. The slow
steps are the sympy → float conversions of the 400×400 symbolic
matrices (§2, §8) and the $\hat S^{2}$ / $\hat\eta^{2}$ operator
builds (§5, §6), a few seconds each.
""")

# =====================================================================
md(r"""
## 1. Setup: load the cached symbolic matrices

We reuse the cache built in the Hubbard-to-Heisenberg notebook (this
folder). If you skipped that notebook, the cell below will rebuild the
matrices (~10 s).
""")

code(r"""
import os, pickle, time
import numpy as np
import sympy as sp
from sympy import init_printing
from collections import Counter
init_printing()

from symvb import Molecule, SlaterDet, symmetry
from symvb.spin import s_squared_matrix, eta_squared_matrix

CACHE = '/tmp/benzene_hubbard_matrices.pkl'

m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'],
    subst={'h': ('H_ab', 'H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af'),
           's': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af')},
    subst_2e={'U': ('1111',)},
    max_2e_centers=1,
)
m.generate_basis(3, 3, 6)
det_strings = [fp.dets[0].det_string for fp in m.basis]
N_full = len(det_strings)
print(f"Full FCI basis: {N_full} determinants")

if os.path.exists(CACHE):
    with open(CACHE, 'rb') as f:
        H1_sym, S_sym, _H2_sym = pickle.load(f)
    print(f"Loaded cached H1, S from {CACHE}")
else:
    print("Building 400×400 symbolic H1, S (~10 s)...")
    t0 = time.time()
    H1_sym = m.build_matrix(m.basis, op='H')
    S_sym  = m.build_matrix(m.basis, op='S')
    print(f"  done in {time.time()-t0:.1f}s")
""")

# =====================================================================
md(r"""
## 2. Discovery mode: degeneracies tell you the irreps

If you knew nothing about the molecular symmetry, you could *discover*
it by diagonalising $H$ once and looking at the eigenvalue
degeneracies. Each multiplet of size $d$ is a candidate for a
$d$-dimensional irreducible representation.

We do this on the one-electron Hamiltonian ($U = 0$) at $s = 0.2$;
non-zero overlap lifts some of the extra coincidences of the
integer-valued $s = 0$ spectrum.
""")

code(r"""
h, s, Usym = sp.symbols('h s U')
H_num = np.array(sp.Matrix(H1_sym).subs({h: -1, s: 0.2}).tolist(), dtype=float)
S_num = np.array(sp.Matrix(S_sym).subs({h: -1, s: 0.2}).tolist(), dtype=float)

t0 = time.time()
evals, evecs, blocks = symmetry.degenerate_block_basis(H_num, S_num, tol=1e-6)
print(f"Diagonalised 400×400 in {time.time()-t0:.1f}s")
print(f"Distinct eigenvalues: {len(blocks)}")

deg_hist = Counter(len(b[1]) for b in blocks)
print("Degeneracy histogram (multiplet size : # of multiplets):")
for d, c in sorted(deg_hist.items()):
    print(f"  {d:>3}-fold : {c:>4}")
""")

md(r"""
The histogram exposes conserved structure, but more than the point
group alone. $D_{6}$ has irreps of dimension 1 ($A_{1}, A_{2}, B_{1},
B_{2}$) and 2 ($E_{1}, E_{2}$), and spin adds multiplet structure, yet
the histogram shows 22-, 25-, 44-, even 60-fold multiplets. Those
large degeneracies are *free-fermion* accidentals: at $U = 0$ every
FCI eigenvalue is a sum of occupied MO energies, and many distinct
occupations of the six MOs (two of which form degenerate $E$ pairs)
share the same total at any $s$.

**This is "discovery mode":** zero group theory required — one
diagonalisation reveals every degeneracy at once, spatial, spin, and
accidental. What it cannot do is disentangle them; for that we use
the constructive route below.
""")

# =====================================================================
md(r"""
## 3. Constructive mode: orbital permutations $\to$ basis permutations

The complementary strategy is to specify the symmetry up front. Define
each generator as a *map of orbital labels*; `symvb.symmetry` lifts it
to a permutation of the determinant basis (with sign tracking from
canonical reordering), and verifies the lift commutes with $H$.

For benzene we need two generators of $D_{6}$: a $C_{6}$ rotation and
a $\sigma_{v}$ reflection.
""")

code(r"""
def canon(ds):
    fp = SlaterDet(ds).get_sorted()
    return fp.dets[0].det_string, fp.coefs[0]

C6    = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'e', 'e': 'f', 'f': 'a'}
sigma = {'a': 'a', 'b': 'f', 'c': 'e', 'd': 'd', 'e': 'c', 'f': 'b'}

perms = []
for name, om in [('C_6', C6), ('sigma_v', sigma)]:
    perm, signs = symmetry.apply_orbital_permutation(om, det_strings, canon)
    P = np.zeros_like(H_num)
    for a, b in enumerate(perm):
        P[a, b] = 1.0
    err = np.max(np.abs(P @ H_num @ P.T - H_num))
    print(f"  {name}: induced basis permutation, "
          f"||P H P.T − H||_∞ = {err:.1e}")
    perms.append(perm)
""")

md(r"""
Each generator is a $400 \times 400$ permutation matrix (zero-one,
exactly one nonzero per row/column). The check $P\,H\,P^{\top} = H$
verifies the lift is consistent with the Hamiltonian — a sanity check
that catches sign errors in `canon()` or wrong orbital maps.
""")

# =====================================================================
md(r"""
## 4. Closing the group, building orbits

`generate_group` closes the generators under composition. For
$D_{6} = \langle C_{6}, \sigma_{v} \rangle$ the order is 12.
`totally_symmetric_basis` then computes the orbits of the group
acting on the 400 determinants and returns:

- a $400 \times 38$ matrix $U$ whose columns are *orbit-sum
  indicator vectors* (one column per orbit, ones on orbit members),
- the explicit list of 38 orbits.

The 38 orbits give the dimension of the totally-symmetric ($A_{1g}$)
representation in the determinant basis.
""")

code(r"""
group = symmetry.generate_group(perms, N=N_full)
print(f"Group order |D_6| = {len(group)}")

U_a, orbits = symmetry.totally_symmetric_basis(perms, N_full)
print(f"A_1g dimension: {U_a.shape[1]}")

orbit_size_hist = Counter(len(o) for o in orbits)
print(f"Orbit-size histogram: {dict(sorted(orbit_size_hist.items()))}")
total = sum(len(o) for o in orbits)
print(f"Sum of orbit sizes: {total}  (must equal {N_full})")
""")

md(r"""
With 12 group elements and 400 dets, every orbit has size dividing 12.
The size of each orbit is $|G|/|\text{stabiliser}|$, so the largest
orbit can be 12 (no symmetry) and the smallest 1 (totally fixed).
Confirmation: total orbit count × average size = 400.

Reduce $H$ to the $A_{1g}$ block by similarity transform. Note that
the columns of $U_{a}$ are *not* orthonormal (each is an orbit sum
of unit vectors, so its norm is $\sqrt{|\text{orbit}|}$); the
generalised eigenvalue problem is $H_{\rm red}\, c = E\, S_{\rm red}\,
c$ where $S_{\rm red} = U_{a}^{\top}\, S\, U_{a}$.
""")

code(r"""
H_red = U_a.T @ H_num @ U_a
S_red = U_a.T @ S_num @ U_a
from scipy.linalg import eigh
E_red = eigh(H_red, S_red)[0][0]
print(f"Reduced GS energy at s=0.2:  {E_red:.6f}")
print(f"Full FCI GS energy at s=0.2: {evals[0]:.6f}")
print(f"Match within: {abs(E_red - evals[0]):.2e}")
""")

md(r"""
The reduced $38 \times 38$ problem reproduces the full GS energy
exactly (to machine precision). One full diagonalisation $\to$
millisecond-level reduced eigenproblem, and we still have the
*correct* state.
""")

# =====================================================================
md(r"""
## 5. Total spin: $A_{1g}$ split into singlet, triplet, quintet, septet

Within $A_{1g}$, total spin $\hat S^{2}$ is an additional good quantum
number (since $H$ is spin-rotation invariant in the absence of
spin-orbit coupling and Zeeman fields). $S = 0, 1, 2, 3$ are all
present in the half-filled $S_{z}=0$ sector.

`symvb.spin.s_squared_matrix(det_strings)` builds $\hat S^{2}$ as a
$400 \times 400$ matrix in the determinant basis (no symbolic
parameters — $\hat S^{2}$ is a pure spin operator).
""")

code(r"""
print("Building S^2 matrix (~5 s)...")
t0 = time.time()
S2 = s_squared_matrix(det_strings)
print(f"  done in {time.time()-t0:.1f}s")

# S^2 in A_1g: project, then find singlet (S^2 = 0) eigenspace
S2_a = U_a.T @ S2 @ U_a
S2_a = 0.5 * (S2_a + S2_a.T)        # numerical symmetry
ev_s2, vS = np.linalg.eigh(S2_a)
print(f"S^2 eigenvalues in A_1g: "
      f"{sorted(set(np.round(ev_s2, 6).tolist()))}")
print(f"   (S(S+1) for S=0,1,2,3 is 0, 2, 6, 12)")

US = vS[:, np.abs(ev_s2) < 1e-6]
print(f"\nSinglet-A_1g dimension: {US.shape[1]}")
""")

md(r"""
**400 → 38 → 22.** The 22 singlet $A_{1g}$ states contain the FCI
ground state (always a singlet for the closed-shell benzene Hubbard
ring at half-filling).
""")

# =====================================================================
md(r"""
## 6. $\eta$-pairing: a second SU(2) on the half-filled lattice

On a *bipartite* lattice (benzene's hexagon: alternating sublattices
$\{a, c, e\}$ with sign $+1$ and $\{b, d, f\}$ with sign $-1$), one can
construct three operators

$$
\hat\eta_{+} = \sum_{i} \varepsilon_{i}\, \hat c^{\dagger}_{i\uparrow}
\hat c^{\dagger}_{i\downarrow}, \quad
\hat\eta_{-} = \hat\eta_{+}^{\dagger}, \quad
\hat\eta_{z} = \tfrac{1}{2}(\hat N - L),
$$

where $\varepsilon_{i} = \pm 1$ is the sublattice sign. They obey the
SU(2) algebra $[\hat\eta_{+}, \hat\eta_{-}] = 2\hat\eta_{z}$,
$[\hat\eta_{z}, \hat\eta_{\pm}] = \pm \hat\eta_{\pm}$ — *but only if
the underlying fermion modes are mutually orthogonal*.

At half-filling $\hat\eta_{z} = 0$, so the eigenvalue of $\hat\eta^{2}$
is $\eta(\eta+1)$ for $\eta = 0, 1, 2, 3$. The spectrum of $\hat\eta^{2}$
on the full 400-dim basis decomposes the basis into multiplets.
""")

code(r"""
SITE_SIGNS = {'a': +1, 'b': -1, 'c': +1, 'd': -1, 'e': +1, 'f': -1}
ORBS = list('abcdef')

print("Building eta^2 matrix (~5 s)...")
t0 = time.time()
E2 = eta_squared_matrix(det_strings, SITE_SIGNS, ORBS)
print(f"  done in {time.time()-t0:.1f}s")

ev_eta = np.linalg.eigvalsh((E2 + E2.T) / 2)
eta_mult = Counter(np.round(ev_eta, 6).tolist())
print("eta^2 spectrum on full 400-dim basis:")
print(f"  {'eigenvalue':>12}  {'eta':>5}  {'multiplicity':>12}")
for ev_, mult in sorted(eta_mult.items()):
    eta_q = (np.sqrt(1 + 4*max(ev_, 0)) - 1) / 2
    print(f"  {ev_:>12.4f}  {eta_q:>5.0f}  {mult:>12}")
print(f"  total = {sum(eta_mult.values())}  (must be 400)")
""")

md(r"""
The 400 dets split as $175 + 189 + 35 + 1$ across $\eta = 0, 1, 2, 3$
multiplets. The unique $\eta = 3$ state is $(\hat\eta_{+})^{3}
|\mathrm{vac}\rangle$, an equal-weight signed sum over all triples of
doubly-occupied sites; the $\eta = 0$ block of size 175 contains the
ground state.
""")

# =====================================================================
md(r"""
## 7. Cutting singlet-$A_{1g}$ down to $\eta = 0$

Within the 22-dimensional singlet $A_{1g}$ block, $\hat\eta^{2}$ is
*also* diagonalisable (it commutes with $H$, $\hat S^{2}$, and $D_{6}$
at $s = 0$). Project onto its $\eta = 0$ eigenspace.
""")

code(r"""
H_a  = U_a.T @ H_num @ U_a;   H_a  = 0.5 * (H_a  + H_a.T)
E2_a = U_a.T @ E2  @ U_a;     E2_a = 0.5 * (E2_a + E2_a.T)

# Switch to s=0 for eta^2 to commute with H
H_num0 = np.array(sp.Matrix(H1_sym).subs({h: -1, s: 0}).tolist(), dtype=float)
H_a0 = U_a.T @ H_num0 @ U_a;  H_a0 = 0.5 * (H_a0 + H_a0.T)

# singlet block at s=0
H_s = US.T @ H_a0 @ US
E2_s = US.T @ E2_a @ US
print(f"Singlet-A_1g dim:  {H_s.shape[0]}")

ev_e2, vE = np.linalg.eigh(0.5 * (E2_s + E2_s.T))
print(f"eta^2 eigenvalues in singlet-A_1g block: "
      f"{sorted(set(np.round(ev_e2, 6).tolist()))}")

# Project to eta = 0
U_eta0 = vE[:, np.abs(ev_e2) < 1e-6]
H_se = U_eta0.T @ H_s @ U_eta0
ev_se = np.linalg.eigvalsh(0.5 * (H_se + H_se.T))
print(f"\neta = 0 singlet-A_1g dim:  {U_eta0.shape[1]}")
print(f"H eigenvalues in this 14-block (s=0, U=0):")
print(f"  {np.round(ev_se, 4).tolist()}")
""")

md(r"""
**400 → 38 → 22 → 14.** The 14-dimensional final block contains the
benzene FCI ground state at $E_{0} = -8\, t$ (which you saw in the
Hubbard-to-Heisenberg notebook).

The eigenvalues are *integers* — the same nine distinct values
$\{-8, -4, -2, -1, 0, 1, 2, 4, 8\}$ that appeared there, with the
multiplicity structure now matching the 14-dim subspace.
""")

# =====================================================================
md(r"""
## 8. The crack: $\eta$-pairing breaks at $s \ne 0$

The SU(2) algebra of $\hat\eta_{\pm}, \hat\eta_{z}$ relied on the
orthogonal-AO anticommutator $\{\hat c_{i\sigma}, \hat
c^{\dagger}_{j\tau}\} = \delta_{ij}\delta_{\sigma\tau}$. With AO
overlap, this picks up the overlap matrix $S_{ij}$ and the algebra no
longer closes. The diagnostic is the commutator $[\hat H, \hat\eta^{2}]$
— which must vanish if $\hat\eta^{2}$ is conserved.
""")

code(r"""
print(f"  {'s':>5}  {'||[H, eta^2]||_∞':>20}")
for sval in (0.0, 0.1, 0.2, 0.3):
    H_n = np.array(sp.Matrix(H1_sym).subs({h: -1, s: sval}).tolist(), dtype=float)
    comm = H_n @ E2 - E2 @ H_n
    print(f"  {sval:>5.2f}  {np.max(np.abs(comm)):>20.4f}")
""")

md(r"""
$\|[\hat H, \hat\eta^{2}]\|_{\infty}$ grows from $0$ to $\sim 0.7$ as
$s$ rises from $0$ to $0.3$. The $\eta$-pairing symmetry is *not* an
exact conservation law in the full non-orthogonal Hamiltonian. The
$400 \to 14$ reduction we just did is restricted to $s = 0$.

**Compare to $\hat S^{2}$.** Total spin commutes with $\hat H$ at *any*
$s$, since the spatial part of the Hamiltonian doesn't see spin. The
$400 \to 38 \to 22$ reductions ($D_{6}$ and singlet) are robust;
only the $22 \to 14$ ($\eta = 0$) step needs orthogonal AOs.
""")

# =====================================================================
md(r"""
## 9. The $V_{U}$ commutator: $\eta^{2}$ survives, but $\eta_{\pm}$ does not

A subtler point: even though $[\hat V_{U}, \hat\eta_{\pm}] =
\pm U\,\hat\eta_{\pm}$ (the on-site Coulomb term breaks $\eta$-rotation
invariance), one can show $[\hat V_{U}, \hat\eta^{2}] = 0$. So
$\hat\eta^{2}$ remains a good quantum number through the *Hubbard
perturbation theory* even though the $\eta_{\pm}$ ladder operators do
not stay good.

This is why the 14-dim $\eta = 0$ block in §7 captures the entire
small-$U$ Taylor series of the Hubbard-to-Heisenberg notebook: the
perturbation $\hat V_{U}$ does not mix between $\eta$ multiplets even
though it's not $\eta$-rotation invariant.
""")

code(r"""
def double_occ(ds):
    occ = {}
    for c in ds:
        occ.setdefault(c.lower(), [False, False])
        if c.islower():
            occ[c.lower()][0] = True
        else:
            occ[c.lower()][1] = True
    return sum(1 for ab in occ.values() if ab[0] and ab[1])

V_U = np.diag([double_occ(d) for d in det_strings]).astype(float)
comm_U = V_U @ E2 - E2 @ V_U
print(f"||[V_U, eta^2]||_∞ = {np.max(np.abs(comm_U)):.2e}  "
      "(must be 0 — Hubbard PT preserves eta)")
""")

# =====================================================================
md(r"""
## 10. Wrap-up

You traced benzene's symmetry cascade end-to-end:

1. **400 → 38** ($D_{6}$ spatial). Two strategies: discovery via
   degeneracy histogram, construction via orbital permutations and
   `totally_symmetric_basis`.
2. **38 → 22** ($\hat S^{2} = 0$ singlet). Builds on $\hat
   S^{2}$ commuting with $H$ at any $s$ — robust.
3. **22 → 14** ($\hat\eta^{2} = 0$). Bipartite-lattice pseudospin SU(2);
   *only valid at $s = 0$*. The break-down at $s \ne 0$ shows
   directly in $[\hat H, \hat\eta^{2}]$.

The 14-dim block is small enough that closed-form perturbation
theory (the Hubbard-to-Heisenberg notebook) and Pade resummation
become practical. Without symmetry, none of that would fit in memory.

### Take-home exercises

1. **Confirm the $\eta$ multiplet sizes.** The decomposition $175 +
   189 + 35 + 1 = 400$ over $\eta = 0, 1, 2, 3$ is a representation
   theory statement about $D_{6} \times \mathrm{SU}(2)_{\rm spin}
   \times \mathrm{SU}(2)_{\eta}$ on six sites. Compute the same
   decomposition for $L = 4$ (cyclobutadiene) and $L = 8$.
   Predict the answer first using SU(2)$_{\eta}$ Clebsch–Gordan
   counting on $L$ sites at half-filling.

2. **Triplet block.** Project to $\hat S^{2} = 2$ (triplet) within
   $A_{1g}$. What is its dimension? What's the lowest triplet energy
   at $U = 0$? Compare to the textbook $\pi$-$\pi^{*}$ excitation of
   benzene.

3. **Cation symmetry.** Repeat the chain for the benzene radical
   cation ($N = 5$). Without half-filling, $\hat\eta_{z} \ne 0$ and
   the role of $\eta$-pairing changes. What replaces it?

4. **Beyond $D_{6}$.** Add inversion ($i$) to the symmetry group, so
   the orbital permutation set generates $D_{6h}$ of order 24. Does
   the $A_{1g}$ block dimension drop further? Why or why not?

### Where this leaves you

You have the full toolkit:

- `01_h2_2c2e` (main set): matrix elements from Slater determinants,
  the $H_{2}$ closed form, MO/VB equivalence, AO overlap softening.
- `allyl_uj_identity` (this folder): two-electron integrals
  $U, J, K, M$, the $(U=J)$ operator identity, its breakdown at
  $s \ne 0$.
- `benzene_hubbard_to_heisenberg` (this folder): rational PT series
  for benzene, MP2 decomposition of $E_{2} = -29/288$, large-$U$
  Heisenberg derivation.
- this notebook: symmetry projection (group orbits, spin,
  $\eta$-pairing) and the $400 \to 14$ cascade.

These notebooks cover the spine of symvb as a research and teaching
tool. The library has more: open-shell systems, polyatomic
generalisations, the $(\text{H}_{2})_{n}^{+}$ chain, Padé resummation,
and others. The accompanying manuscript develops the research
applications systematically; the notebooks here are a friendly entry
point.
""")


NB.cells = cells
out_path = os.path.join(os.path.dirname(__file__), '..', 'symmetry_as_a_tool.ipynb')
out_path = os.path.normpath(out_path)
nbf.write(NB, out_path)
print(f"Wrote {out_path}  ({len(cells)} cells)")
