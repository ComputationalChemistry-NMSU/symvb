"""Build notebooks/02_allyl_long_bond.ipynb from cell definitions below.

Main-set notebook 2: the three-center four-electron (3c4e) pi system
(allyl anion), reproducing the manuscript "long-bond Rumer structure as a
biradical signature" section (Eqs 8-11).

Run from anywhere:  python3 notebooks/_build/build_nb2.py
"""
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
# Notebook 2 — The allyl anion: a long-bond Rumer structure as a biradical signature

**Goal.** Take `symvb` from two electrons to four, and from one bond to a
three-center chain. The symmetric **three-center four-electron (3c4e)**
$\pi$ system, the class that contains the allyl anion, ozone, and the azide
anion, is the smallest $\pi$ frame that supports a non-trivial *long-bond*
Rumer structure. We will

1. enumerate the nine-determinant $S_z = 0$ basis of the allyl $\pi$ system,
2. build the three covalent Rumer structures, two short-bond (Kekulé) and
   one **long-bond** (Dewar), and the $3\times 3$ covalent Hamiltonian,
3. *derive* the long-bond energy gain $\Delta_{\rm lb} = -\sqrt{2}\,|h|$ and
   the covalent-sector weights $(\tfrac14, \tfrac14, \tfrac12)$ in closed form,
4. watch the long-bond weight in the full configuration interaction (FCI)
   ground state rise from $1/8$ to $1/2$ as correlation grows, tracking an
   independent natural-orbital biradical index, and
5. close the symmetric story with an exact overlap-only superexchange
   through a strictly closed-shell bridge, a singlet–triplet gap with **no**
   $1/U$ term, and
6. break the symmetry with a **site-energy offset** on the bridge (a
   heteroatom at the central position), keeping both limits of the long-bond
   weight in closed form and testing whether the closed-shell superexchange
   feels the offset.

This is the companion notebook to the manuscript's allyl section (Eqs 8–11
and Figure 2).

**Prerequisites.** Notebook 1 (the H₂ covalent/ionic 2×2). Linear algebra
and Slater determinants at the level of one quantum-chemistry course.

**Runtime.** Under fifteen seconds; the correlation scan solves the full FCI
ground state at 200 values of $U$.
""")

# =====================================================================
md(r"""
## 1. Setup

We keep `sympy` for the symbolic algebra and `numpy`/`matplotlib` for the
correlation scan. From `symvb` we need `Molecule` (the matrix-element
engine), `generate_dets` (the determinant basis), `FixedPsi` (a labelled
linear combination of determinants, used to *write down* a VB structure),
and two spin tools, `s_squared_matrix` and `project_onto_S`. For the routine
steps, solving a small block and reading off structure weights, we use the
`symvb` high-level facade: `System`, `ground_state`, `chirgwin_coulson`, and
`structure_vector` (the last expands a VB structure over the determinant
basis with the canonical-reordering fermion sign handled for us).
""")

code(r"""
import sympy as sp
from sympy import init_printing
init_printing()
import numpy as np

# Render 'label = expr' as LaTeX in Jupyter; falls back to print without IPython.
try:
    from IPython.display import display, Math
    def show(label, expr):
        display(Math(label + ' = ' + sp.latex(expr)))
except ImportError:
    def show(label, expr):
        print(label, '=', expr)

from symvb import Molecule, FixedPsi, System
from symvb.fixed_psi import generate_dets
from symvb.spin import s_squared_matrix, project_onto_S
from symvb.system import ground_state, chirgwin_coulson, structure_vector
""")

# =====================================================================
md(r"""
## 2. The 3c4e π chain and its nine-determinant basis

The model is a three-atomic-orbital $\pi$ chain $\{a, b, c\}$ with
nearest-neighbor edges $a$–$b$ and $b$–$c$ (`interacting_orbs=['ab','bc']`).
Each edge carries the resonance integral $h$ and overlap $s$; each center
carries the on-site Hubbard repulsion $U$ (the only two-electron integral,
`subst_2e={'U': ('1111',)}`, `max_2e_centers=1`). We set the site energies
to zero (`zero_ii=True`).

The closed-shell allyl **anion** has four $\pi$ electrons. In the
$S_z = 0$ sector (two $\alpha$, two $\beta$) the space is
$\binom{3}{2}\binom{3}{2} = 9$ determinants, obtained from
`generate_dets(2, 2, 3)`.
""")

code(r"""
m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab', 'bc'],
    subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
    subst_2e={'U': ('1111',)},
    max_2e_centers=1,
)
P = generate_dets(2, 2, 3)
det_strings = [p.dets[0].det_string for p in P]
print(f"{len(det_strings)}-determinant Sz=0 basis:")
print(det_strings)
""")

md(r"""
The one-electron (Hückel) problem on the chain has molecular-orbital
eigenvalues $\{-\sqrt{2}\,h,\; 0,\; +\sqrt{2}\,h\}$, with the central
**nonbonding** orbital $\psi_2 = (a - c)/\sqrt{2}$ doubly occupied in the
closed-shell reference. We confirm the spectrum from the $3\times 3$ AO
Hückel matrix.
""")

code(r"""
h, s, U = sp.symbols('h s U')
Huckel = sp.Matrix([[0, h, 0], [h, 0, h], [0, h, 0]])   # a-b-c chain, s = 0
sp.simplify(Huckel.eigenvals())   # {-sqrt(2) h, 0, +sqrt(2) h}
""")

# =====================================================================
md(r"""
## 3. Building H and S, and the three Rumer structures

We assemble the symbolic $9\times 9$ Hamiltonian and overlap exactly as in
Notebook 1: a one-electron part, a two-electron part, and the metric.
""")

code(r"""
H_det = sp.Matrix(m.build_matrix(P, op='H')) + sp.Matrix(m.o2_matrix(P))
S_det = sp.Matrix(m.build_matrix(P, op='S'))
H_det.shape, S_det.shape
""")

md(r"""
Three covalent (spin-paired) singlet structures span the covalent sector
(manuscript Eq. 8):

$$
\Phi_{ab} = [a\cdot b]_s\,c^2,\qquad
\Phi_{bc} = [b\cdot c]_s\,a^2,\qquad
\Phi_{ac} = [a\cdot c]_s\,b^2,
$$

where $[p\cdot q]_s = (p_\alpha q_\beta - p_\beta q_\alpha)/\sqrt{2}$ is the
Heitler–London singlet pair on orbitals $(p, q)$. $\Phi_{ab}$ and
$\Phi_{bc}$ are the **short-bond** (Kekulé) structures, each pairing
adjacent atoms with a closed-shell lone pair on the third. $\Phi_{ac}$ is
the **long-bond** (Dewar) structure: it pairs the *terminal* electrons
across the full chain while the bridging atom $b$ holds a lone pair. Its
weight is the VB signature of biradical character, because an $a$–$c$ pair
binds two electrons a full bond length apart.

`FixedPsi` lets us write each structure by naming a parent determinant and
the orbital positions to spin-couple.
""")

code(r"""
# parent determinant + the two positions to couple into an HL singlet pair
Phi_ab = FixedPsi('aBcC', coupled_pairs=[(0, 1)])   # HL(a,b) x c^2
Phi_bc = FixedPsi('aAbC', coupled_pairs=[(2, 3)])   # HL(b,c) x a^2
Phi_ac = FixedPsi('abBC', coupled_pairs=[(0, 3)])   # HL(a,c) x b^2  (long bond)

for name, fp in [('Phi_ab', Phi_ab), ('Phi_bc', Phi_bc), ('Phi_ac', Phi_ac)]:
    g = FixedPsi(fp); g.canonicalize()
    print(f"{name} = {g}")
""")

# =====================================================================
md(r"""
## 4. From structures to vectors: the canonical-ordering subtlety

To get a $3\times 3$ covalent block we need each structure as a coefficient
vector over the nine standard determinants. There is one genuine subtlety.
`symvb` stores determinants in a **canonical spin-orbital order**
(orbital-alphabetical, $\alpha$ before $\beta$). The long-bond coupling
`(0, 3)` on the parent `abBC` produces a determinant whose spins read
"$\alpha\alpha\beta\beta$" rather than the interleaved
"$\alpha\beta\alpha\beta$" of the stored basis. Reordering fermionic
spin-orbitals flips the sign of the determinant, so we must standardize each
determinant string *and track that reorder sign* before adding its
coefficient into the vector.

The facade's `structure_vector` does exactly that: it expands a `FixedPsi`
structure over the standard determinant basis and tracks the $\pm 1$ fermion
sign of the canonical reordering for us (via `standardize_det`), so we no
longer hand-derive it. We keep one small `_inversions` helper here only
because the natural-orbital density-matrix cell further down reuses it.
""")

code(r"""
orbs = 'abc'

def _inversions(lst):                       # reused by the density-matrix cell below
    return sum(1 for i in range(len(lst)) for j in range(i + 1, len(lst))
               if lst[i] > lst[j])

# structure_vector folds in the alpha-alpha-beta-beta vs interleaved reorder sign
V = sp.Matrix.hstack(*[structure_vector(st, det_strings)
                       for st in (Phi_ab, Phi_bc, Phi_ac)])
V    # 9 x 3 : columns are the three Rumer structures in the standard basis
""")

# =====================================================================
md(r"""
## 5. The covalent 3×3 block and its spectrum

At orthogonal AOs ($s = 0$) the metric is the identity and the three
structures are mutually orthogonal (they populate disjoint determinant
pairs), each with norm$^2 = 2$. We form the covalent Hamiltonian
$H_{\rm cov} = \tfrac12 V^{\!\top} H\, V$, the factor $\tfrac12$ putting us
in an orthonormal structure basis. We build the block by projecting the
determinant-basis $H$ through $V$, rather than handing the three structures
straight to `Molecule.build_matrix`: the long-bond structure carries the
$\alpha\alpha\beta\beta$ ordering that the matrix builder cannot evaluate
directly, which is exactly why we expanded the structures with
`structure_vector` first.
""")

code(r"""
H_det_s0 = H_det.subs(s, 0)
S_det_s0 = S_det.subs(s, 0)
assert S_det_s0 == sp.eye(9), "determinant basis is orthonormal at s = 0"

H_cov = sp.simplify((V.T * H_det_s0 * V) / 2)   # orthonormal Rumer basis
S_cov = sp.simplify((V.T * S_det_s0 * V) / 2)
assert S_cov == sp.eye(3)
H_cov
""")

md(r"""
This is the manuscript's covalent block: a common diagonal $U$ (each
structure is a covalent configuration with no doubly-occupied site, so it
carries only the bookkeeping $U$ shift), an off-diagonal $-h$ coupling each
Kekulé structure to the long-bond structure, and a **vanishing** $(1,2)$
entry, the two Kekulé structures do not couple directly. $\Phi_{ac}$ is
therefore the only structure that links them.

Diagonalizing gives the spectrum $\{U - \sqrt{2}|h|,\; U,\; U + \sqrt{2}|h|\}$,
the Hückel MO levels of §2 shifted by $U$ (a three-center coincidence).
""")

code(r"""
spectrum = sp.simplify(H_cov.eigenvals())
spectrum    # {U - sqrt(2) h, U, U + sqrt(2) h};  with h = -|h| the ground level is U - sqrt(2)|h|
""")

# =====================================================================
md(r"""
## 6. The long-bond energy gain Δ_lb = −√2 |h|

How much energy does the long-bond structure buy? Compare the ground state
of the full $3\times 3$ block with that of the **two-structure** Kekulé-only
block (drop $\Phi_{ac}$). The difference is the variational gain on
admitting the long bond (manuscript Eq. 9),

$$
\Delta_{\rm lb} \equiv E_{3\text{-cov}} - E_{2\text{-cov}} = -\sqrt{2}\,|h|,
\qquad\text{independent of } U.
$$
""")

code(r"""
def E_lo(Mat):
    return min(Mat.eigenvals().keys(), key=lambda r: float(r.subs({U: 0, h: -1})))

E_3 = E_lo(H_cov)                 # full covalent block
E_2 = E_lo(H_cov[:2, :2])         # Kekulé-only block (drop Phi_ac)
Delta_lb = sp.simplify(E_3 - E_2)
show(r'E_{3\text{-cov}}', E_3)
show(r'E_{2\text{-cov}}', E_2)
show(r'\Delta_{\rm lb}', Delta_lb)   # = -sqrt(2)|h| for h = -|h|
""")

md(r"""
The two Kekulé structures do not couple (the $(1,2)$ entry vanished in §5)
and are degenerate at $U$, so the Kekulé-only ground state is simply
$E_{2\text{-cov}} = U$. Admitting the long-bond structure lowers it to
$E_{3\text{-cov}} = U - \sqrt2\,|h|$, the long bond by itself deepening the
covalent ground state by $\sqrt2\,|h|$. The gain is **$U$-independent**, a
property of the covalent subspace, not of correlation strength. (In
Figure 2B the guide line at $E - U = -\sqrt2$ is this covalent ground level;
the line at $-2\sqrt2$ is the full closed-shell Hückel energy
$2(-\sqrt2|h|) + 2\cdot 0$.)
""")

# =====================================================================
md(r"""
## 7. Covalent-sector weights (1/4, 1/4, 1/2)

The covalent-sector ground state is, in closed form (manuscript Eq. 10),

$$
|\Psi_{\rm cov}\rangle = \tfrac12 \Phi_{ab} + \tfrac12 \Phi_{bc}
   - \tfrac{\sqrt 2}{2}\Phi_{ac},
\qquad
w_{ab} = w_{bc} = \tfrac14,\quad w_{ac} = \tfrac12,
$$

again independent of $U$ on the covalent sector. The facade's `ground_state`
returns the ground root and its (un-normalized) eigenvector, and
`chirgwin_coulson` turns that eigenvector into structure weights under the
$S_{\rm cov}$ metric.
""")

code(r"""
_, gs = ground_state(H_cov, S_cov)                     # facade: ground root + eigenvector
weights = chirgwin_coulson(gs, S_cov, simplify=True)   # facade: Chirgwin-Coulson weights
assert list(weights) == [sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 2)]

gs = sp.simplify(gs / sp.sqrt((gs.T * gs)[0]))         # normalize for display
show(r'|\Psi_{\rm cov}\rangle\ \ (\Phi_{ab},\ \Phi_{bc},\ \Phi_{ac})', gs.T)
show(r'(w_{ab},\ w_{bc},\ w_{ac})', weights.T)          # (1/4, 1/4, 1/2)
""")

# =====================================================================
md(r"""
## 8. The full FCI: long-bond weight as a biradical signature

So far we restricted to the three covalent structures. The full nine-
determinant FCI also contains the doubly-ionic singlets (both pairs
condensed onto single centers, e.g. $a^2 b^2$) and the triplet components.
On the **full** ground state the long-bond weight $w_{ac}$ is no longer
fixed: it rises continuously from $1/8$ at $U = 0$ to the covalent-sector
value $1/2$ as $U \to \infty$ and the ionic admixture decays.

We solve the full nine-determinant ground state with the facade's
`ground_state` in numeric mode (pass a `subs` dict, and it diagonalizes with
scipy) and project it onto the orthonormal Rumer structures. At $s = 0$ the
structures are orthonormal, so these Chirgwin–Coulson weights are plain
squared projections, and the part not captured, $1 - \sum w$, is the ionic
admixture. We keep the squared projections here rather than calling
`System.weights(structures=...)`, which renormalizes over the structure space
(its weights sum to one) and would instead report the covalent *composition*.
""")

code(r"""
from functools import lru_cache

sys_fci = System(m, P)              # the full 9-determinant FCI system (carries Hubbard U)
V_hat = np.array(V, float)
V_hat = V_hat / np.sqrt((V_hat**2).sum(axis=0))     # orthonormal structure columns

@lru_cache(maxsize=None)            # the weight scan and the n_3 scan share each solve
def fci_ground(Uv):
    "FCI ground-state vector at (U, h=-1, s=0), solved by the facade."
    _, psi = sys_fci.ground_state(subs={U: Uv, h: -1, s: 0})
    return psi

def rumer_weights(Uv):
    return (V_hat.T @ fci_ground(Uv))**2            # (w_ab, w_bc, w_ac)

for Uv in [0, 1, 4.4, 16, 64, 1024]:
    w_ab, w_bc, w_ac = rumer_weights(Uv)
    print(f"U/|h| = {Uv:>7g}:  w_ab = {w_ab:.4f}  w_bc = {w_bc:.4f}  "
          f"w_ac = {w_ac:.4f}  (covalent sum = {w_ab+w_bc+w_ac:.4f})")
""")

md(r"""
At $U = 0$, $w_{ac} = 1/8$; at large $U$ it saturates at $1/2$. The PPP value
for a carbon $\pi$ system, $U/|h| \approx 4.4$, lands at $w_{ac} \approx 0.31$:
within the model the gas-phase allyl anion already carries appreciable
biradical character, the long bond contributing about a third of the
covalent weight.

The long-bond weight tracks an *independent* biradical measure, the
natural-orbital index $n_3/2$, where $n_3$ is the occupation of the Hückel
antibonding orbital $\psi_3 = (a - \sqrt2\,b + c)/2$ in the correlated ground
state ($n_3 = 0$ for a closed shell). We build the one-particle density
matrix in the AO basis, rotate to the Hückel MOs, and read off $n_3$.
""")

code(r"""
# spin-summed AO one-particle density matrix gamma_pq = <Psi| sum_sigma a+_{p s} a_{q s} |Psi>
def _canon_idx(ds):
    return [2 * orbs.index(c.lower()) + (0 if c.islower() else 1) for c in ds]
_sig = np.array([(-1) ** _inversions(_canon_idx(d)) for d in det_strings], float)
_occ2idx = {tuple(sorted(_canon_idx(d))): i for i, d in enumerate(det_strings)}

def _apply_pq(occ, p, q, spin):
    qi, pi = 2 * orbs.index(q) + spin, 2 * orbs.index(p) + spin
    if qi not in occ: return None
    k = occ.index(qi); sg = (-1) ** k
    rest = occ[:k] + occ[k+1:]
    if pi in rest: return None
    j = 0
    while j < len(rest) and rest[j] < pi: j += 1
    return sg * (-1) ** j, tuple(rest[:j] + [pi] + rest[j:])

rho = np.zeros((3, 3, 9, 9))
for Jc, d in enumerate(det_strings):
    occ = sorted(_canon_idx(d))
    for ip, p in enumerate(orbs):
        for iq, q in enumerate(orbs):
            for spin in (0, 1):
                res = _apply_pq(occ, p, q, spin)
                if res is None: continue
                sg, oI = res
                Ic = _occ2idx.get(oI)
                if Ic is not None:
                    rho[ip, iq, Ic, Jc] += sg
rho = _sig[None, None, :, None] * rho * _sig[None, None, None, :]

r2 = np.sqrt(2.0)
C_mo = np.array([[0.5, 1/r2, 0.5], [1/r2, 0, -1/r2], [0.5, -1/r2, 0.5]]).T  # cols psi_1,2,3

def n3_over_2(Uv):
    psi = fci_ground(Uv)                             # same facade FCI ground state
    gamma = np.array([[psi @ rho[i, j] @ psi for j in range(3)] for i in range(3)])
    gamma = 0.5 * (gamma + gamma.T)
    return 0.5 * float(C_mo[:, 2] @ gamma @ C_mo[:, 2])

for Uv in [0, 4.4, 1e6]:
    wac = rumer_weights(Uv)[2]; n3 = n3_over_2(Uv)
    print(f"U/|h| = {Uv:>9g}:  w_ac = {wac:.4f}   n_3/2 = {n3:.4f}   diff = {wac-n3:+.4f}")
""")

md(r"""
The two scales differ by exactly $1/8$ at both limits ($w_{ac} - n_3/2 =
\tfrac18 - 0$ at $U = 0$, and $\tfrac12 - \tfrac38$ at $U \to \infty$), with
the gap widening to $\approx 0.2$ in the crossover. The VB long-bond weight
and the natural-orbital biradical index are two faces of the same
correlation. We reproduce the manuscript Figure 2 panels A and C.
""")

code(r"""
import matplotlib.pyplot as plt
U_grid = np.logspace(-2, 4, 200)
W  = np.array([rumer_weights(Uv) for Uv in U_grid])     # columns w_ab, w_bc, w_ac
N3 = np.array([n3_over_2(Uv) for Uv in U_grid])

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(U_grid, W[:, 0], 'C0-',  lw=2,   label=r'$w_{ab}$ (Kekulé)')
ax[0].plot(U_grid, W[:, 1], 'C0--', lw=1.4, label=r'$w_{bc}$ (Kekulé)')
ax[0].plot(U_grid, W[:, 2], 'C3-',  lw=2.4, label=r'$w_{ac}$ (long-bond)')
ax[0].plot(U_grid, 1 - W.sum(1), 'C7:', lw=1.6, label='ionic (remainder)')
for y in (0.125, 0.25, 0.375, 0.5): ax[0].axhline(y, color='k', lw=0.3, alpha=0.3)
ax[0].set_xscale('log'); ax[0].set_xlabel(r'$U/|h|$'); ax[0].set_ylabel('weight')
ax[0].set_title('(A) Rumer weights vs U'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].plot(U_grid, W[:, 2], 'C3-',  lw=2.4, label=r'$w_{ac}$ (VB long-bond)')
ax[1].plot(U_grid, N3,      'C2--', lw=2,   label=r'$n_3/2$ (NO biradical index)')
ax[1].plot(U_grid, W[:, 2] - N3, 'C7:', lw=1.5, label='difference')
for y in (0.125, 0.375, 0.5): ax[1].axhline(y, color='k', lw=0.3, alpha=0.3)
ax[1].set_xscale('log'); ax[1].set_xlabel(r'$U/|h|$'); ax[1].set_ylabel('biradical diagnostic')
ax[1].set_title('(C) VB vs natural-orbital'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

# =====================================================================
md(r"""
## 9. Overlap-only superexchange through a closed-shell bridge

The long-bond manifold isolates a subtle effect: the magnetic coupling
between the two terminal electrons when the bridge $b$ is held *strictly*
doubly occupied. Constraining $b$ to a closed shell leaves a two-
dimensional covalent manifold, the long-bond singlet $\Phi_{ac}$ and its
triplet partner, spanned by the two determinants with $b^2$.

Spin symmetry splits the manifold into two one-dimensional blocks, so each
energy is an exact Rayleigh quotient $\langle\psi|H|\psi\rangle /
\langle\psi|S|\psi\rangle$, no perturbation theory needed. We now keep the
edge overlap $s$ symbolic.
""")

md(r"""
First, the explicit $2\times2$. Restricting the full $H$ and $S$ to the two
strict-$b^2$ determinants, $D_1 = $ `aBbC` and $D_2 = $ `bAcB`, gives the
strict-$b^2$ long-bond Hamiltonian and metric the manuscript displays just
above Eq. 11,

$$
H_{\rm lb} = \begin{pmatrix} U + 4hs^3 - 4hs & 4hs^3 \\ 4hs^3 & U + 4hs^3 - 4hs \end{pmatrix},
\qquad
S_{\rm lb} = \begin{pmatrix} 1 - 2s^2 + s^4 & s^4 \\ s^4 & 1 - 2s^2 + s^4 \end{pmatrix}.
$$

No ionic charge-transfer determinant enters this space, so the on-site
repulsion $U$ appears only on the diagonal as a double-occupancy cost, never
as an off-diagonal hopping. Diagonalizing this $2\times2$ generalized
eigenproblem, whose symmetric and antisymmetric combinations are the singlet
and triplet, gives the singlet–triplet gap of Eq. 11 below. We pull the block
straight out of the determinant matrices and check it against the manuscript
form.
""")

code(r"""
i1 = det_strings.index('aBbC')      # D1: a(alpha) b^2 c(beta)
i2 = det_strings.index('bAcB')      # D2: a(beta)  b^2 c(alpha)

H_lb = sp.simplify(H_det[[i1, i2], [i1, i2]])   # 2x2 over the strict-b^2 dets
S_lb = sp.simplify(S_det[[i1, i2], [i1, i2]])

H_lb_ref = sp.Matrix([[U + 4*h*s**3 - 4*h*s, 4*h*s**3],
                      [4*h*s**3, U + 4*h*s**3 - 4*h*s]])
S_lb_ref = sp.Matrix([[1 - 2*s**2 + s**4, s**4],
                      [s**4, 1 - 2*s**2 + s**4]])
assert sp.simplify(H_lb - H_lb_ref) == sp.zeros(2, 2), "H_lb must match manuscript"
assert sp.simplify(S_lb - S_lb_ref) == sp.zeros(2, 2), "S_lb must match manuscript"

show(r'H_{\rm lb}', H_lb)
show(r'S_{\rm lb}', S_lb)
""")

md(r"""
Diagonalizing $H_{\rm lb}$ against $S_{\rm lb}$ is what the next cell does, in
the spin-adapted form: the symmetric and antisymmetric combinations of $D_1$
and $D_2$ are the singlet and triplet, so each energy is an exact Rayleigh
quotient.
""")

code(r"""
i1 = det_strings.index('aBbC')      # a(alpha) b^2 c(beta)
i2 = det_strings.index('bAcB')      # a(beta)  b^2 c(alpha)

S2_9 = np.array(s_squared_matrix(det_strings, orbs='abc'), float)
for sign, expect in ((+1, 0.0), (-1, 2.0)):   # symmetric = singlet, antisym = triplet
    v = np.zeros(9); v[i1] = 1; v[i2] = sign
    assert abs(v @ S2_9 @ v / (v @ v) - expect) < 1e-12
print("D1 + D2 is the singlet (<S^2>=0); D1 - D2 is the triplet (<S^2>=2)")

E = {}
for sign, name in ((+1, 'S'), (-1, 'T')):
    v = sp.zeros(9, 1); v[i1] = 1; v[i2] = sign
    E[name] = sp.simplify(sp.cancel((v.T * H_det * v)[0] / (v.T * S_det * v)[0]))
show(r'E_S', E['S'])
show(r'E_T', E['T'])
""")

md(r"""
Their difference is the manuscript's Eq. 11,

$$
E_T - E_S = \frac{2 s^3\,[\,U s + 4|h|(1 - s^2)\,]}
                 {(1 - 2 s^2)(1 - 2 s^2 + 2 s^4)}
          = 8|h|\,s^3 + 2 U s^4 + \mathcal{O}(s^5).
$$
""")

code(r"""
gap = sp.simplify(sp.factor(sp.together(E['T'] - E['S'])))
show(r'E_T - E_S', gap)

t = sp.Symbol('t', positive=True)               # write h = -t, t > 0
assert gap.subs(s, 0) == 0                       # vanishes identically at every U
series = sp.series(gap.subs(h, -t), s, 0, 6).removeO().expand()
show(r'E_T - E_S\ \ (h = -t)', series)
assert series.coeff(s, 3) == 8 * t               # leading 8|h| s^3
assert series.coeff(s, 4) == 2 * U               # U enters at +2U s^4
print("checks: gap(s=0)=0;  leading 8|h| s^3;  U first appears as +2U s^4  -> OK")
""")

md(r"""
Three things to read off:

- **At $s = 0$ the gap vanishes identically, at every $U$.** The only
  coupling route between singlet and triplet is kinetic (Anderson)
  superexchange, a virtual hop that transfers a radical electron to make a
  doubly-occupied site at cost $U$. The closed-shell-bridge constraint
  excludes that ionic intermediate, so with orthogonal AOs no splitting can
  appear.
- **At $s \ne 0$ an antiferromagnetic (singlet-favoring) coupling appears
  anyway**, carried entirely by the overlap cofactors, leading order
  $8|h|\,s^3$ through the composite $s_{ab}s_{bc}$ path. It is the three-
  center counterpart of the *direct overlap* term in the H₂ singlet–triplet
  gap of Notebook 1, one power of $s$ higher.
- **There is no $1/U$ term anywhere**; with the ionic route closed the
  splitting is a pure metric effect, and $U$ enters only at $\mathcal{O}(s^4)$
  with a *positive* sign, so raising $U$ *strengthens* the singlet
  preference. (The metric singularity at $s^2 = 1/2$ bounds the trust
  region.)
""")

# =====================================================================
md(r"""
## 10. A heteroatom on the bridge: site-energy asymmetry

The allyl anion, ozone, and azide are all *symmetric* 3c4e frames: their
three $\pi$ centers share one site energy, which we set to zero above.
Replace the central atom by a more (or less) electronegative one, the
2-position of allyl by nitrogen or oxygen, and the bridge orbital acquires a
one-electron site-energy offset $\varepsilon$ ($\varepsilon < 0$ for a more
electronegative, electron-stabilizing bridge). We turn the diagonal site
energies on with `zero_ii=False` and unify the central one to the symbol
$\varepsilon$, leaving the two terminals at zero.

The offset does something the covalent block of §5 did not anticipate. Each
Rumer structure carries exactly one doubly-occupied orbital, and only the
**long-bond** structure $\Phi_{ac} = [a\cdot c]_s\,b^2$ double-occupies the
*bridge*. So $\Phi_{ac}$ feels $2\varepsilon$ while the two Kekulé structures
(bridge singly occupied) feel only $\varepsilon$: the covalent block is no
longer $\varepsilon$-invariant, its long-bond diagonal picks up $2\varepsilon$.
""")

code(r"""
eps = sp.symbols('eps')
m_eps = Molecule(
    zero_ii=False,                                   # turn the diagonal site energies on
    interacting_orbs=['ab', 'bc'],
    subst={'h': ('H_ab', 'H_bc'), 'eps': ('H_bb',),  # central site energy -> symbol eps
           's': ('S_ab', 'S_bc')},
    subst_2e={'U': ('1111',)},
    max_2e_centers=1,
)
H_eps, S_eps = System(m_eps, P).hamiltonian()        # facade: (H, S), 2e block already folded in
H_aa, H_cc = sp.symbols('H_aa H_cc')
H_eps = H_eps.subs({H_aa: 0, H_cc: 0})               # terminals alpha_a = alpha_c = 0

# same structure matrix V as §4; the long bond (column 2) double-occupies the bridge
H_cov_eps = sp.simplify((V.T * H_eps.subs(s, 0) * V) / 2)
H_cov_ref = sp.Matrix([[U + eps, 0, -h],
                       [0, U + eps, -h],
                       [-h, -h, U + 2*eps]])
assert sp.simplify(H_cov_eps - H_cov_ref) == sp.zeros(3, 3)
H_cov_eps    # diagonal (U+eps, U+eps, U+2eps): only the long bond carries 2 eps
""")

md(r"""
Both ends of the correlation trajectory stay in closed form, and in the
*same* single variable. Let

$$
q^2 = \frac12\left(1 - \frac{\varepsilon}{\sqrt{\varepsilon^2 + 8 h^2}}\right)
\in [0, 1],
$$

the squared bridge-orbital weight of the lowest symmetric Hückel MO: it runs
from $q^2 = 1$ for a deep bridge ($\varepsilon \to -\infty$) to $q^2 = 0$ for
an electron-poor one ($\varepsilon \to +\infty$), through $q^2 = \tfrac12$ at
$\varepsilon = 0$. Then the long-bond weight in the full FCI ground state is

$$
w_{ac}(U = 0) = \tfrac12 q^4, \qquad w_{ac}(U \to \infty) = q^2,
$$

recovering the symmetric $1/8$ and $1/2$ at $\varepsilon = 0$. The
uncorrelated value is a single closed-shell determinant (the $q^4$ is two
bridge factors squared); the strong-correlation limit is the covalent block
above, from which $U$ has dropped out. We assert both against the
nine-determinant FCI, in $|h|$ units ($h = -1$).
""")

code(r"""
def w_ac_eps(Uv, epsv):
    "Long-bond Chirgwin-Coulson weight in the 9-det FCI with bridge offset eps (h=-1, s=0)."
    _, psi = ground_state(H_eps, S_eps, subs={U: Uv, h: -1, s: 0, eps: epsv})
    return float((V_hat[:, 2] @ psi)**2)             # V_hat: orthonormal Rumer columns (§8)

q2    = (1 - eps / sp.sqrt(eps**2 + 8)) / 2           # |h| units; general root is eps^2 + 8 h^2
w_U0  = q2**2 / 2                                     # uncorrelated closed shell
w_inf = q2                                            # covalent-sector (U -> infinity) limit

# exact rationals at eps = 0, -1, +1
assert sp.nsimplify(w_U0.subs(eps, 0))  == sp.Rational(1, 8)
assert sp.nsimplify(w_U0.subs(eps, -1)) == sp.Rational(2, 9)
assert sp.nsimplify(w_U0.subs(eps, 1))  == sp.Rational(1, 18)
# closed forms vs FCI at both limits
for epsv in (0, -1, 1):
    assert abs(w_ac_eps(0.0, epsv) - float(w_U0.subs(eps, epsv)))  < 1e-9     # U = 0    -> q^4/2
    assert abs(w_ac_eps(1e5, epsv) - float(w_inf.subs(eps, epsv))) < 1e-3     # U -> inf -> q^2
print("w_ac(U=0)    at eps = 0, -1, +1:  1/8, 2/9, 1/18   (= q^4/2, matches FCI)")
print("w_ac(U->inf) at eps = 0, -1, +1:  1/2, 2/3, 1/3    (= q^2,   matches FCI)")
""")

md(r"""
The dependence is strong, not a small correction. At the carbon-$\pi$ point
$U/|h| = 4.4$, one unit of bridge offset moves the long-bond weight by more
than a third of its symmetric value:
""")

code(r"""
print("Bridge offset at the PPP point U/|h| = 4.4:")
for epsv, ref in [(-1, 0.494), (0, 0.311), (1, 0.180)]:
    w = w_ac_eps(4.4, epsv)
    assert abs(w - ref) < 1e-3
    tag = "more electronegative" if epsv < 0 else ("symmetric" if epsv == 0 else "electron-poor")
    print(f"  eps/|h| = {epsv:+d}  ({tag:>20s} bridge):  w_ac = {w:.3f}")
""")

md(r"""
A more electronegative bridge ($\varepsilon < 0$) stabilizes the
doubly-occupied-bridge long bond more than the Kekulé structures and
*enhances* the biradical weight; an electron-poor bridge ($\varepsilon > 0$)
suppresses it. The weight that leaves the long bond flows into the Kekulé
structures and, among the ionic determinants, specifically into the
bridge-*empty* $a^2c^2$ configuration. The biradical signature of a 3c4e
$\pi$ frame is therefore not a fixed number: it is tuned by the
electronegativity of the central atom relative to the terminals.

Finally, the closed-shell-bridge superexchange of §9 with the offset present.
Both strict-$b^2$ determinants are $b^2$, so each carries the *same* site
content $2\varepsilon$, and naively $\varepsilon$ should cancel in the
singlet–triplet gap. It does at leading order, but the overlap metric leaks
it back one power of $s$ higher.
""")

code(r"""
i1 = det_strings.index('aBbC')      # D1: a(alpha) b^2 c(beta)
i2 = det_strings.index('bAcB')      # D2: a(beta)  b^2 c(alpha)
t = sp.Symbol('t', positive=True)   # write h = -t, t = |h| > 0

E_eps = {}
for sign, name in ((+1, 'S'), (-1, 'T')):
    v = sp.zeros(9, 1); v[i1] = 1; v[i2] = sign
    E_eps[name] = sp.cancel((v.T * H_eps * v)[0] / (v.T * S_eps * v)[0])
gap_eps = sp.simplify(sp.together(E_eps['T'] - E_eps['S']))

# reduces to Eq. 11 at eps = 0
eq11 = (2*s**3 * (U*s + 4*(-h)*(1 - s**2))
        / ((1 - 2*s**2) * (1 - 2*s**2 + 2*s**4)))
assert sp.simplify(gap_eps.subs(eps, 0) - eq11) == 0

series_eps = sp.series(gap_eps.subs(h, -t), s, 0, 5).removeO().expand()
assert series_eps.coeff(s, 3) == 8 * t                    # leading 8|h| s^3 is eps-free
assert sp.expand(series_eps.coeff(s, 4)) == 2*U + 4*eps   # eps enters with U at O(s^4)
show(r'[E_T - E_S]_{s^3}', series_eps.coeff(s, 3))              # 8|h|, unchanged by eps
show(r'[E_T - E_S]_{s^4}', sp.expand(series_eps.coeff(s, 4)))   # 2U + 4 eps
""")

md(r"""
The leading $8|h|\,s^3$ superexchange is untouched by $\varepsilon$: it is a
pure two-overlap ($s_{ab}s_{bc}$) path that does not see the site energy. The
offset first appears at $\mathcal{O}(s^4)$, where the on-site-like coefficient
is $2U + 4\varepsilon$: the overlap metric leaks the full
doubly-occupied-bridge content $2\varepsilon$ in exact lockstep with $U$, the
same $U + 2\varepsilon$ that sits on the $\Phi_{ac}$ diagonal of the covalent
block above. Correlation and bridge electronegativity enter the closed-shell
superexchange through one and the same channel.
""")

# =====================================================================
md(r"""
## 11. Wrap-up

You took `symvb` from a single bond to a three-center chain and found that
the **long-bond Rumer structure** is the algebraic carrier of biradical
character:

1. Built the nine-determinant 3c4e basis and the three Rumer structures,
   handling the canonical spin-ordering sign explicitly.
2. Derived the covalent $3\times 3$ block, its spectrum
   $\{U - \sqrt2|h|, U, U + \sqrt2|h|\}$, the long-bond gain
   $\Delta_{\rm lb} = -\sqrt2|h|$, and the covalent weights
   $(\tfrac14, \tfrac14, \tfrac12)$, all in closed form and all
   $U$-independent on the covalent sector.
3. Showed that on the full FCI ground state the long-bond weight rises
   $1/8 \to 1/2$ with correlation, tracking the natural-orbital biradical
   index $n_3/2$ to within a constant $1/8$.
4. Derived an exact overlap-only superexchange through a closed-shell
   bridge, a singlet–triplet gap $8|h|s^3 + 2Us^4 + \cdots$ with no $1/U$
   term.
5. Broke the mirror symmetry with a bridge site-energy offset $\varepsilon$
   and found both limits of the long-bond weight in one variable,
   $w_{ac}(U=0) = \tfrac12 q^4$ and $w_{ac}(\infty) = q^2$ with
   $q^2 = \tfrac12(1 - \varepsilon/\sqrt{\varepsilon^2 + 8h^2})$, while the
   closed-shell superexchange stayed $\varepsilon$-blind at leading order
   ($\varepsilon$ first enters the gap as $2U + 4\varepsilon$ at
   $\mathcal{O}(s^4)$).

### Take-home exercises

1. **Ozone and azide.** The same three-AO model describes O₃ and N₃⁻. Only
   the integrals change. Re-evaluate $w_{ac}$ at a parameter set with larger
   $U/|h|$ and confirm the long-bond weight grows toward $1/2$.

2. **Triplet crossing.** Add the exchange integral $K$
   (`subst_2e={'U': ('1111',), 'K': ('1122',)}`, `max_2e_centers=2`) and find
   the $K$ at which the FCI ground state turns from singlet to triplet.

3. **Bridge gap at finite $s_{ac}$.** Restore a small direct terminal
   overlap $s_{ac} \ne 0$ and recompute $E_T - E_S$. Does the $8|h|s^3$
   leading term survive?

### Up next

**Notebook 3** adds a fourth center and a charge: the $(\mathrm{H}_2)_2^{\bullet +}$
disphenoid, a mixed-valence cluster where a four-structure VB model locates
the Robin–Day Class II/III crossover in closed form.
""")


NB.cells = cells
out_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '02_allyl_long_bond.ipynb'))
nbf.write(NB, out_path)
print(f"Wrote {out_path}  ({len(cells)} cells)")
