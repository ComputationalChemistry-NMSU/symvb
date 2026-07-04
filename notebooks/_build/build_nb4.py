"""Build notebooks/04_benzene_covalent_only.ipynb from cell definitions below.

Main-set notebook 4: benzene, reproducing the manuscript "a covalent-only
picture inverts the sign of the energy response" section (Eqs 17-18,
Figures 4-5).

Run from anywhere:  python3 notebooks/_build/build_nb4.py
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
# Notebook 4 — Benzene: a covalent-only picture inverts the sign of the energy response

**Goal.** Scale `symvb` up to the aromatic ring and use it to expose a
failure mode of qualitative VB reasoning. Benzene has six $\pi$ electrons on
six centers; the $S_z = 0$ space is 400 determinants. We will

1. build the 400-determinant configuration interaction with `symvb`,
2. decompose the wavefunction by **ionicity** and find that the closed-shell
   Hückel determinant is overwhelmingly *ionic*, the covalent Heitler–London
   sector carrying only $5/72 \approx 7\%$ (manuscript Eq. 17),
3. watch correlation invert that balance, the covalent class taking over as
   $U$ grows (Figure 4),
4. then weaken one ring bond and discover that a **covalent-only** Rumer
   model predicts the *wrong sign* of the energy response, a maximum where
   the exact energy is monotone (Eq. 18, Figure 5), and trace it to AO
   non-orthogonality.

This is the companion notebook to the manuscript's benzene section.

**Prerequisites.** Notebooks 1–2 (the covalent/ionic split and Rumer
structures). The symmetry reduction $400 \to 38 \to 22$ used here in passing
is built explicitly in `additional/symmetry_as_a_tool.ipynb`.

**Runtime.** Building the 400-dimensional symbolic Hamiltonian and reducing
it to numeric form takes about two to three minutes (this is the cost of
working symbolically at this size); the analyses afterward are fast.
""")

# =====================================================================
md(r"""
## 1. The benzene π model and its 400-determinant space

Six $\pi$ orbitals $\{a, b, c, d, e, f\}$ on a ring, uniform nearest-neighbor
resonance $h$ and overlap $s$, on-site Hubbard $U$. We keep the $a$–$b$ edge
integral $H_{ab}$ **free** (the other five edges are $h$): the same build
then serves the uniform ring (set $H_{ab} = h$) and the single-edge scan of
§5. Half filling in the $S_z = 0$ sector is three $\alpha$ and three $\beta$
electrons, $\binom{6}{3}^2 = 400$ determinants.
""")

code(r"""
import os
# Pin BLAS to one thread BEFORE importing numpy: the scans below make ~250 small
# (400-dim) eigensolves, where per-call thread spawning costs more than it saves
# and can thrash a busy machine. setdefault keeps any limit you set yourself.
for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')

import sympy as sp
from sympy import init_printing
init_printing()
import numpy as np
from itertools import combinations
from scipy.linalg import eigh, eigvalsh

# Render 'label = expr' as LaTeX in Jupyter; falls back to print without IPython.
try:
    from IPython.display import display, Math
    def show(label, expr):
        display(Math(label + ' = ' + sp.latex(expr)))
except ImportError:
    def show(label, expr):
        print(label, '=', expr)

from symvb import Molecule, FixedPsi, hamiltonian, ground_state, chirgwin_coulson
from symvb.fixed_psi import generate_dets

h, s, U = sp.symbols('h s U')
Hab = sp.Symbol('H_ab')

m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'],
    subst={'h': ('H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af'),     # H_ab kept free
           's': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af')},
    subst_2e={'U': ('1111',)}, max_2e_centers=1,
)
P = generate_dets(3, 3, 6)
det_strings = [p.dets[0].det_string for p in P]
print(f"{len(det_strings)} determinants in the half-filled Sz = 0 sector")
""")

md(r"""
The full 400-dimensional space reduces by symmetry to 38 ($D_6$), then to 22
($\hat S^2 = 0$, the singlet-$A_{1g}$ block holding the ground state). We do
not need the reduction here, but it is what makes the closed forms in the
manuscript tractable; see the additional symmetry notebook for the full
$400 \to 38 \to 22 \to 14$ cascade.

Now the three 400-dimensional builds (one-electron $H$, overlap $S$, and the
on-site two-electron $H_2$). This is the slow step.
""")

code(r"""
import time
t0 = time.time()
H1 = sp.Matrix(m.build_matrix(P, op='H'))    # one-electron, symbolic in h, s, H_ab
S  = sp.Matrix(m.build_matrix(P, op='S'))    # overlap, symbolic in s
H2 = sp.Matrix(m.o2_matrix(P))               # on-site Hubbard, symbolic in s, U
print(f"three 400x400 symbolic builds: {time.time() - t0:.1f}s")

# Numeric anchor matrices. H is first-order in H_ab and in U, so a handful of
# substitutions then cover every scan with plain numpy -- much faster than
# lambdifying a 400x400 symbolic matrix.
# Reduce to numeric anchor matrices. H is first-order in H_ab and in U, so a
# few substitutions then cover every scan below with plain numpy.
def num(M): return np.array(M.tolist(), dtype=float)
t0 = time.time()
SV = 0.2                                            # working overlap for the single-edge scan
# uniform ring at s = 0 (for the ionicity decomposition)
H1_unif0 = num(H1.subs({h: -1, s: 0, Hab: -1}))
H2_perU0 = num(H2.subs({s: 0, U: 1}))              # coefficient of U at s = 0
S0       = np.eye(len(det_strings))
# single edge free at s = 0.2 (H1 is linear in H_ab)
H1_base  = num(H1.subs({h: -1, s: SV, Hab: 0}))    # a-b edge off
H1_dHab  = num(H1.subs({h: -1, s: SV, Hab: 1})) - H1_base   # d H1 / d H_ab
H2_perUs = num(H2.subs({s: SV, U: 1}))             # coefficient of U at s = 0.2
S_s      = num(S.subs(s, SV))
print(f"reduction to numeric anchors: {time.time() - t0:.0f}s")
""")

# =====================================================================
md(r"""
## 2. Ionicity decomposition: the Hückel determinant is mostly ionic

Classify each determinant by its **ionicity** $n_d$, the number of
doubly-occupied sites. A purely covalent (Heitler–London) determinant has
one electron per site, $n_d = 0$; each doubly-occupied site adds one unit of
ionicity. At $s = 0$ the determinant basis is orthonormal, so the
Chirgwin–Coulson weight of ionicity class $k$ is the plain sum of squared CI
coefficients over determinants with $n_d = k$.
""")

code(r"""
def ionicity(ds):
    low = {c for c in ds if c.islower()}
    up  = {c.lower() for c in ds if c.isupper()}
    return len(low & up)

nd = np.array([ionicity(d) for d in det_strings])
class_idx = [np.flatnonzero(nd == k).tolist() for k in range(4)]   # determinant indices per class
print("determinant counts by ionicity class n_d:")
for k in range(4):
    print(f"  n_d = {k}:  {len(class_idx[k]):>3} determinants")

def class_weights(Uval):                             # uniform ring, s = 0
    Hn = H1_unif0 + Uval * H2_perU0                  # H2 is linear in U
    c = eigh(Hn, S0, subset_by_index=[0, 0])[1][:, 0]    # lowest eigenpair (scipy, fast)
    # Chirgwin-Coulson weights via the facade, summed per ionicity class (S0 = I at s = 0)
    return chirgwin_coulson(c, S0, groups=class_idx)

w0 = class_weights(0.0)
print("\nU = 0, s = 0 ionicity-class weights (w_0, w_1, w_2, w_3):")
print("  numeric :", np.round(w0, 6))
print("  exact   : (5, 31, 31, 5)/72 =", np.round(np.array([5, 31, 31, 5]) / 72, 6))
assert np.allclose(w0, np.array([5, 31, 31, 5]) / 72, atol=1e-9)
""")

md(r"""
The covalent ($n_d = 0$) class carries only $5/72 \approx 7\%$ of the
wavefunction: the closed-shell Hückel determinant is overwhelmingly **ionic**
at the AO level (manuscript Eq. 17). The weights are *palindromic*
($w_0 = w_3$, $w_1 = w_2$), a consequence of the particle–hole symmetry of
the alternant ring at $s = U = 0$.
""")

md(r"""
### The ionicity weights hold at any overlap

That decomposition used $s = 0$, where the determinant basis is orthonormal.
At finite overlap nothing changes: the $U = 0$ weights are $(5, 31, 31, 5)/72$
at *every* $s$, not only at $s = 0$. The reason is structural. The occupied
Hückel MOs $k = 0, \pm 1$ are fixed by ring symmetry, independent of $s$
(overlap only rescales their norms, it does not rotate them), so the $U = 0$
ground state is the **same determinant** $c$ at every overlap. That determinant
is an eigenvector of the 400-dimensional determinant-space overlap matrix,

$$ S(s)\,c = (1 + 2s)^2 (1 + s)^4 \, c, $$

the eigenvalue being the product of the occupied-MO norms squared. Because $c$
is an eigenvector of the metric, every Chirgwin–Coulson weight
$w_I = c_I (S c)_I / (c^\top S c)$ collapses to its $s = 0$ value
$c_I^2 / \lVert c\rVert^2$, and the class weights cannot move. This is the
ring-scale version of the H₂ tie of Notebook 1, where $w_{\rm cov} = w_{\rm ion}
= 1/2$ at every overlap: there $\sigma = (a + b)$ gives $c = (1, 1, 1, 1)$ and
$S(s)\,c = (1 + s)^2 c$.
""")

code(r"""
# The U=0 ground state as an exact rational vector: the closed-shell determinant
# of the three occupied Hückel MOs, expanded in the AO determinant basis. The
# 3x6 MO coefficient matrix is fixed by ring symmetry, so c is s-independent.
from symvb.spin import _symvb_to_canonical_sign

def closed_shell_vector(det_strings):
    site = {ch: i for i, ch in enumerate('abcdef')}
    C = sp.Matrix([                                    # occupied MOs k = 0, +1, -1 (rows)
        [1, 1, 1, 1, 1, 1],
        [sp.Rational(1), sp.Rational(1, 2), sp.Rational(-1, 2),
         sp.Rational(-1), sp.Rational(-1, 2), sp.Rational(1, 2)],
        [0, sp.Rational(1, 2), sp.Rational(1, 2), 0,
         sp.Rational(-1, 2), sp.Rational(-1, 2)]])
    minor = {T: C[:, list(T)].det() for T in combinations(range(6), 3)}   # 3x3 MO minors
    v = sp.zeros(len(det_strings), 1)
    for I, ds in enumerate(det_strings):
        Ta, Tb = [], []
        for ch in ds:                                  # split det into alpha / beta site sets
            (Ta if ch.islower() else Tb).append(site[ch.lower()])
        Ta, Tb = tuple(sorted(Ta)), tuple(sorted(Tb))
        seq = [2 * j for j in Ta] + [2 * j + 1 for j in Tb]     # interleave to canonical order
        inv = sum(seq[a] > seq[b] for a in range(len(seq)) for b in range(a + 1, len(seq)))
        sign = _symvb_to_canonical_sign(ds, site) * (1 if inv % 2 == 0 else -1)
        v[I, 0] = sign * minor[Ta] * minor[Tb]
    return v

c = closed_shell_vector(det_strings)
lam = (1 + 2 * s) ** 2 * (1 + s) ** 4                  # product of occupied-MO norms squared
Sc = S * c                                             # 400 polynomials in s (a few seconds)
n_off = sum(1 for i in range(len(det_strings))
            if sp.simplify(sp.expand(Sc[i, 0] - lam * c[i, 0])) != 0)
print(f"S(s) c - (1+2s)^2 (1+s)^4 c :  {n_off} / 400 nonzero entries")
assert n_off == 0                                      # c is an exact eigenvector of S(s)
print("the U=0 ground-state vector is an eigenvector of the overlap metric.")
""")

code(r"""
# Class weights from the full Chirgwin-Coulson definition, exact rational, at a
# few overlaps including a negative one. All stay (5, 31, 31, 5)/72.
target = [sp.Rational(5, 72), sp.Rational(31, 72), sp.Rational(31, 72), sp.Rational(5, 72)]
print(f"{'s':>6}   " + "   ".join(f"w_{k}" for k in range(4)))
for sv in [sp.Integer(0), sp.Rational(1, 5), sp.Rational(-1, 5)]:
    Sc_sv = [Sc[i, 0].subs(s, sv) for i in range(len(det_strings))]   # cheap: subs on a vector
    D = sum(c[i, 0] * Sc_sv[i] for i in range(len(det_strings)))
    w = [sp.nsimplify(sum(c[i, 0] * Sc_sv[i] for i in class_idx[k]) / D) for k in range(4)]
    print(f"{str(sv):>6}   " + "   ".join(str(x) for x in w))
    assert w == target
print("ionicity weights are exactly (5, 31, 31, 5)/72 at every overlap s.")
""")

# =====================================================================
md(r"""
## 3. The covalent share shrinks with system size

The covalent share of a single closed-shell Hückel determinant needs no FCI:
expand the determinant of the occupied Hückel MOs in the localized AO basis.
A covalent (one-electron-per-site) configuration assigns $k = N/2$ sites to
$\alpha$ and the complement to $\beta$, with coefficient
$\det C[\alpha] \cdot \det C[\beta]$, so $w_{\rm cov}$ is a sum of squared
minors of the occupied-MO matrix. This reproduces benzene's $5/72$ exactly
and shows the covalent share collapsing as the ring grows.
""")

code(r"""
def covalent_share(adjacency, n_electrons):
    A = np.array(adjacency, float); n = A.shape[0]; k = n_electrons // 2
    w, V = np.linalg.eigh(A)                          # Hückel H = -A, occupied = top-A eigvecs
    C = V[:, np.argsort(w)[::-1][:k]]
    return sum((np.linalg.det(C[list(Aset), :]) *
                np.linalg.det(C[[x for x in range(n) if x not in Aset], :]))**2
               for Aset in combinations(range(n), k))

def ring(n):
    return [[1 if (abs(i - j) % n) in (1, n - 1) else 0 for j in range(n)] for i in range(n)]
nap_edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,0),(4,9)]
nap_adj = [[1 if (i, j) in nap_edges or (j, i) in nap_edges else 0
            for j in range(10)] for i in range(10)]

print("covalent (n_d = 0) share of the closed-shell Hückel determinant, s = 0:")
print(f"  H2          (2 centers) : {covalent_share([[0,1],[1,0]], 2):.6f}   (= 1/2)")
print(f"  benzene     (6 centers) : {covalent_share(ring(6), 6):.6f}   (= 5/72 = {5/72:.6f})")
print(f"  naphthalene (10 centers): {covalent_share(nap_adj, 10):.6f}   (~ 1%)")
assert abs(covalent_share(ring(6), 6) - 5/72) < 1e-10
""")

# =====================================================================
md(r"""
## 4. Correlation inverts the balance (Figure 4)

Turning on $U$ penalizes the doubly-occupied (ionic) determinants, so the
ground state sheds ionic character. The covalent and singly-ionic classes
exchange dominance near the carbon $\pi$ value $U/|h| \approx 4$; by
$U/|h| = 16$ the covalent class carries about 94%. Correlation acts on the
ground state chiefly by driving this inversion.
""")

code(r"""
import matplotlib.pyplot as plt
U_grid = np.logspace(-1, 1.4, 60)
Wc = np.array([class_weights(Uv) for Uv in U_grid])

print(f"{'U/|h|':>7} {'w_0 (cov)':>10} {'w_1':>8} {'w_2':>8} {'w_3':>8}")
for Uv in [0, 1, 2, 4, 8, 16]:
    w = class_weights(float(Uv))
    print(f"{Uv:>7} " + " ".join(f"{x:>8.4f}" for x in w))

fig, ax = plt.subplots(figsize=(6.2, 3.8))
labels = [r'$n_d=0$ (covalent)', r'$n_d=1$', r'$n_d=2$', r'$n_d=3$']
for k in range(4):
    ax.plot(U_grid, Wc[:, k], lw=2, label=labels[k])
ax.axvline(4.0, color='gray', lw=0.8, ls=':'); ax.text(4.2, 0.8, 'PPP', color='gray', fontsize=9)
ax.set_xscale('log'); ax.set_xlabel(r'$U/|h|$'); ax.set_ylabel('ionicity-class weight')
ax.set_title('Benzene FCI ionicity composition ($s = 0$)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

# =====================================================================
md(r"""
## 5. The covalent-only wrong sign (Eq. 18, Figure 5)

Now reason with covalent Rumer structures *alone*, as one does with paper
arrows. Benzene's covalent sector is the five Rumer structures: two Kekulé
and three Dewar (long-bond). We weaken one ring edge, scaling its resonance
integral by $\lambda \in [0, 1]$ ($H_{ab} = \lambda h$, the others fixed), and
compare the covalent-only ground state with the full FCI.
""")

code(r"""
PARENT = 'aBcDeF'
rumer = [
    FixedPsi(PARENT, coupled_pairs=[(0, 1), (2, 3), (4, 5)]),   # Kekulé 1 (pairs a-b)
    FixedPsi(PARENT, coupled_pairs=[(0, 5), (1, 2), (3, 4)]),   # Kekulé 2
    FixedPsi(PARENT, coupled_pairs=[(0, 1), (2, 5), (3, 4)]),   # Dewar 1
    FixedPsi(PARENT, coupled_pairs=[(0, 3), (1, 2), (4, 5)]),   # Dewar 2
    FixedPsi(PARENT, coupled_pairs=[(0, 5), (1, 4), (2, 3)])]   # Dewar 3
Hcov = m.build_matrix(rumer, op='H')        # symbolic 5x5 in h, s, H_ab
Scov = m.build_matrix(rumer, op='S')

# the covalent block is exactly the zero matrix at s = 0 -> independent of H_ab
assert sp.simplify(Hcov.subs(s, 0)).is_zero_matrix
assert sp.simplify(sp.diff(Hcov, Hab).subs(s, 0)).is_zero_matrix
print("s = 0:  covalent 5x5 Hamiltonian is the ZERO matrix  ->  E_cov is independent")
print("        of the weakened-edge integral H_ab (indeed of every integral).")
""")

md(r"""
At $s = 0$ the $a$–$b$ hop turns a covalent determinant into an ionic one,
which lies *outside* the covalent space, so $H_{ab}$ has no matrix element
within the covalent block and the covalent-only energy is flat. At finite
overlap the covalent and ionic determinants are no longer orthogonal, and
$H_{ab}$ re-enters the covalent block through the metric. To see the sign of
that re-entry, read the energy response itself.
""")

md(r"""
### Per-structure bond order: the sign of the re-entry

By the Hellmann–Feynman theorem, the slope of the covalent-only energy is set
by the $a$–$b$ bond order
$\rho_{ab} = \langle\Psi_{\rm cov}|\,\partial H/\partial h_{ab}\,|\Psi_{\rm cov}\rangle$.
Evaluate it first for each Rumer structure on its own. A structure in which
$a$ and $b$ are *directly paired* has $\rho_{ab}\approx+2s$, a bonding
response across the edge. A structure in which $a$ and $b$ are instead paired
to *other* neighbors has $\rho_{ab}\approx-s$, a Pauli (antibonding) response
between the two occupied bonds that meet at $a$ and at $b$. Both vanish at
$s=0$, recovering the flat block above. The Kekulé-2 value is exact:
$\rho_{ab} = 4s(s^2-1)/(7s^4+2s^2+4)$.
""")

code(r"""
# Cell A - per-structure a-b bond order: +2s (a-b paired) vs -s (a-b unpaired)
import sympy as sp
from symvb import Molecule, FixedPsi, hamiltonian
h, s = sp.symbols('h s'); Hab = sp.Symbol('H_ab')
m = Molecule(zero_ii=True,
    subst={'s': ('S_ab','S_bc','S_cd','S_de','S_ef','S_af'),
           'h': ('H_bc','H_cd','H_de','H_ef','H_af')},
    interacting_orbs=['ab','bc','cd','de','ef','af'])
P = 'aBcDeF'
structs = [('Kek1',[(0,1),(2,3),(4,5)],'a-b paired'),
           ('Kek2',[(0,5),(1,2),(3,4)],'a,b apart'),
           ('Dew1',[(0,1),(2,5),(3,4)],'a-b paired'),
           ('Dew2',[(0,3),(1,2),(4,5)],'a,b apart'),
           ('Dew3',[(0,5),(1,4),(2,3)],'a,b apart')]
for name, cp, tag in structs:
    Hf, Sf = hamiltonian(m, [FixedPsi(P, coupled_pairs=cp)], two_electron=False)   # 1x1 H, S (1e-only model)
    rho = sp.simplify(sp.diff(Hf[0,0], Hab)/Sf[0,0])
    print(f'{name:5s} ({tag:10s}): leading {sp.series(rho,s,0,2).removeO()}   exact {rho}')
Hk2, Sk2 = hamiltonian(m, [FixedPsi(P, coupled_pairs=[(0,5),(1,2),(3,4)])], two_electron=False)
rho_k2 = sp.simplify(sp.diff(Hk2[0,0], Hab) / Sk2[0,0])
assert sp.simplify(rho_k2 - 4*s*(s**2-1)/(7*s**4+2*s**2+4)) == 0
""")

md(r"""
### Which structures dominate when the edge is off

With the $a$–$b$ coupling removed ($\lambda=0$), reduce the covalent ground
state to its two Kekulé structures. The bond-avoiding Kekulé, the one that
does *not* pair $a$ with $b$, carries weight $(5+\sqrt{10})/10\approx0.82$.
The negative-$\rho_{ab}$ structures therefore dominate the covalent ground
state.
""")

code(r"""
# Cell B - two-Kekule reduction: w_Kek2 = (5+sqrt10)/10 (leading order in s)
Kek1 = FixedPsi(P, coupled_pairs=[(0,1),(2,3),(4,5)])
Kek2 = FixedPsi(P, coupled_pairs=[(0,5),(1,2),(3,4)])
Hk = sp.Matrix(m.build_matrix([Kek1,Kek2],'H')).subs({Hab:0, h:-1})
Sk = sp.Matrix(m.build_matrix([Kek1,Kek2],'S'))
H1 = sp.diff(Hk, s).subs(s, 0); S0 = Sk.subs(s, 0)
E = sp.symbols('E'); roots = sp.solve((H1 - E*S0).det(), E)
Egs = min(roots, key=lambda r: float(r)); c = (H1 - Egs*S0).nullspace()[0]
wK = chirgwin_coulson(c, S0, simplify=True)          # CC weights of the 2x2 GS via the facade
show(r'(w_{\rm Kek1},\ w_{\rm Kek2})', sp.Matrix([[sp.nsimplify(x) for x in wK]]))
assert sp.simplify(wK[1] - (5+sp.sqrt(10))/10) == 0
""")

md(r"""
### Sign by dominance, not by averaging

Because the bond-avoiding structures dominate, the covalent ground-state
$\rho_{ab}$ is negative. Since $\partial E_{\rm cov}/\partial\lambda =
h\,\rho_{ab}$ with $h<0$, the slope is *positive* (about $+0.09\,|h|$ at
$\lambda=0$, $s=0.2$), the wrong sign for bond formation. This ground-state
$\rho_{ab}$ is **not** the weight-average of the per-structure values: the
non-orthogonal metric sets the sign through the dominant structures, not
through an average of structure responses.
""")

code(r"""
# Cell C - covalent ground-state rho_ab is negative (sign-by-dominance, NOT a weighted average)
import numpy as np
rumer = [FixedPsi(P, coupled_pairs=cp) for _,cp,_ in structs]
Hc, Sc = hamiltonian(m, rumer, two_electron=False)   # 5x5 covalent (1e-only) block, symbolic in h, s, H_ab
Tab = sp.diff(sp.Matrix(Hc), Hab)            # a-b hopping operator dH/dh_ab
ssub = {s: sp.Rational(1, 5)}
S0n = np.array(sp.Matrix(Sc).subs(ssub).tolist(), float)
Tn  = np.array(Tab.subs({h: -1, **ssub}).tolist(), float)
# covalent ground state via the facade (numeric scipy path)
_, cgs = ground_state(Hc, Sc, subs={Hab: 0, h: -1, s: sp.Rational(1, 5)})
# rho_ab is a bond-order EXPECTATION <Psi|dH/dh_ab|Psi>, not a CC weight -> contract by hand
rho_gs = (cgs@Tn@cgs)/(cgs@S0n@cgs)
wcc = chirgwin_coulson(cgs, S0n)             # per-structure CC weights of the covalent GS
rho_i = np.array([Tn[i,i]/S0n[i,i] for i in range(5)])
print(f'covalent GS rho_ab = {rho_gs:+.4f}  ->  slope dE_cov/dlam = {-rho_gs:+.4f}|h|  (wrong sign)')
print(f'naive weight-average sum_i w_i rho_i = {float(wcc@rho_i):+.4f}  (NOT equal: sign set by dominant structures)')
assert rho_gs < 0 < -rho_gs
""")

md(r"""
With the mechanism in hand, scan the full covalent-only ground state against
the FCI. We set $s = 0.2$ to make the overlap effect visible.
""")

code(r"""
Hcov_fn = sp.lambdify((h, s, Hab), Hcov, 'numpy')
Scov_fn = sp.lambdify((h, s, Hab), Scov, 'numpy')
lam = np.linspace(1.0, 0.0, 201)

def E_cov(sv):
    return np.array([eigvalsh(np.asarray(Hcov_fn(-1.0, sv, l * -1.0), float),
                              np.asarray(Scov_fn(-1.0, sv, l * -1.0), float))[0] for l in lam])
def E_fci(Uv=0.0, grid=None):                       # s = 0.2 anchors; H_ab = lam * h
    g = lam if grid is None else grid
    H2c = Uv * H2_perUs
    return np.array([eigh(H1_base + (l * -1.0) * H1_dHab + H2c, S_s,
                          eigvals_only=True, subset_by_index=[0, 0])[0] for l in g])

Ecov02, Efci02 = E_cov(0.2), E_fci(0.0)
im = int(np.argmax(Ecov02))
print(f"covalent-only maximum at  lambda* = {lam[im]:.3f},  "
      f"rise above broken-bond = {Ecov02[im] - Ecov02[-1]:.4f} |h|")
assert abs(lam[im] - 0.322) < 0.01 and abs((Ecov02[im] - Ecov02[-1]) - 0.0142) < 5e-4
print(f"covalent-ionic resonance gap E_cov - E_FCI: "
      f"{(Ecov02 - Efci02).min():.2f} to {(Ecov02 - Efci02).max():.2f} |h|")
print("FCI is monotone in lambda; the covalent-only curve is not -> wrong sign near lambda=0.")
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.7))
ax[0].plot(lam, Efci02, 'k-',  lw=2.2, label=r'full FCI ($U=0$)')
ax[0].plot(lam, Ecov02, 'C3-', lw=2.2, label='covalent-only (5 Rumer)')
ax[0].set_xlabel(r'$\lambda = h_{ab}/h$  (1 = intact, 0 = broken)')
ax[0].set_ylabel(r'ground-state energy / $|h|$')
ax[0].set_title('(A) covalent-only vs FCI'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[0].invert_xaxis()

# inset-style second panel: covalent-only relative to broken-bond value, vs overlap
for sv, col in [(0.0, 'C7'), (0.2, 'C0'), (0.4, 'C3')]:
    e = E_cov(sv); ax[1].plot(lam, e - e[-1], color=col, lw=2, label=f'$s = {sv}$')
ax[1].axhline(0, color='k', lw=0.4)
ax[1].set_xlabel(r'$\lambda = h_{ab}/h$'); ax[1].set_ylabel(r'$E_{\rm cov}(\lambda) - E_{\rm cov}(0)$')
ax[1].set_title('(A, inset) the maximum grows with overlap'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
ax[1].invert_xaxis()
plt.tight_layout(); plt.show()
""")

md(r"""
### The mono-ionic class repairs the sign

The covalent-only failure does not demand the full 400-determinant space to
fix. Ask instead which determinants must be *added back* to restore the correct
sign. Project the FCI onto ionicity-truncated subspaces by determinant index
and read the slope $dE/d\lambda$ at the broken-bond end $\lambda = 0$
(Hellmann–Feynman: $dE/d\lambda = c^\top (\partial H/\partial\lambda)\,c$ at
fixed $s = 0.2$, $U = 0$). The covalent block alone ($n_d = 0$, 20
determinants) gives the wrong sign, $+0.087\,|h|$. Adding just the singly-ionic
determinants ($n_d \le 1$, 200 determinants) flips the slope to $-0.210\,|h|$,
monotone, recovering 76% of the FCI slope $-0.277\,|h|$. The single
charge-transfer (superexchange) class alone repairs the qualitative failure;
the on-site $U$ never enters the covalent block, so covalent-only is
$U$-independent.

The finite-$U$ wrong-sign *onset* follows from the same edge slope (section
6): the interior maximum grows out of the $\lambda = 0$ edge when the edge
population $\rho_{ab}(0, U^\ast)$ changes sign, at $U^\ast/|h| = 7.05$ for
$n_d \le 1$ and $8.21$ for $n_d \le 2$ and the FCI. See
`examples/benzene_ionicity_repair.py`.
""")

code(r"""
# Slope dE/dlambda at lambda = 0 (s = 0.2, U = 0) on ionicity-truncated subspaces.
# H(lambda) = H1_base + (lambda * h) * H1_dHab with h = -1, so dH/dlambda = -H1_dHab;
# reuse the single-edge anchors built in section 5.
def slope_at_broken_bond(keep):                        # keep = determinant indices, None = FCI
    if keep is None:
        Hb, Ss, dH = H1_base, S_s, H1_dHab
    else:
        ix = np.ix_(keep, keep); Hb, Ss, dH = H1_base[ix], S_s[ix], H1_dHab[ix]
    c = eigh(Hb, Ss, subset_by_index=[0, 0])[1][:, 0]  # ground state at lambda = 0
    return -(c @ dH @ c) / (c @ Ss @ c)                # Hellmann-Feynman edge response

keep_cov = np.array(class_idx[0])                      # n_d = 0   (20 dets)
keep_nd1 = np.array(class_idx[0] + class_idx[1])       # n_d <= 1  (200 dets)
sl_cov = slope_at_broken_bond(keep_cov)
sl_nd1 = slope_at_broken_bond(keep_nd1)
sl_fci = slope_at_broken_bond(None)
print(f"covalent-only (n_d=0, 20) : slope@0 = {sl_cov:+.3f} |h|   (wrong sign)")
print(f"mono-ionic (n_d<=1, 200)  : slope@0 = {sl_nd1:+.3f} |h|   "
      f"(repaired, {sl_nd1 / sl_fci:.0%} of FCI)")
print(f"full FCI (400)            : slope@0 = {sl_fci:+.3f} |h|")
assert abs(sl_cov - 0.087) < 3e-3 and sl_cov > 0      # covalent-only: wrong sign
assert abs(sl_nd1 + 0.210) < 3e-3 and sl_nd1 < 0      # n_d<=1: sign repaired
assert abs(sl_fci + 0.277) < 3e-3 and sl_fci < 0      # FCI reference
""")

# =====================================================================
md(r"""
## 6. The wrong sign is the strong-correlation limit

The covalent five-structure model is the $U \to \infty$ projection of the
ring: the ionic determinants are suppressed by the cost of double occupancy,
so the FCI and the covalent-only model coincide as $U \to \infty$. At finite
$U$ the ionic structures relax at each $\lambda$ and the FCI stays monotone
until the correlation is strong enough for the maximum to appear. Because the
maximum grows out of the $\lambda = 0$ edge, locating the onset needs no
$\lambda$ scan at all: one eigensolve per $U$ gives the broken-bond slope,
and the onset $U^\ast$ is where that slope changes sign.
""")

code(r"""
def slope_fci_at0(Uv):                      # dE/dlambda at lambda = 0, any U
    c = eigh(H1_base + Uv * H2_perUs, S_s, subset_by_index=[0, 0])[1][:, 0]
    return -(c @ H1_dHab @ c) / (c @ S_s @ c)

for Uv in [0.0, 4.0, 8.0, 8.5, 16.0]:
    sl = slope_fci_at0(Uv)
    tag = "monotone" if sl < 0 else "wrong-sign maximum"
    print(f"U/|h| = {Uv:>4}:  slope@0 = {sl:+.4f} |h|   ({tag})")

# bisect the sign change (one 400-dim eigensolve per step)
lo, hi = 8.0, 8.6
for _ in range(12):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if slope_fci_at0(mid) < 0 else (lo, mid)
Ustar = (lo + hi) / 2
print(f"\nFCI wrong-sign onset:  U*/|h| = {Ustar:.2f}   (manuscript ~ 8.25, argmax-grid criterion)")
assert abs(Ustar - 8.21) < 0.05

# the sign change is real: just above U*, an interior maximum exists
lam_c = np.linspace(1.0, 0.0, 41)
e16 = E_fci(16.0, grid=lam_c); im = int(np.argmax(e16))
assert 0 < im < len(e16) - 1 and e16[im] > e16[-1]
print(f"check at U/|h| = 16: interior maximum at lambda* = {lam_c[im]:.2f}")
""")

# =====================================================================
md(r"""
## 7. Wrap-up

You scaled `symvb` to the 400-determinant benzene ring and used it to check
a piece of qualitative reasoning:

1. Built the half-filled benzene FCI and decomposed it by ionicity, finding
   the closed-shell Hückel determinant is only $5/72 \approx 7\%$ covalent
   (Eq. 17), with palindromic class weights.
2. Showed the covalent share collapses with ring size ($1/2 \to 5/72 \to
   \sim 1\%$) from a sum of squared MO minors.
3. Watched correlation invert the balance, the covalent class taking over
   past $U/|h| \approx 4$ (Figure 4).
4. Found that a covalent-only Rumer model gives the **wrong sign** of the
   energy response to weakening one bond, a maximum near $\lambda \approx
   0.32$ absent from the exact FCI (Eq. 18, Figure 5), and traced it to AO
   non-orthogonality, the covalent block being exactly zero at $s = 0$.
5. Identified the bump as the strong-correlation ($U \to \infty$) limit, the
   exact FCI staying monotone up to $U/|h| \approx 8.25$.

The lesson for VB practice: covalent Rumer structures are an incomplete basis
even for benzene; the ionic structures they omit carry most of the
wavefunction and all of the qualitative trend at physical correlation.

### Take-home exercises

1. **Naphthalene.** Build the 10-center covalent share and confirm the $\sim
   1\%$ figure; predict the trend toward fullerene.

2. **Where does the FCI bump land?** Above the onset, track $\lambda^*$ and
   the bump height as $U \to \infty$ and confirm they approach the
   covalent-only values.

3. **Ring family.** Repeat the ionicity decomposition for the
   cyclobutadiene dianion (6 electrons in 4 orbitals) and the
   cyclopentadienyl anion (6 in 5); above half filling the low-ionicity
   classes are emptied by counting alone.

### Series recap

Across the four notebooks `symvb` derived, not quoted: the H₂ covalent/ionic
balance and charge-shift split (Notebook 1), the allyl long-bond biradical
signature (Notebook 2), the disphenoid Robin–Day crossover (Notebook 3), and
benzene's covalent-only failure (here). Each is a closed-form or exact
statement about VB structure weights and energies that floating-point
single-point calculations cannot establish. The `additional/` notebooks
extend the toolkit: the $U = J$ operator identity, the Hubbard-to-Heisenberg
mapping, and symmetry projection as a tool.
""")


NB.cells = cells
out_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '04_benzene_covalent_only.ipynb'))
nbf.write(NB, out_path)
print(f"Wrote {out_path}  ({len(cells)} cells)")
