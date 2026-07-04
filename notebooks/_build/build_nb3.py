"""Build notebooks/03_h2h2_plus_disphenoid.ipynb from cell definitions below.

Main-set notebook 3: the (H2)2+ disphenoid, a four-center three-electron
mixed-valence cluster, reproducing the manuscript "worked symvb application"
section (Eqs 12-16, Figure 3).

Run from anywhere:  python3 notebooks/_build/build_nb3.py
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
# Notebook 3 — The (H₂)₂⁺ disphenoid: a Robin–Day Class II/III crossover

**Goal.** A worked `symvb` application on a four-center, three-electron
**mixed-valence** cluster: two $\mathrm{H}_2$ units sharing a single positive
charge, with the four atoms at the corners of a *disphenoid* (a tetrahedron
with two short opposite edges and four long ones). The shared hole either
spreads over both units (delocalized, **Class III** in the Robin–Day
classification) or traps on one under an asymmetric distortion (**Class II**).
We will

1. build the full $S_z = \tfrac12$ configuration interaction with `symvb`,
2. reduce it to a four-structure covalent/ionic VB model and read off the
   $4\times 4$ Hamiltonian (manuscript Eq. 12),
3. block-diagonalize at the symmetric point to get the ground-state energy
   $E_0(U)$ in closed form (Eq. 13),
4. take the one-electron limit, couple it to Marcus–Hush diabatic parabolas,
   and *derive* the critical coupling $|h_l|_{\rm crit} = \lambda/4$ (Eqs 14–15),
5. turn on correlation and show the crossover moves to smaller coupling, so
   the shared hole survives to longer inter-pair distances (Eq. 16,
   Figure 3B), validating the four-structure model against the full CI.

**Prerequisites.** Notebook 1 (the H₂ covalent/ionic 2×2 is the building
block here). Each unit is exactly that bond.

**Runtime.** A few seconds.
""")

# =====================================================================
md(r"""
## 1. Setup and the disphenoid model

Four hydrogen 1s orbitals $\{a, b, c, d\}$. The two **short intra-pair
edges** $a$–$b$ and $c$–$d$ carry the resonance integral $h_s$; the four
**long inter-pair edges** $a$–$c$, $a$–$d$, $b$–$c$, $b$–$d$ carry the much
weaker $h_l$ ($|h_l| \ll |h_s|$), the coupling that weakens toward zero as
the units separate. Every atom carries the on-site Hubbard repulsion $U$.

The cluster has three electrons (one neutral $\mathrm{H}_2$, one
$\mathrm{H}_2^{\bullet +}$). In the $S_z = \tfrac12$ sector the space is
$\binom{4}{2}\binom{4}{1} = 24$ determinants, from `generate_dets(2, 1, 4)`.
We keep the two short-bond integrals **independent** ($h_{s,1}, h_{s,2}$) so
we can later distort one bond against the other.
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

from symvb import Molecule, System
from symvb.fixed_psi import FixedPsi, generate_dets

hs1, hs2, hl, s, U = sp.symbols('h_s_1 h_s_2 h_l s U')

m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab', 'cd', 'ac', 'ad', 'bc', 'bd'],
    subst={'h_s_1': ('H_ab',), 'h_s_2': ('H_cd',),
           'h_l': ('H_ac', 'H_ad', 'H_bc', 'H_bd'),
           's': ('S_ab', 'S_cd', 'S_ac', 'S_ad', 'S_bc', 'S_bd')},
    subst_2e={'U': ('1111',)}, max_2e_centers=1,
)
P = generate_dets(2, 1, 4)
det_strings = [p.dets[0].det_string for p in P]
print(f"{len(det_strings)} determinants in the Sz = 1/2 sector")
""")

code(r"""
# full CI Hamiltonian at orthogonal AOs (s = 0); symbolic in h_s_1, h_s_2, h_l, U
Hf = (sp.Matrix(m.build_matrix(P, op='H')) + sp.Matrix(m.o2_matrix(P))).subs(s, 0)
Hf.shape
""")

# =====================================================================
md(r"""
## 2. Four VB structures and the 4×4 Hamiltonian

In each charge-localized diabatic state, one unit is a neutral $\mathrm{H}_2$
and the other a radical cation $\mathrm{H}_2^{\bullet +}$ holding its single
electron in the bonding combination $\sigma_k = (\,\cdot + \cdot\,)/\sqrt2$.
Following the H₂ bond of Notebook 1, the neutral two-electron bond is a
resonance of a Heitler–London **covalent** structure $[a\!\cdot\!b]_s$ and the
symmetric **ionic** structure $(a^2 + b^2)/\sqrt2$. The dominant VB space is
therefore the four structures of manuscript Eq. 12: $\mathrm{cov}_1$,
$\mathrm{ion}_1$ (neutral unit 1 covalent or ionic, hole on unit 2) and
$\mathrm{cov}_2$, $\mathrm{ion}_2$ (the mirror).

We assemble the Hamiltonian over these four structures with the `System`
facade, which builds and projects the non-orthogonal VB matrices for us. The
determinants are written in their natural creation order with $\pm\tfrac12$
coefficients that normalize each structure at $s = 0$; `symvb` reorders them
into its canonical convention internally, so the four structures read straight
off the chemistry.
""")

code(r"""
# The four covalent/ionic VB structures of manuscript Eq. 12, over ATOMIC
# orbitals. Each is a 3-electron Sz = +1/2 doublet: a neutral two-electron bond
# on one unit plus the hole electron in the other unit's bonding MO sigma. The
# +/- 1/2 coefficients normalize each structure at s = 0.
half = sp.Rational(1, 2)

def vb(*terms):
    "A VB structure as a FixedPsi from (determinant, coefficient) terms."
    p = FixedPsi()
    for ds, cf in terms:
        p.add_str_det(ds, coef=cf)
    return p

cov1 = vb(("aBc", half), ("aBd", half), ("Abc", -half), ("Abd", -half))  # HL bond a-b, hole on c/d
ion1 = vb(("aAc", half), ("aAd", half), ("bBc", half), ("bBd", half))    # ionic a^2+b^2, hole on c/d
cov2 = vb(("acD", half), ("aCd", -half), ("bcD", half), ("bCd", -half))  # HL bond c-d, hole on a/b
ion2 = vb(("acC", half), ("adD", half), ("bcC", half), ("bdD", half))    # ionic c^2+d^2, hole on a/b
structs = [cov1, ion1, cov2, ion2]

# System facade: assemble (H, S) over the four VB structures. The facade builds
# and projects the non-orthogonal VB matrices, folding the two-electron block
# into H. The natural-order determinants above are canonicalized inside the
# build, so no reordering bookkeeping is needed here.
sysCI = System.from_structures(m, structs)
Hci_raw, Sci = sysCI.hamiltonian()
Hci = sp.nsimplify(sp.Matrix(Hci_raw).subs(s, 0))
assert sp.simplify(sp.Matrix(Sci).subs(s, 0) - sp.eye(4)) == sp.zeros(4, 4), \
    "structures are orthonormal at s = 0"

print("Four-structure Hamiltonian in basis {cov1, ion1, cov2, ion2} (s = 0):")
show(r'H', Hci)
""")

md(r"""
Each diagonal $2\times2$ block is the H₂ bond of Notebook 1 for the neutral
unit: the covalent and ionic structures are coupled by the short-bond
resonance $2h_{s,i}$, and the ionic structure is raised by the on-site
repulsion $U$. The off-diagonal blocks, which move the hole between the two
units, are a uniform $-h_l$: in this minimal symmetric model every structure
on one unit couples to every structure on the other with the same inter-unit
matrix element. The next cell checks this against manuscript Eq. 12,
entry for entry.
""")

code(r"""
# manuscript Eq. 12, in the ordered basis {cov1, ion1, cov2, ion2}
Eq12 = sp.Matrix([
    [hs2,    2*hs1,  -hl,    -hl  ],
    [2*hs1,  U+hs2,  -hl,    -hl  ],
    [-hl,    -hl,    hs1,    2*hs2],
    [-hl,    -hl,    2*hs2,  U+hs1]])
assert sp.simplify(Hci - Eq12) == sp.zeros(4, 4)
print("verified: H over {cov1, ion1, cov2, ion2} equals manuscript Eq. 12 entry for entry")
""")

md(r"""
For the symmetric-point block-diagonalization in the next section it is
convenient to rotate each unit's covalent/ionic pair into its **molecular
orbitals**: the doubly-occupied bonding configuration
$\sigma^2 = (\mathrm{cov} + \mathrm{ion})/\sqrt2$ and the doubly-occupied
antibonding configuration $\sigma^{*2} = (\mathrm{ion} - \mathrm{cov})/\sqrt2$.
This is a unitary rotation of the Eq. 12 basis, so it represents the same
operator. In the MO basis the within-unit covalent/ionic resonance $2h_{s,i}$
is replaced by the $\sigma^2/\sigma^{*2}$ coupling $U/2$, which **vanishes at
$U = 0$** (so the bonding block is exact in the one-electron limit), and the
hole transfer becomes $-2h_l$ between the two bonding configurations.
""")

code(r"""
# The MO basis is a rotation of the {cov1, ion1, cov2, ion2} basis. R has each
# MO configuration as a row, expressed in the cov/ion basis:
#   sigma_i^2  = (cov_i + ion_i)/sqrt2      (bonding doubly occupied)
#   sigma_i*^2 = (ion_i - cov_i)/sqrt2      (antibonding doubly occupied)
# so the same operator in the MO basis is H4 = R Hci R^T.
r2 = 1 / sp.sqrt(2)
R = r2 * sp.Matrix([
    [ 1,  1,  0,  0],     # sigma1^2  = (cov1 + ion1)/sqrt2
    [ 0,  0,  1,  1],     # sigma2^2  = (cov2 + ion2)/sqrt2
    [-1,  1,  0,  0],     # sigma1*^2 = (ion1 - cov1)/sqrt2
    [ 0,  0, -1,  1]])    # sigma2*^2 = (ion2 - cov2)/sqrt2
assert sp.simplify(R * R.T - sp.eye(4)) == sp.zeros(4, 4), "rotation is orthogonal"

H4 = sp.simplify(R * Hci * R.T)
print("Four-structure Hamiltonian H4 in the bonding/antibonding MO basis")
print("{sigma1^2, sigma2^2, sigma1*^2, sigma2*^2} (s = 0):")
show(r'H_4', H4)
""")

code(r"""
# entries of the MO-basis Hamiltonian (fixes the multiplication-side convention)
assert sp.simplify(H4[0, 0] - (2*hs1 + hs2 + U/2)) == 0    # sigma1^2 diagonal:  2 h_s1 + hole + U/2
assert sp.simplify(H4[2, 2] - (-2*hs1 + hs2 + U/2)) == 0   # sigma1*^2 diagonal: -2 h_s1 + hole + U/2
assert sp.simplify(H4[0, 1] - (-2*hl)) == 0                # sigma1^2 - sigma2^2 hole transfer
assert sp.simplify(H4[0, 2] - U/2) == 0                    # sigma1^2 - sigma1*^2 coupling = U/2 (0 at U=0)
assert H4[0, 3] == 0 and H4[2, 3] == 0
print("verified: sigma^2/sigma*^2 diagonals +/-2 h_s,i + U/2;  hole transfer -2 h_l;")
print("          sigma^2-sigma*^2 coupling U/2 (zero at U = 0);  cross terms zero.")
""")

# =====================================================================
md(r"""
## 3. The symmetric point: ground-state energy in closed form

At the symmetric geometry the two short bonds are equal,
$h_{s,1} = h_{s,2} = h_s$, and the units are equivalent. The pair-swap
symmetry (unit 1 $\leftrightarrow$ unit 2) block-diagonalizes the $4\times 4$
into two $2\times 2$ blocks, an *even* and an *odd* combination of the
structures. The ground state is the lower root of the **odd** block.
""")

code(r"""
T = sp.sqrt(sp.Rational(1, 2)) * sp.Matrix([[1, 0, 1, 0], [1, 0, -1, 0],
                                            [0, 1, 0, 1], [0, 1, 0, -1]])
Hb = sp.simplify(T.T * H4.subs({hs2: hs1}) * T)     # {even_C, even_I | odd_C, odd_I}
assert Hb[0, 2] == 0 and Hb[1, 3] == 0 and Hb[0, 3] == 0 and Hb[1, 2] == 0
print("pair-swap block-diagonalizes the symmetric-point H4 into 2x2 (even) + 2x2 (odd):")
show(r'H_b', Hb)
""")

code(r"""
odd = Hb[2:, 2:]
p, q, c = odd[0, 0], odd[1, 1], odd[0, 1]
E0U = sp.simplify((p + q) / 2 - sp.sqrt((p - q)**2 / 4 + c**2))

# manuscript Eq. 13:  E0 = h_s + h_l + (U - R)/2,  R = sqrt((4 h_s + 2 h_l)^2 + U^2)
R = sp.sqrt((4*hs1 + 2*hl)**2 + U**2)
E0_eq13 = hs1 + hl + (U - R) / 2
assert sp.simplify(E0U - E0_eq13) == 0
print("symmetric-point ground state (Eq. 13):")
show(r'E_0(U)', E0_eq13)
""")

# =====================================================================
md(r"""
## 4. The one-electron limit and the Marcus–Hush picture

At $U = 0$ there is no penalty for double occupancy, so the covalent and
ionic structures of each bond merge into the bonding MO. The four structures
collapse to **two intact-bond states**, the hole on unit 1 or unit 2,
between which it hops with the effective integral $t = 2h_l$.

In the Marcus–Hush picture the two charge-localized diabatic states are
parabolas $(\lambda/4)(1 \pm X)^2$ in a dimensionless charge-transfer
coordinate $X = \eta/\lambda$, where $\eta = h_{s,1} - h_{s,2}$ is the bond
detuning produced by stretching one short bond and compressing the other. We
take the parabolas (the framework reorganization $\lambda$) as given, and use
the VB model for the electronic coupling. Coupling the two diabats gives the
lower adiabatic surface (manuscript Eq. 14)

$$
E_-(\eta) = 3h_s + \frac{\eta^2}{4\lambda} - \sqrt{\frac{\eta^2}{4} + 4h_l^2}.
$$
""")

code(r"""
eta, lam, hlm = sp.symbols('eta lambda h_lm', positive=True)   # hlm = |h_l|
# average parabola + avoided-crossing dip; electronic coupling t = 2 h_l, so 4 h_l^2 under the root
E_minus = 3*sp.Symbol('h_s') + eta**2 / (4*lam) - sp.sqrt(eta**2 / 4 + 4*hlm**2)
curvature = sp.simplify(sp.diff(E_minus, eta, 2).subs(eta, 0))
show(r'\left.\dfrac{\partial^2 E_-}{\partial \eta^2}\right|_0', curvature)   # Eq. 15
""")

md(r"""
The curvature at the symmetric point is $\dfrac{1}{2\lambda} - \dfrac{1}{8|h_l|}$
(manuscript Eq. 15). It changes sign at the **critical coupling**

$$
|h_l|_{\rm crit} = \frac{\lambda}{4},
$$

the Marcus–Hush balance of electronic coupling against framework
reorganization. For $|h_l| > \lambda/4$ (short inter-pair distance) the
curvature is positive, a single well with the hole **shared** (Class III);
for $|h_l| < \lambda/4$ (the units pulled apart) it is negative, a double
well with the hole **trapped** (Class II).
""")

code(r"""
hl_crit = sp.solve(sp.Eq(curvature, 0), hlm)[0]
show(r'|h_l|_{\rm crit}', hl_crit)   # = lambda/4
""")

# =====================================================================
md(r"""
## 5. Correlation protects the shared hole (Eq. 16, Figure 3B)

For $U > 0$ the covalent–ionic coupling $U/2$ admixes the ionic structures
and *weakens* the localizing electronic term. Second-order perturbation
theory on the four-structure ground state gives the symmetric-point
curvature (manuscript Eq. 16)

$$
\left.\frac{\partial^2 E_-}{\partial\eta^2}\right|_{0}
 = \frac{1}{2\lambda} - \frac{1}{8|h_l|}
   \left[\,1 - \frac{14|h_s| - 11|h_l|}{16\,(2|h_s| + |h_l|)^3}\,U^2
         + \mathcal{O}(U^4)\,\right],
$$

with the bracket dropping below one as $U$ grows. The electronic curvature
$k(U) \equiv \tfrac{1}{8|h_l|}[\,\cdots]$ therefore falls with $U$. We
compute it numerically from the four-structure model **and** from the full
24-determinant CI, and confirm the two agree to about one percent.
""")

code(r"""
syms = [hs1, hs2, hl, U]
# numeric expansion matrices for the FULL CI: H(h1,h2,hl,U) = H0 + sum x_i M_i  (Hf is linear)
H0 = np.array(Hf.subs({x: 0 for x in syms}).tolist(), float)
Ms = [np.array(sp.diff(Hf, x).tolist(), float) for x in syms]
# ... and for the four-structure model
H4_0 = np.array(H4.subs({x: 0 for x in syms}).tolist(), float)
H4_M = [np.array(sp.diff(H4, x).tolist(), float) for x in syms]

def Hfull(h1, h2, hlv, Uv):
    A = H0 + h1*Ms[0] + h2*Ms[1] + hlv*Ms[2] + Uv*Ms[3]; return (A + A.T) / 2
def H4num(h1, h2, hlv, Uv):
    A = H4_0 + h1*H4_M[0] + h2*H4_M[1] + hlv*H4_M[2] + Uv*H4_M[3]; return (A + A.T) / 2

def curv(f, d=1e-4): return (f(d) + f(-d) - 2*f(0.0)) / d**2
def gs_curv(Hbuild, hlv, Uv):
    return -curv(lambda e: np.linalg.eigvalsh(Hbuild(-1 + e/2, -1 - e/2, hlv, Uv))[0])

hl_val = -0.3
print(f"electronic curvature k(U) at h_s = -1, h_l = {hl_val}   (k(0) = 1/(8|h_l|) = {1/(8*abs(hl_val)):.4f})")
print(f"{'U/|h_s|':>8} {'4-structure':>12} {'full CI':>10} {'rel. error':>11}")
for Uv in [0.0, 1.0, 2.0, 4.0, 8.0]:
    k4 = gs_curv(H4num, hl_val, Uv)
    kF = gs_curv(Hfull, hl_val, Uv)
    print(f"{Uv:>8.1f} {k4:>12.5f} {kF:>10.5f} {abs(k4-kF)/kF*100:>10.2f}%")
    assert abs(k4 - kF) / kF < 0.012
print("four-structure k(U) within 1.2% of the full CI; both fall with U.")
""")

md(r"""
That curvature has an **exact closed form**. Second-order perturbation theory
on the four-structure model gives the electronic curvature $k(U)$ as a single
rational function of the integrals; the manuscript's Eq. 16 is its small-$U$
expansion. We write the (rationalized, manifestly finite at $U = 0$) form,
check it against the finite-difference values above, and recover Eq. 16.
""")

code(r"""
Rc = sp.sqrt((4*hs1 + 2*hl)**2 + U**2)
R0 = sp.sqrt((4*hs1 + 2*hl)**2)              # = |4 h_s + 2 h_l|
k_exact = (36*(hl**2 - 4*hs1**2)/(Rc + R0) + 4*(2*hl - 5*hs1) - Rc) / (4*hl*Rc)
k_fn = sp.lambdify((hs1, hl, U), k_exact, 'numpy')

# (i) matches the finite-difference four-structure curvature of the previous cell
for Uv in [1.0, 4.0, 8.0]:
    assert abs(float(k_fn(-1, -0.3, Uv)) - gs_curv(H4num, -0.3, Uv)) < 1e-3
# (ii) its small-U expansion is exactly manuscript Eq. 16
ahs, ahl = sp.symbols('a_hs a_hl', positive=True)      # |h_s|, |h_l|
ser = sp.series(k_exact.subs({hs1: -ahs, hl: -ahl}), U, 0, 4)
assert sp.simplify(ser.coeff(U, 0) - 1/(8*ahl)) == 0
assert sp.simplify(ser.coeff(U, 2) + (14*ahs - 11*ahl)/(128*ahl*(2*ahs + ahl)**3)) == 0
print("k(U) closed form verified vs finite difference and reduced to Eq. 16:")
show(r'k(U)', ser)   # small-U expansion in |h_s|, |h_l|; O(U^4) tail retained
""")

md(r"""
The Robin–Day phase boundary follows. For a fixed framework $\lambda$, the
total curvature $\tfrac{1}{2\lambda} - k(U)$ vanishes at the critical coupling
$|h_l|_{\rm crit}$ where $k(U; |h_l|) = \tfrac{1}{2\lambda}$. At $U = 0$,
$k = 1/(8|h_l|)$ gives $|h_l|_{\rm crit} = \lambda/4$ exactly; as $U$ grows
$k$ shrinks at fixed coupling, so a smaller $|h_l|$ reaches the threshold and
$|h_l|_{\rm crit}$ **falls toward zero**: the shared hole survives to longer
inter-pair distances. We reproduce manuscript Figure 3B ($\lambda = 1$).
""")

code(r"""
import matplotlib.pyplot as plt
from scipy.optimize import brentq

lam_val = 1.0
target = 1.0 / (2 * lam_val)                 # curvature balance: k(U; |h_l|_crit) = 1/(2 lambda)
def hl_crit_of_U(Uv):
    g = lambda x: float(k_fn(-1, -x, Uv)) - target     # h_s = -1
    return brentq(g, 1e-5, 0.25) if g(1e-5) > 0 else 0.0

U_grid = np.linspace(0, 8, 33)
hl_crit = np.array([hl_crit_of_U(Uv) for Uv in U_grid])
assert abs(hl_crit[0] - lam_val/4) < 1e-3
print(f"|h_l|_crit:  U = 0 -> {hl_crit[0]:.4f} (= lambda/4),   U = 8 -> {hl_crit[-1]:.4f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(U_grid, hl_crit, 'C0-', lw=2.2)
ax.axhline(lam_val/4, color='gray', lw=0.8, ls=':')
ax.text(2.0, lam_val/4 + 0.006, r'$\lambda/4$ (one-electron limit)', color='gray', fontsize=9)
ax.fill_between(U_grid, hl_crit, 0.30, alpha=0.10, color='C0')
ax.fill_between(U_grid, hl_crit, 0.0,  alpha=0.10, color='C3')
ax.text(6.2, 0.20, 'shared hole\n(Class III)', color='C0', fontsize=9, ha='center')
ax.text(2.0, 0.04, 'trapped hole (Class II)', color='C3', fontsize=9, ha='center')
ax.set_xlabel(r'$U / |h_s|$'); ax.set_ylabel(r'$|h_l|_{\rm crit}\,/\,|h_s|$')
ax.set_title(r'Robin–Day phase boundary ($\lambda = 1$)')
ax.set_ylim(0, 0.30); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

# =====================================================================
md(r"""
## 6. Wrap-up

You used `symvb` as a *worked tool* on a mixed-valence cluster:

1. Built the 24-determinant $(\mathrm{H}_2)_2^{\bullet +}$ configuration
   interaction symbolically.
2. Reduced it to the four covalent/ionic VB structures and recovered the
   manuscript $4\times 4$ Hamiltonian (Eq. 12) entry for entry: each diagonal
   $2\times2$ an H₂ bond (covalent and ionic coupled by $2h_{s,i}$, the ionic
   structure raised by $U$), the inter-unit hole transfer $-h_l$. Rotating to
   the bonding/antibonding MO basis turns the within-unit coupling into the
   $\sigma^2/\sigma^{*2}$ term $U/2$, which vanishes at $U = 0$.
3. Block-diagonalized at the symmetric point to get
   $E_0(U) = h_s + h_l + (U - R)/2$ in closed form (Eq. 13).
4. Took the one-electron limit, coupled it to Marcus–Hush parabolas, and
   derived the curvature $\tfrac{1}{2\lambda} - \tfrac{1}{8|h_l|}$ and the
   critical coupling $|h_l|_{\rm crit} = \lambda/4$ (Eqs 14–15).
5. Showed that correlation lowers $|h_l|_{\rm crit}$ (Eq. 16, Figure 3B), so
   charge correlation keeps the hole shared to longer distances, with the
   four-structure model tracking the full CI to about one percent.

### Take-home exercises

1. **The avoided-crossing surface.** Plot $E_-(\eta)$ of Eq. 14 for three
   couplings $|h_l| < \lambda/4$, $= \lambda/4$, $> \lambda/4$ and watch the
   single well split into a double well (manuscript Figure 3A).

2. **Where is the crossover for real?** Pick $|h_s|$ and $\lambda$ from a
   real oxidized organic mixed-valence dimer and read off the $U$ at which
   the system would switch class.

3. **The dropped structures.** Add the two fully charge-separated structures
   ($\mathrm{H}_2^- \cdots \mathrm{H}_2^{2+}$) and check they shift
   $|h_l|_{\rm crit}$ by only about a percent.

### Up next

**Notebook 4** scales up to six centers: benzene, where a covalent-only
$\pi$ model inverts the *sign* of the energy response to weakening one bond,
and the ionic structures the model omits carry most of the wavefunction.
""")


NB.cells = cells
out_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '03_h2h2_plus_disphenoid.ipynb'))
nbf.write(NB, out_path)
print(f"Wrote {out_path}  ({len(cells)} cells)")
