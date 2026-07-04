"""Build notebooks/additional/allyl_uj_identity.ipynb."""
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
# Additional notebook — Allyl and the (U = J) operator identity

**The puzzle.** A standard objection to the Hückel method is that it
ignores electron–electron repulsion. Adding a Hubbard $U$ then
introduces correlation and the simple Hückel orbitals stop being
eigenstates. So far, so familiar.

But on a **single line in parameter space** — $U = J$, $K = M = 0$,
orthogonal AOs — *the Hückel ground state remains an exact eigenstate
of the full FCI Hamiltonian, regardless of the value of $U$.* The
correlation potential along this line acts as a constant scalar shift
on every state in the $N$-electron sector.

In this notebook you will discover that identity by inspecting the
$H_{2e}$ matrix, prove it operator-algebraically, verify it numerically
across several spin sectors, and then *break it* by turning on AO
overlap. The break is the lesson: identities that look basis-
independent on Fock space need not survive when the basis is non-
orthogonal.

**System.** The allyl anion: three carbon $p_{\pi}$ orbitals
$\{a, b, c\}$ in a chain, four $\pi$ electrons, nearest-neighbour
hopping. 9-dimensional $S_{z}=0$ FCI.

**Prerequisites.** The main H₂ notebook `01_h2_2c2e` (basis convention,
`Molecule` setup, and Löwdin cofactor matrix elements).
""")

# =====================================================================
md(r"""
## 1. The allyl anion in `symvb`

Three orbitals in a row, two bonded pairs $\{ab, bc\}$. With $N_{\alpha}
= N_{\beta} = 2$, the $S_{z} = 0$ basis has $\binom{3}{2}^{2} = 9$
determinants.
""")

code(r"""
import sympy as sp
from sympy import init_printing
init_printing()

import numpy as np
from symvb import Molecule
from symvb.fixed_psi import generate_dets
""")

code(r"""
P = generate_dets(2, 2, 3)
det_strings = [p.dets[0].det_string for p in P]
print(f"{len(P)} determinants in the Sz=0 sector:")
det_strings
""")

# =====================================================================
md(r"""
## 2. The four classes of two-electron integrals

For two-centre cutoff (`max_2e_centers=2`), the chemist-notation
two-electron integrals $(\mu \nu | \rho \sigma) =
\int\!\int \mu(1)\nu(1)\,\frac{1}{r_{12}}\,\rho(2)\sigma(2)$ split
into four symmetry classes:

| symbol | integrals | physical meaning |
|---|---|---|
| $U$ | $(11\|11)$ | on-site Coulomb |
| $J$ | $(12\|12)$ | direct Coulomb between two atoms |
| $K$ | $(11\|22)$ | inter-atomic exchange |
| $M$ | $(11\|12)$, $(11\|21)$, $(12\|22)$ | hybrid "ionic" integrals (charge-transfer flavoured) |

(Conventions vary across the literature. We follow the manuscript:
$U > J > 0$ for chemically reasonable values, $K \sim 0.1\,J$, $M$
small.)

`subst_2e` registers the four symbols, and `max_2e_centers=2` keeps
only one- and two-centre integrals in the build.
""")

code(r"""
m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab', 'bc'],
    subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
    subst_2e={'U': ('1111',), 'J': ('1212',),
              'K': ('1122',), 'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)

H1 = sp.Matrix(m.build_matrix(P, op='H'))
S  = sp.Matrix(m.build_matrix(P, op='S'))
H2 = sp.Matrix(m.o2_matrix(P))
print("Built H1, S, H2 — each 9×9 symbolic in (h, s, U, J, K, M).")
""")

# =====================================================================
md(r"""
## 3. Look hard at $H_{2e}$ along $K = M = 0$, $s = 0$

Switch off the smaller integrals ($K, M$) and the AO overlap. The
resulting $H_{2e}$ depends only on $U$ and $J$. *Before reading on*,
look at the matrix below and try to spot a pattern between the $U$
contribution and the $J$ contribution.
""")

code(r"""
h, s, U, J, K, M = sp.symbols('h s U J K M')
H2_simple = sp.simplify(H2.subs({s: 0, K: 0, M: 0}))
H2_simple
""")

md(r"""
**Observation.** The matrix is diagonal. Each diagonal entry decomposes
the $\binom{N}{2} = 6$ electron pairs into

- $n_{D}$ pairs that sit on the *same* orbital (one $\alpha$, one
  $\beta$ on the same atom), contributing $U$ each;
- $\binom{N}{2} - n_{D}$ pairs that sit on *different* orbitals,
  contributing $J$ each.

So the diagonal entry of det $D$ is $n_{D}\, U + (6 - n_{D})\, J$, where
$n_{D}$ counts the number of doubly-occupied orbitals. With four
electrons in three orbitals, the pigeonhole principle forces $n_{D}
\ge 1$, and inspection of the basis shows only $n_{D} \in \{1, 2\}$
configurations exist.
""")

code(r"""
def n_doubly_occ(ds):
    return sum(1 for c in 'abc' if (c in ds) and (c.upper() in ds))

print(f"  {'det':>6}  {'n_D':>4}  {'H2 diag':>14}  {'n_D·U + (6-n_D)·J':>20}")
for i, ds in enumerate(det_strings):
    nD = n_doubly_occ(ds)
    expect = nD * U + (6 - nD) * J
    print(f"  {ds:>4}   {nD:>4}  {str(sp.simplify(H2_simple[i,i])):>14}  "
          f"{str(expect):>20}")
""")

# =====================================================================
md(r"""
## 4. The $U = J$ line — a constant on the diagonal

If we further set $U = J$, every diagonal entry collapses to the same
number: $\binom{N}{2} U = 6 U$.
""")

code(r"""
H2_at_UJ = sp.simplify(H2_simple.subs(J, U))
H2_at_UJ
""")

md(r"""
That is

$$
H_{2e}\bigl|_{U=J,\,K=M=0,\,s=0} \;=\; \binom{N}{2}\, U \cdot \mathbb{I}.
$$

A scalar multiple of the identity *commutes with anything*. So the
**full** Hamiltonian along this line,

$$
H = H_{1e}^{\text{Hückel}} \;+\; \binom{N}{2}\, U\,\mathbb{I},
$$

shares its eigenvectors exactly with the Hückel one-electron problem:
turning on $U = J$ Coulomb repulsion shifts every energy by
$\binom{N}{2} U$ but does not mix the configurations at all.
""")

# =====================================================================
md(r"""
## 5. Numerical verification: linearity of $E_{\rm gs}$ in $U$

If the operator identity holds, the FCI ground-state energy along
$U = J$ should be *exactly* $E_{\rm Hückel,gs} + 6 U$. We diagonalise
the full $9 \times 9$ Hamiltonian at several $U$ values and check
linearity.
""")

code(r"""
H_full = H1 + H2
H_at_UJ_line = H_full.subs({s: 0, h: -1, K: 0, M: 0})
H_fn = sp.lambdify((U, J), H_at_UJ_line, 'numpy')

print(f"  {'U':>5}  {'E_gs':>11}  {'E_1st':>11}  {'gap':>8}")
Us = np.array([0.0, 0.5, 1, 2, 4, 8, 16])
E_gs_arr = []
for Uv in Us:
    Hn = np.array(H_fn(Uv, Uv), dtype=float)
    Hn = 0.5 * (Hn + Hn.T)
    ev = np.linalg.eigvalsh(Hn)
    E_gs_arr.append(ev[0])
    print(f"  {Uv:>5.1f}  {ev[0]:>11.6f}  {ev[1]:>11.6f}  {ev[1]-ev[0]:>8.4f}")

E_gs_arr = np.array(E_gs_arr)
slope, intercept = np.polyfit(Us, E_gs_arr, 1)
print(f"\nLinear fit:  E_gs(U) = {slope:.6f} * U + {intercept:.6f}")
print(f"Expected:    slope = 6 = C(4,2),  intercept = -2*sqrt(2) = {-2*np.sqrt(2):.6f}")
print(f"Max residual: {np.max(np.abs(E_gs_arr - slope*Us - intercept)):.2e}")
""")

md(r"""
The residuals are at machine precision: the FCI ground-state energy is
*exactly* $-2\sqrt{2} + 6 U$ along the entire $U = J$ line. Slope $6$
is $\binom{4}{2}$ for $N = 4$ electrons; intercept $-2\sqrt{2}$ is the
Hückel ground-state energy of the allyl anion at $h = -1$.
""")

# =====================================================================
md(r"""
## 6. The eigenvector check

Linearity proves that *some* eigenvector at every $U$ has energy
$-2\sqrt{2} + 6U$, but does not pin that eigenvector down. We verify it
is the *same* one — the Hückel ground state $|\psi_{1}^{2}\,
\psi_{2}^{2}\rangle$ where $\psi_{1}, \psi_{2}$ are the two lowest
Hückel MOs.

The Hückel MOs of allyl at $h = -1$ are
$\psi_{1} = \tfrac{1}{2}(a + \sqrt{2}\,b + c)$ (bonding),
$\psi_{2} = \tfrac{1}{\sqrt{2}}(a - c)$ (non-bonding),
$\psi_{3} = \tfrac{1}{2}(a - \sqrt{2}\,b + c)$ (antibonding).
The closed-shell ground state has $\psi_{1}$ and $\psi_{2}$ doubly
occupied. Below we expand $|\psi_{1}^{2}\,\psi_{2}^{2}\rangle$ in the
9-determinant `symvb` AO basis and overlap it with the numerical FCI
ground state.
""")

code(r"""
from symvb.mo_projection import mo_determinant_in_ao

# 1-electron Hückel matrix at h=-1, s=0
h1_1e = np.array([[0, -1, 0], [-1, 0, -1], [0, -1, 0]], dtype=float)
mo_E, C_mo = np.linalg.eigh(h1_1e)
order = np.argsort(mo_E)
C_mo = C_mo[:, order]    # columns: psi_1, psi_2, psi_3
mo_E_sorted = mo_E[order]
print(f"Hückel MO energies: {mo_E_sorted}")

# Expand |psi_1 psi_1 psi_2 psi_2> in the 9-det symvb basis.
# mo_determinant_in_ao does the cofactor expansion and the fermionic
# sign bookkeeping (spin-orbital interleaving + symvb string order).
# Rows of mo_coeffs are MOs, so pass C_mo.T; occupation lists the
# occupied MO indices for alpha and beta.
psi_huckel = mo_determinant_in_ao(C_mo.T, ([0, 1], [0, 1]), det_strings)

print(f"|psi_Huckel| = {np.linalg.norm(psi_huckel):.6f}  (should be 1)")
""")

code(r"""
# Overlap with FCI ground state along U = J
print(f"  {'U':>5}  {'<psi_H | gs>':>14}  {'overlap^2':>12}")
for Uv in [0, 0.5, 1, 2, 4, 8, 16]:
    Hn = np.array(H_fn(Uv, Uv), dtype=float); Hn = 0.5*(Hn+Hn.T)
    _, vec = np.linalg.eigh(Hn)
    ovlp = float(psi_huckel @ vec[:, 0])
    print(f"  {Uv:>5.1f}  {ovlp:>14.10f}  {ovlp**2:>12.10f}")
""")

md(r"""
Overlap is $\pm 1$ at every $U$ (the sign is just an arbitrary phase
convention of `np.linalg.eigh`). The Hückel ground state is *literally*
the FCI ground state along the entire $U = J$ line.
""")

# =====================================================================
md(r"""
## 7. Why? An operator-algebra view

In second-quantised form, the two-electron piece at $U = J$, $K = M =
0$ on the orthogonal-AO basis can be written

$$
\hat{V}_{U} + \hat{V}_{J}\bigl|_{U=J,K=M=0}
= U \sum_{\mu} \hat{n}_{\mu\uparrow}\hat{n}_{\mu\downarrow}
+ U \sum_{\mu < \nu} \hat{n}_{\mu}\hat{n}_{\nu}
= U\, \binom{\hat{N}}{2},
$$

since $\sum_{\mu} \hat{n}_{\mu\uparrow}\hat{n}_{\mu\downarrow} +
\sum_{\mu<\nu} \hat{n}_{\mu}\hat{n}_{\nu} = \tfrac{1}{2}(\hat{N}^{2} -
\hat{N}) = \binom{\hat{N}}{2}$ in any *fixed-$N$* sector. So on a
fixed-$N$ subspace the operator is literally a scalar.

This is reference- and spin-sector-independent: the same identity must
hold for the cation, anion, neutral, triplet, ... — anywhere the
electron count is fixed.
""")

# =====================================================================
md(r"""
## 8. Cross-check: the identity holds in other sectors

Test it on the allyl cation ($N = 3$, $C(3,2) = 3$) and the allyl
triplet $M_{S}=+1$ ($N = 4$, $C(4,2) = 6$). Both should give exact
linearity in $U$ along the $U = J$, $K = M = 0$, $s = 0$ line, with
slopes $3$ and $6$ respectively.
""")

code(r"""
def fci_slope_test(Na, Nb, label):
    P_ = generate_dets(Na, Nb, 3)
    H1_ = sp.Matrix(m.build_matrix(P_, op='H'))
    H2_ = sp.Matrix(m.o2_matrix(P_))
    H_  = (H1_ + H2_).subs({s: 0, h: -1, K: 0, M: 0})
    H_fn_ = sp.lambdify((U, J), H_, 'numpy')
    Us_ = np.array([0.0, 1, 4, 16])
    Es = []
    for Uv in Us_:
        Hn = np.array(H_fn_(Uv, Uv), dtype=float); Hn = 0.5*(Hn+Hn.T)
        Es.append(np.linalg.eigvalsh(Hn)[0])
    Es = np.array(Es)
    slope, intercept = np.polyfit(Us_, Es, 1)
    N = Na + Nb
    expected = N*(N-1)//2
    resid = float(np.max(np.abs(Es - slope*Us_ - intercept)))
    print(f"  {label:25s}  N={N}, dim={len(P_)}, "
          f"slope={slope:.4f} (expected {expected}), max_resid={resid:.1e}")

print("FCI ground-state slope along U = J  (K = M = 0, s = 0):")
fci_slope_test(2, 1, "allyl cation (3 e-)")
fci_slope_test(2, 2, "allyl anion (4 e-)")
fci_slope_test(3, 1, "allyl triplet Ms=+1")
fci_slope_test(3, 2, "allyl tetraplet Ms=+1/2 (5e-)")
""")

md(r"""
Perfect linearity, slopes match $\binom{N}{2}$. The identity is exact
in *every* spin sector.
""")

# =====================================================================
md(r"""
## 9. The break: turn on AO overlap

The operator-algebra derivation in §7 used the orthogonal-AO Fock space
$\{|\mu \sigma\rangle\}$ where the standard anticommutator
$\{c_{\mu\sigma}, c^{\dagger}_{\nu\tau}\} = \delta_{\mu\nu}
\delta_{\sigma\tau}$ holds. With non-orthogonal AOs the algebra picks
up overlap factors. The naive guess for the matrix-element analogue
would be

$$
H_{2e}\bigl|_{U=J,\,K=M=0} \;\stackrel{?}{=}\; \binom{N}{2}\, S
\quad (\text{at any } s).
$$

We can test this directly. Compute the residual matrix
$R = H_{2e}|_{U=J=1,K=M=0} - \binom{N}{2}\, S$ symbolically in $s$ and
look at its entries.
""")

code(r"""
def residual_test(Na, Nb, label, show=2):
    P_ = generate_dets(Na, Nb, 3)
    H2_ = sp.Matrix(m.o2_matrix(P_))
    S_  = sp.Matrix(m.build_matrix(P_, op='S'))
    H2_at = H2_.subs({U: 1, J: 1, K: 0, M: 0})
    N = Na + Nb
    CN2 = N*(N-1)//2
    R = sp.simplify(H2_at - CN2 * S_)
    Ndet = len(P_)
    nz = [(i, j) for i in range(Ndet) for j in range(Ndet) if R[i, j] != 0]
    print(f"\n{label}  (N={N}, C(N,2)={CN2}, dim={Ndet})")
    print(f"  Residual H_2e(U=J=1, K=M=0) - C(N,2)*S has {len(nz)} nonzero entries")
    if nz:
        print(f"  First {min(show, len(nz))} residuals (polynomial in s):")
        for (i, j) in nz[:show]:
            print(f"    R[{i},{j}] = {sp.simplify(R[i, j])}")
""")

code(r"""
# H2 dimer for comparison (rebuild Molecule with single bond)
m_h2 = Molecule(
    zero_ii=True,
    interacting_orbs=['ab'],
    subst={'h': ('H_ab',), 's': ('S_ab',)},
    subst_2e={'U': ('1111',), 'J': ('1212',),
              'K': ('1122',), 'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)
P_h2 = generate_dets(1, 1, 2)
H2_h2 = sp.Matrix(m_h2.o2_matrix(P_h2))
S_h2  = sp.Matrix(m_h2.build_matrix(P_h2, op='S'))
R_h2  = sp.simplify(H2_h2.subs({U: 1, J: 1, K: 0, M: 0}) - 1 * S_h2)
nz_h2 = [(i, j) for i in range(4) for j in range(4) if R_h2[i, j] != 0]
print(f"H2 dimer (N=2, C(2,2)=1, dim=4)")
print(f"  Residual H_2e - 1*S has {len(nz_h2)} nonzero entries")
for (i, j) in nz_h2[:2]:
    print(f"    R[{i},{j}] = {sp.simplify(R_h2[i, j])}")

# Allyl
residual_test(2, 2, "Allyl anion")
residual_test(3, 1, "Allyl triplet Ms=+1")
""")

md(r"""
The residuals are *not* zero — they are explicit polynomials in $s$,
starting linearly. The naive overlap-dressed identity is wrong.
""")

# =====================================================================
md(r"""
## 10. What went wrong with the operator argument?

The Fock-space derivation in §7 implicitly assumed $\{c_{\mu\sigma},
c^{\dagger}_{\nu\tau}\} = \delta_{\mu\nu}\delta_{\sigma\tau}$. With
non-orthogonal AOs that anticommutator picks up the overlap matrix,

$$
\{a_{\mu\sigma}, a^{\dagger}_{\nu\tau}\} = S_{\mu\nu}\, \delta_{\sigma\tau},
$$

and number operators $\hat n_{\mu}$ for non-orthogonal modes don't even
satisfy $\hat n_{\mu}^{2} = \hat n_{\mu}$. The combinatorial bookkeeping
that produced $\binom{\hat N}{2}$ in §7 simply fails.

Equivalently, on the determinant matrix-element side: Löwdin's
cofactor formula expresses $\langle \Phi | \hat V | \Phi'\rangle$ as a
sum over orbital substitutions, each weighted by a *minor* of the
spin-orbital overlap matrix between $\Phi$ and $\Phi'$. At $s = 0$
those minors collapse to delta functions and the only surviving terms
are diagonal ones that combine to give $\binom{N}{2} U$. At $s \ne 0$
the sub-leading minors contribute extra $s^{n}$ pieces that are *not*
captured by the single matrix $S$ — they live in higher-rank tensor
structures specific to which orbitals overlap with which.

> **Lesson.** "Operator identity in fixed-$N$ sector" → "matrix
> identity proportional to $S$" is a tempting shortcut, but it relies
> on the orthogonal Fock-space algebra. With non-orthogonal AOs, the
> identity dissolves into explicit s-polynomial corrections. `symvb`
> exposes those corrections directly because it never converts
> determinants to an orthogonal MO basis under the hood.
""")

# =====================================================================
md(r"""
## 11. Wrap-up

You have:

1. Built the allyl anion's 9-dimensional FCI Hamiltonian symbolically
   in $(h, s, U, J, K, M)$.
2. Discovered that on the line $U = J$, $K = M = 0$, $s = 0$ the
   two-electron part is a *scalar multiple of the identity*.
3. Verified numerically that the FCI ground state stays Hückel and the
   energy is exactly linear in $U$ with slope $\binom{N}{2}$.
4. Generalised across spin sectors (cation, anion, triplet, …) — the
   identity is reference-independent.
5. Tested the natural extension to $s \ne 0$ ($H_{2e} \stackrel{?}{=}
   \binom{N}{2} S$) and watched it *fail* with explicit residual
   polynomials in $s$.
6. Traced the failure to non-orthogonal Fock-space algebra.

### Take-home exercises

1. **High-spin slope.** Repeat the slope test at $U = J$ for the
   maximally polarised 3-electron sector ($M_{S} = +3/2$, all three
   electrons $\alpha$). Why is the dimension only $1$? Confirm
   slope $= 3$.

2. **Off-line behaviour.** Move off the $U = J$ line: scan $J = U/2$
   instead. Is $E_{\rm gs}$ still linear in $U$? Plot for several
   ratios $J/U$ to see how the linearity fails.

3. **Add $K \ne 0$.** Now keep $U = J$ but switch on $K = U/4$. Does
   the linearity survive? What does $K$ measure physically that $J$
   does not?

4. **Map the residual at $s = 0.1$.** For the allyl anion at $s = 0.1$
   numerically, what fraction of the FCI ground state still lies along
   $|\psi_{1}^{2}\,\psi_{2}^{2}\rangle$? At what $s$ does the
   projection drop below $0.99$?

5. **PPP bridge.** In PPP-type models the on-site and inter-atomic
   Coulomb integrals often enter observables only through the
   effective combination $U - J$. Explain how the identity you've
   just derived produces that reduction, and why it is exact only
   at $s = 0$.

### Up next

The **Hubbard-to-Heisenberg** notebook (this folder) moves to benzene:
building the rational perturbation series in $U/t$, taking the large-$U$
limit, and *deriving* the Heisenberg coupling $J = 4t^{2}/U$ from the full
400-dimensional FCI.
""")


NB.cells = cells

out_path = os.path.join(os.path.dirname(__file__), '..', 'allyl_uj_identity.ipynb')
out_path = os.path.normpath(out_path)
nbf.write(NB, out_path)
print(f"Wrote {out_path}  ({len(cells)} cells)")
