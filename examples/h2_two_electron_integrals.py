"""
Distance dependence of H2 two-electron integrals (Slater 1s, zeta=1).

Integrals are computed with the STO-6G expansion of an exact Slater 1s
(zeta=1) orbital. Pyscf's built-in STO-6G uses Slater's-rule zeta=1.24
for hydrogen; we rescale the exponents by 1/1.24^2 to recover the
zeta=1 integrals used elsewhere in the manuscript (so that on-site
(aa|aa) = 5/8 exactly).

symvb / PPP integral-pattern naming (see h2_hubbard_ujk.py):
    U  = (aa|aa)  pattern 1111   on-site repulsion        (R-independent)
    K  = (aa|bb)  pattern 1122   two-center direct Coulomb
    J  = (ab|ab)  pattern 1212   two-center exchange
    M  = (aa|ab)  pattern 1112   three-index hybrid

Asymptotics:
    K(R) -> 1/R as R -> infty  (point-charge limit)
    J(R), M(R) decay exponentially as e^{-2R}, e^{-R} respectively
    U is exactly R-independent
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyscf import gto

# STO-6G expansion of a Slater 1s orbital, rescaled to zeta = 1
# (pyscf's STO-6G for H uses zeta = 1.24; alpha -> alpha / zeta^2).
ZETA_PYSCF = 1.24
EXPS_124 = [35.52322122, 6.513143725, 1.822142904,
            0.625955266, 0.243076747, 0.100112428]
COEFS    = [0.00916359628, 0.04936149294, 0.16853830490,
            0.37056279970, 0.41649152980, 0.13033408410]
EXPS_1 = [a / ZETA_PYSCF**2 for a in EXPS_124]
BASIS_ZETA1 = {'H': [[0] + list(zip(EXPS_1, COEFS))]}


def integrals_at_R(R):
    """Return (U, K, J, M, S_ab) for H-H at distance R (au)."""
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {R}',
                basis=BASIS_ZETA1, verbose=0, unit='Bohr')
    eri = mol.intor('int2e')
    S = mol.intor('int1e_ovlp')
    U = eri[0, 0, 0, 0]          # (aa|aa)
    K = eri[0, 0, 1, 1]          # (aa|bb)
    J = eri[0, 1, 0, 1]          # (ab|ab)
    M = eri[0, 0, 0, 1]          # (aa|ab)
    return U, K, J, M, S[0, 1]


def K_roothaan(R):
    """Closed-form (aa|bb) for Slater 1s, zeta=1 (Roothaan 1951)."""
    return (1.0 - (1.0 + 11*R/8 + 3*R**2/4 + R**3/6) * np.exp(-2*R)) / R


# Compute on a grid
R_grid = np.linspace(0.3, 7.0, 80)
data = np.array([integrals_at_R(R) for R in R_grid])
U_vals, K_vals, J_vals, M_vals, S_vals = data.T

# Verify against closed-form K(R)
K_cf = np.array([K_roothaan(R) for R in R_grid])
print(f"max |K_numeric - K_Roothaan| = {np.max(np.abs(K_vals - K_cf)):.2e}")
print(f"U(R) numerical: min={U_vals.min():.6f}, max={U_vals.max():.6f}, "
      f"expected 5/8 = {5/8:.6f}")

# Notable values
def at(Rv):
    i = np.argmin(np.abs(R_grid - Rv))
    return R_grid[i], U_vals[i], K_vals[i], J_vals[i], M_vals[i]

print("\n  R       U        K        J        M")
for Rv in [1.0, 1.4, 2.0, 3.0, 5.0]:
    R_, U_, K_, J_, M_ = at(Rv)
    print(f" {R_:.2f}  {U_:.4f}  {K_:.4f}  {J_:.4f}  {M_:.4f}")


# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

R_eq = 1.4   # Slater zeta=1 best-fit equilibrium distance for H2

# Panel (a): linear scale
ax[0].axhline(5/8, color='C3', lw=1.6, ls='-',
              label=r'$U = (aa|aa) = 5/8$  (pattern 1111)')
ax[0].plot(R_grid, K_vals, 'C0-',  lw=2.0,
           label=r'$K(R) = (aa|bb)$  (pattern 1122)')
ax[0].plot(R_grid, M_vals, 'C2-',  lw=2.0,
           label=r'$M(R) = (aa|ab)$  (pattern 1112)')
ax[0].plot(R_grid, J_vals, 'C4-',  lw=2.0,
           label=r'$J(R) = (ab|ab)$  (pattern 1212)')
ax[0].plot(R_grid, 1.0/R_grid, 'k:', lw=1.0, alpha=0.6,
           label=r'$1/R$  (point-charge limit)')
ax[0].axvline(R_eq, color='k', lw=0.4, ls=':', alpha=0.5)
ax[0].text(R_eq + 0.05, 1.18, r'$R_e^{\rm H_2}$', fontsize=9)
ax[0].set_xlabel(r'$R$ (au)')
ax[0].set_ylabel(r'integral value (au)')
ax[0].set_title(r'(a)  H$_2$ two-electron integrals  (Slater 1s, $\zeta=1$)')
ax[0].set_xlim(0.3, 7.0)
ax[0].set_ylim(0.0, 1.3)
ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=9, loc='upper right')

# Panel (b): semi-log, to expose exponential vs power-law decay
ax[1].axhline(5/8, color='C3', lw=1.6, ls='-', label=r'$U = 5/8$')
ax[1].semilogy(R_grid, K_vals, 'C0-',  lw=2.0, label=r'$K(R) = (aa|bb)$')
ax[1].semilogy(R_grid, M_vals, 'C2-',  lw=2.0, label=r'$M(R) = (aa|ab)$')
ax[1].semilogy(R_grid, J_vals, 'C4-',  lw=2.0, label=r'$J(R) = (ab|ab)$')
ax[1].semilogy(R_grid, 1.0/R_grid, 'k:', lw=1.0, alpha=0.6, label=r'$1/R$')
ax[1].axvline(R_eq, color='k', lw=0.4, ls=':', alpha=0.5)
ax[1].set_xlabel(r'$R$ (au)')
ax[1].set_ylabel(r'integral value (au, log)')
ax[1].set_title(r'(b)  Asymptotic decay: $K \sim 1/R$,  $M \sim e^{-R}$,  $J \sim e^{-2R}$')
ax[1].set_xlim(0.3, 7.0)
ax[1].set_ylim(1e-4, 2.0)
ax[1].grid(alpha=0.3, which='both')
ax[1].legend(fontsize=9, loc='lower left')

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'vbt-3', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, 'fig_h2_two_electron_integrals.png')
out_pdf = os.path.join(out_dir, 'fig_h2_two_electron_integrals.pdf')
plt.savefig(out_png, dpi=140)
plt.savefig(out_pdf)
plt.close()
print(f"\nFigure saved: {out_png}")
print(f"             {out_pdf}")
