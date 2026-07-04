"""
Does the H2 singlet-triplet crossing of E(trip) = J - K and the A_1
ground state actually happen at any R?

Uses Slater-1s integrals (zeta = 1, STO-6G expansion) Lowdin-
orthogonalised so the AO frame has s = 0 (matching the closed-form
crossing condition  2 h^2 = K (U - J + 2 K)  derived in
h2_hubbard_ujk.py).

Run:  PYTHONPATH=. python3 examples/h2_st_crossing_check.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyscf import gto

# zeta = 1 Slater 1s expansion (rescaled STO-6G)
ZETA = 1.24
EXPS = [35.52322122, 6.513143725, 1.822142904,
        0.625955266, 0.243076747, 0.100112428]
COEFS = [0.00916359628, 0.04936149294, 0.16853830490,
         0.37056279970, 0.41649152980, 0.13033408410]
BASIS = {'H': [[0] + list(zip([a / ZETA**2 for a in EXPS], COEFS))]}


def model_at_R(R):
    """Return (h_orth, U, J, K) in the Lowdin-orthogonalised AO basis."""
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {R}', basis=BASIS,
                verbose=0, unit='Bohr')
    S = mol.intor('int1e_ovlp')
    H1 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    eri = mol.intor('int2e')

    sval, svec = np.linalg.eigh(S)
    X = svec @ np.diag(sval**-0.5) @ svec.T   # S^(-1/2)

    H1o = X @ H1 @ X
    erio = np.einsum('ip,jq,ijkl,kr,ls->pqrs', X, X, eri, X, X)

    h_orth = H1o[0, 1]                        # off-diagonal one-electron hop
    U = erio[0, 0, 0, 0]
    J = erio[0, 0, 1, 1]
    K = erio[0, 1, 0, 1]
    eps_diag = H1o[0, 0]    # one-electron diagonal (same for both H by symmetry)
    return h_orth, U, J, K, eps_diag


def symvb_eigs(h, U, J, K, eps):
    """4x4 Sz=0 FCI in the {cov, ion+, ion-, trip} basis at s=0, M=0.
    Includes the one-electron diagonal eps on all states (2*eps for the
    two-electron diagonals, since both electrons sit somewhere).
    """
    cov_ip = np.array([[J + K + 2*eps, 2*h],
                       [2*h, U + K + 2*eps]])
    e_gs, e_ex = np.linalg.eigvalsh(cov_ip)
    e_im = U - K + 2*eps
    e_t  = J - K + 2*eps
    return e_gs, e_ex, e_im, e_t


# -------------------- scan over R --------------------
R_grid = np.linspace(0.4, 8.0, 200)
h_, U_, J_, K_, eps_ = np.array([model_at_R(R) for R in R_grid]).T

E_gs = np.zeros_like(R_grid)
E_ex = np.zeros_like(R_grid)
E_im = np.zeros_like(R_grid)
E_t  = np.zeros_like(R_grid)
for i, R in enumerate(R_grid):
    E_gs[i], E_ex[i], E_im[i], E_t[i] = symvb_eigs(
        h_[i], U_[i], J_[i], K_[i], eps_[i])

gap_t_gs = E_t - E_gs        # triplet above ground singlet?
crossing_lhs = 2 * h_**2
crossing_rhs = K_ * (U_ - J_ + 2 * K_)

print("R(au)   h_orth     U      J      K     E_gs    E_trip   E_t-E_gs"
      "    2h^2     K(U-J+2K)")
for i in range(0, len(R_grid), 20):
    print(f"{R_grid[i]:5.2f}  {h_[i]:+.4f}  {U_[i]:.3f}  {J_[i]:.3f}  "
          f"{K_[i]:.3f}  {E_gs[i]:+.4f}  {E_t[i]:+.4f}  {gap_t_gs[i]:+.4f}"
          f"   {crossing_lhs[i]:.4f}   {crossing_rhs[i]:.4f}")

print(f"\nmin(E_trip - E_gs) over scan = {gap_t_gs.min():.4f} au "
      f"at R = {R_grid[np.argmin(gap_t_gs)]:.2f} au")
print(f"max ratio  K(U-J+2K) / (2 h^2) = "
      f"{(crossing_rhs/np.maximum(crossing_lhs, 1e-30)).max():.4f} "
      f"at R = {R_grid[np.argmax(crossing_rhs/np.maximum(crossing_lhs,1e-30))]:.2f} au")
print("(triplet becomes ground iff this ratio exceeds 1)")


# -------------------- figure --------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

ax[0].plot(R_grid, E_gs, 'k-',  lw=2.0, label=r'$E_{\rm gs}$  (A$_1$ singlet)')
ax[0].plot(R_grid, E_t,  'C3-', lw=2.0, label=r'$E_{\rm trip}$ (3$\Sigma_u^+$)')
ax[0].plot(R_grid, E_im, 'C2-', lw=1.4, label=r'$E_{\rm B_1}$  (1$\Sigma_u^+$)')
ax[0].plot(R_grid, E_ex, 'C0--', lw=1.4, label=r'$E_{\rm A_1, ex}$')
ax[0].axhline(0, color='k', lw=0.3, alpha=0.4)
ax[0].set_xlabel(r'$R$ (au)')
ax[0].set_ylabel(r'energy (au)')
ax[0].set_title(r'(a) H$_2$ FCI eigenvalues  (orth. AOs, $\zeta = 1$)')
ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=9)
ax[0].set_xlim(0.4, 8.0)

ax[1].plot(R_grid, crossing_lhs, 'k-',  lw=2.0, label=r'$2 h(R)^2$')
ax[1].plot(R_grid, crossing_rhs, 'C3-', lw=2.0, label=r'$K\,(U - J + 2K)$')
ax[1].axvline(R_grid[np.argmin(gap_t_gs)], color='k', lw=0.4, ls=':', alpha=0.5)
ax[1].set_xlabel(r'$R$ (au)')
ax[1].set_ylabel(r'value (au$^2$)')
ax[1].set_title(r'(b) Crossing test:  $2 h^2 \;\overset{?}{=}\; K(U - J + 2K)$')
ax[1].set_yscale('log')
ax[1].grid(alpha=0.3, which='both')
ax[1].legend(fontsize=9)
ax[1].set_xlim(0.4, 8.0)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', '..', 'vbt-3', 'figures',
                   'fig_h2_st_crossing_check.png')
plt.savefig(out, dpi=140)
plt.close()
print(f"\nFigure saved: {out}")
