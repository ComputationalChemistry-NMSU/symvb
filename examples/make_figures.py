"""
Generate all figures in the manuscript as PDFs.

Each figure pins to a specific worked example (see Appendix A of
manuscript.md).  Data is computed on the fly for the small systems;
for benzene Hubbard (Figure 4) the expensive symbolic H1/S/H2 matrices
are loaded from /tmp/benzene_hubbard_matrices.pkl if present, else
Figure 4 is skipped with a diagnostic message.

Run:
    python3 examples/make_figures.py [out_dir]

Produces fig3_h2.pdf ... fig7_aromaticity.pdf in out_dir (default
./figures).
"""
import os
import sys
import pickle
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from vbt3 import Molecule, SlaterDet, FixedPsi, symmetry
from vbt3.fixed_psi import generate_dets

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'vbt-3', 'figures')
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})


# =======================================================================
# Figure 1 - H2 Hubbard: energy and bond character
# =======================================================================
def figure_1_h2():
    m = Molecule(zero_ii=True, interacting_orbs=['ab'],
                 subst={'h': ('H_ab',), 's': ('S_ab',)},
                 subst_2e={'U': ('1111',)}, max_2e_centers=1)
    P = generate_dets(1, 1, 2)
    H = sp.Matrix(m.build_matrix(P, op='H') + m.o2_matrix(P))
    S = sp.Matrix(m.build_matrix(P, op='S'))
    h, s, U = sp.symbols('h s U')
    H0 = H.subs({h: -1, s: 0})

    # Closed form (via 2x2 A_1 block symbolic work)
    U_vals = np.linspace(0, 20, 201)
    t = 1.0
    E_gs = U_vals / 2 - np.sqrt((U_vals / 2) ** 2 + 4 * t ** 2)
    w_cov = 16 * t ** 2 / (16 * t ** 2 + (U_vals - np.sqrt(U_vals ** 2 + 16 * t ** 2)) ** 2)
    w_ion = 1 - w_cov

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))

    ax[0].plot(U_vals, E_gs, 'k-', lw=1.6)
    ax[0].axhline(-2, ls=':', c='C0', lw=1)
    ax[0].text(12, -1.85, r'$-2t$  (Hückel)', color='C0', fontsize=10)
    U_large = np.linspace(5, 20, 100)
    ax[0].plot(U_large, -4 / U_large, ls='--', c='C3', lw=1)
    ax[0].text(14, -0.6, r'$-4t^2/U$  (Heisenberg)', color='C3', fontsize=10)
    ax[0].set_xlabel(r'$U/t$'); ax[0].set_ylabel(r'$E_{\rm gs} / t$')
    ax[0].set_title('(a)  H$_2$ ground-state energy')
    ax[0].grid(alpha=0.3)

    ax[1].plot(U_vals, w_cov, 'C0-', lw=1.6, label=r'$w_{\rm cov}$')
    ax[1].plot(U_vals, w_ion, 'C3-', lw=1.6, label=r'$w_{\rm ion}$')
    ax[1].axhline(0.5, ls=':', c='k', lw=0.8)
    ax[1].axvline(4, ls=':', c='k', lw=0.8)
    ax[1].text(0.4, 0.45, 'RHF 50/50', fontsize=10)
    ax[1].text(4.3, 0.7, r'crossover  $U\sim4t$', fontsize=10)
    ax[1].set_xlabel(r'$U/t$'); ax[1].set_ylabel('weight')
    ax[1].set_title('(b)  VB character of the bond')
    ax[1].legend(loc='center right'); ax[1].grid(alpha=0.3)
    ax[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig3_h2.pdf'))
    plt.close()
    print('  fig3_h2.pdf')


# =======================================================================
# Figure 2 - Allyl: block structure + PT convergence
# =======================================================================
def figure_2_allyl():
    # Build 9x9 allyl H
    m = Molecule(zero_ii=True, interacting_orbs=['ab', 'bc'],
                 subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
                 subst_2e={'U': ('1111',)}, max_2e_centers=1)
    P = generate_dets(2, 2, 3)
    H = sp.Matrix(m.build_matrix(P, op='H') + m.o2_matrix(P))
    h, s, Usym = sp.symbols('h s U')
    # Numerical H(U, h=-1, s=0) at various U
    H0_num = np.array(H.subs({h: -1, s: 0, Usym: 0}).tolist(), dtype=float)
    V_num  = np.array((H.subs({h: -1, s: 0, Usym: 1}) -
                       H.subs({h: -1, s: 0, Usym: 0})).tolist(), dtype=float)

    # Exact and partial Taylor sums
    U_vals = np.linspace(0, 8, 121)
    coefs = [-2 * np.sqrt(2), 11 / 8, -21 * np.sqrt(2) / 512, 3 / 1024,
             537 * np.sqrt(2) / 1048576, -15 / 131072,
             -11661 * np.sqrt(2) / 1073741824]

    def E_exact(U):
        return eigh(H0_num + U * V_num)[0][0]

    E_ex = np.array([E_exact(u) for u in U_vals])
    E_t = [sum(coefs[k] * U_vals ** k for k in range(order + 1)) for order in [2, 4, 6]]

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))

    # (a)  Heatmap of the full 9x9 at U=2 showing block structure:
    #      permute rows/cols so sigma=+1 dets appear first
    det_strings = [p.dets[0].det_string for p in P]
    def canon(ds):
        fp = SlaterDet(ds).get_sorted()
        return fp.dets[0].det_string, fp.coefs[0]
    sig = {'a': 'c', 'b': 'b', 'c': 'a'}
    perm, signs = symmetry.apply_orbital_permutation(sig, det_strings, canon)
    order = []
    seen = [False] * 9
    for i in range(9):
        if seen[i]: continue
        j = perm[i]
        if j == i:
            order.append(i); seen[i] = True
        else:
            order.append(i); order.append(j); seen[i] = seen[j] = True
    # Build symmetry-adapted matrix via U-transformation with 5+4 blocks
    # (same construction as examples/allyl_hubbard_pt.py)
    U_plus, U_minus = [], []
    seen = [False] * 9
    for i in range(9):
        if seen[i]: continue
        j = perm[i]; sj = signs[i]
        if j == i:
            seen[i] = True
            v = np.zeros(9); v[i] = 1
            (U_plus if sj == 1 else U_minus).append(v)
        else:
            seen[i] = seen[j] = True
            vp = np.zeros(9); vp[i] = 1; vp[j] = sj; vp /= np.linalg.norm(vp)
            vm = np.zeros(9); vm[i] = 1; vm[j] = -sj; vm /= np.linalg.norm(vm)
            U_plus.append(vp); U_minus.append(vm)
    Umat = np.column_stack(U_plus + U_minus)
    H_U2 = H0_num + 2.0 * V_num
    H_sym = Umat.T @ H_U2 @ Umat
    im = ax[0].imshow(H_sym, cmap='RdBu_r',
                      vmin=-np.abs(H_sym).max(), vmax=np.abs(H_sym).max())
    ax[0].axhline(len(U_plus) - 0.5, color='k', lw=1)
    ax[0].axvline(len(U_plus) - 0.5, color='k', lw=1)
    ax[0].set_title(r'(a)  $H$ in $\sigma$-adapted basis  ($U=2$)')
    n_plus = len(U_plus)
    ax[0].text(0.5 * (n_plus - 1), n_plus / 2 - 0.5, r'$\sigma=+1$' + '\n' + r'$(5\times 5)$',
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.85))
    ax[0].text(n_plus + 0.5 * (9 - n_plus - 1), n_plus + 0.5 * (9 - n_plus - 1),
               r'$\sigma=-1$' + '\n' + r'$(4\times 4)$',
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.85))
    plt.colorbar(im, ax=ax[0], fraction=0.045, pad=0.03)

    # (b) Taylor convergence
    ax[1].plot(U_vals, E_ex, 'k-', lw=1.8, label='exact')
    for order, (E, ls) in enumerate(zip(E_t, ['--', '-.', ':']), start=2):
        ax[1].plot(U_vals, E, ls=ls, lw=1.2, label=f'Taylor order {2 * order}')
    ax[1].set_xlabel(r'$U/t$'); ax[1].set_ylabel(r'$E / t$')
    ax[1].set_title('(b)  Allyl PT convergence')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    ax[1].set_ylim(-3.5, 2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig4_allyl.pdf'))
    plt.close()
    print('  fig4_allyl.pdf')


# =======================================================================
# Figure 3 - benzene FCI degeneracy spectrum
# =======================================================================
def figure_3_degeneracy():
    m = Molecule(zero_ii=True,
                 subst={'s': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af'),
                        'h': ('H_ab', 'H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af')},
                 interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'])
    m.generate_basis(3, 3, 6)
    H = m.build_matrix(m.basis, op='H')
    S = m.build_matrix(m.basis, op='S')
    h, s = sp.symbols('h s')
    Hn = np.array(H.subs({h: -1, s: 0.2}).tolist(), dtype=float)
    Sn = np.array(S.subs({h: -1, s: 0.2}).tolist(), dtype=float)

    evals, evecs, blocks = symmetry.degenerate_block_basis(Hn, Sn, tol=1e-6)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    degs = [len(b[1]) for b in blocks]
    Es = [b[0] for b in blocks]
    colors = ['C3' if d == 1 else 'C0' for d in degs]
    ax.bar(range(len(blocks)), degs, color=colors, edgecolor='k', lw=0.6)
    for i, (E, d) in enumerate(zip(Es, degs)):
        ax.text(i, d + 1.5, f'{d}', ha='center', fontsize=9)
        ax.text(i, -3.5, f'{E:.2f}', ha='center', fontsize=8, rotation=45)
    ax.set_xlabel('irrep cluster (increasing energy →)')
    ax.set_ylabel('degeneracy')
    ax.set_title(r'Figure 3. Benzene FCI spectrum at $h=-1$, $s=0.2$,  $U=0$'
                 '\n' + 'Highlighted: non-degenerate (1D) clusters '
                 r'include the ground state ($A_{1g}$).')
    ax.set_ylim(-5, max(degs) + 8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig1_degeneracy.pdf'))
    plt.close()
    print('  fig1_degeneracy.pdf')


# =======================================================================
# Figure 4 - benzene Hubbard Taylor vs Pade vs exact
# =======================================================================
def figure_4_pade():
    CACHE = '/tmp/benzene_hubbard_matrices.pkl'
    if not os.path.exists(CACHE):
        print('  fig4: skipping (need /tmp/benzene_hubbard_matrices.pkl; '
              'run examples/benzene_hubbard_pt.py once to generate)')
        return
    with open(CACHE, 'rb') as f:
        H1, S, H2 = pickle.load(f)
    h, s, U = sp.symbols('h s U')
    H = sp.Matrix(H1 + H2).subs({h: -1, s: 0})
    H0 = np.array(H.subs(U, 0).tolist(), dtype=float)
    V = np.array((H.subs(U, 1) - H.subs(U, 0)).tolist(), dtype=float)

    def E_exact(Uv):
        return eigh(H0 + Uv * V)[0][0]

    # Taylor
    C = [sp.Rational(-8), sp.Rational(3, 2), sp.Rational(-29, 288),
         sp.Rational(0), sp.Rational(-2855, 5971968), sp.Rational(0),
         sp.Rational(855791, 61917364224)]
    Cf = [float(c) for c in C]

    # Pade using the same solver as examples/benzene_hubbard_pade.py
    def pade(c, n, m):
        x = sp.Symbol('x')
        A = sp.zeros(m, m); rhs = sp.zeros(m, 1)
        for k in range(n + 1, n + m + 1):
            row = k - (n + 1); rhs[row, 0] = -c[k]
            for j in range(1, m + 1):
                if 0 <= k - j < len(c): A[row, j - 1] = c[k - j]
        b = A.solve(rhs)
        Q = 1 + sum(b[j - 1] * x ** j for j in range(1, m + 1))
        f = sum(c[k] * x ** k for k in range(n + m + 1))
        P_full = sp.expand(f * Q)
        P = sum(P_full.coeff(x, k) * x ** k for k in range(n + 1))
        return sp.lambdify(x, P / Q, 'numpy')

    pade24 = pade(C, 2, 4)
    pade33 = pade(C, 3, 3)

    U_vals = np.geomspace(0.1, 500, 80)
    E_ex = np.array([E_exact(u) for u in U_vals])
    E_t6 = sum(Cf[k] * U_vals ** k for k in range(7))
    E_p24 = np.array([float(pade24(u)) for u in U_vals])
    E_p33 = np.array([float(pade33(u)) for u in U_vals])

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

    ax[0].plot(U_vals, E_ex, 'k-', lw=1.8, label='exact')
    ax[0].plot(U_vals, E_t6, 'C0--', lw=1.2, label='Taylor(6)')
    ax[0].plot(U_vals, E_p24, 'C3-', lw=1.2, label='Padé [2/4]')
    ax[0].plot(U_vals, E_p33, 'C2:', lw=1.2, label='Padé [3/3]')
    ax[0].set_xscale('log')
    ax[0].set_xlabel(r'$U/t$'); ax[0].set_ylabel(r'$E / t$')
    ax[0].set_title('(a)  Benzene Hubbard ground-state energy')
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3, which='both')
    ax[0].set_ylim(-9, 10)

    err_t6  = np.abs(E_t6 - E_ex)
    err_p24 = np.abs(E_p24 - E_ex)
    err_p33 = np.abs(E_p33 - E_ex)
    for arr in (err_t6, err_p24, err_p33):
        arr[arr == 0] = np.nan
    ax[1].loglog(U_vals, err_t6, 'C0--', lw=1.2, label='Taylor(6)')
    ax[1].loglog(U_vals, err_p24, 'C3-', lw=1.2, label='Padé [2/4]')
    ax[1].loglog(U_vals, err_p33, 'C2:', lw=1.2, label='Padé [3/3]')
    ax[1].set_xlabel(r'$U/t$'); ax[1].set_ylabel(r'$|E_{\rm approx} - E_{\rm exact}|$')
    ax[1].set_title('(b)  absolute error, log-log')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig2_pade.pdf'))
    plt.close()
    print('  fig2_pade.pdf')


# =======================================================================
# Figure 5 - aromaticity loss under a-b attack
# =======================================================================
def figure_5_aromaticity():
    m = Molecule(
        zero_ii=True,
        subst={'s': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af'),
               'h': ('H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af')},   # H_ab free
        interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'])
    PARENT = 'aBcDeF'
    rumer = [
        FixedPsi(PARENT, coupled_pairs=[(0, 1), (2, 3), (4, 5)]),
        FixedPsi(PARENT, coupled_pairs=[(0, 5), (1, 2), (3, 4)]),
        FixedPsi(PARENT, coupled_pairs=[(0, 1), (2, 5), (3, 4)]),
        FixedPsi(PARENT, coupled_pairs=[(0, 3), (1, 2), (4, 5)]),
        FixedPsi(PARENT, coupled_pairs=[(0, 5), (1, 4), (2, 3)])]
    names = ['Kek$_1$', 'Kek$_2$', 'Dew$_1$', 'Dew$_2$', 'Dew$_3$']

    Hs = m.build_matrix(rumer, op='H')
    Ss = m.build_matrix(rumer, op='S')
    h, s = sp.symbols('h s')
    Hab = sp.Symbol('H_ab')
    H_VAL, S_VAL = -1.0, 0.2
    lambdas = np.linspace(1.0, 0.0, 81)
    weights = np.zeros((5, len(lambdas)))
    Es = np.zeros((5, len(lambdas)))
    E_cov = np.zeros(len(lambdas))
    for i, lam in enumerate(lambdas):
        subs = {h: H_VAL, s: S_VAL, Hab: lam * H_VAL}
        Hn = np.array(Hs.subs(subs).tolist(), dtype=float)
        Sn = np.array(Ss.subs(subs).tolist(), dtype=float)
        evals, evecs = eigh(Hn, Sn)
        E_cov[i] = evals[0]
        c0 = evecs[:, 0]
        weights[:, i] = c0 * (Sn @ c0)
        Es[:, i] = [Hn[j, j] / Sn[j, j] for j in range(5)]

    RE_Kek = np.min(Es[:2], axis=0) - E_cov

    # --- full 400-det FCI scan via H(lambda) = H0 + lambda * dH --------------
    # The full 6-orbital Sz=0 basis built with H_ab kept symbolic; H is linear
    # in H_ab so two numerical builds (at H_ab=0 and H_ab=h) span the scan.
    CACHE_FULL = '/tmp/benzene_full_aromaticity_HS.pkl'
    if os.path.exists(CACHE_FULL):
        with open(CACHE_FULL, 'rb') as f:
            H_sym, S_sym = pickle.load(f)
    else:
        m_full = Molecule(
            zero_ii=True,
            subst={'s': ('S_ab','S_bc','S_cd','S_de','S_ef','S_af'),
                   'h': ('H_bc','H_cd','H_de','H_ef','H_af')},
            interacting_orbs=['ab','bc','cd','de','ef','af'])
        m_full.generate_basis(3, 3, 6)
        H_sym = m_full.build_matrix(m_full.basis, op='H')
        S_sym = m_full.build_matrix(m_full.basis, op='S')
        with open(CACHE_FULL, 'wb') as f:
            pickle.dump((H_sym, S_sym), f)
    H0 = np.array(H_sym.subs({h: H_VAL, s: S_VAL, Hab: 0.0}).tolist(), dtype=float)
    H1 = np.array(H_sym.subs({h: H_VAL, s: S_VAL, Hab: H_VAL}).tolist(), dtype=float)
    S_f = np.array(S_sym.subs({s: S_VAL}).tolist(), dtype=float)
    dH_full = H1 - H0
    E_fci = np.zeros(len(lambdas))
    rho_ab = np.zeros(len(lambdas))
    for i, lam in enumerate(lambdas):
        evals, evecs = eigh(H0 + lam * dH_full, S_f, subset_by_index=[0, 0])
        E_fci[i] = evals[0]
        c0 = evecs[:, 0]
        rho_ab[i] = -(c0 @ dH_full @ c0)   # Hellmann-Feynman: rho = -dE/dlam

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    ls = ['-', '-', '--', '--', '--']
    for j in range(5):
        ax[0].plot(lambdas, weights[j], ls[j], color=colors[j], lw=1.5, label=names[j])
    ax[0].set_xlabel(r'$\lambda = h_{ab}/h$ (reaction coordinate)')
    ax[0].set_ylabel('Chirgwin–Coulson weight')
    ax[0].set_title('(a)  Rumer structure weights')
    ax[0].legend(fontsize=9, ncol=2, loc='upper left')
    ax[0].grid(alpha=0.3); ax[0].invert_xaxis()

    ax[1].plot(lambdas, RE_Kek, 'k-', lw=1.8)
    ax[1].set_xlabel(r'$\lambda$')
    ax[1].set_ylabel(r'$RE_{\rm Kek}$  (units of $|\beta|$)')
    ax[1].set_title(r'(b)  Kekulé resonance energy')
    ax[1].grid(alpha=0.3); ax[1].invert_xaxis()
    ax[1].annotate('aromatic', xy=(1.0, RE_Kek[0]), xytext=(0.8, 0.42),
                   fontsize=10, ha='center')
    ax[1].annotate('broken bond', xy=(0.0, RE_Kek[-1]), xytext=(0.2, 0.22),
                   fontsize=10, ha='center')

    ax[2].plot(lambdas, E_cov, 'k-', lw=1.8)
    ax[2].set_xlabel(r'$\lambda$')
    ax[2].set_ylabel(r'$E_{\rm cov5}  /  |\beta|$')
    ax[2].set_title(r'(c)  covalent 5-structure energy')
    ax[2].grid(alpha=0.3); ax[2].invert_xaxis()
    imax = int(np.argmax(E_cov))
    ax[2].plot(lambdas[imax], E_cov[imax], 'ro', ms=6)
    ax[2].annotate(fr'max @ $\lambda={lambdas[imax]:.2f}$',
                   xy=(lambdas[imax], E_cov[imax]),
                   xytext=(lambdas[imax] - 0.15, E_cov[imax] + 0.015),
                   fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.8))

    ax[3].plot(lambdas, E_fci, 'k-', lw=1.8, label=r'$E_{\mathrm{FCI}}$')
    ax[3].set_xlabel(r'$\lambda$')
    ax[3].set_ylabel(r'$E_{\mathrm{FCI}}  /  |\beta|$')
    ax[3].set_title(r'(d)  full 400-det CI energy')
    ax[3].grid(alpha=0.3); ax[3].invert_xaxis()
    ax3b = ax[3].twinx()
    ax3b.plot(lambdas, rho_ab, 'C3--', lw=1.5, label=r'$\rho_{ab}=-dE/d\lambda$')
    ax3b.set_ylabel(r'$\rho_{ab}$  ($\pi$ bond order on $a$–$b$)', color='C3')
    ax3b.tick_params(axis='y', labelcolor='C3')
    ax[3].legend(loc='upper right', fontsize=9, frameon=False)
    ax3b.legend(loc='lower left', fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig7_aromaticity.pdf'))
    plt.savefig(os.path.join(OUTDIR, 'fig7_aromaticity.png'), dpi=130)
    plt.close()
    print('  fig7_aromaticity.pdf')


# =======================================================================
# Figure 6 - (H2)2+ disphenoid: pseudo-JT instability and III <-> II
# =======================================================================
def figure_6_disphenoid():
    # Build 3e and 4e disphenoid Hamiltonians once
    m = Molecule(
        zero_ii=True,
        interacting_orbs=['ab', 'cd', 'ac', 'ad', 'bc', 'bd'],
        subst={'h_s_1': ('H_ab',), 'h_s_2': ('H_cd',),
               'h_l':   ('H_ac', 'H_ad', 'H_bc', 'H_bd'),
               's':     ('S_ab', 'S_cd', 'S_ac', 'S_ad', 'S_bc', 'S_bd')},
        subst_2e={'U': ('1111',)}, max_2e_centers=1,
    )
    hs1, hs2, hl, s_sym, Usym = sp.symbols('h_s_1 h_s_2 h_l s U')
    syms = [hs1, hs2, hl, Usym]

    def decomp(Na, Nb):
        P = generate_dets(Na, Nb, 4)
        H = sp.Matrix(m.build_matrix(P, op='H') + m.o2_matrix(P))
        H0 = np.array(H.subs({s_sym: 0, hs1: 0, hs2: 0, hl: 0, Usym: 0}).tolist(), dtype=float)
        coefs = [np.array(sp.diff(H.subs(s_sym, 0), x).tolist(), dtype=float) for x in syms]
        return H0, coefs

    H3_0, H3_c = decomp(2, 1)   # cation
    H4_0, H4_c = decomp(2, 2)   # neutral

    def Egs(H0, coefs, hs1v, hs2v, hlv, Uv=0):
        H = H0 + hs1v * coefs[0] + hs2v * coefs[1] + hlv * coefs[2] + Uv * coefs[3]
        H = (H + H.T) / 2
        return np.linalg.eigvalsh(H)[0]

    etas = np.linspace(-1.0, 1.0, 401)
    h_l_vals = [-0.15, -0.3, -0.6]
    colors = ['C0', 'C2', 'C3']

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    # Panel (a):  E_elec(eta) for cation at three h_l values, with closed form
    for hl_v, col in zip(h_l_vals, colors):
        E_num = np.array([Egs(H3_0, H3_c, -1 - e / 2, -1 + e / 2, hl_v, 0) for e in etas])
        ax[0, 0].plot(etas, E_num, color=col, lw=1.8, label=rf'$h_l = {hl_v}$')
        # Overlay closed form 3 h_s - sqrt(eta^2/4 + 4 h_l^2),  h_s = -1
        E_cf = -3 - np.sqrt(etas**2 / 4 + 4 * hl_v**2)
        ax[0, 0].plot(etas, E_cf, color=col, ls=':', lw=1)
    ax[0, 0].set_xlabel(r'$\eta = h_{s,1} - h_{s,2}$')
    ax[0, 0].set_ylabel(r'$E_{\mathrm{elec}}(\eta) / |h_s|$')
    ax[0, 0].set_title(r'(a)  cation $E_{\mathrm{elec}}(\eta)$')
    ax[0, 0].legend(fontsize=9, loc='lower center'); ax[0, 0].grid(alpha=0.3)
    ax[0, 0].text(0.6, -3.05, 'dotted: closed form', fontsize=9,
                  style='italic', color='0.3')

    # Panel (b):  E_tot(eta) = E_elec(eta) + (1/2) k eta^2
    #             at h_l = -0.3 (k_crit = 1/(8*0.3) ~ 0.417), for several k values
    hl_v = -0.3
    k_crit = 1 / (8 * abs(hl_v))
    E_elec_arr = np.array([Egs(H3_0, H3_c, -1 - e / 2, -1 + e / 2, hl_v, 0) for e in etas])
    for k, ls in zip([0.1, 0.3, k_crit, 0.6, 1.0], ['-', '-', '--', '-', '-']):
        Etot = E_elec_arr + 0.5 * k * etas**2
        # Subtract value at eta=0 for visual clarity
        Etot -= Etot[len(etas) // 2]
        label = f'$k = {k:.2f}$'
        if abs(k - k_crit) < 1e-6:
            label = rf'$k = k_{{\mathrm{{crit}}}} = 1/(8|h_l|) = {k_crit:.3f}$'
            ax[0, 1].plot(etas, Etot, 'k', lw=1.6, ls=ls, label=label)
        else:
            ax[0, 1].plot(etas, Etot, lw=1.4, ls=ls, label=label)
    ax[0, 1].axhline(0, color='0.6', lw=0.6)
    ax[0, 1].set_xlabel(r'$\eta = h_{s,1} - h_{s,2}$')
    ax[0, 1].set_ylabel(r'$E_{\mathrm{tot}}(\eta) - E_{\mathrm{tot}}(0)$')
    ax[0, 1].set_title(rf'(b)  cation  $E_{{\mathrm{{tot}}}}$ at $h_l = {hl_v}$')
    ax[0, 1].legend(fontsize=9); ax[0, 1].grid(alpha=0.3)
    ax[0, 1].set_ylim(-0.1, 0.15)

    # Panel (c):  charge-induced reorganisation  --  4e vs 3e curvature
    eps = 0.01
    h_l_scan = np.linspace(-0.95, -0.05, 31)
    curv3 = []; curv4 = []
    for hl_v in h_l_scan:
        E3_0 = Egs(H3_0, H3_c, -1, -1, hl_v, 0)
        E3_p = Egs(H3_0, H3_c, -1 - eps / 2, -1 + eps / 2, hl_v, 0)
        curv3.append(2 * (E3_p - E3_0) / eps**2)
        E4_0 = Egs(H4_0, H4_c, -1, -1, hl_v, 0)
        E4_p = Egs(H4_0, H4_c, -1 - eps / 2, -1 + eps / 2, hl_v, 0)
        curv4.append(2 * (E4_p - E4_0) / eps**2)
    curv3 = np.array(curv3); curv4 = np.array(curv4)
    ax[1, 0].plot(np.abs(h_l_scan), curv4, 'o-', color='C0', lw=1.5, ms=4,
                  label='4e (neutral)')
    ax[1, 0].plot(np.abs(h_l_scan), curv3, 's-', color='C3', lw=1.5, ms=4,
                  label='3e (cation)')
    # closed-form prediction
    ax[1, 0].plot(np.abs(h_l_scan), -1 / (8 * np.abs(h_l_scan)), ':', color='C3',
                  lw=1.2, label=r'$-1/(8|h_l|)$  (Eq. 22)')
    ax[1, 0].axhline(0, color='0.6', lw=0.6)
    ax[1, 0].set_xlabel(r'$|h_l| / |h_s|$')
    ax[1, 0].set_ylabel(r'$\partial^2 E_{\mathrm{elec}} / \partial \eta^2$ at $\eta=0$')
    ax[1, 0].set_title('(c)  charge-induced pseudo-JT')
    ax[1, 0].legend(fontsize=9); ax[1, 0].grid(alpha=0.3)
    ax[1, 0].set_ylim(-6, 0.5)

    # Panel (d):  static-MO (frozen psi_4) vs relaxed-MO, + SOMO hole density
    #   Demonstrates that the pseudo-JT curvature is entirely due to the
    #   psi_1 <-> psi_4 mixing under eta.  Freezing the MOs at eta=0 gives a
    #   flat E(eta) and a 50/50 hole distribution for all eta (Class III at
    #   every geometry), while relaxing the MOs produces both the energy
    #   lowering and the progressive hole localisation onto the weaker H-H pair.
    hs_base = -1.0
    hl_v = -0.3
    etas_d = np.linspace(-1.0, 1.0, 201)
    E_stat_const = 3 * hs_base + 2 * hl_v  # 2*eps_1(0) + eps_4(0), eta-independent
    E_relax_d = np.empty_like(etas_d)
    hole_ab = np.empty_like(etas_d)  # hole density on pair (a,b) in the SOMO
    for j, e in enumerate(etas_d):
        # eta = h_{s,1} - h_{s,2}: e>0 makes |h_s1| smaller -> weaker a-b -> hole on (a,b)
        hs1 = hs_base + e / 2
        hs2 = hs_base - e / 2
        H4_1e = np.array([[0, hs1, hl_v, hl_v],
                          [hs1, 0, hl_v, hl_v],
                          [hl_v, hl_v, 0, hs2],
                          [hl_v, hl_v, hs2, 0]])
        w, V = np.linalg.eigh(H4_1e)
        E_relax_d[j] = 2 * w[0] + w[1]     # doubly occupy lowest, singly occupy next
        somo = V[:, 1]                      # SOMO = 2nd lowest MO
        hole_ab[j] = somo[0]**2 + somo[1]**2

    ax_d = ax[1, 1]
    ax_dr = ax_d.twinx()
    ln1, = ax_d.plot(etas_d, E_relax_d, 'C0-', lw=1.8,
                     label=r'relaxed MO ($\psi_4 \to \psi_4\'(\eta)$)')
    ln2, = ax_d.plot(etas_d, np.full_like(etas_d, E_stat_const), 'C3--', lw=1.4,
                     label=r'frozen $\psi_4$ (Aufbau at $\eta=0$)')
    ln3, = ax_dr.plot(etas_d, hole_ab, color='C2', ls=':', lw=2.0,
                      label='hole on pair $(a,b)$')
    ax_dr.axhline(0.5, color='0.7', lw=0.6, ls=':')
    ax_dr.set_ylim(0, 1)
    ax_d.set_xlabel(r'$\eta = h_{s,1} - h_{s,2}$')
    ax_d.set_ylabel(r'$E_{\mathrm{elec}}(\eta) / |h_s|$', color='0.2')
    ax_dr.set_ylabel(r'hole density on $(a,b)$', color='C2')
    ax_dr.tick_params(axis='y', labelcolor='C2')
    ax_d.set_title(rf'(d)  static vs relaxed MO at $h_l = {hl_v}$')
    ax_d.legend(handles=[ln1, ln2, ln3], fontsize=8, loc='lower center')
    ax_d.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig6_disphenoid.pdf'))
    plt.savefig(os.path.join(OUTDIR, 'fig6_disphenoid.png'), dpi=150)
    plt.close()
    print('  fig6_disphenoid.pdf / .png')


# =======================================================================
# Scheme 1 - vbt3 package architecture and data flow (Methods §3.1)
# =======================================================================
def scheme_1_pipeline():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    ax.set_xlim(-4, 100)
    ax.set_ylim(0, 100)
    ax.set_axis_off()

    # Color palette
    c_input = '#eef2f7'
    c_core  = '#d8e3f0'
    c_sym   = '#e8f0d8'
    c_out   = '#f5e5d8'
    c_edge  = '#3b4a5a'
    c_label = '#6a7380'

    def box(x, y, w, h, text, fc, fontsize=10, weight='normal', zorder=2):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle='round,pad=0.4,rounding_size=1.2',
                               linewidth=1.0, edgecolor=c_edge,
                               facecolor=fc, zorder=zorder)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, zorder=zorder + 1)

    def arrow(x1, y1, x2, y2, style='-|>', lw=1.1, zorder=1):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, lw=lw,
                            color=c_edge, mutation_scale=11, zorder=zorder)
        ax.add_patch(a)

    def layer_label(y, text):
        ax.text(15, y, text, ha='right', va='center',
                fontsize=9, color=c_label, style='italic')

    # Row y-coordinates (top to bottom)
    y5, y4, y3, y2, y1 = 88, 71, 50, 29, 10
    row_h = 10

    # --- Row 5: user inputs ---
    layer_label(y5 + row_h / 2, 'user input')
    box(20, y5, 36, row_h,
        "integral rules\n"
        r"$\mathtt{Molecule(subst,\ subst\_2e,}$" + "\n"
        r"$\mathtt{interacting\_orbs,\ max\_2e\_centers)}$",
        c_input, fontsize=8.5)
    box(62, y5, 26, row_h,
        "electron / orbital counts\n"
        r"$(n_\alpha,\ n_\beta,\ n_{ao})$",
        c_input, fontsize=9)

    # --- Row 4: slaterdet / fixed_psi ---
    layer_label(y4 + row_h / 2, 'vbt3.slaterdet /\nvbt3.fixed_psi')
    box(20, y4, 68, row_h,
        r"$\mathtt{generate\_dets(n_\alpha,\ n_\beta,\ n_{ao})}$   $\longrightarrow$   "
        r"$\mathtt{FixedPsi}$  basis of $N_D$ determinants",
        c_core, fontsize=10, weight='bold')

    # --- Row 3: molecule (matrix assembly) ---
    layer_label(y3 + row_h / 2, 'vbt3.molecule')
    box(20, y3, 30, row_h,
        r"$\mathtt{build\_matrix(P,\,op)}$" + "\n"
        "1e fast path:  Eq. (5)",
        c_core, fontsize=9)
    box(58, y3, 30, row_h,
        r"$\mathtt{o2\_matrix(P)}$" + "\n"
        u"2e Löwdin cofactors:  Eq. (3a)",
        c_core, fontsize=9)

    # --- Row 2: symmetry / spin projections ---
    layer_label(y2 + row_h / 2, 'vbt3.symmetry /\nvbt3.spin')
    box(19, y2, 22, row_h,
        r"$\mathtt{totally\_symmetric}$" + "\n"
        r"$D_n$ / $A_{1g}$  projection",
        c_sym, fontsize=9)
    box(43, y2, 22, row_h,
        r"$\mathtt{s\_squared\_matrix}$" + "\n"
        r"total-spin  $S(S{+}1)$",
        c_sym, fontsize=9)
    box(67, y2, 22, row_h,
        r"$\mathtt{eta\_squared\_matrix}$" + "\n"
        r"pseudospin  $\eta(\eta{+}1)$",
        c_sym, fontsize=9)

    # --- Row 1: substitute / solve ---
    layer_label(y1 + row_h / 2, 'substitute & solve')
    box(20, y1, 68, row_h,
        r"SymPy $\to$ closed-form $E(h,s,U,J,K,\ldots)$   or   "
        r"NumPy $\to$ numerical scan",
        c_out, fontsize=10, weight='bold')

    # --- Arrows between rows ---
    # row5 -> row4
    arrow(38, y5, 38, y4 + row_h)
    arrow(75, y5, 75, y4 + row_h)
    # row4 -> row3 (split into 1e / 2e)
    arrow(42, y4, 35, y3 + row_h)
    arrow(65, y4, 73, y3 + row_h)
    # row3 -> row2 (combine 1e+2e symbolic (H,S) into the symmetry layer)
    # Intermediate "symbolic (H, S, H^{2e})" label floating between rows 3 and 2
    ax.text(54, (y3 + y2 + row_h) / 2 - 0.5,
            r"symbolic   $H^{1e},\ S,\ H^{2e} \in \mathrm{SymPy}$",
            ha='center', va='center', fontsize=10,
            color=c_edge, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35',
                      facecolor='white', edgecolor=c_edge, linewidth=0.8))
    arrow(35, y3, 30, y2 + row_h)
    arrow(54, y3, 54, y2 + row_h)
    arrow(73, y3, 78, y2 + row_h)
    # row2 -> row1
    arrow(30, y2, 40, y1 + row_h)
    arrow(54, y2, 54, y1 + row_h)
    arrow(78, y2, 68, y1 + row_h)

    ax.set_title('Scheme 1.  vbt3 package architecture and data flow',
                 fontsize=12, loc='left', pad=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'scheme1_pipeline.pdf'))
    plt.savefig(os.path.join(OUTDIR, 'scheme1_pipeline.png'), dpi=150)
    plt.close()
    print('  scheme1_pipeline.pdf / .png')


# =======================================================================
# Scheme 2 - systems studied (§1 preamble, orientation)
# =======================================================================
def _atom(ax, x, y, label, r=0.30, fs=9, fill='white'):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor='#2a3644',
                        linewidth=1.0, zorder=3))
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fs, zorder=4)


def _bond(ax, x1, y1, x2, y2, order=1, dashed=False,
          color='#2a3644', r=0.30, lw=1.2, gap=0.14):
    """Draw a bond between two atom centres, starting/ending just outside
    the atom circles (radius r)."""
    import numpy as np
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    # start/end offset
    xs, ys = x1 + r * ux, y1 + r * uy
    xe, ye = x2 - r * ux, y2 - r * uy
    ls = (0, (3, 2)) if dashed else '-'
    if order == 1:
        ax.plot([xs, xe], [ys, ye], color=color, lw=lw,
                linestyle=ls, zorder=2, solid_capstyle='round')
    elif order == 2:
        # perpendicular offset for double bond
        px, py = -uy * gap, ux * gap
        ax.plot([xs + px, xe + px], [ys + py, ye + py],
                color=color, lw=lw, linestyle=ls, zorder=2,
                solid_capstyle='round')
        ax.plot([xs - px, xe - px], [ys - py, ye - py],
                color=color, lw=lw, linestyle=ls, zorder=2,
                solid_capstyle='round')


def _panel_frame(ax, x0, y0, w, h, title):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                 boxstyle='round,pad=0.4,rounding_size=0.6',
                 linewidth=0.8, edgecolor='#9aa6b3',
                 facecolor='#fafbfc', zorder=0))
    ax.text(x0 + 0.5, y0 + h - 0.6, title,
            ha='left', va='top', fontsize=10, fontweight='bold',
            color='#2a3644', zorder=1)


def scheme_2_systems():
    """Schematic overview of every system treated in §4."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(10.0, 6.8))
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 24)
    ax.set_aspect('equal')
    ax.set_axis_off()

    # ---- (a) H2 dimer -----------------------------------------------------
    _panel_frame(ax, 0.5, 16.5, 10, 7, '(a) H₂  (§4.6.1)')
    _atom(ax, 3.2, 19.2, 'a')
    _atom(ax, 7.8, 19.2, 'b')
    _bond(ax, 3.2, 19.2, 7.8, 19.2, order=1)
    ax.text(5.5, 20.8, r'$h,\ s,\ U,\ J,\ K$',
            ha='center', fontsize=9, color='#4a5668')
    ax.text(5.5, 17.8, '2 orbitals,  2 electrons',
            ha='center', fontsize=9, style='italic', color='#6a7380')

    # ---- (b) Allyl (3c4e) -------------------------------------------------
    _panel_frame(ax, 11.5, 16.5, 11, 7, '(b) Allyl anion (3c4e)  (§4.6.2)')
    _atom(ax, 14.0, 19.2, 'a')
    _atom(ax, 17.0, 19.2, 'b')
    _atom(ax, 20.0, 19.2, 'c')
    _bond(ax, 14.0, 19.2, 17.0, 19.2, order=1)
    _bond(ax, 17.0, 19.2, 20.0, 19.2, order=1)
    ax.text(17.0, 20.9, r'$\sigma$-symmetry: $a \leftrightarrow c$',
            ha='center', fontsize=9, color='#4a5668')
    ax.text(17.0, 17.8, '3 orbitals,  4 electrons  (anion)',
            ha='center', fontsize=9, style='italic', color='#6a7380')

    # ---- (c) Benzene — Kekulé + Dewar ------------------------------------
    _panel_frame(ax, 23.5, 16.5, 12, 7, '(c) Benzene  (§4.1–4.5, 4.7)')
    # hexagon centres — compact ring
    cx, cy, R = 29.5, 19.9, 1.6
    hex_xy = []
    labels = list('abcdef')
    for k in range(6):
        theta = np.pi / 2 - k * np.pi / 3          # start from top, clockwise
        hex_xy.append((cx + R * np.cos(theta), cy + R * np.sin(theta)))
    for (x, y), lbl in zip(hex_xy, labels):
        _atom(ax, x, y, lbl, r=0.22, fs=8)
    # Kekulé bond pattern: alternating double/single (a-b double, b-c single, ...)
    for k in range(6):
        x1, y1 = hex_xy[k]
        x2, y2 = hex_xy[(k + 1) % 6]
        _bond(ax, x1, y1, x2, y2,
              order=2 if k % 2 == 0 else 1,
              r=0.22, lw=1.0, gap=0.09)
    ax.text(29.5, 17.2, '6 orbitals,  6 electrons  (400 dets)',
            ha='center', fontsize=9, style='italic', color='#6a7380')

    # ---- (d) (H2)2+ disphenoid -------------------------------------------
    _panel_frame(ax, 0.5, 9.0, 10, 7, r'(d) $(\mathrm{H_2})_2^+$  (§4.6.3)')
    # disphenoid: two H2 pairs with diagonal coupling
    ax2_cx, ax2_cy = 5.5, 12.6
    _atom(ax, ax2_cx - 2.5, ax2_cy + 1.0, 'a', r=0.28)
    _atom(ax, ax2_cx - 2.5, ax2_cy - 1.0, 'b', r=0.28)
    _atom(ax, ax2_cx + 2.5, ax2_cy + 1.0, 'c', r=0.28)
    _atom(ax, ax2_cx + 2.5, ax2_cy - 1.0, 'd', r=0.28)
    # intra-pair strong bonds
    _bond(ax, ax2_cx - 2.5, ax2_cy + 1.0,
          ax2_cx - 2.5, ax2_cy - 1.0, r=0.28, lw=1.6)
    _bond(ax, ax2_cx + 2.5, ax2_cy + 1.0,
          ax2_cx + 2.5, ax2_cy - 1.0, r=0.28, lw=1.6)
    # inter-pair weak coupling (dashed)
    _bond(ax, ax2_cx - 2.5, ax2_cy + 1.0,
          ax2_cx + 2.5, ax2_cy + 1.0,
          r=0.28, dashed=True, lw=0.9, color='#7a8492')
    _bond(ax, ax2_cx - 2.5, ax2_cy - 1.0,
          ax2_cx + 2.5, ax2_cy - 1.0,
          r=0.28, dashed=True, lw=0.9, color='#7a8492')
    ax.text(ax2_cx, ax2_cy - 1.8, r'intra: $h_s$,   inter: $h_l \ll h_s$',
            ha='center', fontsize=9, color='#4a5668')
    ax.text(ax2_cx, ax2_cy - 2.8,
            '4 orbitals,  3 electrons  (Robin–Day)',
            ha='center', fontsize=9, style='italic', color='#6a7380')

    # ---- (e) (H2)n+ chain ------------------------------------------------
    _panel_frame(ax, 11.5, 9.0, 11, 7, r'(e) $(\mathrm{H_2})_n^+$ chain  (§4.6.4–5)')
    # Draw 4 vertical H2 pairs along a horizontal line
    pair_x = [13.5, 16.0, 18.5, 21.0]
    pair_y_top = 13.2
    pair_y_bot = 11.6
    letters = ['a/b', 'c/d', 'e/f', 'g/h']
    for px, lab in zip(pair_x, letters):
        _atom(ax, px, pair_y_top, '', r=0.22, fs=8)
        _atom(ax, px, pair_y_bot, '', r=0.22, fs=8)
        _bond(ax, px, pair_y_top, px, pair_y_bot, r=0.22, lw=1.4)
        ax.text(px, pair_y_bot - 0.9, lab,
                ha='center', fontsize=8, color='#6a7380')
    # inter-pair dashed couplings
    for i in range(3):
        _bond(ax, pair_x[i], pair_y_top, pair_x[i + 1], pair_y_top,
              r=0.22, dashed=True, lw=0.8, color='#7a8492')
        _bond(ax, pair_x[i], pair_y_bot, pair_x[i + 1], pair_y_bot,
              r=0.22, dashed=True, lw=0.8, color='#7a8492')
    ax.text(17.25, 14.2, r'$n-1$ CT coordinates,  Peierls cascade',
            ha='center', fontsize=9, color='#4a5668')
    ax.text(17.25, 10.2,
            r'$2n$ orbitals,  $2n-1$ electrons  ($n=2,3,4$)',
            ha='center', fontsize=9, style='italic', color='#6a7380')

    # ---- (f) benzene + O3 cycloaddition ----------------------------------
    _panel_frame(ax, 23.5, 9.0, 12, 7,
                 r'(f) Benzene + $\mathrm{O_3}$ [3+2]  (§4.7)')
    # Same hexagon as (c), but with O3 hovering above the a-b edge
    cx, cy, R = 28.5, 11.6, 1.3
    hex_xy = []
    for k in range(6):
        theta = np.pi / 2 - k * np.pi / 3
        hex_xy.append((cx + R * np.cos(theta), cy + R * np.sin(theta)))
    for (x, y), lbl in zip(hex_xy, labels):
        _atom(ax, x, y, lbl, r=0.18, fs=7)
    for k in range(6):
        x1, y1 = hex_xy[k]
        x2, y2 = hex_xy[(k + 1) % 6]
        lam_bond_order = 1 if (k == 0) else (2 if k % 2 == 0 else 1)
        dashed_here = (k == 0)                       # a-b edge is the attacked one
        _bond(ax, x1, y1, x2, y2,
              order=lam_bond_order,
              r=0.18, lw=0.9, gap=0.08, dashed=dashed_here,
              color='#7a2828' if dashed_here else '#2a3644')
    # Ozone triangle above
    o_cx, o_cy = 33.0, 12.4
    _atom(ax, o_cx - 0.55, o_cy + 0.35, 'O', r=0.22, fs=7,
          fill='#ffe6c2')
    _atom(ax, o_cx + 0.55, o_cy + 0.35, 'O', r=0.22, fs=7,
          fill='#ffe6c2')
    _atom(ax, o_cx, o_cy - 0.65, 'O', r=0.22, fs=7,
          fill='#ffe6c2')
    _bond(ax, o_cx - 0.55, o_cy + 0.35, o_cx + 0.55, o_cy + 0.35,
          r=0.22, lw=0.9)
    _bond(ax, o_cx - 0.55, o_cy + 0.35, o_cx, o_cy - 0.65,
          r=0.22, lw=0.9)
    _bond(ax, o_cx + 0.55, o_cy + 0.35, o_cx, o_cy - 0.65,
          r=0.22, lw=0.9)
    # reaction arrow
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(
        (hex_xy[0][0] + 0.2, hex_xy[0][1] + 0.2),
        (o_cx - 0.8, o_cy - 0.5),
        arrowstyle='<|-|>', mutation_scale=11, lw=0.9,
        color='#7a2828', zorder=2))
    ax.text((cx + R) / 2 + o_cx / 2 - 3.7, 13.5,
            r'$\lambda = h_{ab}/h$', fontsize=9, color='#7a2828')
    ax.text(29.5, 10.2, 'aromaticity loss along $\\lambda$',
            ha='center', fontsize=9, style='italic', color='#6a7380')

    # ---- bottom orientation strip ----------------------------------------
    ax.text(18, 7.8,
            r'Conventions:  lowercase orbital labels $a,b,c,\ldots$ are '
            r'AOs;  double lines are Kekulé $\pi$-bonds;  '
            r'dashed lines are weak inter-fragment couplings.',
            ha='center', va='center', fontsize=9, color='#4a5668')
    ax.text(18, 6.5,
            r'All systems are minimal-basis, one $\pi$-orbital per atom.  '
            r'vbt3 carries every integral ($h$, $s$, $U$, $J$, $K$, $\ldots$) '
            r'symbolically throughout.',
            ha='center', va='center', fontsize=9, color='#4a5668')

    ax.set_title(r'Scheme 2.  Systems studied in this paper',
                 fontsize=12, loc='left', pad=6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'scheme2_systems.pdf'))
    plt.savefig(os.path.join(OUTDIR, 'scheme2_systems.png'), dpi=150)
    plt.close()
    print('  scheme2_systems.pdf / .png')


# =======================================================================
# Scheme 3 - five covalent Rumer structures of benzene (Appendix A)
# =======================================================================
def scheme_3_rumer():
    """Two Kekulé + three Dewar long-bond structures for benzene."""
    import numpy as np

    fig, axes = plt.subplots(1, 5, figsize=(11.0, 2.7))
    labels = list('abcdef')
    R = 1.0

    # Rumer bond-pair patterns (pairs are indices i < j from 0=a, 1=b, ...)
    rumer = [
        ('Kek$_1$', [(0, 1), (2, 3), (4, 5)]),    # a-b, c-d, e-f
        ('Kek$_2$', [(1, 2), (3, 4), (5, 0)]),    # b-c, d-e, f-a
        ('Dew$_1$', [(0, 3), (1, 2), (4, 5)]),    # a-d long bond
        ('Dew$_2$', [(1, 4), (0, 5), (2, 3)]),    # b-e long bond
        ('Dew$_3$', [(2, 5), (0, 1), (3, 4)]),    # c-f long bond
    ]

    for ax, (title, pairs) in zip(axes, rumer):
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect('equal')
        ax.set_axis_off()

        # Hexagon vertices — start at top, go clockwise
        xs, ys = [], []
        for k in range(6):
            theta = np.pi / 2 - k * np.pi / 3
            xs.append(R * np.cos(theta))
            ys.append(R * np.sin(theta))

        # Faint hexagonal skeleton
        for k in range(6):
            ax.plot([xs[k], xs[(k + 1) % 6]], [ys[k], ys[(k + 1) % 6]],
                    color='#c8d0d8', lw=0.8, zorder=1)

        # Bond pairs drawn as thick curves
        for (i, j) in pairs:
            is_long = (j - i) % 6 == 3         # diametric pair
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                    color='#7a2828' if is_long else '#1f4068',
                    lw=2.2 if is_long else 2.6,
                    linestyle='--' if is_long else '-',
                    zorder=2, solid_capstyle='round')

        # Atom circles + labels
        for k in range(6):
            _atom(ax, xs[k], ys[k], labels[k], r=0.22, fs=9)

        ax.set_title(title, fontsize=11, pad=3)

    plt.suptitle('Scheme 3.  Five covalent Rumer structures of benzene',
                 fontsize=12, y=1.02, x=0.04, ha='left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'scheme3_rumer.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(OUTDIR, 'scheme3_rumer.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  scheme3_rumer.pdf / .png')


# =======================================================================
# TOC graphic  (3.25" x 2"; ACS requirement)
# =======================================================================
def toc_graphic():
    """Graphical abstract (ACS TOC): benzene -> vbt3 -> closed-form + curve.
    Canvas is 3.25" x 1.75", split into three regions plus a title strip."""
    import numpy as np
    from matplotlib.patches import FancyArrowPatch

    fig = plt.figure(figsize=(3.25, 1.75))

    # 1) Title strip
    fig.text(0.5, 0.93,
             'Symbolic valence-bond theory of benzene',
             ha='center', va='top',
             fontsize=7.5, fontweight='bold', color='#2a3644')

    # 2) Left axis: benzene ring
    ax_L = fig.add_axes([0.00, 0.12, 0.30, 0.72])
    ax_L.set_xlim(-1.6, 1.6)
    ax_L.set_ylim(-1.6, 1.6)
    ax_L.set_aspect('equal')
    ax_L.set_axis_off()
    R = 1.1
    hex_xy = [(R * np.cos(np.pi / 2 - k * np.pi / 3),
               R * np.sin(np.pi / 2 - k * np.pi / 3)) for k in range(6)]
    for k in range(6):
        x1, y1 = hex_xy[k]
        x2, y2 = hex_xy[(k + 1) % 6]
        if k % 2 == 0:
            # double bond via twin lines
            dx, dy = x2 - x1, y2 - y1
            L = np.hypot(dx, dy); ux, uy = dx / L, dy / L
            px, py = -uy * 0.09, ux * 0.09
            ax_L.plot([x1 + px, x2 + px], [y1 + py, y2 + py],
                      color='#1f4068', lw=1.1)
            ax_L.plot([x1 - px, x2 - px], [y1 - py, y2 - py],
                      color='#1f4068', lw=1.1)
        else:
            ax_L.plot([x1, x2], [y1, y2], color='#1f4068', lw=1.1)
    for (x, y) in hex_xy:
        ax_L.plot(x, y, 'o', markersize=4.5, markerfacecolor='white',
                  markeredgecolor='#1f4068', markeredgewidth=0.8, zorder=3)
    ax_L.text(0, -1.5, r'benzene $\pi$ (6e / 6o)',
              ha='center', fontsize=6, color='#4a5668')

    # 3) Middle axis: pipeline arrow with vbt3 label + symbolic H
    ax_M = fig.add_axes([0.30, 0.12, 0.28, 0.72])
    ax_M.set_xlim(0, 10)
    ax_M.set_ylim(0, 10)
    ax_M.set_axis_off()
    ax_M.add_patch(FancyArrowPatch((0.5, 5), (9.5, 5),
                   arrowstyle='-|>', mutation_scale=10, lw=1.2,
                   color='#2a3644'))
    ax_M.text(5, 7.2, r'vbt3', fontsize=9, fontweight='bold',
              color='#2a3644', ha='center')
    ax_M.text(5, 6.0, '(SymPy)', fontsize=6,
              style='italic', color='#6a7380', ha='center')
    ax_M.text(5, 3.3,
              r'$H(h,s,U)$',
              fontsize=8.5, fontweight='bold', color='#1f4068',
              ha='center')
    ax_M.text(5, 2.1, r'$400\times 400$  symbolic',
              fontsize=6, color='#4a5668', ha='center')
    ax_M.text(5, 0.9, 'non-orthogonal VB',
              fontsize=6, style='italic', color='#6a7380', ha='center')

    # 4) Right axis: formula + mini plot
    ax_R = fig.add_axes([0.58, 0.12, 0.42, 0.72])
    ax_R.set_axis_off()
    ax_R.set_xlim(0, 10)
    ax_R.set_ylim(0, 10)
    ax_R.text(5, 9.3,
              r'$E/t = -8 + \frac{3}{2}u - \frac{29}{288}u^{2} - \cdots$',
              ha='center', fontsize=6.2, color='#1f4068')
    ax_R.text(5, 8.15, r'$(u = U/t)$',
              ha='center', fontsize=5.5, color='#6a7380')

    # Mini plot of Padé vs Heisenberg asymptote
    ax_in = fig.add_axes([0.65, 0.13, 0.31, 0.48])
    u = np.linspace(0.1, 20, 200)
    # Qualitative [2/4] Padé
    E_pade = -8.0 * (1 + 0.05 * u) / (1 + 0.20 * u + 0.045 * u**2)
    E_heis = -4.0 / u
    ax_in.plot(u, E_pade, color='#1f4068', lw=1.1, label=r'[2/4] Padé')
    ax_in.plot(u, E_heis, color='#7a2828', lw=0.9, ls='--',
               label=r'$-4t^2/U$')
    ax_in.set_xlim(0, 20)
    ax_in.set_ylim(-8.5, 0)
    ax_in.set_xlabel(r'$U/t$', fontsize=5.5, labelpad=0)
    ax_in.set_ylabel(r'$E/t$', fontsize=5.5, labelpad=0)
    ax_in.tick_params(axis='both', labelsize=5, length=2, pad=1)
    ax_in.legend(fontsize=4.5, frameon=False, loc='lower right',
                 handlelength=1.3, handletextpad=0.3)
    for sp in ax_in.spines.values():
        sp.set_linewidth(0.5)

    plt.savefig(os.path.join(OUTDIR, 'toc_graphic.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(OUTDIR, 'toc_graphic.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('  toc_graphic.pdf / .png')


# =======================================================================
if __name__ == '__main__':
    print(f'writing figures to {OUTDIR}/ ...')
    scheme_1_pipeline()
    scheme_2_systems()
    scheme_3_rumer()
    toc_graphic()
    figure_1_h2()
    figure_2_allyl()
    figure_3_degeneracy()
    figure_4_pade()
    figure_5_aromaticity()
    figure_6_disphenoid()
    print('done.')
