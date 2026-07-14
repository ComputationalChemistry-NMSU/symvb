"""Figure generation for manuscript_v2.md (Symbolic Valence Bond Theory
for Chemists: The symvb Package).

Manuscript display items produced HERE (file names follow the manuscript
labels since the 2026-07-08 figures/ reorganization):
  scheme1_pipeline.{pdf,png}   Scheme 1   (scheme_1_pipeline)
  fig1_h2.{pdf,png}            Figure 1   (figure_1_h2)
  fig3_disphenoid.{pdf,png}    Figure 3   (figure_3_disphenoid)
  fig5_aromaticity.{pdf,png}   Figure 5   (figure_5_aromaticity)
  toc_graphic.{pdf,png}        TOC graphic (toc_graphic)

Produced ELSEWHERE:
  fig2_allyl_long_bond         Figure 2   (examples/allyl_long_bond_vb.py)
  fig4_benzene_ionicity        Figure 4   (examples/make_fig_benzene_ionicity.py)

Legacy outputs not used by manuscript_v2 (kept for the archived v1) write
to figures/archive/: fig4_allyl, fig1_degeneracy, fig2_pade,
scheme1a_object_model, scheme2_systems, scheme3_rumer.

Output dir: ../../vbt-3/figures/. Figures 2-5 are drawn at final journal
dimensions with 7-9 pt fonts per manuscript_style_guide.md section 9.
"""
import os
import sys
import pickle
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, SlaterDet, FixedPsi, symmetry
from symvb.fixed_psi import generate_dets

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'vbt-3', 'figures')
os.makedirs(OUTDIR, exist_ok=True)
# legacy display items (archived manuscript v1) live one level down
ARCHIVE_DIR = os.path.join(OUTDIR, 'archive')
os.makedirs(ARCHIVE_DIR, exist_ok=True)
# fonttype 42 = embed TrueType so PDF text stays editable text (not Type-3
# outlines) in Illustrator/Inkscape; the matplotlib default (Type 3) is not.
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


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

    # singlet-triplet gap E_T - E_S (eq 7) at s = 0 and s = 0.25
    def E_sing_num(Uv, sv, tt=1.0):
        hh = -tt
        R = np.sqrt(Uv**2 * (1 + sv**2)**2 + 16 * Uv * hh * sv * (sv**2 - 1)
                    + 16 * hh**2 * (sv**2 - 1)**2)
        return (Uv * (1 + sv**2) + 4 * hh * sv * (sv**2 - 1) - R) / (2 * (1 - sv**2)**2)
    E_trip = lambda sv, tt=1.0: 2 * tt * sv / (1 - sv**2)
    gap_s0 = E_trip(0.0) - E_sing_num(U_vals, 0.0)
    gap_s25 = E_trip(0.25) - E_sing_num(U_vals, 0.25)
    direct25 = 4 * t * 0.25 / (1 - 0.25**4)        # 4ts/(1-s^4) plateau, ~1.004 t
    U_large = np.linspace(5, 20, 100)

    plt.rcParams.update({'font.size': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7})
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
    # bold standalone panel letters + Title-Case centred titles (user layout,
    # matched to the 2026-07-08 touch-up of fig1_h2)
    for a, letter, title in zip(ax, 'ABC', ['Ground-State Energy',
                                            'VB Character of the Bond',
                                            'Singlet–Triplet Gap']):
        a.set_title(title, fontsize=9)
        a.text(-0.22, 1.02, letter, transform=a.transAxes, fontsize=10,
               fontweight='bold', va='bottom', ha='left')

    ax[0].plot(U_vals, E_gs, 'k-', lw=1.6)
    ax[0].axhline(-2, ls=':', c='C0', lw=1)
    ax[0].plot(U_large, -4 / U_large, ls='--', c='C3', lw=1)
    ax[0].set_ylim(-2.12, -0.02)
    ax[0].annotate(r'$-2t$ (Hückel)', xy=(9, -2), xytext=(4.0, -1.58),
                   color='C0', fontsize=7.5,
                   arrowprops=dict(arrowstyle='->', color='C0', lw=0.7))
    ax[0].annotate(r'$-4t^2/U$ (superexch.)', xy=(13, -4 / 13), xytext=(4.0, -1.12),
                   color='C3', fontsize=7.5,
                   arrowprops=dict(arrowstyle='->', color='C3', lw=0.7))
    ax[0].set_xlabel(r'$U/t$'); ax[0].set_ylabel(r'$E_{\rm gs} / t$')
    ax[0].grid(alpha=0.3)

    ax[1].plot(U_vals, w_cov, 'C0-', lw=1.6, label=r'$w_{\rm cov}$')
    ax[1].plot(U_vals, w_ion, 'C3-', lw=1.6, label=r'$w_{\rm ion}$')
    ax[1].axhline(0.5, ls=':', c='k', lw=0.8)
    ax[1].axvline(4, ls=':', c='k', lw=0.8)
    ax[1].text(19.6, 0.53, 'Hückel 50/50', fontsize=7.5, ha='right')
    ax[1].text(3.6, 0.92, r'$4t$', fontsize=7.5, ha='right')
    ax[1].set_xlabel(r'$U/t$'); ax[1].set_ylabel('Weight')
    ax[1].text(16.8, 0.88, r'$w_{\rm cov}$', color='C0', fontsize=8)
    ax[1].text(16.8, 0.085, r'$w_{\rm ion}$', color='C3', fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[1].set_ylim(-0.05, 1.05)

    ax[2].plot(U_vals, gap_s0, 'C0-', lw=1.6, label=r'$s=0$')
    ax[2].plot(U_vals, gap_s25, 'C3-', lw=1.6, label=r'$s=0.25$')
    ax[2].plot(U_large, 4 / U_large, ls=':', c='C0', lw=1)
    ax[2].axhline(direct25, ls='--', c='C3', lw=0.9)
    ax[2].text(6.3, 0.70, r'$4t^2/U$', color='C0', fontsize=7.5)
    ax[2].text(4.6, 1.06, r'$4ts/(1-s^4)$', color='C3', fontsize=7.5)
    ax[2].set_xlabel(r'$U/t$'); ax[2].set_ylabel(r'$(E_{\rm T}-E_{\rm S}) / t$')
    ax[2].text(19.5, 1.29, r'$s=0.25$', color='C3', fontsize=8, ha='right')
    ax[2].text(19.5, 0.40, r'$s=0$', color='C0', fontsize=8, ha='right')
    ax[2].grid(alpha=0.3)
    ax[2].set_ylim(0, 2.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig1_h2.pdf'))
    plt.savefig(os.path.join(OUTDIR, 'fig1_h2.png'), dpi=150)
    plt.close()
    print('  fig1_h2.pdf')


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
    plt.savefig(os.path.join(ARCHIVE_DIR, 'fig4_allyl.pdf'))
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
    plt.savefig(os.path.join(ARCHIVE_DIR, 'fig1_degeneracy.pdf'))
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
    plt.savefig(os.path.join(ARCHIVE_DIR, 'fig2_pade.pdf'))
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
    h, s = sp.symbols('h s')
    Hab = sp.Symbol('H_ab')
    Usym = sp.Symbol('U')
    H_VAL, S_VAL = -1.0, 0.2
    lambdas = np.linspace(1.0, 0.0, 161)

    # covalent-only five-structure model, solved at arbitrary overlap sv
    fHc = sp.lambdify((h, s, Hab), m.build_matrix(rumer, op='H'), 'numpy')
    fSc = sp.lambdify((h, s, Hab), m.build_matrix(rumer, op='S'), 'numpy')

    def cov_solve(sv):
        E = np.zeros(len(lambdas))
        W = np.zeros((5, len(lambdas)))
        for i, lam in enumerate(lambdas):
            Hn = np.asarray(fHc(H_VAL, sv, lam * H_VAL), float)
            Sn = np.asarray(fSc(H_VAL, sv, lam * H_VAL), float)
            ev, vec = eigh(Hn, Sn)
            E[i] = ev[0]
            c0 = vec[:, 0]
            W[:, i] = c0 * (Sn @ c0)
        return E, W
    E_cov, _ = cov_solve(S_VAL)
    resp_cov = -np.gradient(E_cov, lambdas)

    # --- full 400-det FCI:  H = H1(H_ab) + U * H2(on-site) ------------------
    # H1 is the one-electron Hamiltonian (op='H'); the on-site Hubbard U lives
    # in the separate two-electron matrix H2 (o2_matrix). H is linear in both
    # H_ab and U, so a few numerical builds span the (lambda, U) plane.
    CACHE_H1 = '/tmp/benzene_full_aromaticity_HS.pkl'
    CACHE_H2 = '/tmp/benzene_aromaticity_H2_onsite.pkl'
    if os.path.exists(CACHE_H1):
        with open(CACHE_H1, 'rb') as f:
            H1_sym, S_sym = pickle.load(f)
    else:
        mf = Molecule(
            zero_ii=True,
            subst={'s': ('S_ab','S_bc','S_cd','S_de','S_ef','S_af'),
                   'h': ('H_bc','H_cd','H_de','H_ef','H_af')},
            interacting_orbs=['ab','bc','cd','de','ef','af'])
        mf.generate_basis(3, 3, 6)
        H1_sym = mf.build_matrix(mf.basis, op='H')
        S_sym = mf.build_matrix(mf.basis, op='S')
        with open(CACHE_H1, 'wb') as f:
            pickle.dump((H1_sym, S_sym), f)
    if os.path.exists(CACHE_H2):
        with open(CACHE_H2, 'rb') as f:
            H2_sym = pickle.load(f)
    else:
        mu = Molecule(
            zero_ii=True,
            subst={'s': ('S_ab','S_bc','S_cd','S_de','S_ef','S_af'),
                   'h': ('H_bc','H_cd','H_de','H_ef','H_af')},
            subst_2e={'U': ('1111',)}, max_2e_centers=1,
            interacting_orbs=['ab','bc','cd','de','ef','af'])
        mu.generate_basis(3, 3, 6)
        H2_sym = mu.o2_matrix(mu.basis)
        with open(CACHE_H2, 'wb') as f:
            pickle.dump(H2_sym, f)

    A0 = np.array(H1_sym.subs({h: H_VAL, s: S_VAL, Hab: 0.0}).tolist(), float)
    dA = np.array(H1_sym.subs({h: H_VAL, s: S_VAL, Hab: H_VAL}).tolist(), float) - A0
    BU = np.array(H2_sym.subs({s: S_VAL, Usym: 1.0}).tolist(), float)   # per unit U
    S_f = np.array(S_sym.subs({s: S_VAL}).tolist(), float)

    def fci_E(U):
        return np.array([eigh(A0 + lam * dA + U * BU, S_f, eigvals_only=True,
                              subset_by_index=[0, 0])[0] for lam in lambdas])
    U_LIST = [0, 8, 16, 64]                 # bump onsets in the FCI at U/|h| ~ 8.25
    efci = {U: fci_E(U) for U in U_LIST}
    E_fci = efci[0]
    resp_fci = {U: -np.gradient(efci[U], lambdas) for U in U_LIST}

    plt.rcParams.update({'font.size': 8})
    # 7.0 in = ACS double-column width, so fonts print at their true pt sizes
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.85))
    cov_c, fci_c = '#c0504d', '#1a1a1a'

    # (A) energies: covalent-only sits well above FCI; the vertical gap is the
    #     covalent-ionic resonance energy. Inset: the covalent-only bump,
    #     E_cov(lambda) - E_cov(0), is absent at s=0 and grows with overlap s.
    axA = ax[0]
    axA.plot(lambdas, E_fci, '-', color=fci_c, lw=1.8)
    axA.plot(lambdas, E_cov, '-', color=cov_c, lw=1.8)
    axA.set_xlabel(r'$\lambda = h_{ab}/h$')
    axA.set_ylabel(r'Energy / $|h|$')
    axA.set_title('Energies', fontsize=9)
    axA.text(-0.22, 1.02, 'A', transform=axA.transAxes, fontsize=10,
             fontweight='bold', va='bottom', ha='left')
    axA.invert_xaxis(); axA.grid(alpha=0.3)
    # both curves labelled directly (no legend); the inset sits in the empty
    # band between them
    axA.text(0.97, -1.25, 'covalent-only', color=cov_c, fontsize=7.5,
             ha='left', va='top')
    axA.text(0.97, -5.55, 'FCI ($U=0$)', color=fci_c, fontsize=7.5,
             ha='left', va='top')
    axin = axA.inset_axes([0.26, 0.24, 0.52, 0.44])
    for sv, col in zip([0.0, 0.2, 0.4], ['#9a9a9a', '#5b9bd5', '#1f5fa8']):
        Ev = E_cov if sv == S_VAL else cov_solve(sv)[0]
        dE = Ev - Ev[-1]
        axin.plot(lambdas, dE, '-', color=col, lw=1.3)
        if np.ptp(Ev) > 1e-9:   # ndarray.ptp() method removed in NumPy 2.0
            axin.plot(lambdas[np.argmax(Ev)], dE.max(), 'o', color=col, ms=2.5)
    axin.axhline(0, color='0.6', lw=0.6); axin.invert_xaxis()
    axin.set_xlim(0.62, 0.0); axin.set_ylim(-0.006, 0.033)
    axin.set_xticks([]); axin.set_yticks([])
    # the bump-vs-overlap explanation lives in the caption; every curve gets
    # a direct label, each inside its own parabola (clear of all lines)
    axin.text(0.60, 0.03, '$s=0$', transform=axin.transAxes, fontsize=6.0, color='0.5')
    axin.text(0.52, 0.90, '0.4', transform=axin.transAxes, fontsize=6.0,
              ha='center', color='#1f5fa8')
    axin.text(0.52, 0.38, '0.2', transform=axin.transAxes, fontsize=6.0,
              ha='center', color='#5b9bd5')
    for _spine in axin.spines.values():
        _spine.set_color('0.6'); _spine.set_linewidth(0.6)
    axin.patch.set_alpha(0.9)

    # (B) bond-formation response -dE/dlam = pi bond order. Positive (correct
    #     sign) for the FCI at small U; turns negative near lambda=0 only above
    #     U/|h| ~ 8.25, converging to the covalent-only (U -> infinity) limit.
    axB = ax[1]
    axB.axhline(0, color='0.55', lw=0.8, zorder=1)
    cols = plt.cm.viridis(np.linspace(0.05, 0.78, len(U_LIST)))
    for U, c in zip(U_LIST, cols):
        axB.plot(lambdas, resp_fci[U], '-', color=c, lw=1.5, label=f'$U={U}$')
    axB.plot(lambdas, resp_cov, '--', color=cov_c, lw=1.6,
             label=r'cov ($U\!\to\!\infty$)')
    axB.fill_between(lambdas, resp_cov, 0, where=(resp_cov < 0),
                     color=cov_c, alpha=0.13, zorder=0)
    axB.set_xlabel(r'$\lambda$')
    axB.set_ylabel(r'$-\,dE/d\lambda$  (bond order) / $|h|$', fontsize=7.5)
    axB.set_title('Response vs Correlation $U$', fontsize=9)
    axB.text(-0.22, 1.02, 'B', transform=axB.transAxes, fontsize=10,
             fontweight='bold', va='bottom', ha='left')
    axB.invert_xaxis(); axB.grid(alpha=0.3)
    axB.legend(fontsize=6.5, ncol=2, loc='upper right', frameon=False)
    axB.set_ylim(-0.18, 1.32)   # bottom: room for the note; top: legend rows
                                # sit above the U=0 curve (text is invisible
                                # to autoscale)
    axB.text(0.22, -0.115, 'wrong sign', fontsize=6.3,
             color='0.25', ha='center', va='center')

    fig.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig5_aromaticity.pdf'))
    plt.savefig(os.path.join(OUTDIR, 'fig5_aromaticity.png'), dpi=450)
    plt.close()
    print('  fig5_aromaticity.pdf')


# =======================================================================
# Figure 6 - (H2)2+ disphenoid: pseudo-JT instability and III <-> II
# =======================================================================
def figure_3_disphenoid():
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

    # Two-panel figure 3 (fixed framework reorganization energy lambda = 1).
    # A: lower adiabatic surface along X = eta/lambda for three inter-pair couplings
    #    |h_l| at FIXED lambda -- a single well (hole shared, Class III, large |h_l| /
    #    short inter-pair distance) through the flat critical case |h_l| = lambda/4 to a
    #    double well (hole trapped, Class II, small |h_l| / long distance). The diabatic
    #    parabolas are fixed; only the avoided-crossing dip (set by |h_l|) changes.
    # B: Robin-Day phase boundary |h_l|_crit vs U for the 3-electron cation. |h_l|_crit
    #    = lambda/4 at U = 0 and falls with U: correlation keeps the hole shared down to
    #    smaller |h_l| (longer distance), widening the Class III region.
    plt.rcParams.update({'font.size': 8})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.85))

    # --- Panel A: Marcus-Hush surfaces (pure analytic schematic) ---
    lam = 1.0
    Xg = np.linspace(-1.7, 1.7, 600)
    eA = (lam / 4) * (1 + Xg) ** 2
    eB = (lam / 4) * (1 - Xg) ** 2
    meanX = (eA + eB) / 2
    halfX = np.abs(eA - eB) / 2
    axA.plot(Xg, eA, '--', color='0.72', lw=0.9)
    axA.plot(Xg, eB, '--', color='0.72', lw=0.9)
    # coupling t = 2|h_l|; |h_l| = lambda/4 = 0.25 is the critical (flat-bottom) case
    A_cases = [(0.13, '#9e4b4b', 'trapped (II)'),
               (0.50, '0.35',    'critical'),
               (0.90, '#3f7d5f', 'shared (III)')]
    for _t, _col, _lab in A_cases:
        EmA = meanX - np.sqrt(halfX ** 2 + _t ** 2)
        axA.plot(Xg, EmA, '-', color=_col, lw=2.0)
        _mins = np.where((EmA[1:-1] < EmA[:-2]) & (EmA[1:-1] <= EmA[2:]))[0] + 1
        axA.plot(Xg[_mins], EmA[_mins], 'o', color=_col, ms=4, zorder=5)
    axA.axvline(0, color='0.9', lw=0.7, zorder=0)
    # curve labels placed in analytically clear windows (no legend: the box
    # cannot avoid the diabat/adiabat crossings; conditions are in the caption)
    axA.text(-1.45, 0.18, 'trapped (II)', color='#9e4b4b', fontsize=7)
    axA.text(0.35, -0.185, 'critical', color='0.35', fontsize=7)
    axA.text(-0.55, -0.52, 'shared (III)', color='#3f7d5f', fontsize=7)
    axA.text(0.35, 0.92, 'diabats', color='0.55', fontsize=7)
    axA.set_xlabel(r'distortion  $X = \eta/\lambda$')
    axA.set_ylabel(r'Energy  (units of $\lambda$)')
    axA.set_title('Lower Adiabatic Surface', fontsize=9)
    axA.set_xlim(-1.7, 1.7); axA.set_ylim(-0.80, 1.20)
    axA.set_xticks([-1, 0, 1])
    axA.text(-0.17, 1.02, 'A', transform=axA.transAxes, fontsize=10, fontweight='bold', va='bottom')

    # --- Panel B: Robin-Day phase boundary |h_l|_crit(U) at fixed lambda = 1 ---
    eps = 0.01
    lam_frame = 1.0

    def _curv(H0c, coefs, hl_v, Uv):
        E0 = Egs(H0c, coefs, -1.0, -1.0, hl_v, Uv)
        Ep = Egs(H0c, coefs, -1.0 - eps / 2, -1.0 + eps / 2, hl_v, Uv)
        return 2 * (Ep - E0) / eps ** 2

    def _hlcrit(Uv, lam):
        # solve 1/(2 lam) = k(|h_l|, U) = -curvature; k is decreasing in |h_l|
        target = 1.0 / (2.0 * lam)
        f = lambda ahl: (-_curv(H3_0, H3_c, -ahl, Uv)) - target
        lo, hi = 0.004, 0.46
        flo = f(lo)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if flo * f(mid) <= 0:
                hi = mid
            else:
                lo, flo = mid, f(mid)
        return 0.5 * (lo + hi)

    U_grid = np.linspace(0.0, 6.0, 49)
    hlc_curve = np.array([_hlcrit(U, lam_frame) for U in U_grid])
    yhi = 0.30

    axB.fill_between(U_grid, hlc_curve, yhi, color='#7fbf9f', alpha=0.16, zorder=0)
    axB.fill_between(U_grid, 0.0, hlc_curve, color='#d98c8c', alpha=0.16, zorder=0)
    axB.plot(U_grid, hlc_curve, '-', color='0.15', lw=2.1, zorder=4)

    hlc0 = lam_frame / 4   # U = 0: |h_l|_crit = lambda/4 (exact)
    axB.plot([0.0], [hlc0], 'o', color='0.15', ms=4.5, zorder=5)
    axB.text(0.15, 0.253, r'$|h_l|_{\mathrm{crit}} = \lambda/4$',
             fontsize=8, color='0.15', ha='left', va='bottom')

    axB.text(0.75, 0.42, 'delocalized (Class III)\nlarge $|h_l|$ (short $R$)', fontsize=7,
             color='#2f6f4f', ha='center', va='center', transform=axB.transAxes)
    axB.text(0.29, 0.16, 'localized (Class II)\nsmall $|h_l|$ (long $R$)', fontsize=7,
             color='#9e4b4b', ha='center', va='center', transform=axB.transAxes)

    axB.set_xlabel(r'on-site repulsion  $U / |h_s|$')
    axB.set_ylabel(r'Critical coupling  $|h_l|_{\mathrm{crit}} / |h_s|$')
    axB.set_title('Robin–Day Phase Boundary', fontsize=9)
    axB.set_xlim(0.0, 6.0); axB.set_ylim(0.0, yhi)
    axB.grid(alpha=0.3)
    axB.text(-0.17, 1.02, 'B', transform=axB.transAxes, fontsize=10, fontweight='bold', va='bottom')

    # 3D structure inset on panel B (depth-cued): disphenoid, bold short H2 edges (h_s),
    # four equivalent long inter-pair edges (h_l).
    _ibox = [0.66, 0.52, 0.33, 0.43]
    _bg = axB.inset_axes(_ibox)
    _bg.set_xticks([]); _bg.set_yticks([])
    for _sp in _bg.spines.values():
        _sp.set_visible(False)
    _bg.patch.set_facecolor('white'); _bg.patch.set_alpha(0.9)
    axin = axB.inset_axes(_ibox, projection='3d')
    axin.set_axis_off(); axin.patch.set_alpha(0.0)
    _xyz = {'a': (0.85, 0.0, 0.9), 'b': (-0.85, 0.0, 0.9),
            'c': (0.0, 0.85, -0.9), 'd': (0.0, -0.85, -0.9)}
    _az, _el = np.radians(-58), np.radians(18)
    _cam = np.array([np.cos(_el) * np.cos(_az), np.cos(_el) * np.sin(_az), np.sin(_el)])
    _dep = {_k: float(np.dot(np.array(_v), _cam)) for _k, _v in _xyz.items()}
    _dlo, _dhi = min(_dep.values()), max(_dep.values())
    def _front(*_ks):
        return (sum(_dep[_k] for _k in _ks) / len(_ks) - _dlo) / (_dhi - _dlo + 1e-9)
    for _u, _v in sorted([('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd')], key=lambda e2: _front(*e2)):
        _ff = _front(_u, _v); _p, _q = _xyz[_u], _xyz[_v]
        axin.plot([_p[0], _q[0]], [_p[1], _q[1]], [_p[2], _q[2]], '-',
                  lw=0.6 + 0.9 * _ff, color=str(0.74 - 0.44 * _ff), alpha=0.7 + 0.3 * _ff)
    for _u, _v in sorted([('a', 'b'), ('c', 'd')], key=lambda e2: _front(*e2)):
        _ff = _front(_u, _v); _p, _q = _xyz[_u], _xyz[_v]
        axin.plot([_p[0], _q[0]], [_p[1], _q[1]], [_p[2], _q[2]], '-',
                  lw=2.0 + 1.4 * _ff, color=str(0.42 - 0.36 * _ff), alpha=0.85 + 0.15 * _ff)
    for _lab in sorted(_xyz, key=lambda k: _dep[k]):
        _ff = _front(_lab); _p = _xyz[_lab]
        axin.scatter([_p[0]], [_p[1]], [_p[2]], s=80 + 75 * _ff, color='white',
                     edgecolors=str(0.42 - 0.34 * _ff), linewidths=0.7 + 0.6 * _ff, depthshade=False)
        axin.text(_p[0], _p[1], _p[2], _lab, fontsize=6.5, color='0.1', ha='center', va='center')
    axin.text(0.0, 0.0, 1.32, r'$h_s$', fontsize=7, color='0.12', ha='center')
    axin.text(0.62, 0.62, 0.05, r'$h_l$', fontsize=7, color='#7a8492')
    axin.view_init(elev=18, azim=-58)
    try:
        axin.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    axin.set_xlim(-1, 1); axin.set_ylim(-1, 1); axin.set_zlim(-1, 1)

    fig.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig3_disphenoid.pdf'))
    plt.savefig(os.path.join(OUTDIR, 'fig3_disphenoid.png'), dpi=450)
    plt.close()
    print('  fig3_disphenoid.pdf / .png')
    plt.rcParams.update({'font.size': 11})


# =======================================================================
# Scheme 1 - symvb package architecture and data flow (Methods §3.1)
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

    def box(x, y, w, h, text, fc, fontsize=11.5, weight='normal', zorder=2):
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
                fontsize=11, color=c_label, style='italic')

    # Row y-coordinates (top to bottom)
    y5, y4, y3, y2, y1 = 88, 71, 50, 29, 10
    row_h = 10

    # --- Row 5: user inputs ---
    layer_label(y5 + row_h / 2, 'user input')
    box(20, y5, 36, row_h,
        "integral rules\n"
        r"$\mathtt{Molecule(subst,\ subst\_2e,}$" + "\n"
        r"$\mathtt{interacting\_orbs,\ max\_2e\_centers)}$",
        c_input, fontsize=10.5)
    box(62, y5, 26, row_h,
        "electron / orbital counts\n"
        r"$(n_\alpha,\ n_\beta,\ n_{ao})$",
        c_input, fontsize=11.5)

    # --- Row 4: slaterdet / fixed_psi ---
    layer_label(y4 + row_h / 2, 'symvb.slaterdet /\nsymvb.fixed_psi')
    box(20, y4, 68, row_h,
        r"$\mathtt{generate\_dets(n_\alpha,\ n_\beta,\ n_{ao})}$   $\longrightarrow$   "
        r"$\mathtt{FixedPsi}$  basis of $N_D$ determinants",
        c_core, fontsize=11.5, weight='bold')

    # --- Row 3: molecule (matrix assembly) ---
    layer_label(y3 + row_h / 2, 'symvb.molecule')
    box(20, y3, 30, row_h,
        r"$\mathtt{build\_matrix(P,\,op)}$" + "\n"
        "1e matrix elements:  Eqs. (2)-(3)",
        c_core, fontsize=11)
    box(58, y3, 30, row_h,
        r"$\mathtt{o2\_matrix(P)}$" + "\n"
        u"2e Löwdin cofactors:  Eq. (4)",
        c_core, fontsize=11)

    # --- Row 2: symmetry / spin projections ---
    layer_label(y2 + row_h / 2, 'symvb.symmetry /\nsymvb.spin')
    box(25, y2, 26, row_h,
        r"$\mathtt{totally\_symmetric}$" + "\n"
        r"$D_n$ / $A_{1g}$  projection",
        c_sym, fontsize=11)
    box(57, y2, 26, row_h,
        r"$\mathtt{s\_squared\_matrix}$" + "\n"
        r"total-spin  $S(S{+}1)$",
        c_sym, fontsize=11)

    # --- Row 1: substitute / solve ---
    layer_label(y1 + row_h / 2, 'substitute & solve')
    box(20, y1, 68, row_h,
        r"SymPy $\to$ closed-form $E(h,s,U,J,K,\ldots)$   or   "
        r"NumPy $\to$ numerical scan",
        c_out, fontsize=11.5, weight='bold')

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
            ha='center', va='center', fontsize=11.5,
            color=c_edge, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35',
                      facecolor='white', edgecolor=c_edge, linewidth=0.8))
    # both matrix builders feed the symbolic-(H, S, H2e) collection point,
    # which feeds the optional projections and, directly, the solve row
    sym_top, sym_bot = 46.3, 41.7
    arrow(35, y3, 50, sym_top)
    arrow(73, y3, 58, sym_top)
    arrow(50, sym_bot, 32, y2 + row_h)
    arrow(58, sym_bot, 76, y2 + row_h)
    arrow(54, sym_bot, 54, y1 + row_h)   # projection bypass (between green boxes)
    # row2 -> row1
    arrow(30, y2, 40, y1 + row_h)
    arrow(78, y2, 68, y1 + row_h)

    ax.set_title('Scheme 1.  symvb package architecture and data flow',
                 fontsize=13, loc='left', pad=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'scheme1_pipeline.pdf'))
    plt.savefig(os.path.join(OUTDIR, 'scheme1_pipeline.png'), dpi=450)
    plt.close()
    print('  scheme1_pipeline.pdf / .png')


# =======================================================================
# Scheme 1a - object model (states / operators / Hamiltonian)
# =======================================================================
def scheme_1a_object_model():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(10.0, 6.6))
    ax.set_xlim(-4, 100)
    ax.set_ylim(0, 100)
    ax.set_axis_off()

    # Palette matches Scheme 1 for visual consistency.
    c_state = '#eef2f7'    # states
    c_op    = '#e8f0d8'    # orthogonal-AO operators
    c_mol   = '#d8e3f0'    # non-orthogonal-AO Hamiltonian
    c_out   = '#f5e5d8'    # outputs
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

    def arrow(x1, y1, x2, y2, style='-|>', lw=1.1, zorder=1, ls='-'):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, lw=lw, linestyle=ls,
                            color=c_edge, mutation_scale=11, zorder=zorder)
        ax.add_patch(a)

    def layer_label(y, text):
        ax.text(7, y, text, ha='right', va='center',
                fontsize=9, color=c_label, style='italic')

    # Row geometry (top to bottom).
    y_states, h_state = 80, 14
    y_ops,    h_op    = 38, 30
    y_out,    h_out   = 8,  18

    # --- Row 1: states ---
    layer_label(y_states + h_state / 2, 'states')
    box(18, y_states, 64, h_state,
        r"$\mathtt{symvb.SlaterDet}$    $|D\rangle$ — one det, e.g. $\mathtt{|aB|}$" + "\n"
        r"$\mathtt{symvb.FixedPsi}$    $|\psi\rangle = \sum_i c_i\, |D_i\rangle$ — wavefunction" + "\n"
        r"builders:  $\mathtt{generate\_dets},\ \mathtt{couple\_orbitals},\ +,\ -,\ c\!\cdot$",
        c_state, fontsize=9)

    # --- Row 2: two operator branches ---
    layer_label(y_ops + h_op / 2, 'operators on states')
    box(10, y_ops, 38, h_op,
        r"$\mathtt{symvb.operators.Operator}$" + "\n"
        r"orthogonal AO   ($s = 0$)" + "\n"
        "\n"
        "  spin:    s_squared,  s_z,  s_dot\n"
        "  occupation:    number,  double_occ\n"
        "  hopping:    hop\n"
        "  symmetry:    orbital_perm,\n"
        "       reynolds_projector\n"
        "  VB:    singlet_proj,\n"
        "       bond_singlet_creator\n"
        "\n"
        r"$\mathtt{apply},\ \mathtt{matrix},\ \mathtt{expectation}$",
        c_op, fontsize=8.5)
    box(52, y_ops, 38, h_op,
        r"$\mathtt{symvb.Molecule}$" + "\n"
        r"non-orthogonal AO   ($s \neq 0$)" + "\n"
        "\n"
        "  integral patterns:\n"
        r"     $\mathtt{subst},\ \mathtt{subst\_2e}$" + "\n"
        r"     $\mathtt{interacting\_orbs}$" + "\n"
        r"     $\mathtt{max\_2e\_centers}$" + "\n"
        "\n"
        "one element:   Eq. (3a)\n"
        r"   $\mathtt{op\_det}(D_1, \widehat{H}, D_2)$" + "\n"
        "full pair  $(H, S)$:\n"
        r"   $\mathtt{build\_matrix}$",
        c_mol, fontsize=8.5)

    # --- Row 3: outputs ---
    layer_label(y_out + h_out / 2, 'outputs')
    box(10, y_out, 38, h_out,
        r"spin labels   $S(S{+}1)$" + "\n"
        r"irrep blocks   ($D_n$ / $A_{1g}$ / …)" + "\n"
        r"local correlations  $\langle\widehat S_i\!\cdot\!\widehat S_j\rangle$" + "\n"
        "occupation / ionicity diagnostics",
        c_out, fontsize=9)
    box(52, y_out, 38, h_out,
        r"symbolic   $(H,\, S)$   pair" + "\n"
        r"$H\,\psi_i = E_i\, S\,\psi_i$" + "\n"
        r"energies $E_i$, eigenvectors $\psi_i$" + "\n"
        "(SymPy or NumPy back-end)",
        c_out, fontsize=9)

    # --- Arrows ---
    # States -> Operators (down-left) and States -> Molecule (down-right)
    arrow(40, y_states, 29, y_ops + h_op)
    arrow(60, y_states, 71, y_ops + h_op)
    ax.text(50, (y_states + y_ops + h_op) / 2,
            r"basis of   $\mathtt{FixedPsi}$",
            ha='center', va='center', fontsize=9,
            color=c_edge, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', edgecolor=c_edge, linewidth=0.7))
    # Operators -> output
    arrow(29, y_ops, 29, y_out + h_out)
    # Molecule -> output
    arrow(71, y_ops, 71, y_out + h_out)
    # Interpretation loop: eigenvectors $\psi_i$ flow from the right output
    # back up to the Operators box (which then produces the diagnostics in
    # the left output).  Curved arc routed under the central column.
    from matplotlib.patches import FancyArrowPatch
    back = FancyArrowPatch((60, y_out + h_out), (48, y_ops + 4),
                           arrowstyle='-|>', lw=1.0, linestyle='--',
                           color=c_edge, mutation_scale=11,
                           connectionstyle='arc3,rad=-0.4', zorder=1)
    ax.add_patch(back)
    ax.text(57, y_out + h_out + 6,
            r"$\psi_i$  $\to$  diagnostics",
            ha='center', va='center', fontsize=8.5,
            color=c_label, style='italic')

    ax.set_title('Scheme 1a.  symvb object model — states, operators, Hamiltonian',
                 fontsize=12, loc='left', pad=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'scheme1a_object_model.pdf'))
    plt.savefig(os.path.join(OUTDIR, 'scheme1a_object_model.png'), dpi=150)
    plt.close()
    print('  scheme1a_object_model.pdf / .png')


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
            r'symvb carries every integral ($h$, $s$, $U$, $J$, $K$, $\ldots$) '
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
    """Graphical abstract (ACS TOC): benzene -> symvb -> closed-form + curve.
    Canvas is 3.25" x 1.75", split into three regions plus a title strip."""
    import numpy as np
    from matplotlib.patches import FancyArrowPatch

    fig = plt.figure(figsize=(3.25, 1.75))

    # 1) Title strip
    fig.text(0.5, 0.93,
             'Closed-form valence-bond structure weights',
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
        ax_L.plot([x1, x2], [y1, y2], color='#1f4068', lw=1.1)
    th = np.linspace(0, 2 * np.pi, 100)
    ax_L.plot(0.62 * R * np.cos(th), 0.62 * R * np.sin(th),
              color='#1f4068', lw=0.9)
    for (x, y) in hex_xy:
        ax_L.plot(x, y, 'o', markersize=4.5, markerfacecolor='white',
                  markeredgecolor='#1f4068', markeredgewidth=0.8, zorder=3)
    ax_L.text(0, -1.5, r'benzene $\pi$ (6e / 6o)',
              ha='center', fontsize=6, color='#4a5668')

    # 3) Middle axis: pipeline arrow with symvb label + symbolic H
    ax_M = fig.add_axes([0.30, 0.12, 0.28, 0.72])
    ax_M.set_xlim(0, 10)
    ax_M.set_ylim(0, 10)
    ax_M.set_axis_off()
    ax_M.add_patch(FancyArrowPatch((0.5, 5), (9.5, 5),
                   arrowstyle='-|>', mutation_scale=10, lw=1.2,
                   color='#2a3644'))
    ax_M.text(5, 7.2, r'symvb', fontsize=9, fontweight='bold',
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

    # 4) Right region: headline (figure coords, clear of the canvas edge)
    fig.text(0.63, 0.815,
             r'$(w_0, w_1, w_2, w_3) = (5, 31, 31, 5)/72$ at $U=0$',
             ha='center', fontsize=6.4, color='#1f4068')
    fig.text(0.63, 0.725, r'ionic at $U=0$, covalent at strong coupling',
             ha='center', fontsize=6.0, color='#6a7380')

    # Mini plot: covalent vs ionic Chirgwin-Coulson weight crossover
    # (same data pipeline as examples/make_fig_benzene_ionicity.py)
    import pickle
    from symvb.fixed_psi import generate_dets as _gdets
    with open('/tmp/benzene_hubbard_matrices.pkl', 'rb') as fh:
        _H1, _S2, _H2 = pickle.load(fh)
    _h, _s, _U = sp.symbols('h s U')
    _H0 = np.array(_H1.subs({_h: -1, _s: 0}).tolist(), dtype=float)
    _MU = np.array(sp.diff(_H2, _U).subs({_s: 0}).tolist(), dtype=float)
    _dets = [p.dets[0].det_string for p in _gdets(3, 3, 6)]
    _cls = np.array([sum(1 for c in set(d) if c.islower() and c.upper() in d)
                     for d in _dets])
    _Us = np.logspace(-1, 2, 31)
    _w0 = []
    for _u in _Us:
        _Hn = _H0 + _u * _MU
        _Hn = (_Hn + _Hn.T) / 2
        _c0 = np.linalg.eigh(_Hn)[1][:, 0]
        _w0.append(float(np.sum(_c0[_cls == 0] ** 2)))
    _w0 = np.array(_w0)
    ax_in = fig.add_axes([0.66, 0.18, 0.30, 0.42])
    ax_in.semilogx(_Us, _w0, color='#1f4068', lw=1.2)
    ax_in.semilogx(_Us, 1 - _w0, color='#7a2828', lw=1.2, ls='--')
    ax_in.axhline(0.5, color='0.75', lw=0.5, ls=':')
    ax_in.text(0.15, 0.10, 'covalent', fontsize=6.0, color='#1f4068')
    ax_in.text(0.15, 0.82, 'ionic', fontsize=6.0, color='#7a2828')
    ax_in.text(28, 0.62, r'$s = 0$', fontsize=5.5, color='#6a7380')
    ax_in.set_ylim(0, 1.0)
    ax_in.set_xlabel(r'$U/|h|$', fontsize=6.0, labelpad=0)
    ax_in.set_ylabel('weight', fontsize=6.0, labelpad=1)
    ax_in.tick_params(axis='both', labelsize=5.5, length=2, pad=1)
    for _spine in ax_in.spines.values():
        _spine.set_linewidth(0.5)

    plt.savefig(os.path.join(OUTDIR, 'toc_graphic.pdf'))
    plt.savefig(os.path.join(OUTDIR, 'toc_graphic.png'), dpi=300)
    plt.close()
    print('  toc_graphic.pdf / .png')


# =======================================================================
if __name__ == '__main__':
    print(f'writing figures to {OUTDIR}/ ...')
    scheme_1_pipeline()
    scheme_1a_object_model()
    scheme_2_systems()
    scheme_3_rumer()
    toc_graphic()
    figure_1_h2()
    figure_2_allyl()
    figure_3_degeneracy()
    figure_4_pade()
    figure_5_aromaticity()
    figure_3_disphenoid()
    print('done.')
