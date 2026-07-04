"""(H2)2+ disphenoid: Coulson-Fischer-like breathing structures at finite U.

Extends the one-electron Hueckel analysis of the disphenoid cation
(disphenoid_4o3e.py, manuscript Sec. 7) to finite on-site repulsion U,
and asks the charge analog of the Coulson-Fischer question:

  When does a single, hole-LOCALIZED structure with relaxed (breathing)
  orbitals drop below the symmetric, hole-DELOCALIZED mean-field state?
  And how does the breathing renormalize the effective inter-pair
  coupling that controls the Robin-Day III <-> II threshold
  k_crit = 1/(8|h_l|)?

Setup (same as Sec. 7): AOs {a,b} = fragment 1, {c,d} = fragment 2,
short-edge integrals h_s1 (a-b) and h_s2 (c-d), four equal long edges
h_l, s = 0, on-site Hubbard U.  3 electrons, Sz = +1/2 (24 dets).
eta = h_s1 - h_s2 is the asymmetric-stretch coordinate.

Variational objects (all closed forms from symvb):

  D_sym        symmetric ROHF determinant |psi_1 psi_4 psi_1-bar|,
               psi_1 = (a+b+c+d)/2, psi_4 = (a+b-c-d)/2 (the Hueckel
               ground configuration of Eq. 14 at eta = 0).
  Psi_L(lam)   CF/breathing hole-on-fragment-1 structure:
               SOMO sigma_1 = a+b; spectator pair on fragment 2 is a
               Coulson-Fischer singlet pair (c + lam*d)(d + lam*c).
               lam = 0: plain localized determinant.  lam = 1: closed
               shell sigma_2^2.  Optimal lam<1 = left-right correlation
               of the spectator H2, i.e. the breathing relaxation that
               a delocalized hole frustrates.
  Psi_R(lam)   mirror image (hole on fragment 2).

Diagnostics:

  (1) electronic CF point U*(h_s, h_l):  min_lam E[Psi_L] crosses
      E[D_sym].  Beyond U* the broken-symmetry single structure is
      variationally below the symmetric mean field -- the charge
      analog of the Coulson-Fischer point / Loewdin symmetry dilemma.
  (2) resonance restoration: 2x2 GHEP over {Psi_L, Psi_R}; the
      splitting defines an effective inter-pair coupling
      h_l_eff(U) = (E_+ - E_-)/4   (Hueckel limit: gap = 4|h_l|).
  (3) vibronic threshold: curvature of the 2x2 lower root along eta
      (lam re-optimized at each eta) -> k_crit_CF(U); compared with
      the exact FCI curvature -> k_crit_exact(U), and with the frozen
      Hueckel value 1/(8|h_l|).

Run from the repo root:  PYTHONPATH=. python3 examples/h2h2_plus_cf_breathing.py
"""
import os
import pickle
import sys
import time

import numpy as np
import sympy as sp
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, hamiltonian
from symvb.fixed_psi import FixedPsi, generate_dets

CACHE = '/tmp/disphenoid_cf_matrices_v2.pkl'
ORBS = 'abcd'

hs1_, hs2_, hl_, s_, U_, lam_ = sp.symbols('h_s_1 h_s_2 h_l s U lam')


def make_molecule():
    return Molecule(
        zero_ii=True,
        interacting_orbs=['ab', 'cd', 'ac', 'ad', 'bc', 'bd'],
        subst={'h_s_1': ('H_ab',),
               'h_s_2': ('H_cd',),
               'h_l':   ('H_ac', 'H_ad', 'H_bc', 'H_bd'),
               's':     ('S_ab', 'S_cd', 'S_ac', 'S_ad', 'S_bc', 'S_bd')},
        subst_2e={'U': ('1111',)},
        max_2e_centers=1,
    )


def det3(cA, cB, cC):
    """FixedPsi for |phi_A^alpha phi_B^alpha phi_C^beta| with AO coefficient
    dicts cA, cB, cC over 'abcd'.  Alpha pair antisymmetrized with i<j;
    every det string is written 'ij' + 'K' so relative signs are consistent
    across all structures."""
    p = FixedPsi()
    for ii, oi in enumerate(ORBS):
        for oj in ORBS[ii + 1:]:
            cf2 = cA.get(oi, 0) * cB.get(oj, 0) - cA.get(oj, 0) * cB.get(oi, 0)
            if cf2 == 0:
                continue
            for ok in ORBS:
                cf = sp.expand(cf2 * cC.get(ok, 0))
                if cf != 0:
                    p.add_str_det(oi + oj + ok.upper(), coef=cf)
    return p


def build_structures():
    """D_sym, Psi_L(lam), Psi_R(lam) as FixedPsi over AO determinants."""
    sig1 = {'a': 1, 'b': 1}              # fragment-1 bonding MO (unnormalized)
    sig2 = {'c': 1, 'd': 1}
    psi1 = {'a': 1, 'b': 1, 'c': 1, 'd': 1}    # inter-pair bonding
    psi4 = {'a': 1, 'b': 1, 'c': -1, 'd': -1}  # inter-pair antibonding (SOMO)

    d_sym = det3(psi1, psi4, psi1)

    # CF pair orbitals on the spectator fragment
    chi_p2 = {'c': 1, 'd': lam_}
    chi_m2 = {'c': lam_, 'd': 1}
    chi_p1 = {'a': 1, 'b': lam_}
    chi_m1 = {'a': lam_, 'b': 1}

    # singlet-coupled pair: (chi+^a chi-^b + chi-^a chi+^b)
    psi_L = det3(sig1, chi_p2, chi_m2)
    psi_L.add_fixedpsi(det3(sig1, chi_m2, chi_p2), coef=1)
    psi_R = det3(sig2, chi_p1, chi_m1)
    psi_R.add_fixedpsi(det3(sig2, chi_m1, chi_p1), coef=1)
    return d_sym, psi_L, psi_R


def build_matrices():
    """Structure-basis 3x3 (H, S) with lam symbolic, and the 24-det FCI H,
    both at s = 0.  Cached: the FCI o2 build dominates the wall time."""
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            return pickle.load(f)

    m = make_molecule()
    d_sym, psi_L, psi_R = build_structures()
    basis = [d_sym, psi_L, psi_R]

    t0 = time.time()
    Hst, Sst = hamiltonian(m, basis)   # 2e block folded into Hst
    Hst = sp.expand(Hst.subs(s_, 0))
    Sst = sp.expand(Sst.subs(s_, 0))
    print(f"  structure-basis 3x3 build: {time.time() - t0:.1f} s")

    t0 = time.time()
    P = generate_dets(2, 1, 4)
    Hf = hamiltonian(m, P)[0].subs(s_, 0)
    print(f"  24-det FCI build (cation): {time.time() - t0:.1f} s")

    t0 = time.time()
    P4 = generate_dets(2, 2, 4)
    Hn = hamiltonian(m, P4)[0].subs(s_, 0)
    print(f"  36-det FCI build (neutral): {time.time() - t0:.1f} s")

    with open(CACHE, 'wb') as f:
        pickle.dump((Hst, Sst, Hf, Hn), f)
    return Hst, Sst, Hf, Hn


def lambdify_fci(Hf):
    """H_FCI = H0 + hs1*M1 + hs2*M2 + hl*M3 + U*M4 (numeric)."""
    syms = [hs1_, hs2_, hl_, U_]
    H0 = np.array(Hf.subs({sym: 0 for sym in syms}).tolist(), dtype=float)
    Ms = [np.array(sp.diff(Hf, sym).tolist(), dtype=float) for sym in syms]

    def H(hs1, hs2, hl, U):
        A = H0 + hs1 * Ms[0] + hs2 * Ms[1] + hl * Ms[2] + U * Ms[3]
        return (A + A.T) / 2

    return H


def fci_ground(Hnum, hs1, hs2, hl, U):
    return np.linalg.eigvalsh(Hnum(hs1, hs2, hl, U))[0]


def curvature_decomposition(Hnum, W, hs, hl, U, nshow=3):
    """Exact curvature at eta = 0 via second-order PT:
    -d2E0/deta2 = 2 sum_n |<0|W|n>|^2 / (E_n - E_0),  W = dH/deta.
    (H is linear in eta, so there is no first-order curvature term.)
    Returns (k_exact_PT, list of (n, E_n - E_0, contribution))."""
    E, V = np.linalg.eigh(Hnum(hs, hs, hl, U))
    w0 = V.T @ (W @ V[:, 0])
    contrib = []
    for n in range(1, len(E)):
        dE = E[n] - E[0]
        c = 2.0 * w0[n] ** 2 / dE if dE > 1e-12 else 0.0
        if c > 1e-12:
            contrib.append((n, dE, c))
    contrib.sort(key=lambda t: -t[2])
    return sum(c for _, _, c in contrib), contrib[:nshow]


def curvature(f, d=1e-3):
    return (f(d) + f(-d) - 2.0 * f(0.0)) / d**2


def main():
    hs_val, hl_val = -1.0, -0.3
    print("=" * 72)
    print("(H2)2+ disphenoid: CF/breathing structures at finite U")
    print(f"  h_s = {hs_val}, h_l = {hl_val}  (units of |h_s|), s = 0")
    print("=" * 72)

    Hst, Sst, Hf, Hn = build_matrices()
    Hnum = lambdify_fci(Hf)
    Hneut = lambdify_fci(Hn)

    # ----- closed forms at the symmetric point ---------------------------
    sym_pt = {hs1_: hs1_, hs2_: hs1_}    # hs1 = hs2 = h_s
    E_sym = sp.simplify(sp.cancel(Hst[0, 0] / Sst[0, 0]).subs(sym_pt))
    H_LL = sp.cancel(Hst[1, 1] / Sst[1, 1])
    E_L = sp.simplify(H_LL.subs(sym_pt))
    print("\nClosed forms (h_s1 = h_s2 = h_s):")
    print("  E_sym(ROHF)      =", sp.nsimplify(E_sym, rational=True))
    print("  E_L(lam)         =",
          sp.nsimplify(sp.simplify(E_L), rational=True))
    print("    (note: E_L carries NO h_l at s = 0 -- the localized structure"
          " pays the full resonance 2|h_l|)")

    # effective 2x2 over {Psi_L, Psi_R}: symmetric/antisymmetric roots
    E_plus = sp.cancel((Hst[1, 1] + Hst[1, 2]) / (Sst[1, 1] + Sst[1, 2]))
    E_minus = sp.cancel((Hst[1, 1] - Hst[1, 2]) / (Sst[1, 1] - Sst[1, 2]))
    gap = sp.simplify((E_minus - E_plus).subs(sym_pt))
    print("  gap(lam) = E_- - E_+ =",
          sp.nsimplify(sp.factor(sp.simplify(gap)), rational=True))

    # ----- numeric sweeps over U -----------------------------------------
    subs0 = {hs1_: hs_val, hs2_: hs_val, hl_: hl_val}
    fE_L = sp.lambdify((lam_, U_), E_L.subs(subs0), 'numpy')
    fE_sym = sp.lambdify(U_, E_sym.subs(subs0), 'numpy')
    fE_res = sp.lambdify((lam_, U_),
                         sp.Min(E_plus, E_minus).subs(sym_pt).subs(subs0),
                         'sympy')
    fgap = sp.lambdify((lam_, U_), gap.subs(subs0), 'numpy')

    # eta-dependent 2x2 lower root (lam re-optimized at each eta)
    Hll, Hlr, Hrr = Hst[1, 1], Hst[1, 2], Hst[2, 2]
    Sll, Slr, Srr = Sst[1, 1], Sst[1, 2], Sst[2, 2]
    eta_ = sp.Symbol('eta')
    eta_subs = {hs1_: hs_val + eta_ / 2, hs2_: hs_val - eta_ / 2, hl_: hl_val}
    a2 = (Sll * Srr - Slr**2)
    b2 = (Hll * Srr + Hrr * Sll - 2 * Hlr * Slr)
    c2 = (Hll * Hrr - Hlr**2)
    fa = sp.lambdify((lam_, U_, eta_), a2.subs(eta_subs), 'numpy')
    fb = sp.lambdify((lam_, U_, eta_), b2.subs(eta_subs), 'numpy')
    fc = sp.lambdify((lam_, U_, eta_), c2.subs(eta_subs), 'numpy')

    def E_res_eta(U, eta):
        def root(lam):
            A, B, C = fa(lam, U, eta), fb(lam, U, eta), fc(lam, U, eta)
            return (B - np.sqrt(B * B - 4 * A * C)) / (2 * A)
        r = minimize_scalar(root, bounds=(0.0, 1.0), method='bounded',
                            options={'xatol': 1e-10})
        return r.fun, r.x

    def lam_opt_single(U):
        r = minimize_scalar(lambda l: fE_L(l, U), bounds=(0.0, 1.0),
                            method='bounded', options={'xatol': 1e-10})
        return r.x, r.fun

    print("\n" + "-" * 100)
    print(f"{'U':>4} | {'lam_opt':>8} {'E_loc':>9} {'E_sym':>9} {'E_res':>9} "
          f"{'E_FCI':>9} | {'h_l_eff':>8} | {'k_crit':>7} {'k_crit':>7} {'k_crit':>7}")
    print(f"{'':>4} | {'':>8} {'(broken)':>9} {'(ROHF)':>9} {'(CF 2x2)':>9} "
          f"{'(exact)':>9} | {'gap/4':>8} | {'exact':>7} {'CF2x2':>7} {'Hueckel':>7}")
    print("-" * 100)

    k_huckel = 1.0 / (8.0 * abs(hl_val))
    rows = []
    for U in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
        lam_o, E_loc = lam_opt_single(U)
        E_s = float(fE_sym(U))
        E_r, lam_r = E_res_eta(U, 0.0)
        E_f = fci_ground(Hnum, hs_val, hs_val, hl_val, U)
        hl_eff = abs(fgap(lam_r, U)) / 4.0

        k_exact = -curvature(lambda e: fci_ground(
            Hnum, hs_val + e / 2, hs_val - e / 2, hl_val, U))
        k_cf = -curvature(lambda e: E_res_eta(U, e)[0])
        rows.append((U, lam_o, E_loc, E_s, E_r, E_f, hl_eff,
                     k_exact, k_cf, k_huckel))
        print(f"{U:>4.1f} | {lam_o:>8.4f} {E_loc:>9.5f} {E_s:>9.5f} {E_r:>9.5f} "
              f"{E_f:>9.5f} | {hl_eff:>8.5f} | {k_exact:>7.4f} {k_cf:>7.4f} "
              f"{k_huckel:>7.4f}")

    # ----- electronic CF point U* ----------------------------------------
    def gap_loc_sym(U):
        return lam_opt_single(U)[1] - float(fE_sym(U))

    print("-" * 100)
    if gap_loc_sym(0.0) > 0 and gap_loc_sym(8.0) < 0:
        U_star = brentq(gap_loc_sym, 0.0, 8.0, xtol=1e-8)
        print(f"\nElectronic CF point (broken single structure crosses ROHF):")
        print(f"  U* = {U_star:.4f} |h_s|    (h_l = {hl_val})")
        lam_s, _ = lam_opt_single(U_star)
        print(f"  lam_opt(U*) = {lam_s:.4f}")
    else:
        print("\nNo E_loc/E_sym crossing in U in [0, 8].")

    # Closed form for U*: eliminate lam between dE_L/dlam = 0 and
    # E_L = E_sym.  The resultant factors as
    #   h_l * (U^2 + 16 h_s^2)^2 * (h_l + 2 h_s) * (U^2 - 16 h_l^2 - 32 h_l h_s)^2
    # whose physical branch gives
    #   U*^2 = 16 h_l^2 + 32 h_l h_s,   U* = 4 sqrt(|h_l| (|h_l| + 2|h_s|)).
    h_l_sym = sp.Symbol('h_l')
    E_L_rat = sp.nsimplify(sp.simplify(E_L), rational=True)
    P1 = sp.numer(sp.together(sp.diff(E_L_rat, lam_)))
    P2 = sp.expand(sp.numer(sp.together(E_L_rat))
                   - (U_ / 2 + 2 * h_l_sym + 3 * hs1_)
                   * sp.denom(sp.together(E_L_rat)))
    R = sp.factor(sp.resultant(sp.expand(P1), P2, lam_))
    print("\n  Closed form for U* (resultant elimination of lam):")
    print("    resultant =", R)
    U_closed = 4 * sp.sqrt(h_l_sym**2 + 2 * h_l_sym * hs1_)
    print("    => U* = 4*sqrt(h_l^2 + 2*h_l*h_s)"
          " = 4*sqrt(|h_l|*(|h_l| + 2|h_s|))")

    # E_L is h_l-free, so U* depends on h_l only via E_sym = U/2 + 2 h_l + 3 h_s
    print("\n  U*(h_l) sweep (E_loc is h_l-independent at s = 0):")
    for hlv in [-0.1, -0.2, -0.3, -0.4, -0.5]:
        def g(U, hlv=hlv):
            return lam_opt_single(U)[1] - (U / 2 + 2 * hlv + 3 * hs_val)
        try:
            Us = brentq(g, 0.0, 20.0, xtol=1e-8)
            Uc = float(U_closed.subs({h_l_sym: hlv, hs1_: hs_val}))
            print(f"    h_l = {hlv:>5.2f}:  U* = {Us:.6f} |h_s|"
                  f"   closed form: {Uc:.6f}")
        except ValueError:
            print(f"    h_l = {hlv:>5.2f}:  no crossing in [0, 20]")

    # ----- which excited state drives the exact softening? ----------------
    def W_of(Hmat):
        Ws = (np.array(sp.diff(Hmat, hs1_).tolist(), dtype=float)
              - np.array(sp.diff(Hmat, hs2_).tolist(), dtype=float)) / 2.0
        return (Ws + Ws.T) / 2.0

    W = W_of(Hf)
    Wn = W_of(Hn)
    print("\nExact curvature decomposed over FCI excited states"
          " (top contributions):")
    print(f"{'U':>4} | {'k_PT':>8} | per-state (n, E_n-E_0, contribution)")
    for U in [0.0, 1.0, 2.0, 4.0, 8.0]:
        kPT, top = curvature_decomposition(Hnum, W, hs_val, hl_val, U)
        terms = "  ".join(f"(n={n}, dE={dE:.3f}, k={c:.4f})"
                          for n, dE, c in top)
        print(f"{U:>4.1f} | {kPT:>8.4f} | {terms}")

    # channel split: mixed-valence (low dE, hole transfer) vs intra-fragment
    # ionic (high dE ~ fragment cov->ion gap).  The fragment-additive
    # reference is exact for non-interacting fragments:
    #   E_H2(h, U) = U/2 - sqrt(U^2/4 + 4 h^2)  ->  k_frag = -E''/4 per
    #   neutral fragment (E_H2+ is linear in h and contributes nothing).
    h_ = sp.Symbol('h')
    E_h2_frag = U_ / 2 - sp.sqrt(U_**2 / 4 + 4 * h_**2)
    d2E_frag = sp.diff(E_h2_frag, h_, 2)
    f_kfrag = sp.lambdify((h_, U_), -d2E_frag / 4, 'numpy')
    print("\nChannel split of the cation curvature, neutral reference, and")
    print("fragment-additive closed form  k_frag = -E_H2''(h_s, U)/4 :")
    print(f"{'U':>4} | {'k_exact':>8} {'k_MV':>8} {'k_ionic':>8} | "
          f"{'k_CF2x2':>8} {'k_frag':>8} | {'k_neut':>8} {'2*k_frag':>8} | "
          f"{'k_cat-k_neut':>12}")
    dE_cut = 2.0 * abs(hs_val)   # separates hole-transfer from ionic channel
    for U in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
        kPT, _ = curvature_decomposition(Hnum, W, hs_val, hl_val, U)
        _, allc = curvature_decomposition(Hnum, W, hs_val, hl_val, U,
                                          nshow=999)
        k_MV = sum(c for _, dE, c in allc if dE < dE_cut)
        k_io = kPT - k_MV
        k_cf = -curvature(lambda e: E_res_eta(U, e)[0])
        k_neut = -curvature(lambda e: np.linalg.eigvalsh(
            Hneut(hs_val + e / 2, hs_val - e / 2, hl_val, U))[0])
        kf = float(f_kfrag(hs_val, U))
        print(f"{U:>4.1f} | {kPT:>8.4f} {k_MV:>8.4f} {k_io:>8.4f} | "
              f"{k_cf:>8.4f} {kf:>8.4f} | {k_neut:>8.4f} {2 * kf:>8.4f} | "
              f"{kPT - k_neut:>12.4f}")

    # sanity anchors
    print("\nSanity checks:")
    print(f"  U=0 exact curvature -1/(8|h_l|) = {-k_huckel:.6f}; "
          f"FCI gives {-rows[0][7]:.6f}")
    E14 = 3 * hs_val - np.sqrt(0.0 + 4 * hl_val**2)
    print(f"  U=0 Eq.(14) E_0 = {E14:.6f}; FCI gives {rows[0][5]:.6f}; "
          f"ROHF gives {rows[0][3]:.6f}")
    print(f"  U=0 gap 4|h_l| = {4 * abs(hl_val):.6f}; "
          f"CF 2x2 gives {4 * rows[0][6]:.6f}")


if __name__ == '__main__':
    main()
