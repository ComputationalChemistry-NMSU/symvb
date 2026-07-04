"""(H2)2+ disphenoid CF/breathing analysis with two-center Coulomb (PPP).

Extends h2h2_plus_cf_breathing.py (Hubbard-only) by the direct two-center
Coulomb integrals, assigned per edge class:

    U    on-site            (aa|aa)
    J_s  short (intra-pair) (aa|bb), (cc|dd)
    J_l  long (inter-pair)  (aa|cc), (aa|dd), (bb|cc), (bb|dd)

Exchange and hybrid two-center integrals are set to zero (ZDO/PPP).
Physical hierarchy: U > J_s > J_l (1/r at increasing distance).

Questions:
  (1) does inter-fragment Coulomb J_l reverse the Hubbard-limit finding
      that correlation lowers the vibronic trapping threshold k_crit?
  (2) what happens to the electronic CF point
      U* = 4 sqrt(h_l^2 + 2 h_l h_s)  (J = 0)?

Operator identity used as an anchor: at U = J_s = J_l (all two-electron
integrals equal) the interaction is U * N(N-1)/2 * Identity for a fixed
electron count, so the Hueckel results must be recovered exactly --
only the DIFFERENCES between U, J_s, J_l can have physical effects.

Run from the repo root:
    PYTHONPATH=. python3 examples/h2h2_plus_cf_breathing_ppp.py
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
from symvb.fixed_psi import generate_dets

from h2h2_plus_cf_breathing import (ORBS, det3, build_structures, curvature,
                                    hs1_, hs2_, hl_, s_, U_, lam_)

CACHE = '/tmp/disphenoid_cf_ppp_matrices.pkl'
Js_, Jl_ = sp.symbols('J_s J_l')
INTRA = ({'a', 'b'}, {'c', 'd'})


def make_molecule_ppp():
    return Molecule(
        zero_ii=True,
        interacting_orbs=['ab', 'cd', 'ac', 'ad', 'bc', 'bd'],
        subst={'h_s_1': ('H_ab',),
               'h_s_2': ('H_cd',),
               'h_l':   ('H_ac', 'H_ad', 'H_bc', 'H_bd'),
               's':     ('S_ab', 'S_cd', 'S_ac', 'S_ad', 'S_bc', 'S_bd')},
        subst_2e={'U': ('1111',)},
        max_2e_centers=2,
    )


def classify_T(name):
    """Map an auto-named two-electron integral T_wxyz to J_s, J_l, or 0.

    symvb auto-names use PHYSICIST index order: the chemist direct Coulomb
    (aa|bb) = <ab|ab> is named T_abab (pattern xyxy), and the chemist
    exchange (ab|ab) = <aa|bb> is named T_aabb (pattern xxyy).  Verified:
    o2_matrix(['aB'])[0,0] = T_abab at s = 0.  Only the direct Coulomb
    survives here (ZDO)."""
    idx = name[2:]
    if len(idx) != 4:
        return None
    if idx[0] == idx[2] and idx[1] == idx[3] and idx[0] != idx[1]:
        pair = {idx[0], idx[1]}
        return Js_ if pair in INTRA else Jl_
    return sp.Integer(0)   # exchange T_xxyy, hybrids T_xxxy etc.


def ppp_subs(M):
    sub = {}
    for sym in M.free_symbols:
        if sym.name.startswith('T_'):
            sub[sym] = classify_T(sym.name)
    return M.subs(sub)


def build_matrices():
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    m = make_molecule_ppp()
    d_sym, psi_L, psi_R = build_structures()
    basis = [d_sym, psi_L, psi_R]

    t0 = time.time()
    Hst, Sst = hamiltonian(m, basis)   # 2e block folded into Hst
    Hst = sp.expand(ppp_subs(Hst).subs(s_, 0))
    Sst = sp.expand(Sst.subs(s_, 0))
    print(f"  structure-basis 3x3 build: {time.time() - t0:.1f} s")

    t0 = time.time()
    P = generate_dets(2, 1, 4)
    Hf = ppp_subs(hamiltonian(m, P)[0]).subs(s_, 0)
    print(f"  24-det FCI build (cation): {time.time() - t0:.1f} s")

    t0 = time.time()
    P4 = generate_dets(2, 2, 4)
    Hn = ppp_subs(hamiltonian(m, P4)[0]).subs(s_, 0)
    print(f"  36-det FCI build (neutral): {time.time() - t0:.1f} s")

    with open(CACHE, 'wb') as f:
        pickle.dump((Hst, Sst, Hf, Hn), f)
    return Hst, Sst, Hf, Hn


def lambdify_fci(Hf):
    syms = [hs1_, hs2_, hl_, U_, Js_, Jl_]
    H0 = np.array(Hf.subs({sym: 0 for sym in syms}).tolist(), dtype=float)
    Ms = [np.array(sp.diff(Hf, sym).tolist(), dtype=float) for sym in syms]

    def H(hs1, hs2, hl, U, Js, Jl):
        A = (H0 + hs1 * Ms[0] + hs2 * Ms[1] + hl * Ms[2]
             + U * Ms[3] + Js * Ms[4] + Jl * Ms[5])
        return (A + A.T) / 2

    return H


def main():
    hs_val, hl_val = -1.0, -0.3
    print("=" * 78)
    print("(H2)2+ disphenoid with PPP two-center Coulomb (U, J_s, J_l), s = 0")
    print(f"  h_s = {hs_val}, h_l = {hl_val}")
    print("=" * 78)

    Hst, Sst, Hf, Hn = build_matrices()
    Hnum = lambdify_fci(Hf)
    Hneut = lambdify_fci(Hn)

    sym_pt = {hs1_: hs1_, hs2_: hs1_}
    E_sym = sp.nsimplify(
        sp.simplify(sp.cancel(Hst[0, 0] / Sst[0, 0]).subs(sym_pt)),
        rational=True)
    E_L = sp.nsimplify(
        sp.simplify(sp.cancel(Hst[1, 1] / Sst[1, 1]).subs(sym_pt)),
        rational=True)
    print("\nClosed forms (h_s1 = h_s2 = h_s):")
    print("  E_sym(ROHF) =", E_sym)
    print("  E_L(lam)    =", sp.simplify(E_L))

    # anchor: U = J_s = J_l must be a rigid shift by 3U
    shift = sp.simplify((E_sym - 3 * U_).subs({Js_: U_, Jl_: U_}))
    print("\n  anchor U=J_s=J_l: E_sym - 3U =", shift,
          " (must be U-free, = 2*h_l + 3*h_s)")

    # ----- closed-form U* with J's ---------------------------------------
    P1 = sp.numer(sp.together(sp.diff(E_L, lam_)))
    P2 = sp.expand(sp.numer(sp.together(E_L))
                   - E_sym * sp.denom(sp.together(E_L)))
    R = sp.factor(sp.resultant(sp.expand(P1), P2, lam_))
    print("\nClosed-form U* condition (resultant, factored):")
    print(" ", R)
    print("  physical branch:  (U - J_s)^2 = 16 h_l^2 + 32 h_l h_s")
    print("  =>  U* = J_s + 4 sqrt(|h_l| (|h_l| + 2 |h_s|))")
    print("  J_l drops out exactly: both E_sym and E_L carry the same 2*J_l")
    print("  (<N_1 N_2> = 2 for the localized structure AND, via exchange-")
    print("  suppressed charge fluctuations, for the symmetric determinant).")

    # ----- numeric: optimal lam, U*, and k_crit trends --------------------
    fE_L = sp.lambdify((lam_, U_, Js_, Jl_),
                       E_L.subs({hs1_: hs_val}), 'numpy')
    fE_sym = sp.lambdify((U_, Js_, Jl_),
                         E_sym.subs({hs1_: hs_val, hl_: hl_val}), 'numpy')

    def lam_opt_single(U, Js, Jl):
        r = minimize_scalar(lambda l: fE_L(l, U, Js, Jl), bounds=(0.0, 1.0),
                            method='bounded', options={'xatol': 1e-10})
        return r.x, r.fun

    def Ustar(Js_ratio, Jl_ratio):
        def g(U):
            return (lam_opt_single(U, Js_ratio * U, Jl_ratio * U)[1]
                    - float(fE_sym(U, Js_ratio * U, Jl_ratio * U)))
        try:
            return brentq(g, 1e-6, 40.0, xtol=1e-9)
        except ValueError:
            return float('nan')

    def k_exact(U, Js, Jl, which='cation'):
        Hm = Hnum if which == 'cation' else Hneut
        ev = np.linalg.eigvalsh(Hm(hs_val, hs_val, hl_val, U, Js, Jl))
        if ev[1] - ev[0] < 1e-9:    # degenerate ground state: |eta| cusp,
            return float('nan')     # curvature undefined (1st-order JT)
        return -curvature(lambda e: np.linalg.eigvalsh(
            Hm(hs_val + e / 2, hs_val - e / 2, hl_val, U, Js, Jl))[0])

    print("\nElectronic CF point U* for repulsion ladders J_s = x*U, "
          "J_l = y*U:")
    for xs, yl in [(0.0, 0.0), (0.0, 0.25), (0.5, 0.25), (0.5, 0.0),
                   (0.75, 0.5), (1.0, 1.0)]:
        print(f"  J_s = {xs:.2f} U, J_l = {yl:.2f} U:  "
              f"U* = {Ustar(xs, yl):8.4f} |h_s|")

    k_huckel = 1.0 / (8.0 * abs(hl_val))
    print(f"\nVibronic threshold k_crit(U) = -d2E/deta2 "
          f"(Hueckel value {k_huckel:.4f}):")
    ladders = [("Hubbard           (J_s=0,     J_l=0)   ", 0.0, 0.0),
               ("inter only        (J_s=0,     J_l=U/4) ", 0.0, 0.25),
               ("PPP hierarchy     (J_s=U/2,   J_l=U/4) ", 0.5, 0.25),
               ("strong inter      (J_s=U/2,   J_l=U/2) ", 0.5, 0.5),
               ("all equal         (J_s=U,     J_l=U)   ", 1.0, 1.0)]
    Us = [0.0, 1.0, 2.0, 4.0, 8.0]
    print(f"{'ladder':>42} | " + "  ".join(f"U={u:<4.0f}" for u in Us))
    for label, xs, yl in ladders:
        ks = [k_exact(u, xs * u, yl * u) for u in Us]
        print(f"{label:>42} | " + "  ".join(f"{k:6.4f}" for k in ks))

    print(f"\nSame, charge-induced part (cation minus neutral curvature):")
    print(f"{'ladder':>42} | " + "  ".join(f"U={u:<4.0f}" for u in Us))
    for label, xs, yl in ladders:
        ks = [k_exact(u, xs * u, yl * u)
              - k_exact(u, xs * u, yl * u, 'neutral') for u in Us]
        print(f"{label:>42} | " + "  ".join(f"{k:6.4f}" for k in ks))


if __name__ == '__main__':
    main()
