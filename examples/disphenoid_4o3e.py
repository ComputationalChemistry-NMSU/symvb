"""Tetrahedral disphenoid (H2)2 cation: minimal Robin-Day mixed valence test.

System geometry: 4 s-orbitals at the vertices of a D_{2d} disphenoid --
a tetrahedron with two short opposite edges (the "H-H bond" pairs) and
four equivalent long edges (inter-pair couplings):

                   a -------- b       <- short edge  (strong h_s)
                   |\        /|
                   | \      / |       <- 4 long edges (weak h_l)
                   |  \    /  |          (a-c, a-d, b-c, b-d)
                   |   \  /   |
                   |    \/    |
                   |    /\    |
                   c -------- d       <- short edge  (strong h_s)

Chemical reading: two hydrogen molecules (a-b and c-d) weakly coupled via
the four equivalent through-space contacts a-c, a-d, b-c, b-d.

Electron counts:
    4e (neutral):   closed-shell (H2)(H2),  both bonds intact.
    3e (cation):    1 hole distributed between the two H-H units.

The 3-electron cation is the MINIMAL Robin-Day mixed-valence system --
smaller than any bridged metal dimer, smaller even than the Creutz-Taube
ion.  The question: is the hole delocalised (Class III) or trapped on
one H-H unit (Class II)?  The answer depends on the competition between:

    * inter-pair coupling  |h_l|  (stabilises the delocalised hole state)
    * elastic restoring force  k   against asymmetric bond distortion

We show two results analytically and numerically:

(1) The symmetric-limit 3-electron ground-state energy has a closed form

        E_0(eta, h_l) = 3 h_s - sqrt( eta^2 / 4 + 4 h_l^2 )             (*)

    where  eta = h_{s,1} - h_{s,2}  is the asymmetric-stretch coordinate
    measured in units of the resonance integral.  (*) is an EXACT
    3-electron Hueckel result -- the ground state is a single closed-shell
    configuration (doubly-occupied bonding MO + singly-occupied inter-pair
    antibonding MO of B_2 symmetry), no CI admixture.

(2) The electronic-energy curvature at the symmetric point is

        d^2 E_elec / d eta^2 |_{eta = 0}  =  -1 / (8 |h_l|)             (**)

    ALWAYS NEGATIVE -- the symmetric disphenoid is pseudo-Jahn-Teller
    unstable against the asymmetric stretch at every |h_l| < |h_s|.
    Adding an elastic restoring term  E_elastic = (1/2) k eta^2  gives
    the Robin-Day III <-> II critical stiffness

        k_crit  =  1 / (8 |h_l|)                                        (***)

    k > k_crit: symmetric minimum -> Class III (fully delocalised).
    k < k_crit: symmetry-broken double-well -> Class II (trapped mixed valence).

The 4-electron neutral has d^2 E_elec / d eta^2 = 0 in the Hueckel limit
(each pair is independently saturated -- no asymmetric-stretch
instability at all), so the pseudo-Jahn-Teller instability is
*charge-induced*: removing the electron turns on the reorganization.

Run from the repo root:  PYTHONPATH=. python3 examples/disphenoid_4o3e.py
"""
import os
import sys
import time

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, hamiltonian
from symvb.fixed_psi import generate_dets


def build_disphenoid_H(Na, Nb, include_U=True):
    """Build symbolic H matrix for (Na, Nb) electrons on the disphenoid.

    Convention: a, b share a short edge; c, d share a short edge.
    Four long edges (a-c, a-d, b-c, b-d) are all equivalent.
    """
    subst_2e = {'U': ('1111',)} if include_U else {}
    m = Molecule(
        zero_ii=True,
        interacting_orbs=['ab', 'cd', 'ac', 'ad', 'bc', 'bd'],
        subst={'h_s_1': ('H_ab',),
               'h_s_2': ('H_cd',),
               'h_l':   ('H_ac', 'H_ad', 'H_bc', 'H_bd'),
               's':     ('S_ab', 'S_cd', 'S_ac', 'S_ad', 'S_bc', 'S_bd')},
        subst_2e=subst_2e,
        max_2e_centers=1,
    )
    P = generate_dets(Na, Nb, 4)
    # hamiltonian() folds the on-site 2e block into H; skip it explicitly
    # for the deliberately one-electron (include_U=False) model
    H, _ = hamiltonian(m, P, two_electron=include_U)
    return P, sp.Matrix(H)


def decompose_linear(H, symbols, fixed_subs):
    """Express H = H0 + sum_i p_i * M_i numerically, given H is linear in
    the symbols.  fixed_subs sets the other symbols (e.g. s -> 0)."""
    H_fixed = H.subs(fixed_subs)
    zeros = {sym: 0 for sym in symbols}
    H0 = np.array(H_fixed.subs(zeros).tolist(), dtype=float)
    coefs = [np.array(sp.diff(H_fixed, sym).tolist(), dtype=float) for sym in symbols]
    return H0, coefs


# ------------------------------------------------------------------------
# 1. Build 3e (cation) and 4e (neutral) Hamiltonians and decompose
# ------------------------------------------------------------------------
h_s_1, h_s_2, h_l, s, U = sp.symbols('h_s_1 h_s_2 h_l s U')
symbols = [h_s_1, h_s_2, h_l, U]

print("Building 3-electron (cation) H ...")
t0 = time.time()
P3, H3 = build_disphenoid_H(2, 1, include_U=True)
print(f"  {len(P3)} dets, build {time.time() - t0:.1f}s")
H3_0, H3_M = decompose_linear(H3, symbols, {s: 0})

print("Building 4-electron (neutral) H ...")
t0 = time.time()
P4, H4 = build_disphenoid_H(2, 2, include_U=True)
print(f"  {len(P4)} dets, build {time.time() - t0:.1f}s")
H4_0, H4_M = decompose_linear(H4, symbols, {s: 0})


def Egs(H0, M, hs1v, hs2v, hlv, Uv=0):
    H = H0 + hs1v * M[0] + hs2v * M[1] + hlv * M[2] + Uv * M[3]
    H = (H + H.T) / 2
    return np.linalg.eigvalsh(H)[0]


# ------------------------------------------------------------------------
# 2. Verify the closed form E_0 = 3 h_s - sqrt(eta^2/4 + 4 h_l^2) at U=0
# ------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Closed-form verification: E_0(eta, h_l) = 3 h_s - sqrt(eta^2/4 + 4 h_l^2)")
print("  (3 electrons, s=0, U=0, Hueckel limit)")
print("=" * 72)
print(f"\n{'h_s_1':>7}  {'h_s_2':>7}  {'h_l':>6}  {'E_num':>12}  {'E_closed':>12}  {'diff':>10}")
for hs1v, hs2v, hlv in [(-1, -1, -0.3), (-1.3, -0.7, -0.3),
                        (-1.5, -0.5, -0.5), (-1, -1, -0.1),
                        (-0.9, -1.1, -0.7), (-1.2, -0.8, 0.0)]:
    E_num = Egs(H3_0, H3_M, hs1v, hs2v, hlv, 0)
    Hs_mean = (hs1v + hs2v) / 2
    eta = hs1v - hs2v
    E_cf = 3 * Hs_mean - np.sqrt(eta**2 / 4 + 4 * hlv**2)
    print(f"{hs1v:7.2f}  {hs2v:7.2f}  {hlv:6.2f}  "
          f"{E_num:+12.6f}  {E_cf:+12.6f}  {E_num - E_cf:+10.2e}")


# ------------------------------------------------------------------------
# 3. Curvature at the symmetric point: cation (3e) vs neutral (4e)
# ------------------------------------------------------------------------
eps = 0.005
print("\n" + "=" * 72)
print("Asymmetric-stretch curvature  d^2 E_elec / d eta^2 |_{eta = 0}")
print("=" * 72)
print(f"\n  neutral (4e) vs cation (3e) at h_s = -1:\n")
print(f"{'h_l':>6}  {'U':>5} | {'4e curv':>12}  {'3e curv':>12}  "
      f"  {'predicted 3e':>14}  {'k_crit':>10}")
for hlv in [-0.1, -0.2, -0.3, -0.5, -0.7, -0.9]:
    for Uv in [0, 2, 5]:
        # Curvature: eta = h_s_1 - h_s_2.  Distortion: h_s_1 = -1 - eps/2, h_s_2 = -1 + eps/2
        E4_0 = Egs(H4_0, H4_M, -1, -1, hlv, Uv)
        E4_p = Egs(H4_0, H4_M, -1 - eps / 2, -1 + eps / 2, hlv, Uv)
        curv4 = 2 * (E4_p - E4_0) / eps**2

        E3_0 = Egs(H3_0, H3_M, -1, -1, hlv, Uv)
        E3_p = Egs(H3_0, H3_M, -1 - eps / 2, -1 + eps / 2, hlv, Uv)
        curv3 = 2 * (E3_p - E3_0) / eps**2

        # Predicted for U=0: -1/(8 |h_l|)
        pred = -1 / (8 * abs(hlv)) if Uv == 0 else np.nan
        # E_elastic = (1/2) k eta^2.  Total curvature: d^2 E_elec/d eta^2 + k = 0 at k_crit.
        k_crit_val = -curv3
        print(f"{hlv:6.2f}  {Uv:5.1f} | {curv4:+12.5f}  {curv3:+12.5f}   "
              f"{pred:+14.5f}  {k_crit_val:10.5f}")


# ------------------------------------------------------------------------
# 4. Equilibrium distortion eta* as function of elastic stiffness k
#    E_tot(eta) = E_elec(eta) + (1/2) k eta^2
# ------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Equilibrium asymmetric stretch eta* at various k (U = 0, h_s = -1)")
print("  eta = 0     -> Class III (fully delocalised hole)")
print("  |eta*| > 0  -> Class II  (trapped mixed valence)")
print("=" * 72)
print(f"\n{'h_l':>6} | " +
      "  ".join(f"{'k=' + str(k):>10}" for k in [0.1, 0.3, 0.5, 1.0, 2.0]))
etas = np.linspace(-1.0, 1.0, 801)
for hlv in [-0.1, -0.2, -0.3, -0.5, -0.7, -0.9]:
    row = [f"{hlv:6.2f} |"]
    # Sweep eta
    E_elec = np.array([Egs(H3_0, H3_M, -1 - e / 2, -1 + e / 2, hlv, 0)
                       for e in etas])
    for kv in [0.1, 0.3, 0.5, 1.0, 2.0]:
        Etot = E_elec + 0.5 * kv * etas**2
        i = np.argmin(Etot)
        row.append(f"  eta* = {etas[i]:+5.2f}")
    print(" ".join(row))


# ------------------------------------------------------------------------
# 5. Site populations: is the hole localised on c-d side at eta > 0?
# ------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Site populations in the 3e ground state vs eta  (h_l = -0.3, U = 2)")
print("=" * 72)
det_strings = [p.dets[0].det_string for p in P3]
hlv, Uv = -0.3, 2.0
print(f"\n{'eta':>6}  {'E_elec':>10}  {'n(a+b)':>8}  {'n(c+d)':>8}  "
      f"{'hole@a+b':>10}  {'hole@c+d':>10}")
for eta in [0, 0.1, 0.2, 0.3, 0.5, 0.7]:
    H = (H3_0 + (-1 - eta / 2) * H3_M[0] + (-1 + eta / 2) * H3_M[1]
         + hlv * H3_M[2] + Uv * H3_M[3])
    H = (H + H.T) / 2
    Ev, V = np.linalg.eigh(H)
    c0 = V[:, 0]
    ns = [0.0] * 4
    for i, ds in enumerate(det_strings):
        for site in range(4):
            ns[site] += c0[i]**2 * ds.lower().count('abcd'[site])
    n_ab = ns[0] + ns[1]
    n_cd = ns[2] + ns[3]
    # Each pair has max 2 electrons;  "hole weight" = 2 - n(pair).
    # With 3 electrons and 2 pairs of max 2 each, the hole is shared 1.0 total
    # (i.e.  (2 - n_ab) + (2 - n_cd) = 4 - 3 = 1  by electron count).
    print(f"{eta:6.2f}  {Ev[0]:+10.5f}  {n_ab:8.5f}  {n_cd:8.5f}  "
          f"{2 - n_ab:10.5f}  {2 - n_cd:10.5f}")

print("\n  At eta = 0 the hole is symmetrically distributed (0.5 on each pair).")
print("  Positive eta stretches the c-d bond (|h_s_2| = |h_s|(1 - eta/|h_s|) smaller);")
print("  the hole migrates to c-d, consistent with the Class II picture.")
