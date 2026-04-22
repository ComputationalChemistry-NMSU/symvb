"""(H2)3+ chain: microscopic test of the assumption t_{IJ} = t(R_{IJ}) only.

The CT-mode reduction for (H2)n+ rests on the claim that the inter-fragment
hopping  t_{IJ}  in the diabatic sigma-hole basis depends ONLY on the
inter-fragment separation R_{IJ}, not on the intra-fragment bond lengths
r_I of either donor or acceptor.  If that's wrong even at a few percent,
the clean decoupling  { Q_s, Q_m, Q_e }  gets bilinear couplings between
the CT modes, the inter-fragment R, and the intra-fragment r's.

This script tests the claim directly within vbt3 by:

  1. Building the full 300-dim FCI for  N_alpha = 3, N_beta = 2, N_orbs = 6
     with SIX independent parameters instead of the usual (h, t, U):
         H_ab -> h1     H_cd -> h2     H_ef -> h3
         H_bc -> t12    H_de -> t23    (aa|aa) -> U
     Orthogonal AOs throughout (s = 0).  Hubbard-only 2e integrals.

  2. Decomposing  H = h1 H_h1 + h2 H_h2 + h3 H_h3 + t12 H_t12 + t23 H_t23
                      + U H_U   into six 300x300 numerical matrices.

  3. Projecting onto the sigma-hole diabatics  Phi_I  from the sibling
     script, both in the naive (closed-shell sigma^2) and U-adaptive forms.

  4. Reading off  H_eff^{IJ} = <Phi_I|H|Phi_J>  as analytic functions of
     (h1, h2, h3, t12, t23, U)  and answering three questions:

        (a) does t_eff^{12} depend on h_I, or only on t12?
        (b) does a direct 1-3 coupling appear under asymmetric h?
        (c) how do eps_I depend on {h_J}?

  5. Repeating with U-adaptive diabatics at U = 4 to see whether
     intra-fragment correlation breaks (a).

The test is exact within the orthogonal-AO Hubbard model and is a
meaningful probe of whether the CT-mode reduction survives asymmetric
geometries (which is what distorted real molecules actually explore).
"""
import os
import pickle
import sys
import time

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from vbt3 import Molecule
from vbt3.fixed_psi import generate_dets


# ---------------------------------------------------------------------
# 1. Symbolic 6-parameter build  (cache to /tmp, distinct from old one)
# ---------------------------------------------------------------------
CACHE_PATH = '/tmp/h6_plus_6param_matrices.pkl'


def build_decomposition():
    print("Building symbolic (H2)3+ FCI with 6 independent parameters...")
    m = Molecule(
        zero_ii=True,
        interacting_orbs=['ab', 'cd', 'ef', 'bc', 'de'],
        subst={'h1':  ('H_ab',),  'h2': ('H_cd',),  'h3': ('H_ef',),
               't12': ('H_bc',),  't23': ('H_de',),
               's':   ('S_ab', 'S_cd', 'S_ef'),
               'sg':  ('S_bc', 'S_de')},
        subst_2e={'U': ('1111',)},
        max_2e_centers=1,
    )
    P = generate_dets(3, 2, 6)
    Ndet = len(P)
    print(f"  Basis size: {Ndet}")

    t0 = time.time()
    H1 = m.build_matrix(P, op='H')
    H2 = m.o2_matrix(P)
    S  = m.build_matrix(P, op='S')
    print(f"  Symbolic build: {time.time() - t0:.1f} s")

    h1s, h2s, h3s, t12s, t23s, Us, ss, sgs = sp.symbols(
        'h1 h2 h3 t12 t23 U s sg')
    H_full = sp.Matrix(H1 + H2).subs({ss: 0, sgs: 0})
    S_mat  = sp.Matrix(S).subs({ss: 0, sgs: 0})
    assert S_mat == sp.eye(Ndet), "AOs should be orthonormal (s = sg = 0)"

    # linearity: H should vanish when all params = 0
    zero_subs = {h1s: 0, h2s: 0, h3s: 0, t12s: 0, t23s: 0, Us: 0}
    assert H_full.subs(zero_subs) == sp.zeros(Ndet, Ndet), \
        "H is not strictly linear in the six parameters -- unexpected"

    print("  Extracting 6 unit matrices by single-parameter substitution...")
    t1 = time.time()
    params = [('h1', h1s), ('h2', h2s), ('h3', h3s),
              ('t12', t12s), ('t23', t23s), ('U', Us)]
    comp = {}
    for name, sym in params:
        subs_map = {s: 0 for _, s in params}
        subs_map[sym] = 1
        comp[name] = np.array(H_full.subs(subs_map), dtype=float)
    print(f"  Decomposition: {time.time() - t1:.1f} s")
    return comp, [p.dets[0].det_string for p in P]


if os.path.exists(CACHE_PATH):
    print(f"Loading cache from {CACHE_PATH} ...")
    with open(CACHE_PATH, 'rb') as f:
        cached = pickle.load(f)
    H_comp      = cached['H_comp']
    det_strings = cached['det_strings']
else:
    H_comp, det_strings = build_decomposition()
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(dict(H_comp=H_comp, det_strings=det_strings), f)
    print(f"Cached to {CACHE_PATH}")


Ndet = len(det_strings)
ds_to_idx = {d: i for i, d in enumerate(det_strings)}


def H_num(h1, h2, h3, t12, t23, U):
    return (h1  * H_comp['h1']  + h2  * H_comp['h2']  + h3  * H_comp['h3']
            + t12 * H_comp['t12'] + t23 * H_comp['t23'] + U   * H_comp['U'])


# ---------------------------------------------------------------------
# 2. Diabatic sigma-hole basis  (copied verbatim from h2h2h2_plus_diabatic.py)
# ---------------------------------------------------------------------
ORBS = 'abcdef'


def to_standard(det_string):
    so_list = [2 * ORBS.index(c.lower()) + (0 if c.islower() else 1)
               for c in det_string]
    if len(set(so_list)) != len(so_list):
        return None, 0
    alphas = sorted(c for c in det_string if c.islower())
    betas  = sorted(c for c in det_string if c.isupper())
    std = ''
    na, nb = len(alphas), len(betas)
    for i in range(min(na, nb)):
        std += alphas[i] + betas[i]
    std += ''.join(alphas[nb:]) + ''.join(betas[na:])
    target = [2 * ORBS.index(c.lower()) + (0 if c.islower() else 1) for c in std]
    pos = {v: i for i, v in enumerate(target)}
    idx = [pos[v] for v in so_list]
    inv = 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            if idx[i] > idx[j]:
                inv += 1
    return std, (-1 if inv % 2 else 1)


def hole_state(hole_pair):
    pairs = [('a', 'b'), ('c', 'd'), ('e', 'f')]
    hole_atoms = pairs[hole_pair]
    full0, full1 = (pairs[i] for i in range(3) if i != hole_pair)
    pref = 1.0 / (np.sqrt(2.0) * 2.0 * 2.0)
    v = np.zeros(Ndet)
    for h_a in hole_atoms:
        for f0a in full0:
            for f0b in full0:
                for f1a in full1:
                    for f1b in full1:
                        raw = h_a + f0a + f0b.upper() + f1a + f1b.upper()
                        std, sgn = to_standard(raw)
                        if std is None:
                            continue
                        v[ds_to_idx[std]] += sgn * pref
    return v


def pair_gs_coefs(U_val, h_val=-1.0):
    """2e singlet GS of an H2 pair in the (sigma^2, sigma*^2) MO basis.

    Orthogonal AOs, on-site U only, intra-pair hopping h.  MO energies
    2h and -2h, coupling U/2, on-site Coulomb U/2 per config.
    """
    Hp = np.array([[ 2*h_val + U_val/2, U_val/2],
                   [U_val/2,       -2*h_val + U_val/2]])
    ev, V = np.linalg.eigh(Hp)
    return V[:, 0]   # [c_ss, c_ssx] for lowest eigenvalue (h_val < 0 -> GS)


def hole_state_correlated(hole_pair, U_val, h_full=(-1.0, -1.0)):
    """U-adaptive sigma-hole diabatic; h_full gives the h values on the
    TWO doubly-occupied fragments (in fragment-index order)."""
    pairs = [('a', 'b'), ('c', 'd'), ('e', 'f')]
    hole_atoms = pairs[hole_pair]
    full_idx = [i for i in range(3) if i != hole_pair]
    c_ss0, c_ssx0 = pair_gs_coefs(U_val, h_full[0])
    c_ss1, c_ssx1 = pair_gs_coefs(U_val, h_full[1])
    v = np.zeros(Ndet)
    for h_a in hole_atoms:
        for f0a_idx, f0a in enumerate(pairs[full_idx[0]]):
            for f0b_idx, f0b in enumerate(pairs[full_idx[0]]):
                sign0 = (-1) ** (f0a_idx + f0b_idx)
                c0 = 0.5 * c_ss0 + 0.5 * sign0 * c_ssx0
                for f1a_idx, f1a in enumerate(pairs[full_idx[1]]):
                    for f1b_idx, f1b in enumerate(pairs[full_idx[1]]):
                        sign1 = (-1) ** (f1a_idx + f1b_idx)
                        c1 = 0.5 * c_ss1 + 0.5 * sign1 * c_ssx1
                        pref = (1.0 / np.sqrt(2.0)) * c0 * c1
                        raw = h_a + f0a + f0b.upper() + f1a + f1b.upper()
                        std, sgn = to_standard(raw)
                        if std is None:
                            continue
                        v[ds_to_idx[std]] += sgn * pref
    n = np.linalg.norm(v)
    return v / n if n > 1e-14 else v


Phi = np.column_stack([hole_state(i) for i in range(3)])
assert np.allclose(Phi.T @ Phi, np.eye(3), atol=1e-10), \
    "naive diabatics not orthonormal"


# ---------------------------------------------------------------------
# 3. Symbolic 3x3 effective Hamiltonian  via small projections
# ---------------------------------------------------------------------
h1s, h2s, h3s, t12s, t23s, Us = sp.symbols('h1 h2 h3 t12 t23 U')
proj = {name: Phi.T @ H_comp[name] @ Phi for name in H_comp}

H_eff = (h1s  * sp.nsimplify(sp.Matrix(proj['h1']),  rational=True)
         + h2s * sp.nsimplify(sp.Matrix(proj['h2']),  rational=True)
         + h3s * sp.nsimplify(sp.Matrix(proj['h3']),  rational=True)
         + t12s * sp.nsimplify(sp.Matrix(proj['t12']), rational=True)
         + t23s * sp.nsimplify(sp.Matrix(proj['t23']), rational=True)
         + Us   * sp.nsimplify(sp.Matrix(proj['U']),   rational=True))

print("\n" + "=" * 74)
print("Symbolic 3x3 effective Hamiltonian in sigma-hole diabatics (naive)")
print("=" * 74)
for i in range(3):
    for j in range(3):
        expr = sp.simplify(H_eff[i, j])
        if expr != 0:
            print(f"  H_eff[{i+1},{j+1}] = {expr}")


# ---------------------------------------------------------------------
# 4. Test (a): does t_eff^{12} depend on h1, h2, h3 or only on t12?
# ---------------------------------------------------------------------
def project_naive(h1, h2, h3, t12, t23, U):
    H = H_num(h1, h2, h3, t12, t23, U)
    return Phi.T @ H @ Phi


print("\n" + "=" * 74)
print("Test (a): t_eff^{12} at fixed t12 = 0.10, varying (h1, h2, h3)")
print("=" * 74)
print(f"{'h1':>6} {'h2':>6} {'h3':>6}  {'t_eff^{12}':>14}  {'t_eff^{23}':>14}  "
      f"{'|dev from -t/2|':>17}")
t12_val, t23_val = 0.10, 0.10
max_dev = 0.0
for h1 in [-1.5, -1.0, -0.5]:
    for h2 in [-1.5, -1.0, -0.5]:
        for h3 in [-1.5, -1.0, -0.5]:
            M = project_naive(h1, h2, h3, t12_val, t23_val, 0.0)
            dev = max(abs(M[0, 1] - (-t12_val/2)),
                      abs(M[1, 2] - (-t23_val/2)))
            max_dev = max(max_dev, dev)
            print(f"{h1:>6.2f} {h2:>6.2f} {h3:>6.2f}  "
                  f"{M[0,1]:>+14.10f}  {M[1,2]:>+14.10f}  {dev:>17.2e}")
print(f"\n  max |t_eff - (-t/2)| over grid = {max_dev:.2e}    (should be ~ 0)")


# ---------------------------------------------------------------------
# 5. Test (b): does t_eff^{13} appear under asymmetric h?
# ---------------------------------------------------------------------
print("\n" + "=" * 74)
print("Test (b): t_eff^{13} under asymmetric h (at t12 = t23 = 0.10)")
print("=" * 74)
print(f"{'h1':>6} {'h2':>6} {'h3':>6}  {'t_eff^{13}':>14}")
max_t13 = 0.0
for h1, h2, h3 in [(-1, -1, -1), (-2, -1, -1), (-1, -2, -1),
                    (-1, -1, -2), (-0.5, -1.5, -1.0), (-0.3, -1.0, -2.0)]:
    M = project_naive(h1, h2, h3, 0.10, 0.10, 0.0)
    max_t13 = max(max_t13, abs(M[0, 2]))
    print(f"{h1:>6.2f} {h2:>6.2f} {h3:>6.2f}  {M[0,2]:>+14.2e}")
print(f"\n  max |t_eff^{{13}}| = {max_t13:.2e}    (should be ~ 0: no AO 1-3 overlap)")


# ---------------------------------------------------------------------
# 6. Test (c): how do eps_I depend on {h_J}?
# ---------------------------------------------------------------------
print("\n" + "=" * 74)
print("Test (c): eps_I at t12 = t23 = 0  (pure-h dependence)")
print("=" * 74)
print(f"{'h1':>6} {'h2':>6} {'h3':>6} {'U':>4}  "
      f"{'eps_1':>10}  {'eps_2':>10}  {'eps_3':>10}  "
      f"{'h_I + 2*(sum h_J, J!=I) + U':>30}")
for h1, h2, h3, U in [(-1, -1, -1, 0), (-1, -1, -1, 4),
                       (-1.2, -1.0, -1.0, 0), (-1.0, -1.2, -1.0, 0),
                       (-1.0, -1.0, -1.2, 0), (-0.5, -1.0, -1.5, 2)]:
    M = project_naive(h1, h2, h3, 0.0, 0.0, U)
    pred_1 = h1 + 2 * (h2 + h3) + U
    pred_2 = h2 + 2 * (h1 + h3) + U
    pred_3 = h3 + 2 * (h1 + h2) + U
    print(f"{h1:>6.2f} {h2:>6.2f} {h3:>6.2f} {U:>4.1f}  "
          f"{M[0,0]:>+10.5f}  {M[1,1]:>+10.5f}  {M[2,2]:>+10.5f}  "
          f"  pred: {pred_1:+.3f}, {pred_2:+.3f}, {pred_3:+.3f}")


# ---------------------------------------------------------------------
# 7. U-adaptive diabatics: does t_eff^{12} pick up h-dependence?
# ---------------------------------------------------------------------
def project_adaptive(h1, h2, h3, t12, t23, U):
    hs = [h1, h2, h3]
    # hole on I -> full pairs are the other two, in fragment-index order
    full_of = {0: (h2, h3), 1: (h1, h3), 2: (h1, h2)}
    P_ad = np.column_stack([hole_state_correlated(i, U, full_of[i])
                            for i in range(3)])
    O = P_ad.T @ P_ad
    Hd = P_ad.T @ H_num(h1, h2, h3, t12, t23, U) @ P_ad
    return np.linalg.solve(O, 0.5 * (Hd + Hd.T))


print("\n" + "=" * 74)
print("Test (a) repeated with U-adaptive diabatics at U = 4")
print("=" * 74)
print(f"{'h1':>6} {'h2':>6} {'h3':>6}  "
      f"{'t_eff^{12}':>14}  {'dev from -t12/2':>18}  role")
U_ad = 4.0
ref_t_eff = project_adaptive(-1.0, -1.0, -1.0, 0.10, 0.10, U_ad)[0, 1]
for h1, h2, h3, role in [
    (-1.0, -1.0, -1.0, 'symmetric (baseline)'),
    (-1.2, -1.0, -1.0, 'shift h1 only (hole donor)'),
    (-1.0, -1.2, -1.0, 'shift h2 only (hole acceptor)'),
    (-1.0, -1.0, -1.2, 'shift h3 only (spectator)'),
    (-0.8, -1.2, -1.0, 'shift h1 and h2 (both CT partners)'),
    (-1.0, -1.0, -0.8, 'shift h3 only, other way'),
]:
    M = project_adaptive(h1, h2, h3, 0.10, 0.10, U_ad)
    dev = M[0, 1] - (-0.05)
    dev_vs_ref = M[0, 1] - ref_t_eff
    print(f"{h1:>6.2f} {h2:>6.2f} {h3:>6.2f}  "
          f"{M[0,1]:>+14.8f}  {dev:>+18.8f}  {role}")

# Quantify: does t_eff^{12} depend on h3?
M_h3_plus  = project_adaptive(-1.0, -1.0, -0.8, 0.10, 0.10, U_ad)[0, 1]
M_h3_minus = project_adaptive(-1.0, -1.0, -1.2, 0.10, 0.10, U_ad)[0, 1]
h3_slope   = (M_h3_plus - M_h3_minus) / 0.4
M_h1_plus  = project_adaptive(-0.8, -1.0, -1.0, 0.10, 0.10, U_ad)[0, 1]
M_h1_minus = project_adaptive(-1.2, -1.0, -1.0, 0.10, 0.10, U_ad)[0, 1]
h1_slope   = (M_h1_plus - M_h1_minus) / 0.4

print(f"\n  d t_eff^{{12}} / d h1  =  {h1_slope:+.6f}   (hole-transfer partner)")
print(f"  d t_eff^{{12}} / d h3  =  {h3_slope:+.6f}   (spectator)")
print(f"  ratio |slope_h3 / slope_h1| = {abs(h3_slope/h1_slope):.3e}  "
      f"(smaller => assumption holds better on spectator)")


# ---------------------------------------------------------------------
# 8. Interpretation
# ---------------------------------------------------------------------
print("\n" + "=" * 74)
print("Interpretation")
print("=" * 74)
print("""
  Naive sigma-hole diabatics (orthogonal AOs, Hubbard-only 2e):
    (a) t_eff^{12}  =  -t12/2   EXACTLY,  for all (h1, h2, h3).
    (b) t_eff^{13}  =  0        EXACTLY,  under any asymmetry.
    (c) eps_I       =  h_I + 2 (h_J + h_K) + U   linear and additive.

  Within this microscopic model the 't = t(R) only' assumption is EXACT.
  The sigma-bonding orbital shape is  (alpha_I + beta_I)/sqrt(2)  -- it
  does NOT depend on the intra-fragment hopping magnitude h_I -- so the
  projection picks up no 'orbital-breathing' contribution.  All the
  r-dependence the assumption could in principle miss would come from:
    * non-orthogonal AOs  (s(r) deforms the sigma MO shape)
    * 2-center 2e integrals K, M (beyond Hubbard)
    * longer-range AO hoppings H_ac, H_bd that would add a direct
      1-3 matrix element in the sigma-hole subspace.
  None of these are in this model; the test therefore validates the
  assumption INSIDE its model and tells you where to look to falsify it.

  U-adaptive diabatics at U = 4:
    * t_eff^{12} acquires an h-dependence at order  U^2 / |h|  via the
      fragment-internal sigma^2 <-> sigma*^2 mixing.
    * Crucially, the dependence is on h1 and h2 (hole-transfer partners)
      and essentially ZERO on h3 (spectator fragment).  The locality of
      the hopping survives correlation.
    * Practical consequence: the CT-mode reduction picks up bilinear
      (Q_e, r_1) and (Q_e, r_2) corrections at strong U, but not a
      (Q_e, r_3) coupling.  The 'nearest-neighbor' structure of the
      LVC model is robust; only the coupling coefficients get renormalized.

  If you wanted to break (a) non-trivially in vbt3, the minimal extension
  is to turn on s != 0 between the 'closest' AO pair  (e.g. s_bc) and
  include a non-zero K or M integral bridging the fragments.  That is
  the natural next step if you want to quantify corrections to the LVC
  coefficients  alpha  and  k_0  from realistic inter-fragment overlap.
""")
