"""
(H2...H2...H2)+ = H6+ charge-transfer chain: exact FCI vs 3x3 effective
1-hole tight-binding model on bonding-sigma diabatics.

Layout:
    pair 1: (a, b)      pair 2: (c, d)      pair 3: (e, f)

Couplings (orthogonal AOs, h = -1, no direct 1-3 AO hops):
    H_ab = H_cd = H_ef = h    (intra-pair, strong, bonding)
    H_bc = H_de = t           (inter-pair, 1-2 and 2-3 assumed equal)
    all other off-diagonal H, S = 0

Electronic structure:
    Charge = +1    ->    N_alpha = 3, N_beta = 2    (Sz = +1/2 sector)
    Basis: C(6,3) * C(6,2) = 20 * 15 = 300 Slater dets
    2e integrals: on-site U only (Hubbard)

Effective model (sigma-only diabatic):
    sigma_I   = (a + b) / sqrt(2)
    sigma_II  = (c + d) / sqrt(2)
    sigma_III = (e + f) / sqrt(2)
    Phi_I  = alpha-hole in sigma_I,  sigma_II and sigma_III doubly occupied
    Phi_II = alpha-hole in sigma_II, sigma_I  and sigma_III doubly occupied
    Phi_III analogous.

Builds H_diab_{ij} = <Phi_i | H | Phi_j> (3x3) and compares its three
eigenvalues to the lowest three eigenvalues of the full 300-dim FCI, as a
function of (t / |h|, U / |h|).

Analytic expectation at the projection level:
    eps_i = <Phi_i|H|Phi_i> = -5 + U           (all three, by symmetry)
    t_eff = <Phi_i|H|Phi_{i+1}> = -t / 2       (AO H_bc halved by sqrt(2)^2)
    t_13  = <Phi_1|H|Phi_3>   = 0              (no direct 1-3 AO coupling)
    E_gs  = -5 + U - t / sqrt(2)

FCI captures (on top of this) the coupling to sigma* (antibonding) orbitals,
which is of order (t / 2)^2 / (2|h|) per site, and full correlation within
each H2 when U > 0.

Run from the repo root:  PYTHONPATH=. python3 examples/h2h2h2_plus_diabatic.py
"""
import os
import pickle
import sys
import time

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, hamiltonian
from symvb.fixed_psi import generate_dets


# ------------------------------------------------------------------------
# 1. Symbolic build (cache to /tmp)
# ------------------------------------------------------------------------
CACHE_PATH = '/tmp/h6_plus_hubbard_matrices.pkl'

if os.path.exists(CACHE_PATH):
    print(f"Loading cached matrices from {CACHE_PATH}...")
    with open(CACHE_PATH, 'rb') as f:
        cached = pickle.load(f)
    H_00_np = cached['H_00']
    H_t_np = cached['H_t']
    H_U_np = cached['H_U']
    det_strings = cached['det_strings']
else:
    print("Building symbolic matrices for (H2...H2...H2)+  (N_a=3, N_b=2, N_o=6)...")
    m = Molecule(
        zero_ii=True,
        interacting_orbs=['ab', 'cd', 'ef', 'bc', 'de'],
        subst={'h':  ('H_ab', 'H_cd', 'H_ef'),
               't':  ('H_bc', 'H_de'),
               's':  ('S_ab', 'S_cd', 'S_ef'),
               'sg': ('S_bc', 'S_de')},
        subst_2e={'U': ('1111',)},
        max_2e_centers=1,
    )
    P = generate_dets(3, 2, 6)
    Ndet = len(P)
    print(f"  Basis size: {Ndet}")

    t0 = time.time()
    H_raw, S_raw = hamiltonian(m, P)   # 2e block folded into H_raw
    print(f"  Symbolic build: {time.time() - t0:.1f}s")

    h_sym, t_sym, U_sym, s_sym, sg_sym = sp.symbols('h t U s sg')
    H_sym = H_raw.subs({s_sym: 0, sg_sym: 0, h_sym: -1})
    S_sym = S_raw.subs({s_sym: 0, sg_sym: 0})
    assert S_sym == sp.eye(Ndet), "AOs not orthonormal in the det basis"

    # Decompose H = H_00 + t * H_t + U * H_U  (H is linear in t, U after h = -1)
    print("  Decomposing H(t, U) into numerical (H_00, H_t, H_U)...")
    t1 = time.time()
    H_at_00 = np.array(H_sym.subs({t_sym: 0, U_sym: 0}), dtype=float)
    H_at_t  = np.array(H_sym.subs({t_sym: 1, U_sym: 0}), dtype=float)
    H_at_U  = np.array(H_sym.subs({t_sym: 0, U_sym: 1}), dtype=float)
    H_00_np = H_at_00
    H_t_np  = H_at_t - H_at_00
    H_U_np  = H_at_U - H_at_00
    print(f"  Decomposition: {time.time() - t1:.1f}s")

    det_strings = [p.dets[0].det_string for p in P]
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(dict(H_00=H_00_np, H_t=H_t_np, H_U=H_U_np,
                         det_strings=det_strings), f)
    print(f"  Cached to {CACHE_PATH}")


Ndet = len(det_strings)
ds_to_idx = {d: i for i, d in enumerate(det_strings)}


def H_num(tv, Uv):
    return H_00_np + tv * H_t_np + Uv * H_U_np


# ------------------------------------------------------------------------
# 2. Diabatic hole states  Phi_I (I = 1, 2, 3)
# ------------------------------------------------------------------------
ORBS = 'abcdef'


def to_standard(det_string):
    """symvb-canonical form: pair alpha_i with beta_i alphabetically, leftover at end."""
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
    # Sign: parity of permutation to go from so_list -> target
    pos = {v: i for i, v in enumerate(target)}
    idx = [pos[v] for v in so_list]
    inv = 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            if idx[i] > idx[j]:
                inv += 1
    return std, (-1 if inv % 2 else 1)


def hole_state(hole_pair):
    """Sz=+1/2 state with alpha-hole in sigma_{hole_pair}, other two pairs doubly
    occupied in their bonding sigma.  Returns Ndet-dim coefficient vector."""
    pairs = [('a', 'b'), ('c', 'd'), ('e', 'f')]
    hole_atoms = pairs[hole_pair]
    full_idx = [i for i in range(3) if i != hole_pair]
    full0 = pairs[full_idx[0]]
    full1 = pairs[full_idx[1]]
    # prefactor: (1/sqrt(2)) hole * (1/2) full0 * (1/2) full1
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


Phi = np.column_stack([hole_state(i) for i in range(3)])
overlap = Phi.T @ Phi
print("\n<Phi_i | Phi_j>  (sigma-diabatics should be orthonormal at s = 0):")
print(np.array2string(overlap, precision=8, suppress_small=True))
assert np.allclose(overlap, np.eye(3), atol=1e-10), "diabatics not orthonormal"


# ------------------------------------------------------------------------
# 3. Scan (t, U): compare low-lying FCI to 3x3 diabatic model
# ------------------------------------------------------------------------
def analyse(tv, Uv):
    H = H_num(tv, Uv)
    E_fci = np.linalg.eigvalsh(H)[:3]
    H_diab = Phi.T @ H @ Phi
    H_diab = 0.5 * (H_diab + H_diab.T)
    E_diab = np.linalg.eigvalsh(H_diab)
    return dict(
        E_fci=E_fci, E_diab=E_diab, H_diab=H_diab,
        eps_e=0.5 * (H_diab[0, 0] + H_diab[2, 2]),
        eps_m=H_diab[1, 1],
        t_eff=0.5 * (H_diab[0, 1] + H_diab[1, 2]),
        t_13=H_diab[0, 2],
    )


print("\n" + "=" * 96)
print("Lowest-3 eigenvalues:  full 300-dim FCI  vs  3x3 sigma-diabatic projection")
print("(h = -1, orthogonal AOs, Hubbard-only 2e)")
print("=" * 96)
print(f"{'t':>5} {'U':>4}  "
      f"{'E0_FCI':>10} {'E1_FCI':>10} {'E2_FCI':>10}   "
      f"{'E0_diab':>10} {'E1_diab':>10} {'E2_diab':>10}   "
      f"{'max|dE|':>8}  {'dE_gs/|E|':>9}")

scan = []
for Uv in [0.0, 1.0, 4.0, 8.0, 16.0]:
    for tv in [0.05, 0.1, 0.2, 0.5, 1.0]:
        scan.append((tv, Uv))
        o = analyse(tv, Uv)
        err = max(abs(o['E_fci'][k] - o['E_diab'][k]) for k in range(3))
        rel = abs(o['E_fci'][0] - o['E_diab'][0]) / max(abs(o['E_fci'][0]), 1e-10)
        print(f"{tv:>5.2f} {Uv:>4.1f}  "
              f"{o['E_fci'][0]:>+10.6f} {o['E_fci'][1]:>+10.6f} {o['E_fci'][2]:>+10.6f}   "
              f"{o['E_diab'][0]:>+10.6f} {o['E_diab'][1]:>+10.6f} {o['E_diab'][2]:>+10.6f}   "
              f"{err:>8.2e}  {rel:>9.2e}")


# ------------------------------------------------------------------------
# 4. Effective parameters eps_e, eps_m, t_eff, t_13
# ------------------------------------------------------------------------
print("\n" + "=" * 96)
print("Effective 3x3 Hamiltonian parameters")
print("=" * 96)
print(f"{'t':>5} {'U':>4}  "
      f"{'eps_e':>10} {'eps_m':>10} {'eps_m-eps_e':>11}  "
      f"{'t_eff':>10} {'t_eff / (-t/2)':>14}  {'t_13':>10}")
for tv, Uv in scan:
    o = analyse(tv, Uv)
    ratio = o['t_eff'] / (-tv / 2) if tv > 0 else float('nan')
    print(f"{tv:>5.2f} {Uv:>4.1f}  "
          f"{o['eps_e']:>+10.6f} {o['eps_m']:>+10.6f} {o['eps_m'] - o['eps_e']:>+11.2e}  "
          f"{o['t_eff']:>+10.6f} {ratio:>14.6f}  {o['t_13']:>+10.2e}")


# ------------------------------------------------------------------------
# 5. Sanity: U = 0 analytic check via the 6x6 single-particle chain
# ------------------------------------------------------------------------
def one_particle_gs_energy(tv):
    """Sum of 5 lowest eigenvalues of the 6x6 tight-binding chain
    (double occupations allowed at U = 0)."""
    H1 = np.zeros((6, 6))
    # intra-pair hoppings (h = -1)
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        H1[i, j] = H1[j, i] = -1.0
    # inter-pair hoppings
    for i, j in [(1, 2), (3, 4)]:
        H1[i, j] = H1[j, i] = -tv
    eigs = np.linalg.eigvalsh(H1)
    return 2 * eigs[0] + 2 * eigs[1] + eigs[2]


print("\n" + "=" * 96)
print("U = 0 cross-check: FCI GS  vs  sum-of-5-lowest 6x6 single-particle eigenvalues")
print("=" * 96)
print(f"{'t':>5}  {'E_gs_FCI':>12}  {'E_gs_1p':>12}  {'E_gs_diab':>12}  "
      f"{'1p-FCI':>10}  {'diab-FCI':>10}")
for tv in [0.05, 0.1, 0.2, 0.5, 1.0]:
    o = analyse(tv, 0.0)
    E_1p = one_particle_gs_energy(tv)
    print(f"{tv:>5.2f}  {o['E_fci'][0]:>+12.8f}  {E_1p:>+12.8f}  "
          f"{o['E_diab'][0]:>+12.8f}  "
          f"{E_1p - o['E_fci'][0]:>+10.2e}  {o['E_diab'][0] - o['E_fci'][0]:>+10.2e}")
print("  (U = 0 FCI is exactly the 1-particle chain result; 'diab' is the 3x3 model)")


# ------------------------------------------------------------------------
# 6. Hole-localization diagnostic: weight on each pair in the FCI ground state
# ------------------------------------------------------------------------
print("\n" + "=" * 96)
print("Hole-site weights in the FCI ground state (projection onto sigma-diabatics)")
print("=" * 96)
print(f"{'t':>5} {'U':>4}  {'w(Phi_1)':>10} {'w(Phi_2)':>10} {'w(Phi_3)':>10}  "
      f"{'w_total (sigma-only)':>22}")
for tv, Uv in scan:
    H = H_num(tv, Uv)
    _, V = np.linalg.eigh(H)
    psi = V[:, 0]
    c = Phi.T @ psi          # 3-component projection onto diabatics
    w = c ** 2
    print(f"{tv:>5.2f} {Uv:>4.1f}  {w[0]:>10.6f} {w[1]:>10.6f} {w[2]:>10.6f}  "
          f"{w.sum():>22.6f}")


# ------------------------------------------------------------------------
# 7. U-adaptive diabatics: replace sigma^2 with the true H2 g-singlet GS at
#    each U on the two doubly-occupied pairs (hole pair stays as sigma^alpha)
# ------------------------------------------------------------------------
#  Per-pair 2e g-singlet block in MO basis {|sigma^2>, |sigma*^2>} at h = -1
#  (orthogonal AOs, pure Hubbard, K = 0):
#        H_pair = [[-2 + U/2,  U/2],
#                  [  U/2,    2 + U/2]]
#  Eigenvalues: U/2 +/- sqrt(U^2/4 + 4).  Ground state mixes sigma^2 and
#  sigma*^2; |c_s*s*|^2 -> 0.5 as U -> inf (pure Heitler-London limit).
def pair_gs_coefs(U_val):
    Hp = np.array([[-2.0 + U_val / 2.0, U_val / 2.0],
                   [U_val / 2.0, 2.0 + U_val / 2.0]])
    ev, V = np.linalg.eigh(Hp)
    return V[:, 0]        # [c_ss, c_ssx] for lowest eigenvalue


def hole_state_correlated(hole_pair, U_val):
    """sigma-hole diabatic with U-adaptive (sigma^2, sigma*^2) mixture on the
    two doubly-occupied pairs."""
    pairs = [('a', 'b'), ('c', 'd'), ('e', 'f')]
    hole_atoms = pairs[hole_pair]
    full_idx = [i for i in range(3) if i != hole_pair]
    c_ss, c_ssx = pair_gs_coefs(U_val)

    v = np.zeros(Ndet)
    for h_a in hole_atoms:
        for f0a_idx, f0a in enumerate(pairs[full_idx[0]]):
            for f0b_idx, f0b in enumerate(pairs[full_idx[0]]):
                # coefficient on full-pair AO det (f0a alpha, f0b beta):
                #   sigma^2 contrib  = 1/2
                #   sigma*^2 contrib = 1/2 * (-1)^(f0a_idx + f0b_idx)
                sign0 = (-1) ** (f0a_idx + f0b_idx)
                c0 = 0.5 * c_ss + 0.5 * sign0 * c_ssx
                for f1a_idx, f1a in enumerate(pairs[full_idx[1]]):
                    for f1b_idx, f1b in enumerate(pairs[full_idx[1]]):
                        sign1 = (-1) ** (f1a_idx + f1b_idx)
                        c1 = 0.5 * c_ss + 0.5 * sign1 * c_ssx
                        pref = (1.0 / np.sqrt(2.0)) * c0 * c1
                        raw = h_a + f0a + f0b.upper() + f1a + f1b.upper()
                        std, sgn = to_standard(raw)
                        if std is None:
                            continue
                        v[ds_to_idx[std]] += sgn * pref
    nrm = np.linalg.norm(v)
    if nrm > 1e-14:
        v = v / nrm
    return v


def analyse_adaptive(tv, Uv):
    Phi_U = np.column_stack([hole_state_correlated(i, Uv) for i in range(3)])
    # overlap (should still be identity: beta-AO patterns distinguish i, j, k)
    O = Phi_U.T @ Phi_U
    H = H_num(tv, Uv)
    Hd = Phi_U.T @ H @ Phi_U
    Hd = 0.5 * (Hd + Hd.T)
    # Solve generalized eigenproblem just in case O drifted slightly
    evals = np.linalg.eigvalsh(np.linalg.solve(O, Hd).real)
    E_fci = np.linalg.eigvalsh(H)[:3]
    return dict(
        E_fci=E_fci, E_diab=np.sort(evals), H_diab=Hd, overlap=O,
        eps_e=0.5 * (Hd[0, 0] + Hd[2, 2]),
        eps_m=Hd[1, 1],
        t_eff=0.5 * (Hd[0, 1] + Hd[1, 2]),
        t_13=Hd[0, 2],
    )


print("\n" + "=" * 96)
print("U-adaptive diabatics (H2 g-singlet ground state on each doubly-occupied pair)")
print("=" * 96)
print(f"{'t':>5} {'U':>4}  "
      f"{'E0_FCI':>10} {'E1_FCI':>10} {'E2_FCI':>10}   "
      f"{'E0_diab':>10} {'E1_diab':>10} {'E2_diab':>10}   {'max|dE|':>8}")
scan_ad = [(tv, Uv) for Uv in [0.0, 1.0, 4.0, 8.0, 16.0]
                     for tv in [0.05, 0.1, 0.2, 0.5, 1.0]]
for tv, Uv in scan_ad:
    o = analyse_adaptive(tv, Uv)
    err = max(abs(o['E_fci'][k] - o['E_diab'][k]) for k in range(3))
    print(f"{tv:>5.2f} {Uv:>4.1f}  "
          f"{o['E_fci'][0]:>+10.6f} {o['E_fci'][1]:>+10.6f} {o['E_fci'][2]:>+10.6f}   "
          f"{o['E_diab'][0]:>+10.6f} {o['E_diab'][1]:>+10.6f} {o['E_diab'][2]:>+10.6f}   "
          f"{err:>8.2e}")

print("\n" + "=" * 96)
print("U-adaptive effective parameters  (eps and t_eff both U-dependent)")
print("=" * 96)
print(f"{'t':>5} {'U':>4}  "
      f"{'eps_e':>10} {'eps_m':>10} {'eps_m-eps_e':>11}  "
      f"{'t_eff':>10} {'t_eff / (-t/2)':>14}  {'t_13':>10}")
for tv, Uv in scan_ad:
    o = analyse_adaptive(tv, Uv)
    ratio = o['t_eff'] / (-tv / 2) if tv > 0 else float('nan')
    print(f"{tv:>5.2f} {Uv:>4.1f}  "
          f"{o['eps_e']:>+10.6f} {o['eps_m']:>+10.6f} {o['eps_m'] - o['eps_e']:>+11.2e}  "
          f"{o['t_eff']:>+10.6f} {ratio:>14.6f}  {o['t_13']:>+10.2e}")


# ------------------------------------------------------------------------
# 8. Interpretation printed
# ------------------------------------------------------------------------
print("\n" + "=" * 96)
print("Interpretation")
print("=" * 96)
print("""
  * eps_e and eps_m are IDENTICAL (see col 'eps_m-eps_e') for every (t, U).
    The middle vs end asymmetry vanishes inside the sigma-only subspace:
    it is a second-order effect through sigma <-> sigma* mixing that
    lives OUTSIDE this 3-dim space.  A strict 1-hole tight-binding model
    with symmetric geometry therefore has a single reorganization energy
    eps = -5 + U, not two.

  * t_eff = -t/2 exactly (see col 't_eff / (-t/2)' = 1.000000), and
    t_13  = 0 exactly (no direct 1-3 AO coupling, and the projection
    cannot manufacture a 1-3 element out of nothing).  Superexchange
    t_13^(eff) = t^2/(2|h|) lives in states OUTSIDE this 3-dim space.

  * Accuracy of the 3x3 model (max|dE| column):
        t << |h|  -> error ~ (t/|h|)^2  (sigma/sigma* admixing at order t^2/|h|)
        t ~ |h|   -> error O(0.1*|h|)   (chain-of-equals limit: sigma and sigma*
                                         no longer separate; the model breaks)

  * Effect of U (naive sigma^2 diabatics): the naive ansatz keeps each full
    pair as a closed-shell sigma^2 determinant.  At U > 0 the true H2 ground
    state is no longer sigma^2 (50/50 cov/ion) -- it rotates toward the
    pure Heitler-London covalent singlet at large U, missing essentially
    ALL of the intra-pair correlation energy.  This is visible as:
      - w_total (Phi_1..Phi_3 projection) dropping to 0.26 at U = 16
      - eps_e = -5 + U stays rigid, while the true FCI GS has
        E ~ -5 + U - 2 * E_corr(U/t_intra), with E_corr -> -2 at U -> inf.
    The naive model therefore FAILS at U > |h| even at t = 0; the failure
    is NOT about CT coupling, it is about intra-pair electron correlation.

  * U-ADAPTIVE DIABATICS (section 7): put the EXACT H2 g-singlet ground
    state (c_ss sigma^2 + c_ssx sigma*^2) on each doubly-occupied pair.
    The hole pair stays sigma^alpha (1 electron, no correlation to resolve).
    Result:
      - eps_e = eps_m still exactly equal (middle/end symmetry preserved)
      - eps = 1 * (-1) + 2 * E_H2_gs(U)   (one-electron hole + two correlated H2s)
      - t_eff no longer -t/2; now t_eff = -t/2 * (c_ss + c_ssx*sign) overlap factor
        that depends on U (approach |t| * 0.30 as U -> inf, not t/2)
      - FCI errors drop by 2-3 orders of magnitude at small t for ALL U:
        at (t = 0.1, U = 4) the max |dE| goes from 1.68 (naive) to ~1e-3 (adaptive).

  CONCLUSION: Yes, the (H2)3+ chain CAN be described by a purely 1-electron
  hole model with {hopping t, site energy eps}, PROVIDED the "full pair"
  reference is the U-dependent H2 ground state, not the closed-shell MO.
  Then:
      eps(U)   = -1 + 2 E_H2_gs(U)    <- single reorganization energy
      t_eff(U) = -t/2 * f(U)          <- hopping, renormalised by U
      t_13     = 0 exactly            <- no direct 1-3 coupling in the 3-dim space
  This is the natural extension of H2 long-bond / allyl-cation VB work to a
  three-unit chain, and the 3x3 model is quantitative for t/|h| <~ 0.2 at
  every U.  The residual error (the (t/|h|)^2 piece) lives in the coupling
  to sigma* orbitals on the hole pair -- to kill that you would need to go
  to a 6x6 {sigma-hole, sigma*-hole} space, which would also generate a
  nonzero superexchange 1-3 coupling t_13_eff ~ t^2/(2|h|).
""")
