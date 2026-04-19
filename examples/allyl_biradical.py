"""
Biradical / long-bond character of the allyl 3c4e chain as a function of
(U, J, K, M) integrals.

Framework:  build the 9-det Sz=0 FCI Hamiltonian symbolically with
(U,J,K,M) as in examples/allyl_hubbard_ujk.py, project to the 5-dim
sigma=+1 (A_1) block, diagonalise numerically over a parameter grid,
and compute six biradical-diagnostic observables.

Observables (baseline values at U=J=K=M=0, closed-shell Huckel):
    n_psi1  = 2       (bonding doubly filled)
    n_psi2  = 2       (nonbonding doubly filled)
    n_psi3  = 0       (antibonding empty)
    w_LB    = 1/8     (combined weight on the two long-bond dets
                       |aBbC> and |bAcB>, i.e. b doubly occ, a,c each singly)
    w_ion   = 3/8     (weight on 2+2+0 ionic dets: |aAbB>, |aAcC>, |bBcC>)
    rho_a   = 3/2     (terminal density, same for c by reflection)
    rho_b   = 1       (central density)
    <d>_a   = 9/16    (double occ at terminal: (rho_a/2)^2 = 0.5625)
    <d>_b   = 1/4     (double occ at center)

The biradical index used in the literature is
    b = n(psi_3) / 2      (0 = closed-shell singlet, 1 = full biradical)
or equivalently 1 - n(psi_2)/2.
"""
import os
import sys
import time

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from vbt3 import Molecule, SlaterDet, symmetry
from vbt3.fixed_psi import generate_dets
from vbt3.spin import s_squared_matrix


# ------------------------------------------------------------------------
# 1. Symbolic H1+H2 with (U,J,K,M)
# ------------------------------------------------------------------------
m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab', 'bc'],
    subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
    subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
              'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)
P = generate_dets(2, 2, 3)
det_strings = [p.dets[0].det_string for p in P]
print("Det basis:", det_strings)

t0 = time.time()
H1 = m.build_matrix(P, op='H')
H2 = m.o2_matrix(P)
print(f"Symbolic build: {time.time()-t0:.1f}s")

H = sp.Matrix(H1 + H2)
h, s, U, J, K, M = sp.symbols('h s U J K M')
H_s0 = H.subs({s: 0, h: -1})


# ------------------------------------------------------------------------
# 2. A_1 (sigma = +1) projector
# ------------------------------------------------------------------------
def canon(ds):
    fp = SlaterDet(ds).get_sorted()
    return fp.dets[0].det_string, fp.coefs[0]

sig = {'a': 'c', 'b': 'b', 'c': 'a'}
perm, signs = symmetry.apply_orbital_permutation(sig, det_strings, canon)

U_plus = []
seen = [False] * 9
for i in range(9):
    if seen[i]:
        continue
    j = perm[i]; sj = signs[i]
    if j == i:
        seen[i] = True
        if sj == 1:
            v = sp.zeros(9, 1); v[i] = 1
            U_plus.append(v)
    else:
        seen[i] = seen[j] = True
        v = sp.zeros(9, 1); v[i] = 1; v[j] = sj
        U_plus.append(v / sp.sqrt(2))

Up = sp.Matrix.hstack(*U_plus)
Up_np = np.array(Up, dtype=float)     # 9 x 5
H_red = Up.T * H_s0 * Up
Nd = H_red.shape[0]
print(f"A_1 block dim: {Nd}")

# Fast numeric evaluator
H_red_fn = sp.lambdify((U, J, K, M), H_red, 'numpy')


# ------------------------------------------------------------------------
# 3. Build observable operators on the 9-dim det basis
# ------------------------------------------------------------------------
def parse_det(ds):
    alpha = sorted([c for c in ds if c.islower()])
    beta  = sorted([c.lower() for c in ds if c.isupper()])
    return alpha, beta

def site_occ(ds, p):
    a, b = parse_det(ds)
    return (p in a) + (p in b)

def is_double(ds, p):
    a, b = parse_det(ds)
    return 1 if (p in a and p in b) else 0

def is_ionic(ds):
    counts = sorted(site_occ(ds, p) for p in 'abc')
    return counts == [0, 2, 2]

def is_long_bond(ds):
    """b doubly occupied, a and c each singly occupied (opposite spins)."""
    a, b = parse_det(ds)
    return ('b' in a and 'b' in b and
            (('a' in a) + ('a' in b)) == 1 and
            (('c' in a) + ('c' in b)) == 1)

# Site occupation (diagonal), double occupation (diagonal) -- 9x9
n9 = np.zeros((3, 9, 9))
d9 = np.zeros((3, 9, 9))
for ip, p in enumerate('abc'):
    for I, ds in enumerate(det_strings):
        n9[ip, I, I] = site_occ(ds, p)
        d9[ip, I, I] = is_double(ds, p)

# Ionic-det and long-bond-det projectors (diagonal) -- 9x9
lb_mask = np.array([is_long_bond(d) for d in det_strings], dtype=float)
ion_mask = np.array([is_ionic(d) for d in det_strings], dtype=float)
Proj_lb9  = np.diag(lb_mask)
Proj_ion9 = np.diag(ion_mask)

# 1-RDM in AO basis (spin-summed):  (rho_ao)_{p,q}  =  sum_sigma <a^+_{p,sigma} a_{q,sigma}>.
# Work in an ORBITAL-INTERLEAVED canonical spin-orbital ordering
#   slot order: (alpha_a, beta_a, alpha_b, beta_b, alpha_c, beta_c)
# vbt3 strings use a different ordering (alpha_i/beta_i zipped alphabetically);
# the sign relating the two is sigma_I for each det, so we transform
#   O_vbt3 = diag(sigma) @ O_canon @ diag(sigma).

def canon_indices(ds):
    """Return list of canonical slot indices (0..5) for the spin-orbitals
    occupied by det string `ds`, in the order the vbt3 string lists them."""
    out = []
    for c in ds:
        orb = 'abc'.index(c.lower())
        spin = 0 if c.islower() else 1
        out.append(2 * orb + spin)
    return out

def vbt3_sign(ds):
    """Sign such that |ds>_vbt3 = sign * |ds>_canonical."""
    idx = canon_indices(ds)
    inv = 0
    n = len(idx)
    for i in range(n):
        for j in range(i + 1, n):
            if idx[i] > idx[j]:
                inv += 1
    return (-1) ** inv

sigmas = np.array([vbt3_sign(ds) for ds in det_strings], dtype=float)

# canonical-sorted occ tuple -> det index
occ_sorted_to_idx = {tuple(sorted(canon_indices(ds))): i
                     for i, ds in enumerate(det_strings)}

def apply_pq_canon(occ_sorted, p, q, spin):
    """<D_I|a^+_p_spin a_q_spin|D_J> in canonical ordering; D_J given by sorted
    canonical-index list occ_sorted.  Returns (sign, sorted occ of D_I) or None."""
    q_idx = 2 * 'abc'.index(q) + spin
    p_idx = 2 * 'abc'.index(p) + spin
    if q_idx not in occ_sorted:
        return None
    k = occ_sorted.index(q_idx)
    sign_q = (-1) ** k
    after = occ_sorted[:k] + occ_sorted[k + 1:]
    if p_idx in after:
        return None
    j = 0
    while j < len(after) and after[j] < p_idx:
        j += 1
    sign_p = (-1) ** j
    new_occ = after[:j] + [p_idx] + after[j:]
    return sign_q * sign_p, tuple(new_occ)

rho_ao_canon_9 = np.zeros((3, 3, 9, 9))
for J, ds_J in enumerate(det_strings):
    occ_J = sorted(canon_indices(ds_J))
    for ip, p in enumerate('abc'):
        for iq, q in enumerate('abc'):
            for spin in (0, 1):
                res = apply_pq_canon(occ_J, p, q, spin)
                if res is None:
                    continue
                sign, occ_I = res
                I = occ_sorted_to_idx.get(occ_I)
                if I is None:
                    continue
                rho_ao_canon_9[ip, iq, I, J] += sign

# Transform to vbt3 basis: O_vbt3 = diag(sigma) O_canon diag(sigma)
rho_ao_9 = (sigmas[None, None, :, None]
            * rho_ao_canon_9
            * sigmas[None, None, None, :])

# S^2 operator (already in vbt3 basis)
S2_9 = s_squared_matrix(det_strings, orbs='abc')


# ------------------------------------------------------------------------
# 4. Project all operators to the A_1 block
# ------------------------------------------------------------------------
def P_A1(op9):
    return Up_np.T @ op9 @ Up_np

n_site = np.stack([P_A1(n9[i]) for i in range(3)])
d_site = np.stack([P_A1(d9[i]) for i in range(3)])
Proj_lb  = P_A1(Proj_lb9)
Proj_ion = P_A1(Proj_ion9)
rho_ao   = np.empty((3, 3, Nd, Nd))
for ip in range(3):
    for iq in range(3):
        rho_ao[ip, iq] = P_A1(rho_ao_9[ip, iq])
S2 = P_A1(S2_9)


# ------------------------------------------------------------------------
# 5. Huckel MO matrix (3-chain, orthogonal AOs, h=-1)
# ------------------------------------------------------------------------
r2 = np.sqrt(2.0)
C_mo = np.array([
    [1/2,     1/r2,  1/2],     # psi_1 bonding   E = -sqrt(2), sigma-even
    [1/r2,    0,    -1/r2],    # psi_2 nonbond.  E =  0,       sigma-odd
    [1/2,    -1/r2,  1/2],     # psi_3 antibond. E = +sqrt(2), sigma-even
]).T  # C_mo[p_AO, mu_MO]


def observe(Uv, Jv, Kv, Mv):
    Hn = np.array(H_red_fn(Uv, Jv, Kv, Mv), dtype=float)
    Hn = 0.5 * (Hn + Hn.T)
    evals, evecs = np.linalg.eigh(Hn)
    c = evecs[:, 0]
    rho_site = np.array([c @ n_site[i] @ c for i in range(3)])
    d_vals   = np.array([c @ d_site[i] @ c for i in range(3)])
    w_lb     = float(c @ Proj_lb  @ c)
    w_ion    = float(c @ Proj_ion @ c)
    gamma    = np.array([[c @ rho_ao[ip, iq] @ c for iq in range(3)]
                         for ip in range(3)])
    gamma    = 0.5 * (gamma + gamma.T)
    nat_occ  = np.sort(np.linalg.eigvalsh(gamma))[::-1]
    huck_occ = np.diag(C_mo.T @ gamma @ C_mo)
    s2 = float(c @ S2 @ c)
    return dict(E=float(evals[0]),
                gap=float(evals[1] - evals[0]),
                rho=rho_site, d=d_vals,
                w_lb=w_lb, w_ion=w_ion,
                nat=nat_occ, huck=huck_occ, s2=s2)


# ------------------------------------------------------------------------
# 6. Baseline check at (0,0,0,0) -- should match the analytic Huckel values
# ------------------------------------------------------------------------
o0 = observe(0, 0, 0, 0)
print("\n=== Huckel baseline (U=J=K=M=0) ===")
print(f"  E0      = {o0['E']:+.6f}      (exact: -2*sqrt(2) = {-2*r2:.6f})")
print(f"  rho_a   = {o0['rho'][0]:.4f}    (exact 3/2)")
print(f"  rho_b   = {o0['rho'][1]:.4f}    (exact 1)")
print(f"  <d>_a   = {o0['d'][0]:.4f}   (exact 9/16 = 0.5625)")
print(f"  <d>_b   = {o0['d'][1]:.4f}   (exact 1/4)")
print(f"  w_LB    = {o0['w_lb']:.4f}   (exact 1/2)")
print(f"  w_ion   = {o0['w_ion']:.4f}   (exact 3/8 = 0.375)")
print(f"  nat_occ = ({o0['nat'][0]:.4f}, {o0['nat'][1]:.4f}, {o0['nat'][2]:.4f})")
print(f"  Huckel  = n(psi_1)={o0['huck'][0]:.4f} "
      f"n(psi_2)={o0['huck'][1]:.4f} n(psi_3)={o0['huck'][2]:.4f}")


def print_row(label, o):
    print(f"  {label:>12s}  {o['E']:>9.4f}  {o['s2']:>5.2f}  "
          f"{o['huck'][0]:>7.4f}  {o['huck'][1]:>7.4f}  {o['huck'][2]:>7.4f}  "
          f"{o['w_lb']:>7.4f}  {o['w_ion']:>7.4f}  "
          f"{o['rho'][0]:>5.3f} {o['rho'][1]:>5.3f}  {o['d'][0]:>6.4f} {o['d'][1]:>6.4f}")

def print_header(title):
    print(f"\n=== {title} ===")
    print(f"  {'param':>12s}  {'E':>9s}  {'S²':>5s}  "
          f"{'n(ψ₁)':>7s}  {'n(ψ₂)':>7s}  {'n(ψ₃)':>7s}  "
          f"{'w_LB':>7s}  {'w_ion':>7s}  {'ρ_a':>5s} {'ρ_b':>5s}  "
          f"{'<d>_a':>6s} {'<d>_b':>6s}")


# ------------------------------------------------------------------------
# 7. Parameter sweeps
# ------------------------------------------------------------------------
grid = [0, 0.25, 0.5, 1, 2, 4, 8, 16, 32]

print_header("Pure U axis (J=K=M=0)")
for v in grid:
    print_row(f"U={v}", observe(v, 0, 0, 0))

print_header("Pure J axis (U=K=M=0)")
for v in grid:
    print_row(f"J={v}", observe(0, v, 0, 0))

print_header("Pure K axis (U=J=M=0)")
for v in grid:
    print_row(f"K={v}", observe(0, 0, v, 0))

print_header("Pure M axis (U=J=K=0)")
for v in [-2, -1, -0.5, 0, 0.5, 1, 2]:
    print_row(f"M={v}", observe(0, 0, 0, v))

print_header("(U-J) axis: U=4 fixed, J swept  (tests (U-J) reduction)")
for Jv in [0, 0.5, 1, 2, 3, 4, 5, 6]:
    print_row(f"J={Jv}", observe(4, Jv, 0, 0))

print_header("Physical PPP direction: U=4, J=0.5, K varied (M=0)")
for Kv in [0, 0.25, 0.5, 1, 2]:
    print_row(f"K={Kv}", observe(4, 0.5, Kv, 0))

print_header("M cross-coupling: U=4, J=0.5, K=1, M varied")
for Mv in [-1, -0.5, 0, 0.5, 1]:
    print_row(f"M={Mv}", observe(4, 0.5, 1, Mv))


# ------------------------------------------------------------------------
# 8. Strong-U decomposition: is the NO-occupation limit (7/4, 3/2, 3/4)?
# ------------------------------------------------------------------------
print_header("Strong-U asymptotics (pure U, J=K=M=0)")
for Uv in [1, 4, 16, 64, 256, 1024, 4096]:
    print_row(f"U={Uv}", observe(Uv, 0, 0, 0))

print("\n  --- breakdown by doubly-occupied site at large U ---")
# Weight on dets grouped by which site carries the double occupation
dbl_mask_by_site = np.zeros((3, 9))
for i, ds in enumerate(det_strings):
    for sidx, p in enumerate('abc'):
        if is_double(ds, p):
            dbl_mask_by_site[sidx, i] = 1
Proj_dbl_at_p = [P_A1(np.diag(dbl_mask_by_site[i])) for i in range(3)]
print(f"  {'U':>6s}  {'w(a²)':>7s}  {'w(b²)':>7s}  {'w(c²)':>7s}   "
      f"{'Σ':>6s}  (note: ionic dets double-count here)")
for Uv in [4, 16, 64, 1024]:
    Hn = np.array(H_red_fn(Uv, 0, 0, 0), dtype=float); Hn = 0.5*(Hn+Hn.T)
    _, ev = np.linalg.eigh(Hn)
    c = ev[:, 0]
    w = [float(c @ Proj_dbl_at_p[i] @ c) for i in range(3)]
    print(f"  {Uv:>6.0f}  {w[0]:>7.4f}  {w[1]:>7.4f}  {w[2]:>7.4f}   "
          f"{sum(w):>6.4f}")


# ------------------------------------------------------------------------
# 9. K-driven GS level crossing (U=4, J=0.5, M=0)
# ------------------------------------------------------------------------
print_header("Fine K scan at U=4, J=0.5, M=0 (locates singlet↔triplet crossing)")
for Kv in [0, 0.5, 1.0, 1.3, 1.5, 1.6, 1.7, 1.8, 2.0, 3.0]:
    print_row(f"K={Kv}", observe(4, 0.5, Kv, 0))


# ------------------------------------------------------------------------
# 10. M-driven GS level crossing (U=4, J=0.5, K=1)
# ------------------------------------------------------------------------
print_header("Fine M scan at U=4, J=0.5, K=1 (locates singlet↔triplet crossing)")
for Mv in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
    print_row(f"M={Mv}", observe(4, 0.5, 1, Mv))
