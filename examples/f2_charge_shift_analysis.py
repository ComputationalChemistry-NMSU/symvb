"""
F2 charge-shift VB analysis.

Loads the 12-dim A_g block built by examples/f2_valence_symbolic.py and
decomposes the symbolic ground state into the three "minimal" sigma-bond
VB structures:

    psi_cov   = (|c_dn d_up> - |c_up d_dn>) / sqrt(2)    Heitler-London (covalent)
    psi_ion_A = |c_up c_dn>                              ionic, c^2 (F^- on atom A)
    psi_ion_B = |d_up d_dn>                              ionic, d^2 (F^- on atom B)

In D_2h the symmetric combination (psi_ion_A + psi_ion_B)/sqrt(2) is A_g,
the antisymmetric one is B_1u, so only two sigma-bond structures live in
A_g: psi_cov and psi_ion_g.  All other 14 - 2 = 12 valence electrons are
"frozen" as 2s^2 lone pairs and pi^4 lone pairs in the chemical basis
states; their excitations populate the other 10 of 12 A_g states.

Measurements:
  1. Overlap structure of {psi_cov, psi_ion_g} in the A_g basis.
  2. Full 12x12 ground-state amplitudes in the A_g basis (numerical).
  3. Chirgwin-Coulson weights of GS within the 2-state {cov, ion_g}
     subspace; norm of the orthogonal residual (= "dynamic correlation").
  4. Scan over s_zz (sigma overlap): cov-vs-ion weight crossover.

This is a chemistry-language counterpart to the Hubbard PT analysis in
the manuscript -- it answers "is F2 a covalent or charge-shift bond?"
quantitatively at the all-valence symbolic level.
"""
import os
import pickle
import sys

import numpy as np
import sympy as sp
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ------------------------------------------------------------------------
# Load A_g cache
# ------------------------------------------------------------------------
CACHE = '/tmp/f2_valence_Ag.pkl'
with open(CACHE, 'rb') as f:
    H_Ag, S_Ag, U_mat, det_strings = pickle.load(f)

dim_full = U_mat.shape[0]
N_Ag = U_mat.shape[1]
print(f'Loaded: full dim = {dim_full}, A_g block = {N_Ag}')

# ------------------------------------------------------------------------
# Identify the three sigma-bond chemical structures in the 64-dim basis
# ------------------------------------------------------------------------
# Orbital labels:  c = 2p_sigma_A, d = 2p_sigma_B
# The other 6 orbitals (a, b, e, f, g, h) form the frozen lone-pair core.
#
# Det strings are 14-character: 7 alpha (lowercase) + 7 beta (uppercase).
# (alpha-hole, beta-hole) classification:
#   D_cov_a  = (alpha hole at c, beta hole at d)  =  c_dn d_up  config
#   D_cov_b  = (alpha hole at d, beta hole at c)  =  c_up d_dn  config
#   D_ionA   = (alpha hole at d, beta hole at d)  =  c^2          (F-A^-)
#   D_ionB   = (alpha hole at c, beta hole at c)  =  d^2          (F-B^-)

ALL_ORBS = 'abcdefgh'

def make_det_string(alpha_hole, beta_hole):
    # symvb generate_det_strings interleaves sorted alpha and beta tuples
    # positionally:  a[0]+b[0]+a[1]+b[1]+...
    a = [c for c in ALL_ORBS if c != alpha_hole]
    b = [c.upper() for c in ALL_ORBS if c != beta_hole]
    return ''.join(ai + bi for ai, bi in zip(a, b))

D_cov_a_str = make_det_string('c', 'd')
D_cov_b_str = make_det_string('d', 'c')
D_ionA_str  = make_det_string('d', 'd')
D_ionB_str  = make_det_string('c', 'c')

# Find indices (raise if not present).
def find_index(s):
    if s not in det_strings:
        raise KeyError(f'det {s!r} not found in basis')
    return det_strings.index(s)

i_cov_a = find_index(D_cov_a_str)
i_cov_b = find_index(D_cov_b_str)
i_ionA  = find_index(D_ionA_str)
i_ionB  = find_index(D_ionB_str)

print(f'\nKey det indices in full 64-dim basis:')
print(f'  D_cov_a (c_dn d_up): {i_cov_a:3d}  -> {D_cov_a_str}')
print(f'  D_cov_b (c_up d_dn): {i_cov_b:3d}  -> {D_cov_b_str}')
print(f'  D_ionA  (c^2):       {i_ionA:3d}  -> {D_ionA_str}')
print(f'  D_ionB  (d^2):       {i_ionB:3d}  -> {D_ionB_str}')

# Build the chemical basis vectors in the 64-dim full basis.
def basis_vec(idx, sign=+1):
    v = np.zeros(dim_full)
    v[idx] = sign
    return v

inv_sqrt2 = 1.0 / np.sqrt(2.0)
# symvb canonical interleaved ordering puts d_alpha^dagger before c_beta^dagger
# in D_cov_a, so c_beta^dagger d_alpha^dagger = -d_alpha^dagger c_beta^dagger
# = -(D_cov_a creator). Hence the singlet open-shell antisymmetrizer
#   (c_alpha^dagger d_beta^dagger - c_beta^dagger d_alpha^dagger)/sqrt(2)
#   = (D_cov_b + D_cov_a)/sqrt(2)
# i.e. the SYMMETRIC sum of these dets is the singlet covalent VB.
v_cov   = inv_sqrt2 * (basis_vec(i_cov_a) + basis_vec(i_cov_b))
v_trip0 = inv_sqrt2 * (basis_vec(i_cov_a) - basis_vec(i_cov_b))
v_ion_g = inv_sqrt2 * (basis_vec(i_ionA)  + basis_vec(i_ionB))
v_ion_u = inv_sqrt2 * (basis_vec(i_ionA)  - basis_vec(i_ionB))

# Project chemical structures into the A_g block.
c_cov   = U_mat.T @ v_cov
c_trip0 = U_mat.T @ v_trip0
c_ion_g = U_mat.T @ v_ion_g
c_ion_u = U_mat.T @ v_ion_u

print(f'\nProjection norms into A_g (expect 1 for cov & ion_g; 0 for triplet & ion_u):')
print(f'  ||P_Ag psi_cov||^2     = {np.dot(c_cov,   c_cov):.6f}')
print(f'  ||P_Ag psi_trip(Sz=0)||^2 = {np.dot(c_trip0, c_trip0):.6f}')
print(f'  ||P_Ag psi_ion_g||^2   = {np.dot(c_ion_g, c_ion_g):.6f}')
print(f'  ||P_Ag psi_ion_u||^2   = {np.dot(c_ion_u, c_ion_u):.6f}')

# ------------------------------------------------------------------------
# Build a representative numerical parameter substitution
# ------------------------------------------------------------------------
# Base parameter point: physical-ish minimal-basis F2 with h_sz=0 (no
# 2s-2p_sigma mixing). Locking out sp mixing gives clean AO-language
# sigma-bond VB analysis: 2s lone pairs frozen, sigma-bond is pure 2p_sigma.
# Turning on h_sz, s_sz lets us study charge-shift corrections from
# sp-rehybridization in a follow-up scan.
test_subs = {
    sp.Symbol('alpha_s'): -3.0,    # 2s deep enough to be a stable lone pair
    sp.Symbol('alpha_z'): -1.0,
    sp.Symbol('alpha_p'): -1.0,
    sp.Symbol('h_ss'):    -0.02,
    sp.Symbol('h_zz'):    -0.40,   # sigma bond
    sp.Symbol('h_pp'):    -0.10,   # pi
    sp.Symbol('h_sz'):     0.0,    # no sp-cross hop (clean AO picture)
    sp.Symbol('s_ss'):     0.0,
    sp.Symbol('s_zz'):     0.10,   # sigma overlap (the physically interesting one)
    sp.Symbol('s_pp'):     0.0,
    sp.Symbol('s_sz'):     0.0,
    sp.Symbol('U_s'):      1.5,
    sp.Symbol('U_z'):      1.0,
    sp.Symbol('U_p'):      1.0,
    sp.Symbol('V_sz'):     0.6,
    sp.Symbol('V_sp'):     0.6,
    sp.Symbol('V_zp'):     0.6,
    sp.Symbol('V_pp'):     0.6,
    sp.Symbol('J_sz'):     0.05,
    sp.Symbol('J_sp'):     0.05,
    sp.Symbol('J_zp'):     0.05,
    sp.Symbol('J_pp'):     0.05,
    sp.Symbol('K_ss'):     0.3,
    sp.Symbol('K_zz'):     0.3,
    sp.Symbol('K_pp_par'): 0.3,
    sp.Symbol('K_pp_perp'):0.3,
    sp.Symbol('K_sz'):     0.3,
    sp.Symbol('K_sp'):     0.3,
    sp.Symbol('K_zp'):     0.3,
    sp.Symbol('k_ss'):     0.0,
    sp.Symbol('k_zz'):     0.0,
    sp.Symbol('k_pp'):     0.0,
    sp.Symbol('k_pp_perp'):0.0,
    sp.Symbol('k_sz'):     0.0,
    sp.Symbol('k_sp'):     0.0,
    sp.Symbol('k_zp'):     0.0,
}

_all_symbols = sorted(H_Ag.free_symbols | S_Ag.free_symbols, key=str)
_H_lamb = sp.lambdify(_all_symbols, H_Ag, modules='numpy')
_S_lamb = sp.lambdify(_all_symbols, S_Ag, modules='numpy')

def numerify(M_lamb, subs):
    args = [float(subs[s]) for s in _all_symbols]
    return np.asarray(M_lamb(*args), dtype=float)


def vb_weights(H_Ag_num, S_Ag_num, c_cov, c_ion_g):
    """Return ground-state energy and Chirgwin-Coulson weights for the
    {cov, ion_g} 2-state subspace of A_g, plus the residual norm."""
    evals, evecs = eigh(H_Ag_num, S_Ag_num)
    psi_GS = evecs[:, 0]                                  # 12-dim A_g vector
    E0 = evals[0]
    norm = psi_GS @ S_Ag_num @ psi_GS                     # = 1 from eigh

    # 2-state subspace coefficients via projection in S metric:
    #   c = (V^T S V)^{-1} V^T S psi_GS
    V = np.column_stack([c_cov, c_ion_g])                 # 12 x 2
    M = V.T @ S_Ag_num @ V                                # 2 x 2 overlap
    rhs = V.T @ S_Ag_num @ psi_GS                         # 2-vector
    c2 = np.linalg.solve(M, rhs)                          # subspace coeffs

    # Chirgwin-Coulson weights (in the non-orthogonal {cov, ion_g} basis):
    #   w_i = c_i * sum_j c_j * S_ij   (sums to ||c2||^2_S)
    w_cov   = c2[0] * (M[0, 0] * c2[0] + M[0, 1] * c2[1])
    w_ion_g = c2[1] * (M[1, 0] * c2[0] + M[1, 1] * c2[1])
    w_2state = w_cov + w_ion_g                            # = ||V c2||^2_S

    # Residual norm: portion of GS outside span(cov, ion_g).
    r = psi_GS - V @ c2
    w_residual = r @ S_Ag_num @ r

    return E0, c2, w_cov, w_ion_g, w_2state, w_residual


# ------------------------------------------------------------------------
# 1. Base parameter point
# ------------------------------------------------------------------------
print(f'\n--- ground-state VB analysis at base parameters ---')
H_num = numerify(_H_lamb, test_subs)
S_num = numerify(_S_lamb, test_subs)
E0, c2, w_cov, w_ion_g, w_2st, w_res = vb_weights(H_num, S_num, c_cov, c_ion_g)
print(f'  E_GS               = {E0:+.4f}  a.u.')
print(f'  coeff (cov)        = {c2[0]:+.4f}')
print(f'  coeff (ion_g)      = {c2[1]:+.4f}')
print(f'  weight w_cov       = {w_cov:+.4f}')
print(f'  weight w_ion_g     = {w_ion_g:+.4f}')
print(f'  sum w (2-state)    = {w_2st:+.4f}')
print(f'  residual ||r||^2_S = {w_res:+.4f}  (= 1 - sum_w if normalized)')
print(f'  ratio  w_cov : w_ion_g  =  {w_cov/(w_cov+w_ion_g):.3f} : '
      f'{w_ion_g/(w_cov+w_ion_g):.3f}  (within 2-state)')

# ------------------------------------------------------------------------
# 2. Scan over s_zz  -- charge-shift signature
# ------------------------------------------------------------------------
print(f'\n--- s_zz scan: cov-vs-ion mixing as overlap is dialed ---')
print(f'{"s_zz":>7} | {"E_GS":>10} | {"w_cov":>8} | {"w_ion_g":>9} | '
      f'{"sum_w":>7} | {"residual":>9}')
print('-' * 65)
for s_zz_val in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]:
    subs = dict(test_subs)
    subs[sp.Symbol('s_zz')] = s_zz_val
    H_n = numerify(_H_lamb, subs)
    S_n = numerify(_S_lamb, subs)
    try:
        E0, c2, wcov, wig, w2, wr = vb_weights(H_n, S_n, c_cov, c_ion_g)
        print(f'{s_zz_val:7.3f} | {E0:+10.4f} | {wcov:+8.4f} | {wig:+9.4f} '
              f'| {w2:+7.4f} | {wr:+9.4f}')
    except np.linalg.LinAlgError as e:
        print(f'{s_zz_val:7.3f} | failed: {e}')

# ------------------------------------------------------------------------
# 3. Scan over h_zz (sigma hop) at fixed overlap -- bond-strength sweep
# ------------------------------------------------------------------------
print(f'\n--- h_zz scan: cov vs ion shift with bond strength ---')
print(f'{"h_zz":>7} | {"E_GS":>10} | {"w_cov":>8} | {"w_ion_g":>9} | {"sum_w":>7}')
print('-' * 55)
for h_zz_val in [-0.05, -0.10, -0.15, -0.20, -0.30, -0.45, -0.60, -0.90]:
    subs = dict(test_subs)
    subs[sp.Symbol('h_zz')] = h_zz_val
    H_n = numerify(_H_lamb, subs)
    S_n = numerify(_S_lamb, subs)
    E0, c2, wcov, wig, w2, wr = vb_weights(H_n, S_n, c_cov, c_ion_g)
    print(f'{h_zz_val:7.3f} | {E0:+10.4f} | {wcov:+8.4f} | {wig:+9.4f} | {w2:+7.4f}')
