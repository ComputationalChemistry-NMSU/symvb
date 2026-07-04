"""
F2 all-valence symbolic VBT calculation.

Active space: full valence on each F atom -- 2s, 2pz (sigma), 2px, 2py.
8 orbitals, 14 electrons. Sz=0 sector dimension is C(8,7)^2 = 64.

Orbital alphabet (forced to be the first 8 lowercase letters by
symvb.functions.generate_det_strings):
    a, b   = 2s on F_A, F_B
    c, d   = 2p_sigma on F_A, F_B   (along bond axis z)
    e, f   = 2p_pi_x on F_A, F_B
    g, h   = 2p_pi_y on F_A, F_B

Parameter grouping (declared in subst* dicts, no post-processing for 1e;
2e symbols use default T_<abcd> names and are grouped by classify_2e):

  1-electron / overlap
    alpha_s, alpha_z, alpha_p        on-site energies
    h_ss, h_zz, h_pp                 inter-atom same-type hops
    h_sz                             inter-atom 2s_A - 2p_sigma_B (& partner)
    s_ss, s_zz, s_pp, s_sz           overlaps with same structure
    (intra-atom 2s-2p_sigma matrix elements set to 0 by interacting_orbs)

  2-electron, max_2e_centers = 2
    U_s, U_z, U_p                    on-site (ii|ii)
    V_sz, V_sp, V_zp, V_pp           one-center direct (ii|jj), i,j same atom
    J_sz, J_sp, J_zp, J_pp           one-center exchange (ij|ij)
    K_ss, K_zz, K_pp_par, K_pp_perp  two-center direct, same-type
    K_sz, K_sp, K_zp                 two-center direct, mixed-type
    k_ss, k_zz, k_pp                 two-center exchange, same-type
    M_*                              one-center 3-index (aa|ab)-type
                                       (kept symbolic; commonly small)
"""
import os
import pickle
import sys
import time

import sympy as sp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, SlaterDet, symmetry

# ------------------------------------------------------------------------
# Flags
# ------------------------------------------------------------------------
ZDO = True   # zero out 3-index "M-type" integrals (PPP/ZDO approximation)

# ------------------------------------------------------------------------
# Orbital -> type and -> atom mapping
# ------------------------------------------------------------------------
ORB_TYPE = {
    'a': 's', 'b': 's',
    'c': 'z', 'd': 'z',
    'e': 'x', 'f': 'x',
    'g': 'y', 'h': 'y',
}
ORB_ATOM = {
    'a': 'A', 'c': 'A', 'e': 'A', 'g': 'A',
    'b': 'B', 'd': 'B', 'f': 'B', 'h': 'B',
}

# 1e: inter-atom couplings only. parse_subst inverts the dict, so
# 'h_ss': ('H_ab',)  -->  self.subst = {'H_ab': 'h_ss'}, etc.
SUBST = {
    # 1e diagonals (on-site alphas)
    'alpha_s': ('H_aa', 'H_bb'),
    'alpha_z': ('H_cc', 'H_dd'),
    'alpha_p': ('H_ee', 'H_ff', 'H_gg', 'H_hh'),
    # 1e off-diagonals
    'h_ss':    ('H_ab',),
    'h_zz':    ('H_cd',),
    'h_pp':    ('H_ef', 'H_gh'),
    'h_sz':    ('H_ad', 'H_bc'),
    # overlap off-diagonals
    's_ss':    ('S_ab',),
    's_zz':    ('S_cd',),
    's_pp':    ('S_ef', 'S_gh'),
    's_sz':    ('S_ad', 'S_bc'),
}

INTERACTING = ['ab', 'cd', 'ef', 'gh', 'ad', 'bc']

# ------------------------------------------------------------------------
# 1. Build symbolic matrices (or load from cache)
# ------------------------------------------------------------------------
CACHE = '/tmp/f2_valence_matrices.pkl'

if not os.path.exists(CACHE):
    print(f'building {CACHE} ...')
    m = Molecule(
        zero_ii=False,                 # keep on-site alphas symbolic
        interacting_orbs=INTERACTING,
        subst=SUBST,
        # subst_2e: empty -> 2e integrals get default T_<abcd> names,
        # so each distinct orbital combination is independently labeled.
        max_2e_centers=2,
        orbitals='abcdefgh',
        o2_method='blocked',
    )
    m.generate_basis(7, 7, 8)
    print(f'  basis dim = {len(m.basis)}')

    t0 = time.time()
    H1 = m.build_matrix(m.basis, op='H')
    print(f'  H1 built in {time.time() - t0:.1f}s')

    t0 = time.time()
    S = m.build_matrix(m.basis, op='S')
    print(f'  S  built in {time.time() - t0:.1f}s')

    t0 = time.time()
    H2 = m.o2_matrix(m.basis)
    print(f'  H2 built in {time.time() - t0:.1f}s')

    with open(CACHE, 'wb') as f:
        pickle.dump((H1, S, H2), f)
    print(f'  cached to {CACHE}')

with open(CACHE, 'rb') as f:
    H1, S, H2 = pickle.load(f)

dim = H1.shape[0]
print(f'\nLoaded matrices: dim = {dim}')

H1_sym = sp.Matrix(H1)
S_sym  = sp.Matrix(S)
H2_sym = sp.Matrix(H2)

print(f'\nH1 free symbols: {sorted(H1_sym.free_symbols, key=str)}')
print(f'S  free symbols: {sorted(S_sym.free_symbols, key=str)}')

# ------------------------------------------------------------------------
# 2. Classify 2e integrals from canonical T_<abcd> names
# ------------------------------------------------------------------------
def classify_2e(name):
    """Return a chemically-grouped symbolic name for a T_<abcd> integral.

    The 4-tuple is in chemist canonical order: (ij|kl) with the 8-fold
    permutation already resolved.
    """
    if not name.startswith('T_'):
        return None
    orbs = name[2:]
    if len(orbs) != 4:
        return None

    types = [ORB_TYPE[o] for o in orbs]
    atoms = [ORB_ATOM[o] for o in orbs]
    unique_orbs = sorted(set(orbs))
    unique_atoms = sorted(set(atoms))
    n_orb = len(unique_orbs)
    n_atom = len(unique_atoms)

    # On-site (ii|ii): all four indices same orbital
    if n_orb == 1:
        t = types[0]
        if t == 's': return 'U_s'
        if t == 'z': return 'U_z'
        return 'U_p'

    # 2-orbital integrals
    if n_orb == 2:
        # Canonical patterns:
        #   'aabb' -- (ii|jj) direct                (orbs[0]==orbs[1])
        #   'abab' -- (ij|ij) exchange              (orbs[0]==orbs[2])
        #   'aaab' -- 3-index "M-type"              (3 of one orbital)
        cnt = {o: orbs.count(o) for o in unique_orbs}
        counts_sorted = sorted(cnt.values())

        if counts_sorted == [2, 2]:
            if orbs[0] == orbs[1] and orbs[2] == orbs[3]:
                # Direct (ii|jj): determine same-atom vs two-atom
                t_i = ORB_TYPE[orbs[0]]
                t_j = ORB_TYPE[orbs[2]]
                same_atom = (atoms[0] == atoms[2])
                pair = tuple(sorted((t_i, t_j)))
                # Distinguish pi-pi parallel vs perpendicular (only matters
                # for x-x and y-y when on different atoms).
                if same_atom:
                    # One-center direct V_<types>
                    if pair == ('s', 'z'): return 'V_sz'
                    if pair in (('s', 'x'), ('s', 'y')): return 'V_sp'
                    if pair in (('x', 'z'), ('y', 'z')): return 'V_zp'
                    if pair == ('x', 'y'): return 'V_pp'
                    return f'V_unclassified_{name}'
                else:
                    # Two-center direct K_<types>
                    if pair == ('s', 's'): return 'K_ss'
                    if pair == ('z', 'z'): return 'K_zz'
                    if pair == ('x', 'x') or pair == ('y', 'y'): return 'K_pp_par'
                    if pair == ('x', 'y'): return 'K_pp_perp'
                    if pair == ('s', 'z'): return 'K_sz'
                    if pair in (('s', 'x'), ('s', 'y')): return 'K_sp'
                    if pair in (('x', 'z'), ('y', 'z')): return 'K_zp'
                    return f'K_unclassified_{name}'
            if orbs[0] == orbs[2] and orbs[1] == orbs[3]:
                # Exchange (ij|ij)
                t_i = ORB_TYPE[orbs[0]]
                t_j = ORB_TYPE[orbs[1]]
                same_atom = (atoms[0] == atoms[1])
                pair = tuple(sorted((t_i, t_j)))
                if same_atom:
                    if pair == ('s', 'z'): return 'J_sz'
                    if pair in (('s', 'x'), ('s', 'y')): return 'J_sp'
                    if pair in (('x', 'z'), ('y', 'z')): return 'J_zp'
                    if pair == ('x', 'y'): return 'J_pp'
                    return f'J_unclassified_{name}'
                else:
                    if pair == ('s', 's'): return 'k_ss'
                    if pair == ('z', 'z'): return 'k_zz'
                    if pair == ('x', 'x') or pair == ('y', 'y'): return 'k_pp'
                    if pair == ('s', 'z'): return 'k_sz'
                    if pair in (('s', 'x'), ('s', 'y')): return 'k_sp'
                    if pair in (('x', 'z'), ('y', 'z')): return 'k_zp'
                    if pair == ('x', 'y'): return 'k_pp_perp'
                    return f'k_unclassified_{name}'
            return f'pair_unclassified_{name}'

        if counts_sorted == [1, 3]:
            # M-type (aaab). Identify the "minority" orbital.
            majority = [o for o, c in cnt.items() if c == 3][0]
            minority = [o for o, c in cnt.items() if c == 1][0]
            t_maj = ORB_TYPE[majority]
            t_min = ORB_TYPE[minority]
            same_atom = (ORB_ATOM[majority] == ORB_ATOM[minority])
            pair = tuple(sorted((t_maj, t_min)))
            # Two cases: both orbitals on same atom (one-center M),
            # or split across atoms (two-center M-mixed).
            prefix = 'M' if same_atom else 'Mcross'
            tag = ''.join(pair) if same_atom else ''.join((t_maj, t_min))
            return f'{prefix}_{tag}'

    return f'unclassified_{name}'


h2_symbols = sorted(H2_sym.free_symbols, key=str)
# Many of these are 1e/overlap symbols (alpha_*, h_*, s_*) that legitimately
# appear in 2e matrix elements via Loewdin cofactors. Only T_* symbols are
# raw 2e integral labels that need grouping.
t_symbols = [s for s in h2_symbols if s.name.startswith('T_')]
print(f'\nH2 has {len(h2_symbols)} free symbols ({len(t_symbols)} T_* integrals)')

group_2e = {}
unclassified = []
for sym in t_symbols:
    grouped = classify_2e(sym.name)
    if grouped is None or 'unclassified' in grouped:
        unclassified.append(sym.name)
        continue
    if ZDO and (grouped.startswith('M_') or grouped.startswith('Mcross_')):
        # ZDO: drop 3-index integrals (aa|ab) etc.
        group_2e[sym] = sp.Integer(0)
    else:
        group_2e[sym] = sp.Symbol(grouped)

classes = sorted(set(str(s) for s in group_2e.values()))
print(f'grouped {len(group_2e)} symbols into {len(classes)} classes'
      + (' (M-types -> 0 by ZDO)' if ZDO else '') + ':')
for cls in classes:
    members = [s.name for s, t in group_2e.items() if str(t) == cls]
    print(f'  {cls:14s} <- {len(members):3d}  e.g. {members[0]}')
if unclassified:
    print(f'WARNING: {len(unclassified)} unclassified T_ symbols: {unclassified}')

# Use xreplace (exact symbol-to-symbol substitution, much faster than subs
# which does pattern matching across the whole expression tree).
print('\napplying substitution (xreplace)...')
t0 = time.time()
H2_g = H2_sym.xreplace(group_2e)
print(f'  done in {time.time() - t0:.1f}s')

# ------------------------------------------------------------------------
# 3. Final summary
# ------------------------------------------------------------------------
H_full = H1_sym + H2_g
all_syms = sorted(H_full.free_symbols | S_sym.free_symbols, key=str)
print(f'\nfinal Hamiltonian uses {len(all_syms)} symbols:')
print(f'  {all_syms}')

OUT = '/tmp/f2_valence_grouped.pkl'
with open(OUT, 'wb') as f:
    pickle.dump((H1_sym, S_sym, H2_g), f)
print(f'\ngrouped matrices saved to {OUT}')

# ========================================================================
# 4. D_{2h} -> A_g projection (singlet sigma_g+ ground state of F2)
# ========================================================================
# F2 ground state is 1-Sigma_g+; in D_{2h} (max abelian subgroup of D_inf_h)
# this is the A_g irrep. Generators: i (inversion), C2(z) (along bond),
# sigma_xz. These give the full 8-element D_{2h}.
#
# Each generator acts on AOs as (relabel, intrinsic_sign):
#   i:        a<->b (+1), c<->d (-1), e<->f (-1), g<->h (-1)
#               relabeling swaps atoms; intrinsic sign comes from
#               r -> -r on each AO type.
#   C2(z):    no relabel; px, py flip sign.
#   sigma_xz: no relabel; py flips sign.

ORB_INV_MAP  = {'a': 'b', 'b': 'a', 'c': 'd', 'd': 'c',
                'e': 'f', 'f': 'e', 'g': 'h', 'h': 'g'}
ORB_INV_SIGN = {'a': +1, 'b': +1, 'c': -1, 'd': -1,
                'e': -1, 'f': -1, 'g': -1, 'h': -1}

ID_MAP = {c: c for c in 'abcdefgh'}
ORB_C2Z_SIGN = {'a': +1, 'b': +1, 'c': +1, 'd': +1,
                'e': -1, 'f': -1, 'g': -1, 'h': -1}
ORB_SXZ_SIGN = {'a': +1, 'b': +1, 'c': +1, 'd': +1,
                'e': +1, 'f': +1, 'g': -1, 'h': -1}

# Rebuild the basis (same as in the build step) to get det strings.
m = Molecule(zero_ii=False, interacting_orbs=INTERACTING, subst=SUBST,
             max_2e_centers=2, orbitals='abcdefgh', o2_method='blocked')
m.generate_basis(7, 7, 8)
det_strings = [fp.dets[0].det_string for fp in m.basis]


def canon(ds):
    fp = SlaterDet(ds).get_sorted()
    return fp.dets[0].det_string, fp.coefs[0]


def intrinsic_sign(det_string, sign_map):
    """Product of intrinsic orbital signs over all spin-orbitals occupied
    in det_string. Each lowercase orbital contributes its alpha occupation,
    each uppercase its beta -- sign_map is per spatial orbital."""
    s = 1
    for c in det_string:
        s *= sign_map[c.lower()]
    return s


def make_signed_generator(orbital_map, sign_map):
    perm, perm_signs = symmetry.apply_orbital_permutation(
        orbital_map, det_strings, canon)
    if perm is None:
        raise ValueError("orbital map does not preserve basis set")
    intrinsic = np.array([intrinsic_sign(ds, sign_map) for ds in det_strings],
                         dtype=int)
    combined = (perm_signs * intrinsic).astype(int)
    return perm, combined


print('\n--- D_{2h} A_g projection ---')
gen_i   = make_signed_generator(ORB_INV_MAP, ORB_INV_SIGN)
gen_c2z = make_signed_generator(ID_MAP,      ORB_C2Z_SIGN)
gen_sxz = make_signed_generator(ID_MAP,      ORB_SXZ_SIGN)

U_mat, group_order = symmetry.signed_totally_symmetric_basis(
    [gen_i, gen_c2z, gen_sxz], dim)
N_Ag = U_mat.shape[1]
print(f'group order = {group_order} (expected 8 for D_2h)')
print(f'A_g block dimension = {N_Ag}  (out of {dim})')

# Project H, S into A_g
U_sp = sp.Matrix(U_mat.tolist())
H_total = H1_sym + H2_g

print('projecting H, S into A_g ...')
t0 = time.time()
H_Ag = sp.Matrix(U_sp.T * H_total * U_sp)
S_Ag = sp.Matrix(U_sp.T * S_sym  * U_sp)
print(f'  {time.time() - t0:.1f}s')

print(f'\nA_g free symbols ({len(H_Ag.free_symbols)}):')
print(f'  {sorted(H_Ag.free_symbols, key=str)}')

OUT_AG = '/tmp/f2_valence_Ag.pkl'
with open(OUT_AG, 'wb') as f:
    pickle.dump((H_Ag, S_Ag, U_mat, det_strings), f)
print(f'\nA_g matrices saved to {OUT_AG} (incl. U_mat, det_strings)')

# ------------------------------------------------------------------------
# 5. Numerical sanity check at a representative parameter point
# ------------------------------------------------------------------------
# Plug in rough literature-style values (atomic units; minimal-basis F2).
# These are NOT meant to be quantitative -- just to verify the projection
# yields a sensible spectrum (real eigenvalues, ground state below
# atomic-fragment limit).
test_subs = {
    sp.Symbol('alpha_s'): -1.6,    # 2s much deeper than 2p
    sp.Symbol('alpha_z'): -0.7,
    sp.Symbol('alpha_p'): -0.7,
    sp.Symbol('h_ss'):    -0.05,
    sp.Symbol('h_zz'):    -0.30,
    sp.Symbol('h_pp'):    -0.10,
    sp.Symbol('h_sz'):    -0.15,
    sp.Symbol('s_ss'):     0.02,
    sp.Symbol('s_zz'):     0.08,
    sp.Symbol('s_pp'):     0.04,
    sp.Symbol('s_sz'):     0.05,
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
    sp.Symbol('K_ss'):     0.5,
    sp.Symbol('K_zz'):     0.5,
    sp.Symbol('K_pp_par'): 0.5,
    sp.Symbol('K_pp_perp'):0.4,
    sp.Symbol('K_sz'):     0.5,
    sp.Symbol('K_sp'):     0.5,
    sp.Symbol('K_zp'):     0.5,
    sp.Symbol('k_ss'):     0.005,
    sp.Symbol('k_zz'):     0.02,
    sp.Symbol('k_pp'):     0.01,
    sp.Symbol('k_pp_perp'):0.005,
    sp.Symbol('k_sz'):     0.01,
    sp.Symbol('k_sp'):     0.005,
    sp.Symbol('k_zp'):     0.01,
}

H_num = np.array(H_Ag.subs(test_subs).evalf().tolist(), dtype=float)
S_num = np.array(S_Ag.subs(test_subs).evalf().tolist(), dtype=float)
print(f'\n--- numerical sanity (test parameter set) ---')
from scipy.linalg import eigh
evals = eigh(H_num, S_num, eigvals_only=True)
print(f'A_g eigenvalues (a.u.):')
for ev in evals:
    print(f'  {ev:+.4f}')
