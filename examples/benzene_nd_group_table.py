"""
Group the 22 singlet-A_1g basis vectors by ionicity n_d (= 0, 1, 2, 3) and
report per-group:
  - sum of self-energies   E_k(h, s) = H22[k,k] / S22[k,k]
  - sum of CC weights      w_k       (matches Table 4 in manuscript)
  - lowest eigenvalue E_drop of the GEVP after removing all rows/columns
    with that n_d, plus the gap E_drop - E_GS(h, s).

Use this to argue which ionicity classes are essential for the closed-shell
GS at finite overlap s.

Cached symbolic (H1, S) is loaded from /tmp/benzene_hubbard_matrices.pkl;
this script does no symbolic re-build.

Run from the repo root: PYTHONPATH=. python3 examples/benzene_nd_group_table.py
"""
import os
import sys
import pickle

import numpy as np
import sympy as sp
import scipy.linalg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from symvb import Molecule, SlaterDet, symmetry
from symvb.spin import s_squared_matrix
from symvb.numerical import decompose_polynomial_matrix
from symvb.mo_projection import mo_determinant_in_ao

CACHE = '/tmp/benzene_hubbard_matrices.pkl'


def double_occ(ds):
    occ = {}
    for c in ds:
        occ.setdefault(c.lower(), [False, False])
        if c.islower():
            occ[c.lower()][0] = True
        else:
            occ[c.lower()][1] = True
    return sum(1 for ab in occ.values() if ab[0] and ab[1])


def closed_shell_in_AO_sp(det_strings):
    """Exact rational coefficient vector of the U=0 closed-shell ground state
    (three occupied real Huckel MOs k=0,+-1) in the AO-determinant basis."""
    M_rat = sp.Matrix([
        [sp.Integer(1)] * 6,
        [sp.Rational(1), sp.Rational(1, 2), sp.Rational(-1, 2),
         sp.Rational(-1), sp.Rational(-1, 2), sp.Rational(1, 2)],
        [sp.Integer(0), sp.Rational(1, 2), sp.Rational(1, 2),
         sp.Integer(0), sp.Rational(-1, 2), sp.Rational(-1, 2)],
    ])
    return mo_determinant_in_ao(M_rat, ([0, 1, 2], [0, 1, 2]),
                                det_strings, site_labels='abcdef')


def a1g_orbit_projector(det_strings):
    def canon(ds):
        fp = SlaterDet(ds).get_sorted()
        return fp.dets[0].det_string, fp.coefs[0]
    C6    = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'e', 'e': 'f', 'f': 'a'}
    sigma = {'a': 'a', 'b': 'f', 'c': 'e', 'd': 'd', 'e': 'c', 'f': 'b'}
    perms = [symmetry.apply_orbital_permutation(om, det_strings, canon)[0]
             for om in (C6, sigma)]
    U_a_norm, orbits = symmetry.totally_symmetric_basis(perms, len(det_strings))
    U_a_int = np.zeros((len(det_strings), len(orbits)), dtype=int)
    for col, orb in enumerate(orbits):
        for idx in orb:
            U_a_int[idx, col] = 1
    return U_a_norm, U_a_int, orbits


def build_22_block():
    """Return (H22_sym, S22_sym, US_sp, U_a_int, orbits, det_strings,
    h_sym, s_sym)."""
    m = Molecule(
        zero_ii=True,
        interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'],
        subst={'h': ('H_ab', 'H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af'),
               's': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af')},
        subst_2e={'U': ('1111',)},
        max_2e_centers=1,
    )
    m.generate_basis(3, 3, 6)
    det_strings = [fp.dets[0].det_string for fp in m.basis]

    if not os.path.exists(CACHE):
        raise RuntimeError(f'cache {CACHE} missing — run benzene_22row_table.py first')
    with open(CACHE, 'rb') as fh:
        H1_sym, S_sym, _ = pickle.load(fh)

    h_sym, s_sym = sp.symbols('h s')
    A = decompose_polynomial_matrix(H1_sym, s_sym, factor=h_sym)
    B = decompose_polynomial_matrix(S_sym,  s_sym)

    U_a_norm, U_a_int, orbits = a1g_orbit_projector(det_strings)

    S2 = s_squared_matrix(det_strings)
    S2_a_int = U_a_int.T @ S2.astype(int) @ U_a_int
    US_sp = sp.Matrix.hstack(*sp.Matrix(S2_a_int.tolist()).nullspace())
    if US_sp.shape[1] != 22:
        raise RuntimeError(f'singlet kernel dim = {US_sp.shape[1]}')

    M_q_38 = [(U_a_int.T @ A[q].astype(int) @ U_a_int) for q in range(len(A))]
    N_q_38 = [(U_a_int.T @ B[q].astype(int) @ U_a_int) for q in range(len(B))]
    M_q_22 = [US_sp.T * sp.Matrix(mq.tolist()) * US_sp for mq in M_q_38]
    N_q_22 = [US_sp.T * sp.Matrix(nq.tolist()) * US_sp for nq in N_q_38]

    H22_sym = sp.zeros(*M_q_22[0].shape)
    for q in range(len(M_q_22)):
        H22_sym = H22_sym + h_sym * (s_sym ** q) * M_q_22[q]
    S22_sym = sp.zeros(*N_q_22[0].shape)
    for q in range(len(N_q_22)):
        S22_sym = S22_sym + (s_sym ** q) * N_q_22[q]

    return H22_sym, S22_sym, US_sp, U_a_int, orbits, det_strings, h_sym, s_sym


def main(h_val=-1.0, s_values=(0.0, 0.1, 0.2, 0.3)):
    print(__doc__)
    print(f'\nParameters: h = {h_val},  s in {list(s_values)}')

    print('\nBuilding 22-dim singlet-A_1g block (loading from cache)...')
    (H22_sym, S22_sym, US_sp, U_a_int, orbits,
     det_strings, h_sym, s_sym) = build_22_block()

    # Per-row dominant n_d: pick the orbit with largest |US_sp[i,k]|.
    # All orbits in a given column carry the same n_d (Table 4 footnote).
    nd_per_row = []
    for k in range(22):
        col_abs = [abs(float(US_sp[i, k])) for i in range(38)]
        i_dom = int(np.argmax(col_abs))
        rep = det_strings[orbits[i_dom][0]]
        nd_per_row.append(double_occ(rep))
    nd_per_row = np.array(nd_per_row)

    # GS in 22-dim basis: v22 = G^{-1} X^T v0_sp (exact rational)
    print('\nBuilding GS vector v22 (exact rational arithmetic)...')
    v0_sp = closed_shell_in_AO_sp(det_strings)
    orbit_sizes = (U_a_int.T @ U_a_int).diagonal()
    D_sp = sp.diag(*[sp.Integer(int(d)) for d in orbit_sizes])
    G = US_sp.T * D_sp * US_sp
    U_a_sp = sp.Matrix(U_a_int.tolist())
    v22_naive_sp = US_sp.T * (U_a_sp.T * v0_sp)
    v22_sp = G.solve(v22_naive_sp)
    v0_norm2_sp = sum(v0_sp[i, 0] ** 2 for i in range(len(det_strings)))

    # CC weights w_k = c_k * (G c)_k / (c^T G c) = c_k * v22_naive_k / ||v0||^2
    w_sp = [v22_sp[k, 0] * v22_naive_sp[k, 0] / v0_norm2_sp for k in range(22)]
    w_sp = [sp.nsimplify(sp.together(w)) for w in w_sp]
    w_total = sum(w_sp)
    print(f'  sum of CC weights = {w_total}  (should be 1)')

    # Self-energies as sympy expressions in (h, s)
    print('\nComputing self-energies E_k(h,s) = H22[k,k]/S22[k,k]...')
    E_sym = []
    for k in range(22):
        E = sp.together(sp.simplify(H22_sym[k, k] / S22_sym[k, k]))
        E_sym.append(E)

    # Reference GS energy
    E_GS_sym = sp.Rational(4) * h_sym * (2 + 3 * s_sym) / \
               ((1 + 2 * s_sym) * (1 + s_sym))

    # ---- Per-n_d aggregation -------------------------------------------------
    # Weights are s-independent (rationals). Self-energies and the
    # "remove this group" eigenvalue depend on (h, s) — evaluate numerically.

    print('\n' + '=' * 92)
    print('Sum of CC weights by ionicity n_d  (s-independent — Table 4 sums)')
    print('=' * 92)
    print(f'  {"n_d":>3}  {"# rows":>6}  {"sum w_k (rational)":<20}  '
          f'{"sum w_k (decimal)":>17}  {"% of GS":>8}')
    print('-' * 92)
    grand = sp.Integer(0)
    for nd in (0, 1, 2, 3):
        idxs = np.where(nd_per_row == nd)[0]
        wsum = sum(w_sp[k] for k in idxs)
        wsum = sp.nsimplify(sp.together(wsum))
        grand = grand + wsum
        print(f'  {nd:>3}  {len(idxs):>6d}  {sp.sstr(wsum):<20}  '
              f'{float(wsum):>17.6f}  {100*float(wsum):>7.3f}%')
    print('-' * 92)
    print(f'  {"sum":>3}  {22:>6d}  {sp.sstr(grand):<20}  {float(grand):>17.6f}'
          f'  {100*float(grand):>7.3f}%')

    print('\n' + '=' * 92)
    print(f'Sum of self-energies  sum_k E_k(h={h_val}, s)   '
          f'and  E_GS(h={h_val}, s) = lowest eigenvalue')
    print('=' * 92)

    # Symbolic sums per group (closed-form in (h, s))
    nd_self_sym = {}
    print('\nSymbolic sum of self-energies per n_d group:')
    for nd in (0, 1, 2, 3):
        idxs = np.where(nd_per_row == nd)[0]
        S_self = sp.together(sum(E_sym[k] for k in idxs))
        S_self = sp.simplify(S_self)
        nd_self_sym[nd] = S_self
        print(f'  n_d = {nd}:  sum_k E_k(h, s) = {sp.sstr(S_self)}')

    # Numerical sum-of-self-energies per group at each s
    print(f'\nNumerical sum_k E_k(h={h_val}, s) per n_d group:')
    print(f'  {"n_d":>3}  {"# rows":>6}   '
          + '   '.join(f'{"s=%g"%s:>14}' for s in s_values))
    print('-' * (16 + 17 * len(s_values)))
    col_totals = {s: 0.0 for s in s_values}
    for nd in (0, 1, 2, 3):
        idxs = np.where(nd_per_row == nd)[0]
        cells = []
        for s_val in s_values:
            val = float(nd_self_sym[nd].subs([(h_sym, h_val), (s_sym, s_val)]))
            col_totals[s_val] += val
            cells.append(f'{val:>14.6f}')
        print(f'  {nd:>3}  {len(idxs):>6d}   ' + '   '.join(cells))
    print('-' * (16 + 17 * len(s_values)))
    print(f'  {"sum":>3}  {22:>6d}   '
          + '   '.join(f'{col_totals[s]:>14.6f}' for s in s_values))
    print('  (Note: sum_k E_k != E_GS — diagonals miss off-diagonal coupling.)')

    # ---- Reduced GEVP: drop one n_d group, get lowest eigenvalue at each s --
    print('\n' + '=' * 92)
    print('Lowest eigenvalue after removing all 22-basis rows with that n_d')
    print('(numerical solve of reduced GEVP H22[mask]c = E S22[mask] c)')
    print('=' * 92)
    print(f'  {"n_d dropped":>11}  {"# rows kept":>11}'
          + '   ' + '   '.join(f'{"s=%g"%s:>14}' for s in s_values))
    print('-' * (28 + 17 * len(s_values)))

    rows_print = []
    rows_print.append(('full (none)', 22, []))
    for nd in (0, 1, 2, 3):
        keep = [k for k in range(22) if nd_per_row[k] != nd]
        rows_print.append((f'drop n_d={nd}', len(keep), keep))

    # For each s, build float H22, S22 once and solve all reduced problems
    s_to_eigs = {}  # s -> dict label -> lowest eigval
    s_to_egs  = {}  # s -> E_GS
    for s_val in s_values:
        H_full = np.array(
            [[float(H22_sym[i, j].subs([(h_sym, h_val), (s_sym, s_val)]))
              for j in range(22)] for i in range(22)])
        S_full = np.array(
            [[float(S22_sym[i, j].subs([(h_sym, h_val), (s_sym, s_val)]))
              for j in range(22)] for i in range(22)])
        H_full = 0.5 * (H_full + H_full.T)
        S_full = 0.5 * (S_full + S_full.T)
        s_to_egs[s_val] = float(E_GS_sym.subs([(h_sym, h_val), (s_sym, s_val)]))
        store = {}
        # Full
        ev = scipy.linalg.eigh(H_full, S_full, eigvals_only=True)
        store['full (none)'] = ev[0]
        # Drops
        for nd in (0, 1, 2, 3):
            keep = [k for k in range(22) if nd_per_row[k] != nd]
            Hr = H_full[np.ix_(keep, keep)]
            Sr = S_full[np.ix_(keep, keep)]
            ev = scipy.linalg.eigh(Hr, Sr, eigvals_only=True)
            store[f'drop n_d={nd}'] = ev[0]
        s_to_eigs[s_val] = store

    for label, n_keep, _ in rows_print:
        cells = []
        for s_val in s_values:
            cells.append(f'{s_to_eigs[s_val][label]:>14.6f}')
        print(f'  {label:>11}  {n_keep:>11d}   ' + '   '.join(cells))

    print('\nReference  E_GS(h, s) = 4h(2+3s)/[(1+2s)(1+s)]:')
    print(f'  {"E_GS":>11}  {"":>11}   '
          + '   '.join(f'{s_to_egs[s]:>14.6f}' for s in s_values))

    print('\nGap  (E_drop - E_GS)  per dropped group:')
    print(f'  {"n_d dropped":>11}  {"":>11}   '
          + '   '.join(f'{"s=%g"%s:>14}' for s in s_values))
    for label, _, _ in rows_print[1:]:
        cells = []
        for s_val in s_values:
            gap = s_to_eigs[s_val][label] - s_to_egs[s_val]
            cells.append(f'{gap:>14.6f}')
        print(f'  {label:>11}  {"":>11}   ' + '   '.join(cells))

    # ---- Per-vector leave-one-out: drop each of the 22 basis vectors --------
    print('\n' + '=' * 92)
    print('Leave-one-out: drop each of the 22 singlet-A_1g basis vectors individually')
    print('(numerical solve of 21-dim reduced GEVP after deleting row/col k)')
    print('=' * 92)

    # For each s in s_values, build float (H22, S22) once, then 22 reduced solves
    s_to_loo = {}  # s -> list of 22 lowest eigvals (one per dropped row)
    for s_val in s_values:
        H_full = np.array(
            [[float(H22_sym[i, j].subs([(h_sym, h_val), (s_sym, s_val)]))
              for j in range(22)] for i in range(22)])
        S_full = np.array(
            [[float(S22_sym[i, j].subs([(h_sym, h_val), (s_sym, s_val)]))
              for j in range(22)] for i in range(22)])
        H_full = 0.5 * (H_full + H_full.T)
        S_full = 0.5 * (S_full + S_full.T)
        eigs_loo = np.empty(22)
        for k in range(22):
            keep = [i for i in range(22) if i != k]
            Hr = H_full[np.ix_(keep, keep)]
            Sr = S_full[np.ix_(keep, keep)]
            eigs_loo[k] = scipy.linalg.eigh(Hr, Sr, eigvals_only=True)[0]
        s_to_loo[s_val] = eigs_loo

    # 22-row table at s=0 (where pairings are cleanest) and at largest s
    s_show_loo = sorted({s_values[0], s_values[-1]})
    for s_val in s_show_loo:
        E_GS_n = s_to_egs[s_val]
        eigs   = s_to_loo[s_val]
        print(f'\n  Table: leave-one-out at h={h_val}, s={s_val}'
              f'   (E_GS = {E_GS_n:.6f})')
        print(f'  {"k":>2}  {"n_d":>3}  {"w_k":>10}  {"w_k (dec)":>10}  '
              f'{"E_k(h, s)":>12}  {"E_drop^(k)":>12}  {"gap":>10}')
        print('  ' + '-' * 76)
        for k in range(22):
            Ek = float(E_sym[k].subs([(h_sym, h_val), (s_sym, s_val)]))
            gap = eigs[k] - E_GS_n
            print(f'  {k+1:>2}  {nd_per_row[k]:>3d}  '
                  f'{sp.sstr(w_sp[k]):>10}  {float(w_sp[k]):>10.6f}  '
                  f'{Ek:>12.6f}  {eigs[k]:>12.6f}  {gap:>10.6f}')
        # group-by-n_d summary: largest gap per class
        print('  ' + '-' * 76)
        print('  Largest leave-one-out gap per ionicity class:')
        for nd in (0, 1, 2, 3):
            idxs = np.where(nd_per_row == nd)[0]
            gaps = eigs[idxs] - E_GS_n
            k_max = int(idxs[int(np.argmax(gaps))])
            print(f'    n_d={nd}:  max gap = {gaps.max():.6f}  '
                  f'(row k={k_max+1}, w_k={float(w_sp[k_max]):.6f})')

    # ---- Symbolic analysis of E_drop ----------------------------------------
    # Honest scope: E_drop is the lowest root of an 11- to 20-dim generalised
    # characteristic polynomial in E with rational coefs in (h, s).  No clean
    # closed form exists in general (degree > 4).  Two computable proxies:
    #
    #   (A) Rayleigh bound  E_R(drop) = <v_kept|H|v_kept> / <v_kept|S|v_kept>
    #       where v_kept = exact GS v22 with dropped components zeroed out.
    #       This is a rational function of (h, s).  It UPPER-BOUNDS the true
    #       lowest eigenvalue of the reduced GEVP (Rayleigh-Ritz with one
    #       trial vector) — i.e. E_R >= E_drop, with equality only if v_kept
    #       happens to be an eigenvector.
    #
    #   (B) The full symbolic characteristic polynomial det(H_red - E S_red)
    #       — print it for the smallest case (drop n_d=2, 11x11) only.

    print('\n' + '=' * 92)
    print('Symbolic E_drop analysis')
    print('=' * 92)
    print('E_drop is the lowest root of a degree-N polynomial in E with')
    print('coefficients in Q(h, s); for N >= 5 (all four cases) there is no')
    print('closed-form root expression in radicals.')
    print()
    print('(A) Rayleigh upper bound  E_R(drop)(h, s)  --  rational closed form:')
    print('    Project the exact GS v22 onto the retained subspace (zero out')
    print('    rows with the dropped n_d), then take the Rayleigh quotient.')
    print('    This BOUNDS the true E_drop from above.')
    print()
    E_R_dict = {}
    for nd in (0, 1, 2, 3):
        keep = [k for k in range(22) if nd_per_row[k] != nd]
        v_kept = sp.Matrix([v22_sp[k, 0] for k in keep])
        H_red = H22_sym.extract(keep, keep)
        S_red = S22_sym.extract(keep, keep)
        num = (v_kept.T * H_red * v_kept)[0, 0]
        den = (v_kept.T * S_red * v_kept)[0, 0]
        E_R = sp.cancel(sp.together(num / den))
        E_R_dict[nd] = E_R
        print(f'  Drop n_d = {nd}  ({len(keep)}-dim retained):')
        print(f'    E_R(h, s) = {sp.sstr(E_R)}')
        print()

    # Sanity-check E_R upper-bounds E_drop numerically
    print('  Numerical check  E_R >= E_drop  at h=%g:' % h_val)
    print(f'  {"n_d dropped":>11}  '
          + '   '.join(f'{"s=%g"%s:>14}' for s in s_values))
    for nd in (0, 1, 2, 3):
        cells = []
        for s_val in s_values:
            E_R_num = float(E_R_dict[nd].subs([(h_sym, h_val), (s_sym, s_val)]))
            E_drop  = s_to_eigs[s_val][f'drop n_d={nd}']
            gap     = E_R_num - E_drop
            cells.append(f'{E_R_num:>7.4f} (+{gap:>5.4f})')
        print(f'  drop n_d={nd:>2}   ' + '   '.join(cells))
    print('  (parenthesis = E_R - E_drop, must be >= 0 by Rayleigh-Ritz)')

    print()
    print('(B) Symbolic characteristic polynomial det(H_red - E S_red) for the')
    print('    smallest case (drop n_d=2, 11x11), at s=0 and as polynomial in (E, h):')
    keep2 = [k for k in range(22) if nd_per_row[k] != 2]
    H11_s0 = H22_sym.subs(s_sym, 0).extract(keep2, keep2)
    S11_s0 = S22_sym.subs(s_sym, 0).extract(keep2, keep2)
    E_var = sp.Symbol('E')
    char_poly_s0 = (H11_s0 - E_var * S11_s0).det()
    char_poly_s0 = sp.expand(char_poly_s0)
    print(f'    char_poly(E; h, s=0) = {sp.sstr(sp.factor(char_poly_s0))}')
    print('    (lowest real root in E gives E_drop at s=0)')

    # ---- 4x4 effective Hamiltonian grouped by n_d ---------------------------
    print('\n' + '=' * 92)
    print('4x4 effective Hamiltonian grouped by n_d')
    print('=' * 92)
    print('Idea: replace the 22-dim singlet block by a 4-dim subspace, one')
    print('representative vector per ionicity class.  Two natural choices:')
    print('  (V1) UNIFORM:  |phi_n> = sum_{k in class n} |b_k>  (equal coefs)')
    print('       -> 4x4 GEVP gives a variational UPPER BOUND on E_GS')
    print('  (V2) GS-WEIGHTED:  |phi_n> = sum_{k in class n} c_k |b_k>')
    print('       -> 4-dim subspace contains v22, so E_GS is recovered EXACTLY')
    print('       (with eigenvector (1,1,1,1) in the phi basis).  Useful for')
    print('       reading off class-class couplings, not for testing accuracy.')

    # Build W matrices (22 -> 4)
    W_uni = sp.zeros(22, 4)
    W_gs  = sp.zeros(22, 4)
    for k in range(22):
        n = int(nd_per_row[k])
        W_uni[k, n] = 1
        W_gs[k, n]  = v22_sp[k, 0]

    print('\n--- V1: uniform-sum basis ---')
    H_eff_uni = sp.simplify(W_uni.T * H22_sym * W_uni)
    S_eff_uni = sp.simplify(W_uni.T * S22_sym * W_uni)
    print('H_eff (4x4, rows/cols indexed by n_d=0,1,2,3):')
    for i in range(4):
        row = '  ' + '   '.join(f'{sp.sstr(H_eff_uni[i, j]):>30}' for j in range(4))
        print(row)
    print('S_eff (4x4):')
    for i in range(4):
        row = '  ' + '   '.join(f'{sp.sstr(S_eff_uni[i, j]):>30}' for j in range(4))
        print(row)

    # 4x4 quartic char poly in E
    char4_uni = sp.expand((H_eff_uni - E_var * S_eff_uni).det())
    print('\nchar_poly(E; h, s) for V1:')
    print(f'  {sp.sstr(sp.collect(char4_uni, E_var))}')

    # Numerical comparison: V1 lowest eigval vs E_GS
    print('\nLowest eigenvalue of V1 4x4 GEVP at h=%g:' % h_val)
    print(f'  {"":>14}  ' + '   '.join(f'{"s=%g"%s:>14}' for s in s_values))
    cells_v1, cells_egs = [], []
    for s_val in s_values:
        H4 = np.array([[float(H_eff_uni[i, j].subs([(h_sym, h_val), (s_sym, s_val)]))
                        for j in range(4)] for i in range(4)])
        S4 = np.array([[float(S_eff_uni[i, j].subs([(h_sym, h_val), (s_sym, s_val)]))
                        for j in range(4)] for i in range(4)])
        H4 = 0.5 * (H4 + H4.T); S4 = 0.5 * (S4 + S4.T)
        ev4 = scipy.linalg.eigh(H4, S4, eigvals_only=True)
        cells_v1.append(f'{ev4[0]:>14.6f}')
        cells_egs.append(f'{s_to_egs[s_val]:>14.6f}')
    print(f'  {"V1 lowest":>14}  ' + '   '.join(cells_v1))
    print(f'  {"E_GS (exact)":>14}  ' + '   '.join(cells_egs))

    print('\n--- V2: GS-weighted basis ---')
    H_eff_gs = sp.simplify(W_gs.T * H22_sym * W_gs)
    S_eff_gs = sp.simplify(W_gs.T * S22_sym * W_gs)
    print('H_eff (4x4):')
    for i in range(4):
        row = '  ' + '   '.join(f'{sp.sstr(H_eff_gs[i, j]):>30}' for j in range(4))
        print(row)
    print('S_eff (4x4):')
    for i in range(4):
        row = '  ' + '   '.join(f'{sp.sstr(S_eff_gs[i, j]):>30}' for j in range(4))
        print(row)

    # Verify (1,1,1,1) is an eigenvector with eigenvalue E_GS
    one4 = sp.Matrix([1, 1, 1, 1])
    Hone = H_eff_gs * one4
    Sone = S_eff_gs * one4
    # Check H_eff * 1 = E_GS * S_eff * 1 component-wise
    print('\nVerify (1,1,1,1) is an eigenvector of V2 with eigenvalue E_GS:')
    residual = [sp.cancel(sp.together(Hone[i] - E_GS_sym * Sone[i])) for i in range(4)]
    if all(r == 0 for r in residual):
        print('  All 4 residual components vanish identically in Q(h, s).  EXACT.')
    else:
        print('  Residuals:')
        for i, r in enumerate(residual):
            print(f'    [{i}] = {sp.sstr(r)}')

    # Diagonal class self-energies (V2): E_self(n) = H_eff[n,n]/S_eff[n,n]
    print('\nClass diagonal self-energies E_n^(V2)(h, s) = H_eff[n,n]/S_eff[n,n]:')
    for n in range(4):
        E_n = sp.cancel(sp.together(H_eff_gs[n, n] / S_eff_gs[n, n]))
        print(f'  n_d = {n}:  {sp.sstr(E_n)}')

    print('\nClass-class off-diagonal couplings H_eff[n,m] / sqrt(S[n,n]*S[m,m]):')
    for n in range(4):
        for m in range(n + 1, 4):
            if H_eff_gs[n, m] == 0:
                print(f'  ({n},{m}): 0  (forbidden by |dn_d|<=1 selection)')
                continue
            ratio = sp.cancel(sp.together(
                H_eff_gs[n, m] / sp.sqrt(S_eff_gs[n, n] * S_eff_gs[m, m])))
            print(f'  ({n},{m}): {sp.sstr(ratio)}')

    # ---- Per-row breakdown for completeness ---------------------------------
    print('\n' + '=' * 92)
    print('Per-row breakdown (n_d, w_k, E_k at h=%g, s=%g):' % (h_val, s_values[-1]))
    print('=' * 92)
    print(f'  {"k":>2}  {"n_d":>3}  {"w_k":>10}  {"w_k (dec)":>10}  '
          f'{"E_k(h, s)":>14}')
    s_show = s_values[-1]
    for k in range(22):
        Ek = float(E_sym[k].subs([(h_sym, h_val), (s_sym, s_show)]))
        print(f'  {k+1:>2}  {nd_per_row[k]:>3d}  '
              f'{sp.sstr(w_sp[k]):>10}  {float(w_sp[k]):>10.6f}  '
              f'{Ek:>14.6f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
