"""
22-row composition table for the benzene pi system, using the singlet-A_1g
basis (valid for all s, not just s=0).

Reduction chain:

    400  (full Sz=0 FCI)
     |  D_6 (C_6 + sigma_v)
    38   A_1g orbits
     |  S^2 = 0             <- s-independent (operator is geometry-blind)
    22   singlet-A_1g       <- target space, valid for ALL s

The closed-shell 1e GS fills MOs k = 0, +/-1  (energies eps_a, eps_b, eps_b):
    E_GS(h, s) = 2 eps_0 + 4 eps_1
              = 4h(2+3s) / [(1+2s)(1+s)]

For each of the 22 basis vectors the table shows:
  - n_d   : dominant doubly-occupied count (from the representative orbit det)
  - VB structure label
  - c_alpha: rational coordinate v22_sp[k] of basis vector k in the GS expansion
             (in the un-normalised orbit-sum basis; related to but not equal to the
             orthonormal coefficient)
  - n_k  : Gram norm G[k,k] = S22_sym[k,k] at s=0  (= sum of orbit sizes in US_sp[:,k])
  - w_alpha (CC) = c_alpha * v22_naive[k] / ||v0||^2: true Chirgwin-Coulson weight,
             sums exactly to 1 for non-orthogonal basis (s-independent)
  - E_alpha(h, s): diagonal self-energy H22[k,k] / S22[k,k], rational in (h, s);
             vanishes at s=0 because H22 is off-diagonal there
  - Non-zero couplings H22[i,j] at s=0

The singlet projector US_sp is built from the exact integer S^2 nullspace, so all
self-energies are genuinely rational in (h, s).  Compare: the 14-dim basis from the
extra eta^2 projection has irrational self-energies (eta eigenvectors introduce sqrt).

CC weight formula: w_k = c_k * (X[:,k]^T v0) / (v0^T v0) where X = U_a_int @ US_sp.
Sum: sum_k w_k = v22_sp^T G v22_sp / ||v0||^2 = ||v0||^2 / ||v0||^2 = 1.  Exact.

Analogous to allyl_1e_analytical_overlap.py (4-row table) but for benzene.
"""
import os
import sys
import pickle
import time

import numpy as np
import sympy as sp
import scipy.linalg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from symvb import Molecule, SlaterDet, symmetry
from symvb.spin import s_squared_matrix
from symvb.numerical import decompose_polynomial_matrix
from symvb.mo_projection import mo_determinant_in_ao

CACHE = '/tmp/benzene_hubbard_matrices.pkl'
ORBS  = list('abcdef')


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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
    """
    Closed-shell MO determinant |k=0>^2 |k=+1>^2 |k=-1>^2 in the symvb
    AO Slater-det basis, using exact rational coefficients.

    Real Huckel MOs (un-normalised; factor 1/sqrt(6) and 1/sqrt(3) dropped,
    these cancel in the eigenvector relation):
        psi_a (k=0):  row = (1,  1,   1,  1,   1,   1)
        psi_+ (real): row = (1, 1/2, -1/2, -1, -1/2, 1/2)
        psi_- (real): row = (0, 1/2,  1/2,  0, -1/2, -1/2)
    """
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
    """Returns (U_a_norm, U_a_int, orbits) for the D_6 A_1g block."""
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


def vb_label(ds):
    """Human-readable VB label from a representative det string."""
    nd = double_occ(ds)
    alpha_occ = sorted([c for c in ds if c.islower()])
    beta_occ  = sorted([c.lower() for c in ds if c.isupper()])
    doubly    = sorted([o for o in ORBS if o in alpha_occ and o in beta_occ])
    if nd == 0:
        if set(alpha_occ) == {'a', 'c', 'e'}:
            return 'Kekule K1 (ace/bdf)'
        elif set(alpha_occ) == {'b', 'd', 'f'}:
            return 'Kekule K2 (bdf/ace)'
        else:
            # Dewar bond pair
            alpha_str = ''.join(sorted(alpha_occ))
            beta_str  = ''.join(sorted(beta_occ))
            return f'Dewar ({alpha_str}/{beta_str})'
    elif nd == 1:
        return f'1-ionic d={doubly[0]}'
    elif nd == 2:
        rel = abs(ord(doubly[0]) - ord(doubly[1]))
        pos = 'adj' if rel in (1, 5) else ('meta' if rel in (2, 4) else 'para')
        return f'2-ionic {pos} d={"".join(doubly)}'
    elif nd == 3:
        return f'3-ionic d={"".join(doubly)}'
    else:
        return f'nd={nd}'


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print(__doc__)
    print()

    # -----------------------------------------------------------------------
    # Step 1: Build basis and load symbolic matrices from cache
    # -----------------------------------------------------------------------
    print('Step 1: Loading benzene FCI basis and symbolic (H1, S) from cache...')
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
    N = len(det_strings)
    print(f'  N = {N} AO-det basis (3 alpha + 3 beta in 6 orbitals)')

    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as fh:
            H1_sym, S_sym, _H2 = pickle.load(fh)
        print('  loaded from cache')
    else:
        print('  cache miss: building ~5s...')
        t0 = time.time()
        H1_sym = m.build_matrix(m.basis, op='H')
        S_sym  = m.build_matrix(m.basis, op='S')
        H2_sym = m.o2_matrix(m.basis)
        print(f'  done in {time.time()-t0:.1f}s')
        with open(CACHE, 'wb') as fh:
            pickle.dump((H1_sym, S_sym, H2_sym), fh)

    # -----------------------------------------------------------------------
    # Step 2: Polynomial decomposition H1 = h * sum_q s^q A_q, S = sum_q s^q B_q
    # -----------------------------------------------------------------------
    print('\nStep 2: Decomposing H1, S into polynomial-in-s coefficient matrices...')
    h_sym, s_sym = sp.symbols('h s')
    t0 = time.time()
    A = decompose_polynomial_matrix(H1_sym, s_sym, factor=h_sym)
    B = decompose_polynomial_matrix(S_sym,  s_sym)
    print(f'  H1: {len(A)} matrices (deg 0..{len(A)-1}),  '
          f'S: {len(B)} matrices (deg 0..{len(B)-1});  {time.time()-t0:.1f}s')

    def H1_eval(h_val, s_val):
        return h_val * sum((s_val ** q) * A[q] for q in range(len(A)))
    def S_eval(s_val):
        return sum((s_val ** q) * B[q] for q in range(len(B)))

    # -----------------------------------------------------------------------
    # Step 3: D_6 A_1g projector -> 38-dim block
    # -----------------------------------------------------------------------
    print('\nStep 3: Building D_6 A_1g orbit-sum projector...')
    U_a, U_a_int, orbits = a1g_orbit_projector(det_strings)
    print(f'  A_1g block dim = {U_a.shape[1]},  # orbits = {len(orbits)}')

    # -----------------------------------------------------------------------
    # Step 4: Singlet projection (S^2 = 0) using exact integer arithmetic
    # -----------------------------------------------------------------------
    print('\nStep 4: Projecting to singlet (S^2 = 0) within A_1g (exact arithmetic)...')
    t0 = time.time()
    S2 = s_squared_matrix(det_strings)
    S2_a_int = U_a_int.T @ S2.astype(int) @ U_a_int   # 38x38 exact int
    S2_a_sp  = sp.Matrix(S2_a_int.tolist())
    US_basis = S2_a_sp.nullspace()                      # exact rational null vectors
    US_sp    = sp.Matrix.hstack(*US_basis)               # 38x22 rational matrix
    print(f'  US_sp shape: {US_sp.shape}  ({time.time()-t0:.1f}s)')
    if US_sp.shape[1] != 22:
        raise RuntimeError(f'singlet kernel dim = {US_sp.shape[1]}, expected 22')

    # Float projector for numerical verification
    S2_a = U_a.T @ S2 @ U_a
    S2_a = 0.5 * (S2_a + S2_a.T)
    ev, V = np.linalg.eigh(S2_a)
    US_float = V[:, np.abs(ev) < 1e-6]   # 38x22 float
    U22 = U_a @ US_float                  # 400x22 float, orthonormal at s=0

    # -----------------------------------------------------------------------
    # Step 5: Build 22x22 symbolic H22 and S22 in the US_sp basis
    # -----------------------------------------------------------------------
    print('\nStep 5: Building 22x22 symbolic H22 and S22 in US_sp basis...')
    t0 = time.time()
    M_q_38 = [(U_a_int.T @ A[q].astype(int) @ U_a_int) for q in range(len(A))]
    N_q_38 = [(U_a_int.T @ B[q].astype(int) @ U_a_int) for q in range(len(B))]
    M_q_22 = [US_sp.T * sp.Matrix(mq.tolist()) * US_sp for mq in M_q_38]
    N_q_22 = [US_sp.T * sp.Matrix(nq.tolist()) * US_sp for nq in N_q_38]
    print(f'  projection done in {time.time()-t0:.1f}s')

    H22_sym = sp.zeros(*M_q_22[0].shape)
    for q in range(len(M_q_22)):
        H22_sym = H22_sym + h_sym * (s_sym ** q) * M_q_22[q]
    S22_sym = sp.zeros(*N_q_22[0].shape)
    for q in range(len(N_q_22)):
        S22_sym = S22_sym + (s_sym ** q) * N_q_22[q]

    # -----------------------------------------------------------------------
    # Step 6: Self-energies (symbolic, rational in h and s)
    # -----------------------------------------------------------------------
    print('\nStep 6: Computing symbolic self-energies E_k(h, s) = H22[k,k] / S22[k,k]...')
    t0 = time.time()
    E_diag_sym = []
    for k in range(22):
        Hkk = H22_sym[k, k]
        Skk = S22_sym[k, k]
        if Skk == 0:
            E_diag_sym.append(sp.nan)
        else:
            E_k = sp.together(sp.simplify(Hkk / Skk))
            E_diag_sym.append(E_k)
    print(f'  done in {time.time()-t0:.1f}s')

    # Verify all self-energies are polynomial-ratio in (h, s) — no irrationals
    n_irrational = 0
    for E in E_diag_sym:
        if E is sp.nan:
            continue
        for sub in sp.preorder_traversal(E):
            if isinstance(sub, sp.Pow) and not sub.exp.is_integer:
                n_irrational += 1
                break
    print(f'  Self-energies with irrational (non-integer power): {n_irrational}  '
          f'(should be 0)')

    # -----------------------------------------------------------------------
    # Step 7: Closed-shell GS in the 400-dim AO basis
    # -----------------------------------------------------------------------
    print('\nStep 7: Building closed-shell GS vector in AO basis...')
    t0 = time.time()
    v0_sp = closed_shell_in_AO_sp(det_strings)
    print(f'  symbolic done in {time.time()-t0:.1f}s')
    v0_num = np.array([float(v0_sp[i, 0]) for i in range(N)])

    # Exact norm squared of v0_sp (rational arithmetic)
    v0_norm2_sp = sum(v0_sp[i, 0] ** 2 for i in range(N))
    v0_norm2    = float(v0_norm2_sp)
    print(f'  ||v0_sp||^2 = {v0_norm2_sp}  (= {v0_norm2:.1f})')
    v0_n = v0_num / np.sqrt(v0_norm2)  # unit-normalised

    # -----------------------------------------------------------------------
    # Step 8: GS in the 22-dim basis (exact metric correction)
    # -----------------------------------------------------------------------
    print('\nStep 8: Projecting GS to 22-dim basis (exact rational arithmetic)...')
    # X = U_a_int @ US_sp is 400x22, columns are orbit-sum singlet basis vectors.
    # Their Gram matrix G = X^T X = US_sp^T D US_sp,  D = diag(orbit sizes).
    # v0_sp = X v22_sp  =>  v22_sp = G^{-1} X^T v0_sp
    t0 = time.time()
    orbit_sizes = (U_a_int.T @ U_a_int).diagonal()
    D_sp   = sp.diag(*[sp.Integer(int(d)) for d in orbit_sizes])
    G      = US_sp.T * D_sp * US_sp              # 22x22 Gram metric (integers)
    U_a_sp = sp.Matrix(U_a_int.tolist())
    v22_naive = US_sp.T * (U_a_sp.T * v0_sp)    # 22x1
    v22_sp    = G.solve(v22_naive)               # exact rational: G^{-1} (X^T v0_sp)
    if all(e == 0 for e in v22_sp):
        raise RuntimeError("v22 is zero: cofactor sign or projector mismatch")
    print(f'  done in {time.time()-t0:.1f}s')

    # Verify: v22_sp^T G v22_sp = ||v0_sp||^2
    G_num   = np.array([[float(G[i, j]) for j in range(22)] for i in range(22)])
    v22_num = np.array([float(v22_sp[k, 0]) for k in range(22)])
    quad    = float(v22_num @ G_num @ v22_num)
    print(f'  v22^T G v22 = {quad:.6f}  (should be {v0_norm2:.6f})')
    assert abs(quad - v0_norm2) < 1e-6, f'Gram quadratic form mismatch: {quad} != {v0_norm2}'

    # Diagonal of Gram metric at s=0 (= S22[k,k] at s=0)
    n_k_arr = [int(G[k, k]) for k in range(22)]

    # True Chirgwin-Coulson weight (non-orthogonal basis):
    #   w_k = c_k * (X[:,k]^T v0_sp) / (v0_sp^T v0_sp)
    #        = v22_sp[k] * v22_naive[k] / ||v0_sp||^2
    #
    # Derivation: expand v0_sp = sum_k c_k X[:,k]  (X = U_a_int @ US_sp).
    # Projector onto X[:,k]: P_k = X[:,k] (X[:,k]^T X[:,k])^{-1} X[:,k]^T
    # (not a rank-1 projector if basis non-orthogonal, but CoulsonCC convention
    # uses the bilinear form v0^T P_k^CC v0 = c_k (X[:,k]^T v0) ).
    # sum_k c_k (X[:,k]^T v0) = sum_k c_k (G v22)_k = v22^T G v22 = ||v0||^2. OK.
    v22_naive_num = np.array([float(v22_naive[k, 0]) for k in range(22)])
    w_alpha_cc    = v22_num * v22_naive_num / v0_norm2   # true CC weights
    w_alpha_sum   = sum(w_alpha_cc)
    print(f'  sum_k w_k (CC) = {w_alpha_sum:.8f}  (should be exactly 1.0)')

    # -----------------------------------------------------------------------
    # Step 9: Verify symbolic GS eigenvector relation
    # -----------------------------------------------------------------------
    print('\nStep 9: Verifying (H22 - E_GS * S22) v22 == 0 symbolically...')
    E_GS_sym = sp.Rational(4) * h_sym * (2 + 3 * s_sym) / \
               ((1 + 2 * s_sym) * (1 + s_sym))
    t0 = time.time()
    residual    = H22_sym * v22_sp - E_GS_sym * (S22_sym * v22_sp)
    simplified  = [sp.cancel(sp.together(e)) for e in residual]
    nonzero     = [(i, e) for i, e in enumerate(simplified) if e != 0]
    print(f'  simplification done in {time.time()-t0:.1f}s')
    if not nonzero:
        print('  ALL 22 residual entries identically zero in Q(h, s).  OK')
    else:
        print(f'  {len(nonzero)} nonzero residual entries:')
        for i, e in nonzero[:3]:
            print(f'    [{i}] = {e}')

    # -----------------------------------------------------------------------
    # Step 10: Verify s-independence via Chirgwin-Coulson (numerical)
    # -----------------------------------------------------------------------
    print('\nStep 10: Verifying s-independence of GS weights...')
    print(f'\n  Chirgwin-Coulson verification (s-independence of w_k = n_k c_k^2 / ||v0||^2):')
    print(f'  Note: w_k use the unnormalised US_sp columns; sum != 1 due to')
    print(f'  off-diagonal G.  True s-independence is verified via the residual')
    print(f'  above and numerically below via the float U22 basis.')
    # Numerical (orthonormal U22 basis) CC weights are exactly s-independent:
    v22_orth = U22.T @ v0_n   # orthonormal coords, s-independent direction
    w_orth   = v22_orth ** 2  # sum = 1
    print(f'\n  Float (U22 orthonormal) weights sum = {sum(w_orth):.8f}')
    print(f'  Verifying CC = w_orth across s values:')
    print(f'  {"s":>5}  {"max|CC - w_orth|":>20}  {"sum CC":>10}')
    for s_val in [0.0, 0.1, 0.2, 0.3]:
        Sn   = S_eval(s_val)
        Sv   = Sn @ v0_n
        denom = v0_n @ Sv
        cc   = v22_orth * (U22.T @ Sv) / denom
        diff = np.max(np.abs(cc - w_orth))
        print(f'  {s_val:>5.2f}  {diff:>20.2e}  {cc.sum():>10.6f}')

    # -----------------------------------------------------------------------
    # Step 11: Per-basis-vector data
    # -----------------------------------------------------------------------
    print('\nStep 11: Computing per-basis-vector data...')

    # Dominant orbit for VB labeling: index with largest |US_sp[i,k]|
    def dominant_orbit(k):
        col = [abs(float(US_sp[i, k])) for i in range(38)]
        return int(np.argmax(col))

    orb_rep = [det_strings[orb[0]] for orb in orbits]

    rows = []
    for k in range(22):
        dom  = dominant_orbit(k)
        rep  = orb_rep[dom]
        nd_k = double_occ(rep)
        c_k  = float(v22_sp[k, 0])
        n_k  = n_k_arr[k]
        w_k  = float(n_k) * c_k ** 2 / v0_norm2
        E_k  = E_diag_sym[k]

        # Exact rational c_k
        c_k_rat = v22_sp[k, 0]

        # Contributing orbits (non-zero US_sp[i,k])
        contrib = [(i, US_sp[i, k]) for i in range(38) if US_sp[i, k] != 0]
        contrib.sort(key=lambda x: abs(float(x[1])), reverse=True)

        # Coupling elements H22[k,j] at s=0 (j != k)
        H22_s0 = H22_sym.subs(s_sym, 0)
        couplings = [(j, H22_s0[k, j]) for j in range(22)
                     if j != k and H22_s0[k, j] != 0]
        couplings.sort(key=lambda x: abs(float(x[1].subs(h_sym, -1))), reverse=True)

        rows.append({
            'k': k, 'dom': dom, 'rep': rep, 'nd': nd_k,
            'vb_str': vb_label(rep),
            'c_alpha': c_k, 'c_alpha_rat': c_k_rat,
            'n_k': n_k, 'w_alpha': w_k,
            'E_self': E_k,
            'contrib': contrib,
            'couplings': couplings,
        })

    # Sort by decreasing w_k
    rows_sorted = sorted(rows, key=lambda r: r['w_alpha'], reverse=True)

    # -----------------------------------------------------------------------
    # Print: main 22-row composition table
    # -----------------------------------------------------------------------
    print()
    print('=' * 170)
    print('BENZENE 22-ROW VB COMPOSITION TABLE  (singlet-A_1g, all s)')
    print('Rows ordered by DECREASING w_alpha = n_k * c_alpha^2 / ||v0||^2')
    print('c_alpha: coordinate in the un-normalised orbit-sum basis (rational)')
    print('n_k = S22[k,k] at s=0 = sum of orbit sizes in US_sp[:,k]')
    print('w_alpha sums to ~1 only for orthogonal US_sp columns (check table)')
    print('Self-energy E_alpha(h,s) = H22[k,k] / S22[k,k]  (rational, vanishes at s=0)')
    print('=' * 170)
    print()
    print(f'  {"#":>2}  {"k":>2}  {"nd":>3}  {"c_alpha":>8}  {"n_k":>4}  '
          f'{"w_alpha":>8}  {"E_alpha(h, s)":<55}  VB structure')
    print('-' * 170)
    for row_num, r in enumerate(rows_sorted, 1):
        E_str = sp.sstr(r['E_self'])
        print(f'  {row_num:>2}  {r["k"]+1:>2}  {r["nd"]:>3d}  '
              f'{r["c_alpha"]:>8.4f}  {r["n_k"]:>4d}  '
              f'{r["w_alpha"]:>8.5f}  '
              f'{E_str:<55}  {r["vb_str"]}')
    print()
    w_total = sum(r['w_alpha'] for r in rows)
    print(f'  Sum of w_alpha = {w_total:.6f}  '
          f'(= sum n_k c_k^2 / ||v0||^2 = {sum(n_k_arr[k]*v22_num[k]**2 for k in range(22)):.1f}'
          f' / {v0_norm2:.1f})')
    print(f'  ||v0||^2 = {v0_norm2:.1f}  (= v22^T G v22 = {quad:.1f})')

    # -----------------------------------------------------------------------
    # Print: coupling table
    # -----------------------------------------------------------------------
    print()
    print('=' * 90)
    print('H22 upper-triangle non-zero couplings at s=0:')
    H22_s0 = H22_sym.subs(s_sym, 0)
    print(f'  {"(i,j)":>8}  {"H22[i,j] at s=0":>28}  {"H22[i,j](h=-1)":>16}')
    any_coup = False
    for i in range(22):
        for j in range(i + 1, 22):
            v = H22_s0[i, j]
            if v != 0:
                any_coup = True
                v_rat = sp.together(sp.simplify(v))
                v_num = float(v_rat.subs(h_sym, -1))
                print(f'  ({i+1:2d},{j+1:2d})  {sp.sstr(v_rat):>28}  {v_num:>16.4f}')
    if not any_coup:
        print('  (none)')

    # -----------------------------------------------------------------------
    # Print: orbit catalogue
    # -----------------------------------------------------------------------
    print()
    print('=' * 100)
    print('D_6 A_1g orbit catalogue:')
    print(f'  {"idx":>3}  {"size":>5}  {"n_d":>3}  {"rep det":<12}  VB label')
    for i, orb in enumerate(orbits):
        rep = det_strings[orb[0]]
        nd  = double_occ(rep)
        print(f'  {i:>3}  {len(orb):>5d}  {nd:>3d}  {rep:<12}  {vb_label(rep)}')

    # -----------------------------------------------------------------------
    # Print: US_sp column decomposition
    # -----------------------------------------------------------------------
    print()
    print('Basis vector -> contributing orbits (column k of US_sp in the 38-orbit basis):')
    print(f'  {"k":>2}  {"c_alpha":>8}  {"E_alpha(h,s)":<48}  contributing orbits (idx: coeff)')
    for r in rows:
        k = r['k']
        parts_str = ',  '.join(
            f'{i}: {float(c):+.4f}' for i, c in r['contrib'][:8])
        print(f'  {k+1:>2}  {r["c_alpha"]:>+8.4f}  {sp.sstr(r["E_self"]):<48}  {parts_str}')

    # -----------------------------------------------------------------------
    # Print: eigenvalues of H22 at s=0 for comparison
    # -----------------------------------------------------------------------
    print()
    print('=' * 80)
    print('EIGENVALUES of H22 at s=0, h=-1  (22-dim GEVP):')
    H22_s0_num = np.array([[float(H22_s0[i, j].subs(h_sym, -1))
                             for j in range(22)] for i in range(22)], dtype=float)
    S22_s0     = S22_sym.subs(s_sym, 0)
    S22_s0_num = np.array([[float(S22_s0[i, j])
                             for j in range(22)] for i in range(22)], dtype=float)
    H22_s0_num = 0.5 * (H22_s0_num + H22_s0_num.T)
    S22_s0_num = 0.5 * (S22_s0_num + S22_s0_num.T)
    eigvals_22 = scipy.linalg.eigh(H22_s0_num, S22_s0_num, eigvals_only=True)
    E_GS_num = float(E_GS_sym.subs([(h_sym, -1), (s_sym, 0)]))
    print(f'  Eigenvalues (h=-1): {np.round(eigvals_22, 6).tolist()}')
    print(f'  Lowest eigenvalue = {eigvals_22[0]:.10f}')
    print(f'  Formula E_GS(h=-1, s=0) = {E_GS_num:.10f}')
    print(f'  Match: {abs(eigvals_22[0] - E_GS_num) < 1e-8}')

    # -----------------------------------------------------------------------
    # Full 400-dim verification
    # -----------------------------------------------------------------------
    print()
    print('=' * 60)
    print('Verification: full 400-dim lowest eigenvalue at s=0, h=-1:')
    H0 = H1_eval(-1.0, 0.0)
    ev_full = np.linalg.eigvalsh(H0)
    print(f'  Full FCI lowest eigenvalue = {ev_full[0]:.10f}')
    print(f'  22-dim block lowest eigenvalue = {eigvals_22[0]:.10f}')
    print(f'  Formula 4h(2+3s)/[(1+2s)(1+s)] = {E_GS_num:.10f}')
    print(f'  Difference (FCI - 22dim) = {abs(ev_full[0] - eigvals_22[0]):.2e}')

    # -----------------------------------------------------------------------
    # Self-energy table
    # -----------------------------------------------------------------------
    print()
    print('=' * 110)
    print('All 22 self-energies E_alpha(h, s) = H22[k,k] / S22[k,k]:')
    print('(All vanish at s=0 because H22 is block-off-diagonal there;')
    print(' the leading behaviour is O(hs) as s -> 0.)')
    print(f'  {"k":>2}  {"S22[k,k]":<36}  {"E_alpha(h,s)":<55}')
    for k in range(22):
        Skk  = sp.sstr(S22_sym[k, k])
        Estr = sp.sstr(E_diag_sym[k])
        print(f'  {k+1:>2}  {Skk:<36}  {Estr}')

    print('\nDone.')


if __name__ == '__main__':
    main()
