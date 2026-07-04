"""
Closed-form 1e ground state of benzene at non-zero AO overlap, via the
22-dim singlet-A_1g VB block.

Setup: tight-binding Huckel benzene, h_{ij} = h on each ring bond,
S_{ij} = s on each ring bond, U = 0 (one-electron only, no on-site
repulsion).

The 1-particle generalized eigenproblem h_AO c = eps S_AO c is solved
simultaneously by Bloch waves c_k(j) ~ exp(2 pi i k j / 6) (for any s),
giving MO energies

    eps_k(s) = h * lambda_k / (1 + s * lambda_k),
    lambda_k = 2 cos(2 pi k / 6) = +2, +1, +1, -1, -1, -2.

The closed-shell ground state, doubly filling k = 0, +-1, has total
energy

    E_GS(h, s) = 2 eps_0 + 4 eps_1
              = 4h/(1+2s) + 4h/(1+s)
              = 4h * (2 + 3s) / [(1+2s)(1+s)].

Reduction chain at s != 0:

    400  Sz=0 AO-det FCI
     |  D_6 + sigma_v       <- s-independent
    38   A_1g
     |  S^2 = 0              <- s-independent (operator is geometry-blind)
    22   singlet-A_1g        <- closed-shell GS lives here, keeping s
     |  eta^2 = 0            <- requires s = 0 (BREAKS at s != 0)
    14

Demonstrates:
  - 22-dim is the smallest reduction that preserves s as a free symbol.
  - Numerical lowest eigenvalue of the 22-dim generalized eigenproblem
    matches E_GS(h, s) at several values of s.
  - Bloch-wave MO eigenvectors are s-independent (only normalization
    changes), so the closed-shell GS direction in AO det space is fixed
    at s = 0 and remains an eigenvector of the (s-dependent) generalized
    eigenproblem at every s.

Run from the repo root: PYTHONPATH=. python3 examples/benzene_1e_analytical_overlap.py
"""
import os
import sys
import pickle
import time

import numpy as np
import sympy as sp
import scipy.linalg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from symvb import Molecule, SlaterDet, symmetry, verify_eigenpair
from symvb.huckel import solve_ring
from symvb.mo_projection import mo_determinant_in_ao
from symvb.numerical import decompose_polynomial_matrix
from symvb.spin import s_squared_matrix


CACHE = '/tmp/benzene_hubbard_matrices.pkl'


def build_basis_and_matrices():
    m = Molecule(
        zero_ii=True,
        interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'],
        subst={'h': ('H_ab', 'H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af'),
               's': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af')},
        subst_2e={'U': ('1111',)},
        max_2e_centers=1,
    )
    m.generate_basis(3, 3, 6)
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            H1, S, _H2 = pickle.load(f)
    else:
        print("Cache miss; building 400x400 symbolic H1, S (~5s)...")
        t0 = time.time()
        H1 = m.build_matrix(m.basis, op='H')
        S  = m.build_matrix(m.basis, op='S')
        H2 = m.o2_matrix(m.basis)
        print(f"  done in {time.time() - t0:.1f}s")
        with open(CACHE, 'wb') as f:
            pickle.dump((H1, S, H2), f)
    return m, H1, S


def a1g_orbit_projector(det_strings):
    """Returns (U_a_norm, U_a_int, orbits) -- the orthonormalized A_1g
    projector (columns 1/sqrt(orbit size)), the integer-{0,1} orbit-sum
    projector, and the orbit lists. Both projectors span the 38-dim
    A_1g block; U_a_int is needed for exact-arithmetic operations."""
    def canon(ds):
        fp = SlaterDet(ds).get_sorted()
        return fp.dets[0].det_string, fp.coefs[0]
    C6    = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'e', 'e': 'f', 'f': 'a'}
    sigma = {'a': 'a', 'b': 'f', 'c': 'e', 'd': 'd', 'e': 'c', 'f': 'b'}
    perms = [symmetry.apply_orbital_permutation(om, det_strings, canon)[0]
             for om in (C6, sigma)]
    U_a_norm, orbits = symmetry.totally_symmetric_basis(
        perms, len(det_strings))
    U_a_int = np.zeros((len(det_strings), len(orbits)), dtype=int)
    for col, orb in enumerate(orbits):
        for idx in orb:
            U_a_int[idx, col] = 1
    return U_a_norm, U_a_int, orbits


def double_occ(ds):
    occ = {}
    for c in ds:
        occ.setdefault(c.lower(), [False, False])
        if c.islower():
            occ[c.lower()][0] = True
        else:
            occ[c.lower()][1] = True
    return sum(1 for ab in occ.values() if ab[0] and ab[1])


def main():
    print(__doc__)
    print()

    print('Loading benzene FCI basis and symbolic (H1, S)...')
    m, H1, S = build_basis_and_matrices()
    det_strings = [fp.dets[0].det_string for fp in m.basis]
    N = len(det_strings)
    print(f'  {N}-dim AO-det basis (3 alpha + 3 beta in 6 orbitals)')

    print('\nBuilding D_6 A_1g orbit-sum projector (s-independent)...')
    U_a, U_a_int, orbits = a1g_orbit_projector(det_strings)
    print(f'  A_1g block dim = {U_a.shape[1]}')

    print('\nProjecting to singlet (S^2 = 0) within A_1g...')
    S2 = s_squared_matrix(det_strings)
    S2_a = U_a.T @ S2 @ U_a
    S2_a = 0.5 * (S2_a + S2_a.T)
    ev, V = np.linalg.eigh(S2_a)
    US = V[:, np.abs(ev) < 1e-6]
    print(f'  singlet-A_1g block dim = {US.shape[1]}')

    U22 = U_a @ US                                # (400, 22), s-independent

    # Symbolic Huckel solve of the hexagon: integer-rational real Bloch
    # basis (a_2u, e_1g cos, e_1g sin, e_2u cos, e_2u sin, b_2g) with
    # eigenvalues lambda = (+2, +1, +1, -1, -1, -2). The closed-shell
    # GS doubly occupies the three lowest MOs.
    huckel_result = solve_ring(6)
    h_sym, s_sym = huckel_result.h_symbol, huckel_result.s_symbol
    E_GS_sym = sp.cancel(huckel_result.energy_of_occupation([2, 2, 2, 0, 0, 0]))
    E_GS = sp.lambdify((h_sym, s_sym), E_GS_sym, 'numpy')

    # H1 = h * sum_q s^q * A_q;  S = sum_q s^q * B_q.
    print('\nDecomposing H1, S into polynomial-in-s coefficient matrices...')
    t0 = time.time()
    A = decompose_polynomial_matrix(H1, s_sym, factor=h_sym)
    B = decompose_polynomial_matrix(S,  s_sym)
    print(f'  H1: {len(A)} matrices (degrees 0..{len(A)-1}),  '
          f'S: {len(B)} matrices (degrees 0..{len(B)-1});  '
          f'{time.time() - t0:.1f}s')

    def H1_eval(h_val, s_val):
        return h_val * sum((s_val ** q) * A[q] for q in range(len(A)))
    def S_eval(s_val):
        return sum((s_val ** q) * B[q] for q in range(len(B)))

    # --- Verification 1: numerical lowest eigenvalue at several s ------
    print(f'\nLowest eigenvalue of the 22-dim generalized eigenproblem '
          f'(h = -1):')
    print(f'  Predicted: E_GS(h, s) = 4h(2+3s) / [(1+2s)(1+s)]\n')
    print(f'  {"s":>5}  {"eig_lowest":>16}  {"E_GS formula":>16}  '
          f'{"diff":>10}')
    # NB: S_AO becomes singular at s = 1/2 (eigenvalue 1 + s*lambda_min
    # vanishes for the antibonding b_2g mode at s = 0.5), so stop short.
    for s_val_f in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45]:
        H_num = H1_eval(-1.0, s_val_f)
        S_num = S_eval(s_val_f)
        H22 = U22.T @ H_num @ U22
        S22 = U22.T @ S_num @ U22
        H22 = 0.5 * (H22 + H22.T)
        S22 = 0.5 * (S22 + S22.T)
        eigs = scipy.linalg.eigh(H22, S22, eigvals_only=True)
        E_pred = E_GS(-1.0, s_val_f)
        print(f'  {s_val_f:>5.2f}  {eigs[0]:>16.10f}  {E_pred:>16.10f}  '
              f'{eigs[0] - E_pred:>10.2e}')

    # --- Verification 2: GS direction in AO-det space is s-independent
    # Build the closed-shell determinant via symvb.mo_projection: complex
    # Bloch waves k in {0, +1, -1}, doubly occupied. The s = 0
    # normalization fixes the direction; s-dependence is a global scalar.
    print(f'\nClosed-shell MO determinant in the AO-det basis '
          f'(s = 0 normalization)...')
    omega = np.exp(2j * np.pi / 6)
    C_complex = np.array([[omega ** (k * j) for j in range(6)]
                           for k in [0, 1, -1]]) / np.sqrt(6)
    v0 = mo_determinant_in_ao(
        C_complex, ([0, 1, 2], [0, 1, 2]),
        det_strings, site_labels='abcdef')
    v22 = U22.T @ v0
    norm22 = np.linalg.norm(v22)
    print(f'  ||v_0||_2 = {np.linalg.norm(v0):.6e},  '
          f'||v_22||_2 = {norm22:.6e}')
    if norm22 < 1e-12:
        raise RuntimeError("v_22 has zero norm: closed-shell vector "
                           "missed the singlet-A_1g block.")

    print(f'\nCheck H_22(h,s) v_22  ==  E_GS(h,s) * S_22(s) v_22 across (h, s):')
    print(f'  {"h":>5}  {"s":>5}  {"max|residual|":>16}')
    for h_val, s_val in [(-1.0, 0.0), (-1.0, 0.1), (-1.0, 0.2), (-1.0, 0.3),
                         (-2.0, 0.25), (1.0, 0.2)]:
        H_num = H1_eval(h_val, s_val)
        S_num = S_eval(s_val)
        H22 = U22.T @ H_num @ U22
        S22 = U22.T @ S_num @ U22
        E = E_GS(h_val, s_val)
        residual = H22 @ v22 - E * (S22 @ v22)
        print(f'  {h_val:>5.1f}  {s_val:>5.2f}  '
              f'{np.max(np.abs(residual)):>16.2e}')

    # --- Symbolic identity check at general (h, s) ---------------------
    # Build exact rational U22, then exact symbolic 22x22 matrices
    # H_22(h, s), S_22(s), and verify the GS eigenvector relation as a
    # polynomial identity (residual identically zero in Q(h, s)^22).
    print(f'\n--- Symbolic identity (H_22 - E_GS S_22) v_22 == 0 ---')
    t0 = time.time()
    # Use integer 0/1 orbit-sum projector for exact arithmetic.
    S2_a_int = U_a_int.T @ S2.astype(int) @ U_a_int     # 38x38 int
    S2_a_sp = sp.Matrix(S2_a_int.tolist())
    US_basis = S2_a_sp.nullspace()
    US_sp = sp.Matrix.hstack(*US_basis) if US_basis else sp.Matrix.zeros(38, 0)
    print(f'  US (rational, exact): shape {US_sp.shape},  '
          f'{time.time() - t0:.1f}s')

    if US_sp.shape[1] != 22:
        raise RuntimeError(f'singlet kernel dim = {US_sp.shape[1]}, expected 22')

    # 22x22 symbolic projections of each polynomial coefficient matrix.
    print('  projecting A_q, B_q to 22-dim (rational)...')
    t0 = time.time()
    M_q_38 = [(U_a_int.T @ A[q].astype(int) @ U_a_int) for q in range(len(A))]
    N_q_38 = [(U_a_int.T @ B[q].astype(int) @ U_a_int) for q in range(len(B))]
    M_q_sp = [US_sp.T * sp.Matrix(m.tolist()) * US_sp for m in M_q_38]
    N_q_sp = [US_sp.T * sp.Matrix(m.tolist()) * US_sp for m in N_q_38]
    print(f'  done in {time.time() - t0:.1f}s')

    # Closed-shell GS in AO basis, exactly, via the real Hückel MOs of
    # huckel_result (sqrt(3)/2 prefactor absorbed in the row scaling).
    print('  building v_0 exactly (real Huckel cofactor)...')
    t0 = time.time()
    v0_sp = mo_determinant_in_ao(
        huckel_result.coefficients, ([0, 1, 2], [0, 1, 2]),
        det_strings, site_labels='abcdef')
    print(f'  done in {time.time() - t0:.1f}s')

    # The 22-dim "basis matrix" is X = U_a_int * US_sp (400 x 22, rational).
    # Its columns are NOT orthonormal -- the orbit-sum projector has non-
    # trivial Gram metric G = X^T X = US^T D US, with D = diag(orbit sizes).
    # Coordinates of v_0 in this basis solve  X v22 = v_0,  giving
    # v22 = G^{-1} (X^T v_0).  Computing v22 directly via X^T v_0 (without
    # G^{-1}) gives G v22_true, which is NOT an eigenvector.
    U_a_sp = sp.Matrix(U_a_int.tolist())
    orbit_sizes = (U_a_int.T @ U_a_int).diagonal()    # = D as a 1-D int
    D_sp = sp.diag(*[sp.Integer(int(d)) for d in orbit_sizes])
    G = US_sp.T * D_sp * US_sp                        # 22x22 metric
    v22_naive = US_sp.T * (U_a_sp.T * v0_sp)
    v22_sp = G.solve(v22_naive)                       # exact rational solve
    if all(e == 0 for e in v22_sp):
        raise RuntimeError("v22 is zero: cofactor sign or projector mismatch")

    # Residual at every (h, s) -- 22-vector of rational functions.
    print('  forming H_22(h,s), S_22(s) and residual...')
    t0 = time.time()
    H22sp = sp.zeros(*M_q_sp[0].shape)
    for q in range(len(M_q_sp)):
        H22sp = H22sp + h_sym * (s_sym ** q) * M_q_sp[q]
    S22sp = sp.zeros(*N_q_sp[0].shape)
    for q in range(len(N_q_sp)):
        S22sp = S22sp + (s_sym ** q) * N_q_sp[q]
    residual = H22sp * v22_sp - E_GS_sym * (S22sp * v22_sp)
    print(f'  matrix ops: {time.time() - t0:.1f}s')

    print('  simplifying 22 residual entries via verify_eigenpair...')
    t0 = time.time()
    try:
        verify_eigenpair(H22sp, S22sp, v22_sp, E_GS_sym)
        print(f'  done in {time.time() - t0:.1f}s')
        print(f'  ALL 22 residual entries identically zero in Q(h, s).  ✓')
        print(f'  E_GS(h, s) = 4h(2+3s)/[(1+2s)(1+s)] is exact for the '
              f'closed-shell GS at every s.')
    except Exception as exc:
        print(f'  done in {time.time() - t0:.1f}s')
        print(f'  FAILED: {exc}')

    # --- AO-VB ionicity weights are genuinely s-INDEPENDENT --------------
    # Each H_1e eigenstate is simultaneously an S(s)-eigenvector (Slater
    # det of one-particle S-eigenvectors), so S(s) v = gamma(s) v. The
    # Chirgwin-Coulson weight w_C(s) = (v^T P_C S v)/(v^T S v) collapses
    # to the Euclidean class fraction v^T P_C v / v^T v, with the
    # gamma(s) factor cancelling in numerator and denominator.
    print('\nVerifying ionicity weights are s-independent for closed-shell GS:')
    classes_arr = np.array([double_occ(ds) for ds in det_strings])
    v0n = v0 / np.linalg.norm(v0)
    w_eucl = np.array([(v0n[classes_arr == c] ** 2).sum() for c in range(4)])
    print(f'  Euclidean (s = 0): {":".join(f"{w:.6f}" for w in w_eucl)}')
    for sv in [0.1, 0.2, 0.3, 0.4]:
        Sn = sum((sv ** q) * B[q] for q in range(len(B)))
        Sv = Sn @ v0
        denom = v0 @ Sv
        w_cc = np.array([v0[classes_arr == c] @ Sv[classes_arr == c]
                         for c in range(4)]) / denom
        print(f'  Chirgwin-Coulson (s={sv}): '
              f'{":".join(f"{w:.6f}" for w in w_cc)}  '
              f'max|diff|={np.max(np.abs(w_cc - w_eucl)):.1e}')

    # --- AO-VB ionicity decomposition of v_0 ---------------------------
    print(f'\nAO-VB ionicity weights of |v_0> (closed-shell GS, s=0 norm):')
    classes = [[] for _ in range(4)]
    for I, ds in enumerate(det_strings):
        c = double_occ(ds)
        classes[c].append((I, v0[I] ** 2))
    total = sum(w for cls in classes for _, w in cls)
    print(f'  ionicity (# doubly-occupied sites)   # AO dets   sum |v|^2')
    for c in range(4):
        s_w = sum(w for _, w in classes[c])
        print(f'  {c:<37d}  {len(classes[c]):>8d}   '
              f'{s_w / total:>10.6f}')

    print(f'\n  (Class sizes: 20 covalent + 180 single-ionic + 180 '
          f'double-ionic + 20 triple-ionic = 400 AO dets.)')

    # --- GS breakdown over the 22 singlet-A_1g basis vectors ----------
    # The 38 -> 22 projection is the existing US_sp (columns = singlet
    # basis vectors in the D_6 orbit-sum coordinates), with v22_sp giving
    # the GS amplitudes in that basis. M_q_sp / N_q_sp are the 22x22
    # symbolic polynomial-coefficient matrices already built above. So
    # H_22(h, s) = h * sum_q s^q M_q_sp[q],  S_22(s) = sum_q s^q N_q_sp[q],
    # and per-row data (self-energy E_k = H_22[k,k]/S_22[k,k], couplings
    # H_kj = H_22[k,j]) is just an entry lookup. No merging or grouping.
    print('\n--- GS breakdown over the 22 singlet-A_1g basis vectors ---')
    n_singlet = US_sp.shape[1]

    def H22_entry(k, j):
        return h_sym * sum(M_q_sp[q][k, j] * (s_sym ** q)
                           for q in range(len(M_q_sp)))

    def S22_entry(k, j):
        return sum(N_q_sp[q][k, j] * (s_sym ** q)
                   for q in range(len(N_q_sp)))

    # GS norm in the singlet basis: v22^T G v22 with G = US^T D US (D =
    # diag of orbit sizes). Use to normalize per-row weights so they
    # sum to 1 (Chirgwin-Coulson on the singlet-basis Gram).
    Gv = G * v22_sp
    norm_sq = (v22_sp.T * Gv)[0, 0]

    print(f'  {"#":>2}  {"D_6 orbit contributions":<32}  {"c_k":>10}  '
          f'{"w_k":>10}  {"E_k(h, s)":<40}  H_kj (->j)')
    for k in range(n_singlet):
        bk = US_sp[:, k]
        contribs = [(i, bk[i, 0]) for i in range(bk.rows) if bk[i, 0] != 0]
        contrib_str = ", ".join(f"{i}({sp.sstr(c)})" for i, c in contribs)

        c_k = v22_sp[k, 0]
        w_k = c_k * Gv[k, 0] / norm_sq
        E_k = sp.together(sp.cancel(H22_entry(k, k) / S22_entry(k, k)))

        coup_parts = []
        for j in range(k + 1, n_singlet):
            H_kj = sp.expand(H22_entry(k, j))
            if H_kj != 0:
                coup_parts.append(f"->{j}: {sp.sstr(H_kj)}")
        coup_str = "; ".join(coup_parts) if coup_parts else "--"

        print(f'  {k:>2}  {contrib_str:<32}  {sp.sstr(c_k):>10}  '
              f'{sp.sstr(w_k):>10}  {sp.sstr(E_k):<40}  {coup_str}')
    print(f'  Sum of weights: {sp.sstr(sp.simplify(sum(v22_sp[k, 0] * Gv[k, 0] / norm_sq for k in range(n_singlet))))}')


if __name__ == '__main__':
    main()
