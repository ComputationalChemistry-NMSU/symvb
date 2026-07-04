"""
14-row composition table for the benzene pi system at s=0 (orthogonal AOs).

Reduction chain:

    400  (full Sz=0 FCI)
     |  D_6 (C_6 + sigma_v)
    38   A_1g orbits
     |  S^2 = 0
    22   singlet-A_1g
     |  eta^2 = 0  (valid at s=0 only)
    14   singlet-A_1g, eta=0  <- target space

For each of the 14 basis vectors (columns of U14), the table shows:
  - n_d: dominant doubly-occupied count
  - VB structure label (Kekule, Dewar, ionic, etc.)
  - c_alpha: coefficient of basis vector k in the 1e (U=0, s=0) GS expansion
             c_alpha_k = (U14.T @ v_gs_400)[k]   (NOT eigenvector of H_14!)
  - w_alpha = c_alpha^2: weight in GS  (sum = 1.0)
  - E_alpha(h, s=0): self-energy = diagonal element H_14[k,k]
                      (NOT the eigenvalue of H_14, since U14 columns are NOT eigenstates)
  - H_14: 14x14 Hamiltonian in this basis at s=0

NOTE: The U14 basis vectors are NOT eigenstates of H.  They are projectors
built from symmetry (D_6, S^2, eta^2 = 0).  The self-energy of basis vector
k is H_14[k,k], the diagonal matrix element.  The eigenvalues of H_14 are
shown separately for comparison.

The self-energies E_alpha_k = H_14[k,k] are in general IRRATIONAL algebraic
numbers (they come from rotating through eigenvectors of eta^2), so they are
displayed as decimal multiples of h.  In contrast, the 14 eigenvalues of H_14
are integer multiples of h (Huckel spectrum: {8,4,4,2,2,1,0,0,-1,-2,-2,-4,-4,-8}).

At s=0: S = I, so the GEVP is a plain eigenvalue problem. The 1e GS energy
is E_GS = 4h*(2+3s)/[(1+2s)(1+s)] = 8h at s=0, h=-1 -> E_GS = -8.
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
from symvb.spin import s_squared_matrix, eta_squared_matrix
from symvb.numerical import decompose_polynomial_matrix
from symvb.mo_projection import mo_determinant_in_ao

CACHE = '/tmp/benzene_hubbard_matrices.pkl'

ORBS = list('abcdef')
# Bipartite sublattice signs: {a,c,e}=+1, {b,d,f}=-1
SITE_SIGNS = {'a': +1, 'b': -1, 'c': +1, 'd': -1, 'e': +1, 'f': -1}


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


def closed_shell_in_AO(det_strings):
    """Bloch-wave MO closed-shell GS direction in AO det basis (s=0 norm)."""
    omega = np.exp(2j * np.pi / 6)
    C = np.array([[omega ** (k * j) for j in range(6)]
                   for k in [0, 1, -1]]) / np.sqrt(6)
    v = mo_determinant_in_ao(C, ([0, 1, 2], [0, 1, 2]),
                             det_strings, site_labels='abcdef')
    return np.asarray(v, dtype=float).ravel()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print(__doc__)

    # -----------------------------------------------------------------------
    # 1. Build basis and load/build matrices
    # -----------------------------------------------------------------------
    print('Loading benzene FCI basis and (H1, S) from cache...')
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
    print(f'  N = {N} dets')

    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as fh:
            H1_sym, S_sym, _H2 = pickle.load(fh)
        print('  loaded from cache')
    else:
        print('  cache miss - building (~5s)...')
        t0 = time.time()
        H1_sym = m.build_matrix(m.basis, op='H')
        S_sym  = m.build_matrix(m.basis, op='S')
        H2_sym = m.o2_matrix(m.basis)
        print(f'  done in {time.time()-t0:.1f}s')
        with open(CACHE, 'wb') as fh:
            pickle.dump((H1_sym, S_sym, H2_sym), fh)

    # -----------------------------------------------------------------------
    # 2. Polynomial decomposition in s (with h as factor)
    # -----------------------------------------------------------------------
    print('\nDecomposing H1, S into polynomial-in-s coefficient matrices...')
    h_sym, s_sym = sp.symbols('h s')
    t0 = time.time()
    A = decompose_polynomial_matrix(H1_sym, s_sym, factor=h_sym)
    B = decompose_polynomial_matrix(S_sym, s_sym)
    print(f'  H1: {len(A)} matrices, S: {len(B)} matrices  {time.time()-t0:.1f}s')

    # At s=0, h=-1:
    H0 = -1.0 * A[0]   # H at s=0, h=-1
    S0 = B[0]           # S at s=0 (= identity)

    # -----------------------------------------------------------------------
    # 3. D_6 A_1g projector -> 38-dim block
    # -----------------------------------------------------------------------
    print('\nBuilding D_6 A_1g projector...')

    def canon(ds):
        fp = SlaterDet(ds).get_sorted()
        return fp.dets[0].det_string, fp.coefs[0]

    C6    = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'e', 'e': 'f', 'f': 'a'}
    sigma = {'a': 'a', 'b': 'f', 'c': 'e', 'd': 'd', 'e': 'c', 'f': 'b'}

    perms = [symmetry.apply_orbital_permutation(om, det_strings, canon)[0]
             for om in (C6, sigma)]

    U_a, orbits = symmetry.totally_symmetric_basis(perms, N)
    # Integer projector for exact arithmetic
    U_a_int = np.zeros((N, len(orbits)), dtype=int)
    for col, orb in enumerate(orbits):
        for idx in orb:
            U_a_int[idx, col] = 1
    print(f'  A_1g dim = {U_a.shape[1]},  # orbits = {len(orbits)}')

    H_a = U_a.T @ H0 @ U_a
    H_a = 0.5 * (H_a + H_a.T)

    # -----------------------------------------------------------------------
    # 4. S^2 -> singlet projection -> 22-dim
    # -----------------------------------------------------------------------
    print('\nBuilding S^2 and projecting to singlet-A_1g...')
    t0 = time.time()
    S2 = s_squared_matrix(det_strings)
    S2_a = U_a.T @ S2 @ U_a
    S2_a = 0.5 * (S2_a + S2_a.T)
    ev_s2, vS = np.linalg.eigh(S2_a)
    US = vS[:, np.abs(ev_s2) < 1e-6]
    print(f'  singlet-A_1g dim = {US.shape[1]}  ({time.time()-t0:.2f}s)')

    U22 = U_a @ US   # 400 x 22

    H_22 = US.T @ H_a @ US
    H_22 = 0.5 * (H_22 + H_22.T)

    # -----------------------------------------------------------------------
    # 5. eta^2 -> eta=0 projection -> 14-dim
    # -----------------------------------------------------------------------
    print('\nBuilding eta^2 and projecting to eta=0...')
    t0 = time.time()
    E2 = eta_squared_matrix(det_strings, SITE_SIGNS, ORBS)
    E2_a = U_a.T @ E2 @ U_a
    E2_a = 0.5 * (E2_a + E2_a.T)
    E2_s = US.T @ E2_a @ US
    E2_s = 0.5 * (E2_s + E2_s.T)
    ev_e2, vE = np.linalg.eigh(E2_s)
    UE = vE[:, np.abs(ev_e2) < 1e-6]
    print(f'  eta=0 dim = {UE.shape[1]}  ({time.time()-t0:.2f}s)')
    assert UE.shape[1] == 14, f'Expected 14, got {UE.shape[1]}'

    U14 = U22 @ UE   # 400 x 14 (columns = basis vectors of target space)

    H_14 = UE.T @ H_22 @ UE
    H_14 = 0.5 * (H_14 + H_14.T)

    # -----------------------------------------------------------------------
    # 6. Diagonalise 14-dim block to get eigenvalues for comparison
    # -----------------------------------------------------------------------
    print('\nDiagonalising 14x14 block at s=0, h=-1, U=0...')
    eigvals_14, eigvecs_14 = np.linalg.eigh(H_14)
    print(f'  Eigenvalues: {np.round(eigvals_14, 6).tolist()}')
    E_GS_formula = 8 * (-1.0)   # 4h(2+3s)/[(1+2s)(1+s)] at s=0, h=-1
    print(f'  Lowest eigenvalue = {eigvals_14[0]:.10f}  '
          f'(formula 8h = {E_GS_formula:.6f})')
    assert abs(eigvals_14[0] - E_GS_formula) < 1e-8, \
        f'GS mismatch: {eigvals_14[0]} != {E_GS_formula}'
    print('  GS energy matches 8h formula.  OK')

    # -----------------------------------------------------------------------
    # 7. Diagnostic: is H_14 diagonal?
    # -----------------------------------------------------------------------
    diag_H14 = np.diag(np.diag(H_14))
    offdiag_norm = np.linalg.norm(H_14 - diag_H14, 'fro')
    is_diagonal = offdiag_norm < 1e-10
    print(f'\nDiagnostic: ||H_14 - diag(H_14)||_F = {offdiag_norm:.6e}')
    print(f'  H_14 is {"DIAGONAL" if is_diagonal else "NOT DIAGONAL"} '
          f'-> U14 columns are '
          f'{"eigenstates" if is_diagonal else "NOT eigenstates"} of H')

    # -----------------------------------------------------------------------
    # 8. GS at s=0, U=0: projection of 400-dim GS onto U14 basis
    # -----------------------------------------------------------------------
    # The 400-dim GS (Bloch closed-shell)
    v_bloch = closed_shell_in_AO(det_strings)
    v_bloch_n = v_bloch / np.linalg.norm(v_bloch)

    # Verify: project onto 14-dim subspace and check it's within
    v_gs_in_14 = U14.T @ v_bloch_n        # 14-dim coordinates: c_k = <u_k | v_GS>
    v_gs_400_reconstructed = U14 @ v_gs_in_14
    overlap_check = np.linalg.norm(v_gs_in_14)
    print(f'\n  GS projection onto 14-dim subspace: ||P_14 v_GS|| = {overlap_check:.8f}')
    print(f'  (should be ~1.0 if GS lies entirely in the 14-dim space)')

    # Also verify using eigvecs_14 route for comparison
    gs_vec_14_eig = eigvecs_14[:, 0]      # eigenvector route (rotation in 14-dim)
    v_gs_400_eig = U14 @ gs_vec_14_eig
    overlap_bloch_eig = abs(v_gs_400_eig @ v_bloch_n)
    print(f'  |<Bloch_GS | eig_GS>| = {overlap_bloch_eig:.8f}  (should be ~1)')

    # The correct GS coefficients in the U14 basis:
    # c_k = (U14.T @ v_gs_400)[k] where v_gs is the 400-dim GS
    # Since U14 is orthonormal (at s=0, S=I), these are just dot products.
    c_gs = v_gs_in_14   # shape (14,)
    # If the GS is fully in the 14-dim space, sum(c_k^2) = 1.0
    print(f'  sum(c_k^2) = {np.sum(c_gs**2):.8f}  (should be 1.0)')

    # -----------------------------------------------------------------------
    # 9. Self-energies: diagonal elements of H_14
    # -----------------------------------------------------------------------
    # E_alpha_k = H_14[k,k]  (not eigenvalue)
    # At h=-1: H_14[k,k] is numerical. At general h:
    #   H_14(h) = h * (U14.T @ A[0] @ U14)  => H_14[k,k](h) = h * (U14[:,k] . A[0] . U14[:,k])
    # Since H0 = -1 * A[0], we have H_14 = (-1) * (U14.T @ A[0] @ U14)
    # So the coefficient of h is: (-1) * H_14[k,k] / h|_{h=-1} = -H_14[k,k] / (-1) = H_14[k,k]
    # Wait: H_14[k,k] at h=-1 gives E_k(h=-1).
    # At general h: E_k(h) = h * (u_k . A[0] . u_k) = h * (-H_14[k,k])
    # because H_14[k,k] = -1 * (u_k . A[0] . u_k)  at h=-1.
    # So the coefficient of h is: (-H_14[k,k])  where H_14[k,k] is at h=-1.
    # Equivalently: E_k(h) = h * (u_k^T A[0] u_k)

    # Compute self-energies as diagonal elements
    E_diag = np.diag(H_14)   # H_14[k,k] at h=-1

    # Coefficient of h for each self-energy: since H_14 = h * K_14 (kinetic matrix)
    # and H_14 is at h=-1, the coefficient is -H_14[k,k]  (flip sign)
    # E_k(h) = h * coeff_k  where coeff_k = -E_diag[k]
    coeff_h = -E_diag   # coefficient of h: E_k(h) = coeff_h[k] * h
    # NOTE: coeff_h values are irrational algebraic numbers (from eta^2 eigenvectors).
    # We store them as floats and display as decimals.

    print('\n14 self-energies E_alpha(h) = coeff_h * h  (diagonal elements of H_14):')
    print(f'  {"k":>2}  {"H_14[k,k] at h=-1":>20}  {"coeff of h":>14}  '
          f'{"E_k(h)":>20}')
    E_sym_diag = []
    for k in range(14):
        E_sym = f'{coeff_h[k]:.6f}*h'
        E_sym_diag.append(E_sym)
        print(f'  {k+1:>2}  {E_diag[k]:>20.10f}  {coeff_h[k]:>14.6f}  '
              f'{E_sym:>20}')

    # -----------------------------------------------------------------------
    # 10. Decompose each 14-dim basis vector into D_6 orbit contributions
    # -----------------------------------------------------------------------
    print('\nAnalysing each of the 14 basis vectors...')

    orbit_sizes = np.array([len(o) for o in orbits])

    # 14-dim coordinates in the 38-orbit basis:
    c38_all = U_a.T @ U14   # shape (38, 14)

    # Class array for ionicity
    nd_arr = np.array([double_occ(ds) for ds in det_strings])

    # Per-state data
    rows = []
    for k in range(14):
        u14 = U14[:, k]   # 400-dim unit vector

        # Identify contributing orbits (threshold)
        c38 = c38_all[:, k]
        tol_c = 1e-8
        contributing = [(i, c38[i]) for i in range(len(orbits))
                        if abs(c38[i]) > tol_c]

        # n_d: weighted average over AO dets
        nd_mean = float((nd_arr * u14**2).sum())

        # dominant nd
        nd_dominant = int(round(nd_mean))

        # representative orbit det: use largest-|coeff| orbit
        rep_orb_idx = max(contributing, key=lambda x: abs(x[1]))[0]
        rep_det = det_strings[orbits[rep_orb_idx][0]]
        nd_rep = double_occ(rep_det)

        alpha_occ = sorted([c for c in rep_det if c.islower()])
        beta_occ  = sorted([c.lower() for c in rep_det if c.isupper()])
        doubly = sorted([o for o in ORBS if o in alpha_occ and o in beta_occ])
        singly_alpha = [o for o in alpha_occ if o not in beta_occ]
        singly_beta  = [o for o in beta_occ  if o not in alpha_occ]

        if nd_rep == 0:
            if set(alpha_occ) == {'a', 'c', 'e'}:
                vb_str = 'Kekule K1 (ace/bdf)'
            elif set(alpha_occ) == {'b', 'd', 'f'}:
                vb_str = 'Kekule K2 (bdf/ace)'
            else:
                vb_str = 'covalent Dewar'
        elif nd_rep == 1:
            vb_str = f'1-ionic d={doubly[0]}'
        elif nd_rep == 2:
            rel = abs(ord(doubly[0]) - ord(doubly[1]))
            if rel == 1 or rel == 5: pos = 'adj'
            elif rel == 2 or rel == 4: pos = 'meta'
            else: pos = 'para'
            vb_str = f'2-ionic {pos} d={doubly[0]}{doubly[1]}'
        elif nd_rep == 3:
            vb_str = f'3-ionic d={doubly[0]}{doubly[1]}{doubly[2]}'
        else:
            vb_str = f'nd={nd_rep}'

        # GS coefficient and weight (from U14 basis projection)
        c_alpha = c_gs[k]
        w_alpha = c_alpha ** 2

        # Self-energy: diagonal element of H_14 (not eigenvalue)
        E_self = E_diag[k]   # at h=-1
        E_sym_k = E_sym_diag[k]   # as "{coeff:.6f}*h" string (irrational)

        # Couplings to other basis vectors: H_14[k,j] for j != k
        couplings = [(j, H_14[k, j]) for j in range(14)
                     if j != k and abs(H_14[k, j]) > 1e-8]
        couplings.sort(key=lambda x: abs(x[1]), reverse=True)

        rows.append({
            'k': k,
            'contributing': contributing,
            'nd_mean': nd_mean,
            'nd_rep': nd_rep,
            'rep_det': rep_det,
            'vb_str': vb_str,
            'c_alpha': c_alpha,
            'w_alpha': w_alpha,
            'E_self': E_self,
            'E_sym': E_sym_k,
            'couplings': couplings,
        })

    # Sort rows by DECREASING GS weight
    rows_sorted = sorted(rows, key=lambda r: r['w_alpha'], reverse=True)

    # -----------------------------------------------------------------------
    # 11. Print the full H_14 matrix first
    # -----------------------------------------------------------------------
    print()
    print('=' * 80)
    print('DIAGNOSTIC: Full 14x14 Hamiltonian H_14 at s=0, h=-1')
    print('(Rows/cols are the U14 basis vectors in their original order)')
    print('=' * 80)
    np.set_printoptions(linewidth=240, precision=5, suppress=True)
    print(H_14)

    print()
    print(f'  Frobenius norm of off-diagonal part: {offdiag_norm:.6e}')
    print(f'  -> H_14 is {"DIAGONAL" if is_diagonal else "NOT DIAGONAL"} '
          f'(U14 columns are {"eigenstates" if is_diagonal else "NOT eigenstates"} of H)')

    # -----------------------------------------------------------------------
    # 12. H_14 upper-triangle (non-zero couplings)
    # -----------------------------------------------------------------------
    print()
    print('H_14 upper-triangle elements (non-zero, |val|>1e-8):')
    print(f'  {"(i,j)":>8}  {"H_ij at h=-1":>14}')
    any_offdiag = False
    for i in range(14):
        for j in range(i + 1, 14):
            v = H_14[i, j]
            if abs(v) > 1e-8:
                any_offdiag = True
                print(f'  ({i+1:2d},{j+1:2d})  {v:>14.8f}')
    if not any_offdiag:
        print('  (none — H_14 is diagonal)')

    # -----------------------------------------------------------------------
    # 13. 14 self-energies E_alpha(h) from diagonal elements
    # -----------------------------------------------------------------------
    print()
    print('=' * 80)
    print('14 SELF-ENERGIES E_alpha(h) at s=0  [= (coeff_h) * h = H_14[k,k]]')
    print('NOTE: these are diagonal elements of H_14, NOT eigenvalues.')
    print('  coeff_h values are IRRATIONAL algebraic numbers (from eta^2 eigenvectors).')
    print('=' * 80)
    print(f'  {"k":>2}  {"H_14[k,k] at h=-1":>20}  {"coeff_h":>14}  '
          f'{"E_k(h)":>20}')
    for k in range(14):
        print(f'  {k+1:>2}  {E_diag[k]:>20.10f}  {coeff_h[k]:>14.6f}  '
              f'{E_sym_diag[k]:>20}')

    # -----------------------------------------------------------------------
    # 14. Eigenvalues for comparison
    # -----------------------------------------------------------------------
    print()
    print('=' * 80)
    print('EIGENVALUES of H_14 at s=0, h=-1  (for comparison with self-energies)')
    print('(These are the physical energy levels; self-energies are H_14[k,k])')
    print('=' * 80)
    print(f'  {"#":>2}  {"eigenvalue":>16}  {"E(h) sym":>20}')
    for k in range(14):
        lam_rat = sp.nsimplify(-eigvals_14[k], rational=True, tolerance=1e-9)
        # E_k(h) = h * lam_rat  (since eigval at h=-1 = -lam_rat => E(h) = lam_rat*h... wait)
        # eigvals_14[k] is E at h=-1. At general h: E_k(h) = eigvals_14[k] * h / (-1)
        # = -eigvals_14[k] * h... no.
        # H_14(h) = h * K where K = U14.T @ A[0] @ U14
        # At h=-1: H_14 = -K, eigvals_14[k] = -k_k  (k_k = eigenval of K)
        # So K eigenval = -eigvals_14[k], and E_k(h) = h * (-eigvals_14[k])
        coeff = -eigvals_14[k]
        coeff_rat = sp.nsimplify(coeff, rational=True, tolerance=1e-9)
        E_sym = coeff_rat * h_sym
        print(f'  {k+1:>2}  {eigvals_14[k]:>16.10f}  {sp.sstr(E_sym):>20}')

    # -----------------------------------------------------------------------
    # 15. Main composition table (rows ordered by decreasing GS weight)
    # -----------------------------------------------------------------------
    print()
    print('=' * 140)
    print('BENZENE 14-ROW VB COMPOSITION TABLE  (s=0, U=0, 1e basis)')
    print('Rows ordered by DECREASING GS weight w_alpha = c_alpha^2')
    print('Self-energy = H_14[k,k], NOT eigenvalue of H_14')
    print('=' * 140)
    print()

    print(f'  {"#":>2}  {"nd":>3}  {"c_alpha":>10}  {"w_alpha":>10}  '
          f'{"E_alpha(h)":>18}  {"VB structure":<28}  {"couplings H_14[k,j]"}')
    print('-' * 150)

    for row_num, r in enumerate(rows_sorted, 1):
        k = r['k']
        # Format couplings as "->j: val" (decimal)
        coup_strs = []
        for j, v in r['couplings'][:5]:
            coup_strs.append(f'→{j+1}:{v:+.4f}')
        coup_str = '  '.join(coup_strs) if coup_strs else '(none)'

        print(f'  {row_num:>2}  {r["nd_rep"]:>3d}  {r["c_alpha"]:>+10.6f}  '
              f'{r["w_alpha"]:>10.6f}  '
              f'{r["E_sym"]:>18}  '
              f'{r["vb_str"]:<28}  '
              f'{coup_str}')

    print()
    total_w = sum(r['w_alpha'] for r in rows)
    print(f'  Sum of weights: {total_w:.8f}  (should be 1.00000000)')
    print(f'  GS energy from diagonalization: {eigvals_14[0]:.10f}')
    print(f'  GS energy from formula 8h:      {8 * (-1.0):.10f}')

    # -----------------------------------------------------------------------
    # 16. Full details including orbit decomposition
    # -----------------------------------------------------------------------
    print()
    print('=' * 100)
    print('DETAILED ORBIT DECOMPOSITION: which D_6 A_1g orbits span each basis vector')
    print('=' * 100)
    print()
    print('D_6 A_1g orbit index -> representative det, orbit size, n_d:')
    print(f'  {"idx":>3}  {"size":>5}  {"n_d":>3}  {"rep det":<12}')
    for i, orb in enumerate(orbits):
        rep = det_strings[orb[0]]
        nd = double_occ(rep)
        print(f'  {i:>3}  {len(orb):>5d}  {nd:>3d}  {rep:<12}')

    print()
    print('Basis vector decomposition (ordered by original k index):')
    print(f'  {"k":>2}  {"E_self(h)":>18}  contributing orbits (index: normalized_coeff)')
    for r in rows:
        k = r['k']
        orb_parts = [(i, v) for i, v in r['contributing'] if abs(v) > 1e-4]
        orb_parts.sort(key=lambda x: abs(x[1]), reverse=True)
        parts_str = ',  '.join(f'{i}: {v:+.6f}' for i, v in orb_parts)
        print(f'  {k+1:>2}  {r["E_sym"]:>18}  {parts_str}')

    # -----------------------------------------------------------------------
    # 17. Markdown table (weight-sorted)
    # -----------------------------------------------------------------------
    print()
    print('=' * 60)
    print('MARKDOWN TABLE (rows ordered by decreasing weight)')
    print('=' * 60)
    print()
    header_cols = ['#', 'n_d', 'VB structure', 'c_alpha', 'w_alpha',
                   'E_alpha(h)', 'H couplings']
    col_widths = [3, 4, 28, 10, 10, 18, 40]

    def md_row(vals, widths):
        return '| ' + ' | '.join(str(v).ljust(w) for v, w in zip(vals, widths)) + ' |'

    def md_sep(widths):
        return '|' + '|'.join('-' * (w + 2) for w in widths) + '|'

    print(md_row(header_cols, col_widths))
    print(md_sep(col_widths))

    for row_num, r in enumerate(rows_sorted, 1):
        k = r['k']
        coup_strs = []
        for j, v in r['couplings'][:4]:
            v_rat = sp.nsimplify(v, rational=True, tolerance=1e-9)
            coup_strs.append(f'→{j+1}:{sp.sstr(v_rat)}')
        coup_str = ', '.join(coup_strs) if coup_strs else '—'
        vals = [
            str(row_num),
            str(r['nd_rep']),
            r['vb_str'],
            f"{r['c_alpha']:+.6f}",
            f"{r['w_alpha']:.6f}",
            sp.sstr(r['E_sym']),
            coup_str,
        ]
        print(md_row(vals, col_widths))

    print()
    print(f'Total GS weight sum = {total_w:.8f}')

    # -----------------------------------------------------------------------
    # 18. Verification: full 400-dim eigenvalue
    # -----------------------------------------------------------------------
    print()
    print('=' * 60)
    print('Verification: full 400-dim lowest eigenvalue at s=0, h=-1:')
    print('=' * 60)
    ev_full = np.linalg.eigvalsh(H0)
    print(f'  Full FCI lowest eigenvalue = {ev_full[0]:.10f}')
    print(f'  14-dim block lowest eigenvalue = {eigvals_14[0]:.10f}')
    print(f'  Formula 8h = {8*(-1):.10f}')
    print(f'  Difference (FCI - 14dim) = {abs(ev_full[0] - eigvals_14[0]):.2e}')


if __name__ == '__main__':
    main()
