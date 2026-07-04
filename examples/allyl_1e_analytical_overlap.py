"""
Closed-form 1e ground state of allyl (4 electrons in 3 orbitals) at
non-zero AO overlap, via the 4-dim singlet-A_1 VB block.

Setup: 3-AO chain a-b-c, h_{ij}=h on adjacent bonds (a-b, b-c only;
no a-c through-space), S_{ij}=s on adjacent bonds, U=0.

Hückel MOs are S(s)-eigenvectors with chain-adjacency eigenvalues
lambda = +sqrt(2), 0, -sqrt(2):

    psi_1 = (a + sqrt(2) b + c) / 2,   eps_1(s) =  sqrt(2) h / (1 + sqrt(2) s)
    psi_2 = (a - c) / sqrt(2),         eps_2(s) =  0
    psi_3 = (a - sqrt(2) b + c) / 2,   eps_3(s) = -sqrt(2) h / (1 - sqrt(2) s)

The closed-shell GS (4 electrons doubly fill psi_1 and psi_2):
    E_GS(h, s) = 2 eps_1 + 2 eps_2 = 2 sqrt(2) h / (1 + sqrt(2) s).

Reduction chain at s != 0:

    9   Sz = 0 AO-det FCI (C(3,2)^2 = 9)
     |  sigma_v (a <-> c)
    5   A_1 (sigma = +1)
     |  S^2 = 0
    4   singlet-A_1   <- closed-shell GS lives here, keeping s symbolic

(E, s) -> (-E, -s) bipartite-style symmetry pairs MO occupation
(n_1, n_2, n_3) with (n_3, n_2, n_1):

  - closed-shell (2, 2, 0) <-> doubly-excited (0, 2, 2)
  - "open-shell" (2, 0, 2) self-paired (E odd in s; vanishes at s = 0)
  - "open-shell" (1, 2, 1) self-paired (E odd in s; vanishes at s = 0)

Note that allyl is *above* half-filling (4 electrons, 3 sites): every
AO determinant has at least one doubly-occupied site, so the AO-VB
ionicity decomposition is over the 1-ionic (6 dets) and 2-ionic (3
dets) classes only -- no purely-covalent structures exist in this
filling.

Demonstrates the same machinery as benzene (§4.3.1), now in Q[sqrt(2)](h, s).
"""
import os
import sys
import time

import numpy as np
import sympy as sp
import scipy.linalg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from symvb import Molecule, SlaterDet, symmetry, verify_eigenpair
from symvb.huckel import solve
from symvb.mo_projection import mo_determinant_in_ao
from symvb.numerical import decompose_polynomial_matrix
from symvb.spin import s_squared_matrix


def build_basis_and_matrices():
    m = Molecule(
        zero_ii=True,
        interacting_orbs=['ab', 'bc'],
        subst={'h': ('H_ab', 'H_bc'),
               's': ('S_ab', 'S_bc')},
    )
    m.generate_basis(2, 2, 3)         # 2 alpha + 2 beta in 3 orbitals -> 9 dets
    H1 = m.build_matrix(m.basis, op='H')
    S  = m.build_matrix(m.basis, op='S')
    return m, H1, S


def a1_orbit_projector(det_strings):
    """sigma_v (a<->c) A_1 orbit-sum projector. Returns (U_a_norm, U_a_int, orbits)."""
    def canon(ds):
        fp = SlaterDet(ds).get_sorted()
        return fp.dets[0].det_string, fp.coefs[0]
    sigma_v = {'a': 'c', 'b': 'b', 'c': 'a'}
    perms = [symmetry.apply_orbital_permutation(sigma_v, det_strings, canon)[0]]
    U_a_norm, orbits = symmetry.totally_symmetric_basis(perms, len(det_strings))
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

    print('Building allyl basis and symbolic (H1, S)...')
    t0 = time.time()
    m, H1, S = build_basis_and_matrices()
    det_strings = [fp.dets[0].det_string for fp in m.basis]
    N = len(det_strings)
    print(f'  {N}-dim AO-det basis (2 alpha + 2 beta in 3 orbitals)  '
          f'{time.time() - t0:.2f}s')

    print('\nBuilding sigma_v A_1 orbit-sum projector (s-independent)...')
    U_a, U_a_int, orbits = a1_orbit_projector(det_strings)
    print(f'  A_1 block dim = {U_a.shape[1]}')

    print('\nProjecting to singlet (S^2 = 0) within A_1...')
    S2 = s_squared_matrix(det_strings)
    S2_a = U_a.T @ S2 @ U_a
    S2_a = 0.5 * (S2_a + S2_a.T)
    ev, V = np.linalg.eigh(S2_a)
    US = V[:, np.abs(ev) < 1e-6]
    print(f'  singlet-A_1 block dim = {US.shape[1]}')
    U4 = U_a @ US

    # Symbolic Huckel solve of the 3-AO open chain. Real MOs in Q[sqrt(2)]:
    # psi_1 (lambda=+sqrt(2)), psi_2 (lambda=0), psi_3 (lambda=-sqrt(2)).
    # Closed-shell GS doubly occupies psi_1 and psi_2.
    huckel_result = solve(sp.Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
                          site_labels='abc')
    h_sym, s_sym = huckel_result.h_symbol, huckel_result.s_symbol
    E_GS_sym = sp.simplify(huckel_result.energy_of_occupation([2, 2, 0]))

    print('\nDecomposing H1, S into polynomial-in-s coefficient matrices...')
    t0 = time.time()
    A = decompose_polynomial_matrix(H1, s_sym, factor=h_sym)
    B = decompose_polynomial_matrix(S, s_sym)
    print(f'  H1: {len(A)} matrices, S: {len(B)} matrices  '
          f'{time.time() - t0:.2f}s')

    def H1_eval(h_val, s_val):
        return h_val * sum(s_val ** q * A[q] for q in range(len(A)))
    def S_eval(s_val):
        return sum(s_val ** q * B[q] for q in range(len(B)))

    # --- Verification 1: lowest eigenvalue at several s ----------------
    print('\nLowest eigenvalue (h = -1):')
    print(f'  Predicted: E_GS(h, s) = 2*sqrt(2)*h / (1 + sqrt(2)*s)\n')
    print(f'  {"s":>5}  {"eig_lowest":>16}  {"E_GS formula":>16}  {"diff":>10}')
    for s_val_f in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
        # singularity of S_AO at s = 1/sqrt(2) ~ 0.707, so stop short
        H_num = H1_eval(-1.0, s_val_f)
        S_num = S_eval(s_val_f)
        H4n = U4.T @ H_num @ U4
        S4n = U4.T @ S_num @ U4
        H4n = 0.5 * (H4n + H4n.T)
        S4n = 0.5 * (S4n + S4n.T)
        try:
            eigs = scipy.linalg.eigh(H4n, S4n, eigvals_only=True)
            E_pred = float(E_GS_sym.subs({h_sym: -1, s_sym: s_val_f}))
            print(f'  {s_val_f:>5.2f}  {eigs[0]:>16.10f}  '
                  f'{E_pred:>16.10f}  {eigs[0] - E_pred:>10.2e}')
        except scipy.linalg.LinAlgError:
            print(f'  {s_val_f:>5.2f}  S(s) singular (1 + sqrt(2)*s passes 0)')

    # --- Closed-form 4-state spectrum ----------------------------------
    print('\nClosed-form 4-state spectrum:')
    eps_sym = lambda label: huckel_result.energies[int(label) - 1]

    h_t, s_t = -1.0, 0.137
    H_n = H1_eval(h_t, s_t); S_n = S_eval(s_t)
    H4n = U4.T @ H_n @ U4;   S4n = U4.T @ S_n @ U4
    H4n = 0.5*(H4n+H4n.T);   S4n = 0.5*(S4n+S4n.T)
    eigvals_full, eigvecs_full = scipy.linalg.eigh(H4n, S4n)

    patterns = [(n1, n2, n3)
                for n1 in range(3) for n2 in range(3) for n3 in range(3)
                if n1 + n2 + n3 == 4]
    eps_t = {str(i + 1): float(huckel_result.energies[i].subs(
                                    {h_sym: h_t, s_sym: s_t}))
             for i in range(3)}

    print(f'  evaluation point: h = {h_t}, s = {s_t}')
    print(f'  {"#":>2}  {"eigenvalue":>16}  {"pattern":<10}  {"E(h, s) symbolic":<55}  PH partner')
    for k in range(4):
        E_num = eigvals_full[k]
        best, bd = None, np.inf
        for pat in patterns:
            E_pat = sum(pat[i] * eps_t[l] for i, l in enumerate('123'))
            if abs(E_num - E_pat) < bd:
                bd, best = abs(E_num - E_pat), pat
        E_sym = sum(best[i] * eps_sym(l) for i, l in enumerate('123'))
        E_sym = sp.together(sp.simplify(E_sym))
        partner = best[::-1]
        partner_label = (f'self-paired (E odd in s)' if best == partner
                         else f'-> {partner} under (E,s)->(-E,-s)')
        print(f'  {k+1:>2}  {E_num:>16.10f}  {str(best):<10}  '
              f'{sp.sstr(E_sym):<55}  {partner_label}')

    # --- Symbolic identity check in Q[sqrt(2)](h, s) -------------------
    print('\nSymbolic identity (H_4 - E_GS S_4) v_4 == 0 in Q[sqrt(2)](h, s):')
    t0 = time.time()
    S2_a_int = U_a_int.T @ S2.astype(int) @ U_a_int
    S2_a_sp = sp.Matrix(S2_a_int.tolist())
    US_basis = S2_a_sp.nullspace()
    US_sp = sp.Matrix.hstack(*US_basis)
    print(f'  US shape {US_sp.shape},  {time.time() - t0:.2f}s')
    if US_sp.shape[1] != 4:
        raise RuntimeError(f'singlet kernel dim = {US_sp.shape[1]}, expected 4')

    M_q_5 = [(U_a_int.T @ A[q].astype(int) @ U_a_int) for q in range(len(A))]
    N_q_5 = [(U_a_int.T @ B[q].astype(int) @ U_a_int) for q in range(len(B))]
    M_q_sp = [US_sp.T * sp.Matrix(m.tolist()) * US_sp for m in M_q_5]
    N_q_sp = [US_sp.T * sp.Matrix(m.tolist()) * US_sp for m in N_q_5]

    print('  building v_0 exactly (real Huckel cofactor in Q[sqrt(2)])...')
    t0 = time.time()
    v0_sp = mo_determinant_in_ao(
        huckel_result.coefficients, ([0, 1], [0, 1]),
        det_strings, site_labels='abc')
    print(f'  done in {time.time() - t0:.2f}s')

    U_a_sp = sp.Matrix(U_a_int.tolist())
    orbit_sizes = (U_a_int.T @ U_a_int).diagonal()
    D_sp = sp.diag(*[sp.Integer(int(d)) for d in orbit_sizes])
    G = US_sp.T * D_sp * US_sp
    v_naive = US_sp.T * (U_a_sp.T * v0_sp)
    v_sp = G.solve(v_naive)
    if all(e == 0 for e in v_sp):
        raise RuntimeError("v4 is zero: cofactor or projector mismatch")

    H4_sp = sp.zeros(*M_q_sp[0].shape)
    for q in range(len(M_q_sp)):
        H4_sp = H4_sp + h_sym * (s_sym ** q) * M_q_sp[q]
    S4_sp = sp.zeros(*N_q_sp[0].shape)
    for q in range(len(N_q_sp)):
        S4_sp = S4_sp + (s_sym ** q) * N_q_sp[q]

    print('  forming residual and simplifying via verify_eigenpair...')
    t0 = time.time()
    try:
        # surd case (Q[sqrt(2)]); sp.cancel won't reduce the radicals,
        # use sp.simplify
        verify_eigenpair(H4_sp, S4_sp, v_sp, E_GS_sym, simplify=sp.simplify)
        print(f'  done in {time.time() - t0:.2f}s')
        print(f'  ALL 4 residual entries identically zero in '
              f'Q[sqrt(2)](h, s).  ✓')
        print(f'  E_GS(h, s) = 2 sqrt(2) h / (1 + sqrt(2) s) is exact for '
              f'the closed-shell allyl GS at every s.')
    except Exception as exc:
        print(f'  done in {time.time() - t0:.2f}s')
        print(f'  FAILED: {exc}')

    # --- Per-state AO-VB ionicity (s-independent) ----------------------
    print('\nPer-state AO-VB ionicity weights (1-ionic : 2-ionic, '
          's-INDEPENDENT):')
    classes_arr = np.array([double_occ(ds) for ds in det_strings])
    print(f'  {"#":>2}  {"E (h=-1, s=0.137)":>16}  {"pattern":<10}  '
          f'{"1i : 2i":<14}  {"as fraction":<12}')
    for k in range(4):
        v_400 = U4 @ eigvecs_full[:, k]
        v_400 = v_400 / np.linalg.norm(v_400)
        E_num = eigvals_full[k]
        best, bd = None, np.inf
        for pat in patterns:
            E_pat = sum(pat[i] * eps_t[l] for i, l in enumerate('123'))
            if abs(E_num - E_pat) < bd:
                bd, best = abs(E_num - E_pat), pat
        w1 = (v_400[classes_arr == 1] ** 2).sum()
        w2 = (v_400[classes_arr == 2] ** 2).sum()
        # Try to recognize as a/b for small denominators
        frac = sp.nsimplify(sp.Rational(int(round(w1*10000)), 10000),
                            rational=True).limit_denominator(20)
        print(f'  {k+1:>2}  {E_num:>16.10f}  {str(best):<10}  '
              f'{w1:.4f}:{w2:.4f}      {frac} : {1 - frac}')

    print('\nVerifying s-independence of the GS via Chirgwin-Coulson:')
    v0_num = np.array([float(v0_sp[i, 0])
                       for i in range(len(det_strings))])
    v0_n = v0_num / np.linalg.norm(v0_num)
    print(f'  Euclidean weights (s = 0):   '
          + ', '.join(f'{c}:{(v0_n[classes_arr==c]**2).sum():.6f}'
                      for c in range(4) if (classes_arr==c).sum() > 0))
    for s_val in [0.1, 0.2, 0.3, 0.4]:
        Sn = S_eval(s_val)
        Sv = Sn @ v0_num
        denom = v0_num @ Sv
        cc = []
        for c in range(4):
            mask = (classes_arr == c)
            if mask.sum() == 0: continue
            cc.append(f'{c}:{(v0_num[mask]@Sv[mask])/denom:.6f}')
        print(f'  Chirgwin-Coulson (s = {s_val}):  ' + ', '.join(cc))

    # --- GS breakdown by sigma_v orbit (VB structure types) ------------
    # Each orbit alpha is a class of sigma_v-equivalent AO Slater dets;
    # the GS amplitude is constant within an orbit (since GS is A_1).
    # Orbit weights w_alpha = n_alpha c_alpha^2 / <v|v> sum to 1 and are
    # s-independent (simultaneous-S-eigenvector argument).  Self-energy
    # E_alpha(h, s) = <orb_alpha|H_1e|orb_alpha> / <orb_alpha|orb_alpha>
    # is a rational function in (h, s), independent of sqrt(2) at the
    # orbit-sum level.
    # --- GS breakdown over the 4 singlet-A_1 basis vectors ------------
    # The 5 -> 4 projection is the existing US_sp (columns = singlet
    # basis vectors in sigma_v-orbit-sum coordinates), v_sp gives the GS
    # amplitudes in that basis, and M_q_sp / N_q_sp are the 4x4 symbolic
    # polynomial-coefficient matrices already built. Per-row data is a
    # direct entry lookup; no merging or grouping.
    print('\n--- GS breakdown over the 4 singlet-A_1 basis vectors ---')
    n_singlet = US_sp.shape[1]

    def H4_entry(k, j):
        return h_sym * sum(M_q_sp[q][k, j] * (s_sym ** q)
                           for q in range(len(M_q_sp)))

    def S4_entry(k, j):
        return sum(N_q_sp[q][k, j] * (s_sym ** q)
                   for q in range(len(N_q_sp)))

    Gv = G * v_sp
    norm_sq = (v_sp.T * Gv)[0, 0]

    print(f'  {"#":>2}  {"sigma_v orbit contribs":<24}  {"c_k":>10}  '
          f'{"w_k":>10}  {"E_k(h, s)":<40}  H_kj (->j)')
    for k in range(n_singlet):
        bk = US_sp[:, k]
        contribs = [(i, bk[i, 0]) for i in range(bk.rows) if bk[i, 0] != 0]
        contrib_str = ", ".join(f"{i}({sp.sstr(c)})" for i, c in contribs)

        c_k = v_sp[k, 0]
        w_k = sp.simplify(c_k * Gv[k, 0] / norm_sq)
        E_k = sp.together(sp.simplify(H4_entry(k, k) / S4_entry(k, k)))

        coup_parts = []
        for j in range(k + 1, n_singlet):
            H_kj = sp.expand(H4_entry(k, j))
            if H_kj != 0:
                coup_parts.append(f"->{j}: {sp.sstr(H_kj)}")
        coup_str = "; ".join(coup_parts) if coup_parts else "--"

        print(f'  {k:>2}  {contrib_str:<24}  {sp.sstr(c_k):>10}  '
              f'{sp.sstr(w_k):>10}  {sp.sstr(E_k):<40}  {coup_str}')
    print(f'  Sum of weights: {sp.sstr(sp.simplify(sum(v_sp[k, 0] * Gv[k, 0] / norm_sq for k in range(n_singlet))))}')


if __name__ == '__main__':
    main()
