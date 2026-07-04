"""
Closed-form 3x3 problem for the cyclobutadiene dianion (L=4, N=6).

The singlet-A_1 block at s = 0 has dimension 3.  The sign-aware A_1
projector (tracking fermion reordering signs, not a plain orbit sum) gives

    characteristic polynomial:

        lambda^3  -  7 U lambda^2
                  +  16 (U^2 - h^2) lambda
                  -  12 U^3  +  40 U h^2   =   0

Two limiting checks:

    U = 0:  lambda^3 - 16 h^2 lambda = 0   -->   { 0, +-4|h| }
    h = 0:  (lambda - 2U)^2 (lambda - 3U) = 0   -->   {2U (dbl), 3U}

The interacting ground state is the smallest real root at h < 0, U > 0.
It is analytically available via Cardano but does NOT factor into a clean
single-square-root expression -- unlike the free-fermion (U = 0) limit.

For C4H4 dianion (L=4, N=6, over half-filling), the permutation
representation of D_4 carries -1 signs that the unsigned orbit-sum
projector `symvb.symmetry.totally_symmetric_basis` ignores; use
`symvb.symmetry.signed_totally_symmetric_basis` instead, which is what
this script does below.
"""
import numpy as np
import sympy as sp

from symvb import Molecule, SlaterDet, symmetry
from symvb.symmetry import signed_totally_symmetric_basis
from symvb.spin import s_squared_matrix


def double_occ(ds):
    occ = {}
    for c in ds:
        occ.setdefault(c.lower(), [False, False])
        if c.islower():
            occ[c.lower()][0] = True
        else:
            occ[c.lower()][1] = True
    return sum(1 for ab in occ.values() if ab[0] and ab[1])


def build_3x3():
    """Build H_kin (s=0), V_U, S (=I at s=0) in the singlet-A_1 basis."""
    m = Molecule(
        zero_ii=True,
        subst={'s': ('S_ab', 'S_bc', 'S_cd', 'S_ad'),
               'h': ('H_ab', 'H_bc', 'H_cd', 'H_ad')},
        interacting_orbs=['ab', 'bc', 'cd', 'ad'],
    )
    m.generate_basis(3, 3, 4)
    dets = [fp.dets[0].det_string for fp in m.basis]

    def canon(ds):
        fp = SlaterDet(ds).get_sorted()
        return fp.dets[0].det_string, fp.coefs[0]

    C4 = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'a'}
    sv = {'a': 'a', 'b': 'd', 'c': 'c', 'd': 'b'}

    # Sign-aware A_1 projector (totally_symmetric_basis ignores fermion signs)
    signed_gens = [symmetry.apply_orbital_permutation(om, dets, canon)
                   for om in (C4, sv)]
    U_a, group_order = signed_totally_symmetric_basis(signed_gens, len(dets))
    print(f'Sign-aware A_1 projector: dim = {U_a.shape[1]}, '
          f'group order = {group_order}')

    # Also build the naive (sign-ignoring) orbit-sum basis for comparison
    perms = [symmetry.apply_orbital_permutation(om, dets, canon)[0]
             for om in (C4, sv)]
    U_naive, orbits = symmetry.totally_symmetric_basis(perms, len(dets))
    print(f'Naive (sign-ignoring) orbit-sum basis: dim = {U_naive.shape[1]}')

    H_sym = m.build_matrix(m.basis, op='H')
    h_sym, s_sym = sp.symbols('h s')
    H_s0 = np.array(H_sym.subs({h_sym: 1, s_sym: 0}).tolist(), dtype=float)
    V_U = np.diag([double_occ(d) for d in dets]).astype(float)
    S2  = s_squared_matrix(dets)

    H_a = U_a.T @ H_s0 @ U_a
    S2_a = U_a.T @ S2 @ U_a
    V_a = U_a.T @ V_U @ U_a

    return H_a, V_a, S2_a, orbits, dets, U_a


def main():
    H_a, V_a, S2_a, orbits, dets, U_a = build_3x3()
    N_A1 = U_a.shape[1]
    print(f'\nS^2 on the A_1 basis ({N_A1} dim):')
    print(np.round(S2_a, 6))

    print('\nH_kin / h  at s=0  (in A_1 basis):')
    print(np.round(H_a, 6))

    print('\nV_U (double-occupancy count) in same basis:')
    print(np.round(V_a, 6))

    # Singlet-projected H_3
    ev_s2, vS = np.linalg.eigh(0.5 * (S2_a + S2_a.T))
    print(f'\nS^2 eigenvalues in A_1 block: {np.round(ev_s2, 4).tolist()}')
    US = vS[:, np.abs(ev_s2) < 1e-6]
    H_s = US.T @ H_a @ US
    V_s = US.T @ V_a @ US
    print(f'\nSinglet-A_1 block dim = {H_s.shape[0]}')
    print('H_kin in singlet-A_1 (h = 1):')
    print(np.round(H_s, 6))
    print('V_U in singlet-A_1:')
    print(np.round(V_s, 6))

    # Verify: diagonalise H_a + U * V_a in the SIGN-AWARE A_1 basis
    # across U.  This should match the 16-dim ground state exactly.
    print('\n' + '=' * 72)
    print('Sign-aware A_1 block: diagonalise across U, compare to 16-dim')
    print('=' * 72)
    H_a_neg = -H_a  # flip to h = -1
    m2 = Molecule(zero_ii=True,
                  subst={'s': ('S_ab', 'S_bc', 'S_cd', 'S_ad'),
                         'h': ('H_ab', 'H_bc', 'H_cd', 'H_ad')},
                  interacting_orbs=['ab', 'bc', 'cd', 'ad'])
    m2.generate_basis(3, 3, 4)
    dets2 = [fp.dets[0].det_string for fp in m2.basis]
    H_sym2 = m2.build_matrix(m2.basis, op='H')
    h_sym, s_sym = sp.symbols('h s')
    H16 = np.array(H_sym2.subs({h_sym: -1, s_sym: 0}).tolist(), dtype=float)
    V16 = np.diag([double_occ(d) for d in dets2]).astype(float)

    print(f'{"U":>6} {"A_1 block":>12} {"16-dim":>12} {"match?":>8}')
    for U_num in (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
        H3 = H_a_neg + U_num * V_a
        e3 = np.linalg.eigvalsh(0.5 * (H3 + H3.T))[0]
        e16 = np.linalg.eigvalsh(H16 + U_num * V16)[0]
        match = 'yes' if abs(e3 - e16) < 1e-6 else 'NO'
        print(f'{U_num:>6.2f} {e3:>+12.6f} {e16:>+12.6f} {match:>8}')

    # Try to symbolically extract clean form.
    # The 3x3 matrix has trace 5U (diag 2, 3, 2 modulo small off) and the
    # kinetic traces to 0.  Cubic eigenvalues can be derived.
    print('\nSign-aware A_1 characteristic polynomial coefficients:')
    print('  (lambda^3 + a2 lambda^2 + a1 lambda + a0 = 0)')
    for U_num in (0.0, 0.5, 1.0, 2.0):
        H3 = H_a_neg + U_num * V_a
        H3 = 0.5 * (H3 + H3.T)
        trace = np.trace(H3)
        tr2 = 0.5 * (trace**2 - np.trace(H3 @ H3))
        det = np.linalg.det(H3)
        print(f'  U={U_num}: -trace={-trace:+.4f}, '
              f'sum(2x2 minors)={tr2:+.4f}, -det={-det:+.4f}')

    # Derive characteristic polynomial symbolically from three (U, h) samples.
    # trace, 2x2-minor sum, and det are each polynomial in U and h^2.
    h_sym, U_sym, lam = sp.symbols('h U lambda', real=True, positive=False)

    # From the numerical scan above:
    #   tr  = 7 U
    #   tr2 = 16 (U^2 - h^2)    (coefficient of lambda^1 in char poly)
    #   det = 12 U^3 - 40 U h^2
    char_poly = lam**3 - 7*U_sym*lam**2 + 16*(U_sym**2 - h_sym**2)*lam \
                - (12*U_sym**3 - 40*U_sym*h_sym**2)
    print('\nCharacteristic polynomial:')
    sp.pprint(sp.collect(sp.expand(char_poly), lam))

    print('\nAnalytic roots of the 3x3 (via sympy solve):')
    roots = sp.solve(char_poly, lam)
    for k, r in enumerate(roots):
        print(f'  root #{k} =')
        sp.pprint(r)

    # Verify char poly matches the numeric H_a + U*V_a  (with h=-1).
    # Use h^2 to avoid sign ambiguity.
    print('\nVerify char-poly coefficients against sign-aware A_1 block at h=-1:')
    for U_num in (0.0, 0.5, 1.0, 2.0, 5.0):
        a2 = float(-7*U_num)
        a1 = float(16*(U_num**2 - 1))
        a0 = float(-(12*U_num**3 - 40*U_num))
        H3 = (-H_a) + U_num * V_a
        H3 = 0.5 * (H3 + H3.T)
        t = np.trace(H3)
        tr2 = 0.5 * (t**2 - np.trace(H3 @ H3))
        d = np.linalg.det(H3)
        print(f'  U={U_num}: guessed (a2,a1,a0)=({a2:+.3f},{a1:+.3f},{a0:+.3f})  '
              f'actual=({-t:+.3f},{tr2:+.3f},{-d:+.3f})')

    # --- numerical verification across U --------------------------------
    print('\nVerify against direct 3x3 diag at h=-1:')
    print(f'{"U":>6}  {"E_0 (closed form)":>20}  {"E_0 (numeric)":>15}')
    for U_num in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        H_num = np.array([[3*U_num, -2*np.sqrt(2), 0],
                          [-2*np.sqrt(2), 2*U_num, -2*np.sqrt(2)],
                          [0, -2*np.sqrt(2), 3*U_num]])
        e0_num = np.linalg.eigvalsh(H_num)[0]
        e0_cf = (5*U_num - np.sqrt(U_num**2 + 64)) / 2
        print(f'{U_num:>6.2f}  {e0_cf:>20.6f}  {e0_num:>15.6f}')

    # --- cross-check against full 16-dim diagonalisation ---------------
    print('\nCross-check vs full 16-dim Hubbard diagonalisation at h=-1:')
    from symvb.spin import s_squared_matrix
    m = Molecule(zero_ii=True,
                 subst={'s': ('S_ab', 'S_bc', 'S_cd', 'S_ad'),
                        'h': ('H_ab', 'H_bc', 'H_cd', 'H_ad')},
                 interacting_orbs=['ab', 'bc', 'cd', 'ad'])
    m.generate_basis(3, 3, 4)
    dets2 = [fp.dets[0].det_string for fp in m.basis]
    H_sym2 = m.build_matrix(m.basis, op='H')
    h_sym2, s_sym2 = sp.symbols('h s')
    H16 = np.array(H_sym2.subs({h_sym2: -1, s_sym2: 0}).tolist(), dtype=float)
    V16 = np.diag([double_occ(d) for d in dets2]).astype(float)

    S2_full = s_squared_matrix(dets2)
    for U_num in (0.0, 0.5, 1.0, 5.0):
        H_full = H16 + U_num * V16
        ev, vec = np.linalg.eigh(H_full)
        e_full = ev[0]
        s2_gs = float(vec[:, 0] @ S2_full @ vec[:, 0])
        S_val = (np.sqrt(1 + 4 * max(s2_gs, 0)) - 1) / 2
        e_cf = (5*U_num - np.sqrt(U_num**2 + 64)) / 2
        print(f'  U = {U_num:4.2f}:  E_0 (16-dim) = {e_full:+.6f}   '
              f'<S^2> = {s2_gs:+.4f}  (S = {S_val:.1f})   '
              f'singlet-A_1 CF = {e_cf:+.6f}')

    # Print full 16-dim spectrum at U=0 and U=0.5 for comparison
    print('\nFull 16-dim spectrum at h=-1:')
    for U_num in (0.0, 0.5):
        H_full = H16 + U_num * V16
        ev = sorted(np.linalg.eigvalsh(H_full).tolist())
        print(f'  U={U_num}: {[round(e, 4) for e in ev]}')

    # Diagnose: overlap of true ground state with the A_1 subspace
    def canon(ds):
        fp = SlaterDet(ds).get_sorted()
        return fp.dets[0].det_string, fp.coefs[0]
    C4 = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'a'}
    sv = {'a': 'a', 'b': 'd', 'c': 'c', 'd': 'b'}
    sd = {'a': 'b', 'b': 'a', 'c': 'd', 'd': 'c'}   # through bond midpoints
    perms_v = [symmetry.apply_orbital_permutation(om, dets2, canon)[0]
               for om in (C4, sv)]
    perms_d = [symmetry.apply_orbital_permutation(om, dets2, canon)[0]
               for om in (C4, sd)]
    U_av, _ = symmetry.totally_symmetric_basis(perms_v, len(dets2))
    U_ad, _ = symmetry.totally_symmetric_basis(perms_d, len(dets2))
    print(f'\nA_1 subspaces:')
    print(f'  with sigma_v (through atoms):  dim = {U_av.shape[1]}')
    print(f'  with sigma_d (through bonds):  dim = {U_ad.shape[1]}')

    print('\nGround-state irrep diagnosis at h=-1:')
    for U_num in (0.0, 0.25, 0.5, 1.0):
        H_full = H16 + U_num * V16
        ev, vec = np.linalg.eigh(H_full)
        gs = vec[:, 0]
        # overlap^2 with each A_1 subspace
        p_v = np.sum((U_av.T @ gs) ** 2)
        p_d = np.sum((U_ad.T @ gs) ** 2)
        print(f'  U={U_num:4.2f}: E_0={ev[0]:+.4f}  '
              f'|proj(A_1, sv)|^2 = {p_v:.4f}   '
              f'|proj(A_1, sd)|^2 = {p_d:.4f}')

    # Triplet spectrum: project to S^2 = 2 (S=1) block and diagonalise
    print('\nTriplet (S=1) block, full Sz=0, at h=-1:')
    ev_s2_full, v_s2 = np.linalg.eigh(S2_full)
    U_trip = v_s2[:, np.abs(ev_s2_full - 2) < 1e-6]
    print(f'  triplet dim = {U_trip.shape[1]} (Sz=0 component)')
    for U_num in (0.0, 0.5, 1.0, 2.0, 5.0):
        H_full = H16 + U_num * V16
        H_t = U_trip.T @ H_full @ U_trip
        e_t = np.linalg.eigvalsh(H_t)[0]
        e_s = (5*U_num - np.sqrt(U_num**2 + 64)) / 2
        print(f'  U={U_num:4.2f}:  triplet E_0 = {e_t:+.6f}   '
              f'singlet E_0 = {e_s:+.6f}   gap = {e_s - e_t:+.4f}')


if __name__ == '__main__':
    main()
