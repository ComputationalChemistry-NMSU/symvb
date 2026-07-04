"""
Refining the benzene A1g block with total-spin (S^2) and eta-pairing pseudospin.

Hierarchy demonstrated here (orthogonal-AO Huckel limit, s=0):

    400  (full Sz=0 FCI)
     |  D_6 (C_6 + sigma_v)
    38   A_1g
     |  S^2 = 0  (singlet)
    22   singlet-A_1g
     |  eta^2 = 0
    14   singlet-A_1g, eta=0    <-- contains the ground state

Also reports:
  - the full eta(eta+1) multiplet decomposition of the 400-dim basis
    (175 / 189 / 35 / 1 for eta = 0,1,2,3),
  - [H_kinetic, eta^2] = 0 at s=0 vs nonzero at s>0,
  - [V_U, eta^2] = 0 (eta is a good quantum number through the Hubbard PT)
    while [V_U, eta_+] = +U eta_+ (the SU(2) action is broken by U).

Reproduces the numerics quoted in manuscript section 4.2.1.
"""
import time
import numpy as np
import sympy as sp
from collections import Counter
from scipy.linalg import eigh

from symvb import Molecule, SlaterDet, symmetry
from symvb.spin import (
    s_squared_matrix,
    eta_squared_matrix,
    _apply_c, _canon_det, _symvb_to_canonical_sign,
)


ORBS = list('abcdef')
SITE_SIGNS = {'a': +1, 'b': -1, 'c': +1, 'd': -1, 'e': +1, 'f': -1}


def deg_hist(ev, tol=1e-6):
    s = sorted(ev.tolist())
    out, i = [], 0
    while i < len(s):
        j = i + 1
        while j < len(s) and abs(s[j] - s[i]) < tol:
            j += 1
        out.append(j - i)
        i = j
    return Counter(out)


def build_basis():
    m = Molecule(zero_ii=True,
                 subst={'s': ('S_ab','S_bc','S_cd','S_de','S_ef','S_af'),
                        'h': ('H_ab','H_bc','H_cd','H_de','H_ef','H_af')},
                 interacting_orbs=['ab','bc','cd','de','ef','af'])
    m.generate_basis(3, 3, 6)
    det_strings = [fp.dets[0].det_string for fp in m.basis]
    return m, det_strings


def a1g_projector(det_strings):
    def canon(ds):
        fp = SlaterDet(ds).get_sorted()
        return fp.dets[0].det_string, fp.coefs[0]
    C6    = {'a':'b','b':'c','c':'d','d':'e','e':'f','f':'a'}
    sigma = {'a':'a','b':'f','c':'e','d':'d','e':'c','f':'b'}
    perms = [symmetry.apply_orbital_permutation(om, det_strings, canon)[0]
             for om in (C6, sigma)]
    U_a, _ = symmetry.totally_symmetric_basis(perms, len(det_strings))
    return U_a


def double_occ(ds):
    """Number of doubly-occupied sites in a symvb det string."""
    occ = {}
    for c in ds:
        occ.setdefault(c.lower(), [False, False])
        if c.islower():
            occ[c.lower()][0] = True
        else:
            occ[c.lower()][1] = True
    return sum(1 for ab in occ.values() if ab[0] and ab[1])


def build_eta_plus(det_strings_with_doping, orbital_index):
    """
    Build  eta_+ = sum_i s_i c+_{i,alpha} c+_{i,beta}  on a basis that spans
    multiple particle-number sectors (otherwise eta_+ has no codomain).
    """
    N = len(det_strings_with_doping)
    index = {d: i for i, d in enumerate(det_strings_with_doping)}
    vsign = np.array([_symvb_to_canonical_sign(d, orbital_index)
                      for d in det_strings_with_doping])
    EP = np.zeros((N, N))
    for col, ds in enumerate(det_strings_with_doping):
        alpha = {c for c in ds if c.islower()}
        beta  = {c.lower() for c in ds if c.isupper()}
        for i in ORBS:
            a1, b1, s1 = _apply_c(alpha, beta, i, 1, create=True)
            if a1 is None: continue
            a2, b2, s2 = _apply_c(a1, b1, i, 0, create=True)
            if a2 is None: continue
            key = _canon_det(a2, b2)
            row = index.get(key)
            if row is None: continue
            EP[row, col] += SITE_SIGNS[i] * s1 * s2
    return vsign[:, None] * EP * vsign[None, :]


def main():
    print('Building benzene FCI basis (3 alpha + 3 beta in 6 orbitals)...')
    m, det_strings = build_basis()
    N = len(det_strings)
    print(f'  N = {N} dets at half-filling (eta_z = 0)')

    print('\nBuilding S^2, eta^2 matrices...')
    t0 = time.time()
    S2 = s_squared_matrix(det_strings)
    E2 = eta_squared_matrix(det_strings, SITE_SIGNS, ORBS)
    print(f'  done in {time.time()-t0:.2f}s')

    # --- eta(eta+1) decomposition of the 400-dim basis -------------------
    ev_eta = np.linalg.eigvalsh((E2 + E2.T) / 2)
    eta_mult = Counter(np.round(ev_eta, 6).tolist())
    print(f'\neta^2 spectrum on full 400-dim basis (eigenvalue: multiplicity):')
    for ev, mult in sorted(eta_mult.items()):
        eta = (np.sqrt(1 + 4 * max(ev, 0)) - 1) / 2
        print(f'  {ev:6.2f}  (eta = {eta:.0f}):  {mult}')
    print(f'  total: {sum(eta_mult.values())}')

    # --- A1g symmetry projection -----------------------------------------
    print('\nBuilding D_6 A_1g projector...')
    U_a = a1g_projector(det_strings)
    print(f'  A_1g dim = {U_a.shape[1]}')

    # --- (h, s)-dependent check at the orthogonal-AO point ---------------
    print('\nSymbolic H, S over the determinantal basis...')
    t0 = time.time()
    H_sym = m.build_matrix(m.basis, op='H')
    S_sym = m.build_matrix(m.basis, op='S')
    print(f'  done in {time.time()-t0:.2f}s')
    h, s = sp.symbols('h s')

    print('\n[H, eta^2] commutator across overlap parameter:')
    for sval in (0.0, 0.1, 0.2, 0.3):
        H_num = np.array(H_sym.subs({h: -1, s: sval}).tolist(), dtype=float)
        comm = H_num @ E2 - E2 @ H_num
        print(f'  s = {sval:.2f}:  ||[H, eta^2]||_inf = {np.max(np.abs(comm)):.2e}')

    # --- the orthogonal-AO chain  A1g -> singlet -> eta=0 ---------------
    print('\nOrthogonal-AO reduction chain  A_1g (38) -> singlet (22) -> eta=0 (14):')
    H0 = np.array(H_sym.subs({h: -1, s: 0}).tolist(), dtype=float)

    H_a, S2_a, E2_a = (U_a.T @ M @ U_a for M in (H0, S2, E2))
    sym = lambda M: 0.5 * (M + M.T)
    H_a, S2_a, E2_a = map(sym, (H_a, S2_a, E2_a))

    # singlet projection (S=I in orthogonal-AO limit, so plain eigh)
    ev_s2, vS = np.linalg.eigh(S2_a)
    US = vS[:, np.abs(ev_s2) < 1e-6]
    H_s, E2_s = US.T @ H_a @ US, US.T @ E2_a @ US
    print(f'  singlet-A_1g dim = {H_s.shape[0]}')
    print(f'    H spectrum degeneracies: {dict(sorted(deg_hist(np.linalg.eigvalsh(sym(H_s))).items()))}')

    ev_e2, vE = np.linalg.eigh(sym(E2_s))
    print(f'  eta^2 eigenvalues in singlet-A_1g block: '
          f'{sorted(set(np.round(ev_e2, 6).tolist()))}')
    for tgt, label in [(0, 'eta=0'), (2, 'eta=1'), (6, 'eta=2'), (12, 'eta=3')]:
        msk = np.abs(ev_e2 - tgt) < 1e-6
        if msk.any():
            Ueta = vE[:, msk]
            H_se = Ueta.T @ H_s @ Ueta
            ev = np.linalg.eigvalsh(sym(H_se))
            print(f'    {label} dim = {int(msk.sum())},  '
                  f'spectrum = {np.round(ev, 4).tolist()}')

    # --- Hubbard term ---------------------------------------------------
    print('\nHubbard V_U  (sum_i n_{i,alpha} n_{i,beta}):')
    V_U = np.diag([double_occ(d) for d in det_strings]).astype(float)
    print(f'  ||[V_U, eta^2]||_inf = {np.max(np.abs(V_U @ E2 - E2 @ V_U)):.2e}'
          f'  (must be 0)')

    # eta_+ on N=4,6,8 basis
    big_dets = []
    for nab in (2, 3, 4):
        m.generate_basis(nab, nab, 6)
        big_dets.extend(fp.dets[0].det_string for fp in m.basis)
    orbital_index = {o: k for k, o in enumerate(ORBS)}
    EP = build_eta_plus(big_dets, orbital_index)
    V_U_big = np.diag([double_occ(d) for d in big_dets]).astype(float)
    diff = (V_U_big @ EP - EP @ V_U_big) - EP
    print(f'  ||[V_U, eta_+] - U*eta_+||_inf = {np.max(np.abs(diff)):.2e}'
          f'  (with U absorbed into V_U_big; must be 0)')


if __name__ == '__main__':
    main()
