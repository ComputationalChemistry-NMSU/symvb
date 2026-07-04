"""
Comparative symmetry reduction for cyclic pi-systems with 6 electrons:

    L = 4   cyclobutadiene dianion  (C_4 H_4 ^2-)     16-dim Sz=0
    L = 5   cyclopentadienyl anion  (Cp-)            100-dim Sz=0
    L = 6   benzene                                  400-dim Sz=0

All three have 6 pi-electrons -> Huckel 4n+2 aromatics.  Table produced at
the end:

    L   full   A_1   singlet-A_1   singlet-A_1, eta=eta_z    bipartite?
    4    16     ?       ?                ?                       yes
    5   100    12       8             (not available)            no
    6   400    38      22               14                       yes

Fixed parameters: orthogonal AOs (s = 0), kinetic h = -1, Hubbard U = 0.
The eta-pairing refinement only applies on bipartite rings; L = 4 and
L = 6 qualify, L = 5 does not.

eta_z depends on filling:
    eta_z = (N_alpha + N_beta - L) / 2
    L = 4 (6e): eta_z = +1   (2e past half-filling)
    L = 5 (6e): eta_z = +0.5 (half-integer, eta multiplets all half-integer)
    L = 6 (6e): eta_z =  0   (half-filling)
"""
import time
import numpy as np
import sympy as sp
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict

from symvb import Molecule, SlaterDet, symmetry
from symvb.spin import s_squared_matrix, eta_squared_matrix


def ring_orbitals(L):
    """orbital labels a, b, c, ...  and nearest-neighbor edges for L-ring."""
    orbs = [chr(ord('a') + i) for i in range(L)]
    edges = [orbs[i] + orbs[(i + 1) % L] for i in range(L)]
    # symvb convention: each edge listed with lowercase in alphabetical order
    edges = [''.join(sorted(e)) for e in edges]
    return orbs, edges


def ring_symmetry_maps(L):
    """C_L rotation and a sigma_v reflection that's representable in symvb."""
    orbs = [chr(ord('a') + i) for i in range(L)]
    C_L = {orbs[i]: orbs[(i + 1) % L] for i in range(L)}
    # sigma_v through orbital 'a' and (for even L) the opposite orbital
    sigma = {orbs[i]: orbs[(-i) % L] for i in range(L)}
    return C_L, sigma


def bipartition_signs(L):
    """Bipartite sublattice signs (None if odd L)."""
    if L % 2:
        return None
    orbs = [chr(ord('a') + i) for i in range(L)]
    return {o: (+1 if i % 2 == 0 else -1) for i, o in enumerate(orbs)}


def double_occ(ds):
    occ = {}
    for c in ds:
        occ.setdefault(c.lower(), [False, False])
        if c.islower():
            occ[c.lower()][0] = True
        else:
            occ[c.lower()][1] = True
    return sum(1 for ab in occ.values() if ab[0] and ab[1])


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


@dataclass
class Result:
    L: int
    N_total: int
    group_order: int
    dim_full: int
    dim_A1: int
    dim_singlet: int
    dim_eta0: int = None            # None if eta-pairing n/a
    bipartite: bool = False
    E0_huckel: float = None         # h = -1, s = 0, U = 0
    singlet_spectrum: List[float] = field(default_factory=list)
    deg_A1: Dict[int, int] = field(default_factory=dict)
    deg_singlet: Dict[int, int] = field(default_factory=dict)


def analyse_ring(L, N_alpha=3, N_beta=3, verbose=True):
    orbs, edges = ring_orbitals(L)
    C_L, sigma = ring_symmetry_maps(L)
    site_signs = bipartition_signs(L)

    # S-sym uses 'S_ab', 'S_bc', etc; substitution list must match edges
    s_labels = tuple(f'S_{e}' for e in edges)
    h_labels = tuple(f'H_{e}' for e in edges)

    m = Molecule(zero_ii=True,
                 subst={'s': s_labels, 'h': h_labels},
                 interacting_orbs=edges)
    m.generate_basis(N_alpha, N_beta, L)
    det_strings = [fp.dets[0].det_string for fp in m.basis]
    N_full = len(det_strings)

    if verbose:
        print(f'\n=== L = {L}, {N_alpha}a + {N_beta}b electrons, '
              f'Sz = {(N_alpha-N_beta)/2:.1f} ===')
        print(f'  full dim = {N_full}  [C({L},{N_alpha}) x C({L},{N_beta}) '
              f'= {len(list(m.basis))}]')

    # Symmetry: C_L + sigma_v  (order 2L)
    def canon(ds):
        fp = SlaterDet(ds).get_sorted()
        return fp.dets[0].det_string, fp.coefs[0]
    perms = [symmetry.apply_orbital_permutation(om, det_strings, canon)[0]
             for om in (C_L, sigma)]
    U_a, orbits = symmetry.totally_symmetric_basis(perms, N_full)
    dim_A1 = U_a.shape[1]

    # S^2 projection
    S2 = s_squared_matrix(det_strings)
    S2_a = 0.5 * (U_a.T @ S2 @ U_a + (U_a.T @ S2 @ U_a).T)
    ev_s2, vS = np.linalg.eigh(S2_a)
    US = vS[:, np.abs(ev_s2) < 1e-6]
    dim_S = US.shape[1]

    # Kinetic H at h = -1, s = 0
    H_sym = m.build_matrix(m.basis, op='H')
    h, ssym = sp.symbols('h s')
    H0 = np.array(H_sym.subs({h: -1, ssym: 0}).tolist(), dtype=float)
    H_a = 0.5 * (U_a.T @ H0 @ U_a + (U_a.T @ H0 @ U_a).T)
    H_s = 0.5 * (US.T @ H_a @ US + (US.T @ H_a @ US).T)

    res = Result(
        L=L, N_total=N_alpha + N_beta, group_order=2 * L,
        dim_full=N_full, dim_A1=dim_A1, dim_singlet=dim_S,
        bipartite=(site_signs is not None),
        E0_huckel=float(np.linalg.eigvalsh(H_s)[0]),
        singlet_spectrum=sorted(np.round(np.linalg.eigvalsh(H_s), 6).tolist()),
        deg_A1=dict(sorted(deg_hist(np.linalg.eigvalsh(H_a)).items())),
        deg_singlet=dict(sorted(deg_hist(np.linalg.eigvalsh(H_s)).items())),
    )

    # eta-pairing refinement only if bipartite
    if site_signs is not None:
        E2 = eta_squared_matrix(det_strings, site_signs, orbs)
        E2_a = 0.5 * (U_a.T @ E2 @ U_a + (U_a.T @ E2 @ U_a).T)
        E2_s = 0.5 * (US.T @ E2_a @ US + (US.T @ E2_a @ US).T)

        # [H, eta^2] should vanish at s = 0
        comm = H0 @ E2 - E2 @ H0
        if verbose:
            print(f'  bipartite -> ||[H, eta^2]||_inf at s=0: '
                  f'{np.max(np.abs(comm)):.2e}')

        # eta_z target: eta(eta+1) >= eta_z^2, so smallest allowed eta*(eta+1)
        eta_z = (N_alpha + N_beta - L) / 2
        eta_min = abs(eta_z)
        target = eta_min * (eta_min + 1)
        ev_e2 = np.linalg.eigvalsh(E2_s)
        mask = np.abs(ev_e2 - target) < 1e-6
        res.dim_eta0 = int(mask.sum())

        if verbose:
            eta_hist = Counter(np.round(ev_e2, 4).tolist())
            print(f'  eta_z = {eta_z:+.1f},  minimum allowed eta = {eta_min:.1f}')
            print(f'  eta^2 spectrum on singlet-A_1 block:')
            for ev, mult in sorted(eta_hist.items()):
                eta = (np.sqrt(1 + 4 * max(ev, 0)) - 1) / 2
                marker = '  <-- ground' if abs(ev - target) < 1e-6 else ''
                print(f'    {ev:7.4f}  (eta = {eta:.1f}):  {mult}{marker}')

    if verbose:
        print(f'  A_1           dim = {dim_A1:3d}   H degeneracies = {res.deg_A1}')
        print(f'  singlet-A_1   dim = {dim_S:3d}   H degeneracies = {res.deg_singlet}')
        if res.dim_eta0 is not None:
            print(f'  + eta-reduced dim = {res.dim_eta0:3d}')
        print(f'  E_0 (Huckel, h=-1, s=0, U=0) = {res.E0_huckel:.6f}')
        print(f'  singlet-A_1 spectrum: {res.singlet_spectrum}')

    return res


def main():
    t0 = time.time()
    print('=' * 72)
    print('Symmetry reduction for cyclic 6pi-electron aromatics (4n+2, n=1)')
    print('=' * 72)

    results = [analyse_ring(L) for L in (4, 5, 6)]

    # ------------------------------------------------------------------
    # Side-by-side summary table
    # ------------------------------------------------------------------
    print('\n' + '=' * 72)
    print('SUMMARY TABLE')
    print('=' * 72)
    print(f'{"L":>3}  {"full":>5}  {"A_1":>5}  {"S-A_1":>6}  '
          f'{"eta=|eta_z|":>11}  {"bipartite?":>11}  {"E_0 (Huckel)":>14}')
    print('-' * 72)
    for r in results:
        eta_col = f'{r.dim_eta0}' if r.dim_eta0 is not None else 'n/a'
        bip = 'yes' if r.bipartite else 'no'
        print(f'{r.L:>3}  {r.dim_full:>5}  {r.dim_A1:>5}  {r.dim_singlet:>6}  '
              f'{eta_col:>11}  {bip:>11}  {r.E0_huckel:>14.6f}')

    # ------------------------------------------------------------------
    # Huckel MO check: closed-form energies
    # ------------------------------------------------------------------
    print('\nHuckel closed-form check:')
    print('  MO energies for L-ring: epsilon_k = 2h cos(2 pi k / L)')
    for r in results:
        L = r.L
        levels = sorted(2 * (-1) * np.cos(2 * np.pi * np.arange(L) / L))
        # fill the lowest 6 spin-orbitals: levels[0] takes 2, levels[1] takes 2, ...
        e_closed, electrons_left = 0.0, r.N_total
        level_i = 0
        while electrons_left > 0 and level_i < L:
            # degeneracy handling: group near-degenerate levels
            j = level_i + 1
            while j < L and abs(levels[j] - levels[level_i]) < 1e-8:
                j += 1
            group_size = j - level_i
            capacity = 2 * group_size
            take = min(capacity, electrons_left)
            e_closed += take * levels[level_i]
            electrons_left -= take
            level_i = j
        match = '<-- OK' if abs(e_closed - r.E0_huckel) < 1e-6 else '<-- MISMATCH'
        print(f'  L={L}: levels = {np.round(levels, 4).tolist()}, '
              f'E_0(closed) = {e_closed:.6f}   {match}')

    # ------------------------------------------------------------------
    # Spectrum comparison
    # ------------------------------------------------------------------
    print('\nSinglet-A_1 spectra, shifted by E_0 (gaps above ground state):')
    for r in results:
        gaps = [round(e - r.E0_huckel, 4) for e in r.singlet_spectrum]
        print(f'  L={r.L}: {gaps}')

    # ------------------------------------------------------------------
    # Pattern commentary
    # ------------------------------------------------------------------
    print('\nObservations:')
    print('  1. Free-electron ground-state energies follow the Huckel rule')
    print('     E_0 = -2 (sum of lowest three |cos|-values, times 2 for double')
    print('     occupation).  L=4: E_0 = -4; L=5: -6.472; L=6: -8.')
    print('  2. A_1 / orbit-count ratio (full/A_1):')
    for r in results:
        print(f'     L={r.L}: {r.dim_full}/{r.dim_A1} = {r.dim_full/r.dim_A1:.2f}'
              f'  (compare group order {r.group_order})')
    print('  3. Only L=4 and L=6 admit eta-pairing as an extra quantum number.')
    print('     L=5 is the control: the odd ring breaks bipartiteness, hence')
    print('     no sublattice phase and eta_+ does not close on H_kin.')

    print(f'\nTotal wall time: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
