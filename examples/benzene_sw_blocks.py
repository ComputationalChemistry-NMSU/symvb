"""Block setup for Schrieffer-Wolff on benzene (s=0, half-filling).

Builds the 14-dim singlet-A_1g, eta=0 block, decomposes it by N_d
(double-occupation count = V_U/U eigenvalue), and projects V_t = h*K into
the resulting partition.

Blocks:
  P       N_d = 0   covalent (low-energy)
  Q_k     N_d = k   k-fold ionic (energy k*U at V_t = 0)

V_t connects |Delta N_d| <= 1 (single hop changes the doublon count by
-1, 0, or +1).  At s=0,  [V_t, S^2] = [V_t, D_6] = [V_t, eta^2] = 0, so
the whole SW chain stays inside this 14-dim block.

Saves the partitioned basis and V_t blocks to /tmp/benzene_sw_blocks.pkl
for use by the 4th-order SW solver.
"""
import os
import sys
import pickle
import time
import numpy as np
import sympy as sp
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from symvb import Molecule, SlaterDet, symmetry
from symvb.spin import s_squared_matrix, eta_squared_matrix


HUBBARD_CACHE = '/tmp/benzene_hubbard_matrices.pkl'
OUT_CACHE = '/tmp/benzene_sw_blocks.pkl'


ORBS = list('abcdef')
SITE_SIGNS = {'a': +1, 'b': -1, 'c': +1, 'd': -1, 'e': +1, 'f': -1}


def build_basis():
    m = Molecule(zero_ii=True,
                 subst={'s': ('S_ab','S_bc','S_cd','S_de','S_ef','S_af'),
                        'h': ('H_ab','H_bc','H_cd','H_de','H_ef','H_af')},
                 interacting_orbs=['ab','bc','cd','de','ef','af'])
    m.generate_basis(3, 3, 6)
    return m, [fp.dets[0].det_string for fp in m.basis]


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
    occ = {}
    for c in ds:
        occ.setdefault(c.lower(), [False, False])
        if c.islower(): occ[c.lower()][0] = True
        else:           occ[c.lower()][1] = True
    return sum(1 for ab in occ.values() if ab[0] and ab[1])


def project_kernel(M, U, tol=1e-8):
    """Return an orthonormal basis (columns) for ker(M) inside span(U)."""
    Mp = U.T @ M @ U
    Mp = 0.5 * (Mp + Mp.T)
    ev, vec = np.linalg.eigh(Mp)
    return U @ vec[:, np.abs(ev) < tol]


def main():
    print("Building benzene 400-dim basis (3a + 3b in 6 orbitals)...")
    m, det_strings = build_basis()
    N = len(det_strings)
    Nd_diag = np.array([double_occ(d) for d in det_strings], dtype=float)
    print(f"  full Sz=0 dim = {N}")
    print(f"  raw N_d distribution: {dict(sorted(Counter(Nd_diag.astype(int)).items()))}")

    print("\nBuilding A_1g projector (D_6 = C_6 + sigma_v)...")
    U_a = a1g_projector(det_strings)
    print(f"  A_1g dim = {U_a.shape[1]}")

    print("\nBuilding S^2, eta^2...")
    t0 = time.time()
    S2 = s_squared_matrix(det_strings)
    E2 = eta_squared_matrix(det_strings, SITE_SIGNS, ORBS)
    print(f"  done in {time.time()-t0:.2f}s")

    print("\nReducing  A_1g (38) -> singlet (22) -> eta=0 (14)...")
    U_s = project_kernel(S2, U_a)
    print(f"  singlet-A_1g dim = {U_s.shape[1]}")
    U_e = project_kernel(E2, U_s)
    print(f"  eta=0 singlet-A_1g dim = {U_e.shape[1]}")

    Nd = np.diag(Nd_diag)

    def dist(U, label):
        Nd_block = U.T @ Nd @ U
        ev = np.linalg.eigvalsh(0.5 * (Nd_block + Nd_block.T))
        ev_int = np.round(ev).astype(int)
        assert np.allclose(ev, ev_int, atol=1e-6), f"N_d non-integer in {label}: {ev}"
        c = Counter(ev_int.tolist())
        return dict(sorted(c.items()))

    print("\nN_d = (double-occupation count) decomposition:")
    print(f"  full A_1g (38)            : {dist(U_a, 'A_1g')}")
    print(f"  singlet-A_1g (22)         : {dist(U_s, 'singlet-A_1g')}")
    print(f"  singlet-A_1g, eta=0 (14)  : {dist(U_e, 'eta=0 singlet-A_1g')}")

    print("\nSW block structure (s=0, the relevant case):")
    d14 = dist(U_e, 'eta=0 singlet-A_1g')
    print(f"  P    = N_d = 0  : dim = {d14.get(0, 0)}    (covalent A_1g singlets)")
    print(f"  Q_1  = N_d = 1  : dim = {d14.get(1, 0)}    (energy denom = -U)")
    print(f"  Q_2  = N_d = 2  : dim = {d14.get(2, 0)}    (energy denom = -2U)")
    print(f"  Q_3  = N_d = 3  : dim = {d14.get(3, 0)}    (energy denom = -3U)")
    print(f"  total                            = {sum(d14.values())}")

    # ------------------------------------------------------------------
    # N_d-sorted basis inside the 14-dim block
    # ------------------------------------------------------------------
    Nd_in_e = U_e.T @ Nd @ U_e
    ev, vec = np.linalg.eigh(0.5 * (Nd_in_e + Nd_in_e.T))
    order = np.argsort(np.round(ev).astype(int), kind='stable')
    ev = ev[order]
    vec = vec[:, order]
    U_14 = U_e @ vec
    Nd_eigs = np.round(ev).astype(int)
    slices = {k: np.where(Nd_eigs == k)[0] for k in range(4)}

    # ------------------------------------------------------------------
    # V_t connectivity matrix K (V_t = h * K) from the Hubbard cache
    # ------------------------------------------------------------------
    print(f"\nLoading {HUBBARD_CACHE} for V_t...")
    if not os.path.exists(HUBBARD_CACHE):
        raise RuntimeError(f"Missing {HUBBARD_CACHE}; "
                           f"run examples/benzene_hubbard_pt.py to build it.")
    with open(HUBBARD_CACHE, 'rb') as f:
        H1_sym, S_sym, H2_sym = pickle.load(f)
    h_s, s_s, U_s = sp.symbols('h s U')
    K_full = np.array(sp.Matrix(H1_sym).subs({h_s: 1, s_s: 0, U_s: 0}),
                      dtype=float)

    K14 = U_14.T @ K_full @ U_14
    K14[np.abs(K14) < 1e-10] = 0.0

    print("\nV_t = h * K, partitioned by N_d (rows = bra, cols = ket):")
    print(f"{'':>6}  " + "  ".join(f"Q_{j}({len(slices[j])})" for j in range(4)))
    for i in range(4):
        row = f"Q_{i}({len(slices[i])})"
        cells = []
        for j in range(4):
            block = K14[np.ix_(slices[i], slices[j])]
            nrm = np.linalg.norm(block)
            cells.append(f"{nrm:>7.3f}" if nrm > 1e-8 else f"{'.':>7}")
        print(f"  {row:>6}  " + "  ".join(cells))

    # Verify tridiagonal-in-N_d structure (|i-j| > 1 must vanish)
    for i in range(4):
        for j in range(4):
            if abs(i - j) > 1:
                block = K14[np.ix_(slices[i], slices[j])]
                assert np.allclose(block, 0, atol=1e-8), \
                    f"V_t leaks N_d={i}<->N_d={j}: ||K||={np.linalg.norm(block)}"
    print("  [verified tridiagonal in N_d]")

    # Within-block hopping: V_{P,P} and V_{Q_3,Q_3} must vanish
    # (every hop from a fully-singly state creates a doublon; every hop
    #  from a 3-doublon-3-hole state moves a doublon -> singly)
    for k in (0, 3):
        block = K14[np.ix_(slices[k], slices[k])]
        assert np.allclose(block, 0, atol=1e-8), \
            f"V_t intra-block at N_d={k} should vanish: ||K||={np.linalg.norm(block)}"
    print("  [verified V_{P,P} = V_{Q_3,Q_3} = 0]")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    rationalize = lambda M: sp.Matrix([[sp.nsimplify(x, rational=True) for x in row]
                                       for row in M.tolist()])
    K14_sym = rationalize(K14)
    if (K14_sym - K14_sym.T).is_zero_matrix is False:
        # numerical noise; force symmetry symbolically
        K14_sym = (K14_sym + K14_sym.T) / 2

    cache = dict(
        U_14=U_14,
        Nd_eigs=Nd_eigs,
        slices=slices,
        K14=K14,
        K14_sym=K14_sym,
        det_strings=det_strings,
        SITE_SIGNS=SITE_SIGNS,
        ORBS=ORBS,
    )
    with open(OUT_CACHE, 'wb') as f:
        pickle.dump(cache, f)
    print(f"\nSaved 14-dim SW block setup to {OUT_CACHE}")


if __name__ == '__main__':
    main()
