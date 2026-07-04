"""
Rational Hubbard perturbation series for cyclic 6-electron pi-systems
at L = 4, 5, 6.  Writes

    E_0(U) = c_0 + c_1 U + c_2 U^2 + c_3 U^3 + ...

at orthogonal AO limit (s = 0), h = -1.  Coefficients are fit from the
true 16/100/400-dim Sz=0 ground state (sidestepping the fermion-sign
subtlety in the A_1 orbit-sum basis).

Pattern we look for:

    c_0  =  free-electron Huckel energy
    c_1  =  <GS_0 | V_U | GS_0>  =  average double-occupancy at U = 0
    c_n  (n >= 2) -- rational-in-L progression?

For an L-site bipartite Hubbard model at half-filling, the strong-U limit
is Heisenberg (J = 4 h^2 / U).  For L = 4 (over-half-filled) and L = 5
(not bipartite) the strong-U story differs.
"""
import time
import numpy as np
import sympy as sp
from fractions import Fraction

from symvb import Molecule


def double_occ(ds):
    occ = {}
    for c in ds:
        occ.setdefault(c.lower(), [False, False])
        if c.islower():
            occ[c.lower()][0] = True
        else:
            occ[c.lower()][1] = True
    return sum(1 for ab in occ.values() if ab[0] and ab[1])


def ring_basis(L):
    orbs = [chr(ord('a') + i) for i in range(L)]
    edges = [''.join(sorted(orbs[i] + orbs[(i + 1) % L])) for i in range(L)]
    m = Molecule(
        zero_ii=True,
        subst={'s': tuple(f'S_{e}' for e in edges),
               'h': tuple(f'H_{e}' for e in edges)},
        interacting_orbs=edges,
    )
    m.generate_basis(3, 3, L)
    dets = [fp.dets[0].det_string for fp in m.basis]
    H_sym = m.build_matrix(m.basis, op='H')
    h, s = sp.symbols('h s')
    H = np.array(H_sym.subs({h: -1, s: 0}).tolist(), dtype=float)
    V = np.diag([double_occ(d) for d in dets]).astype(float)
    return H, V, dets


def pt_coefficients(H0, V, n_terms=8, U_max=0.02, n_samples=64):
    """
    Fit E_0(U) ~ sum_{k=0..n_terms-1} c_k U^k from numerical ground
    state at small U.  Accuracy limited by finite-diff noise.
    """
    Us = np.linspace(0, U_max, n_samples)
    E0s = []
    for U in Us:
        H = H0 + U * V
        E0s.append(np.linalg.eigvalsh(H)[0])
    E0s = np.array(E0s)
    coeffs = np.polyfit(Us, E0s, n_terms - 1)
    return coeffs[::-1]  # low-to-high order


def pt_coefficients_analytic(H0, V, n_terms=6):
    """
    Rayleigh-Schrodinger PT directly from H0 eigenstates.  Handles
    non-degenerate ground state.  Returns (c_0, c_1, c_2, c_3).
    """
    evals, evecs = np.linalg.eigh(H0)
    e0 = evals[0]
    g = evecs[:, 0]

    # c_0 = e0
    c = [e0]

    # c_1 = <0|V|0>
    V00 = g @ V @ g
    c.append(V00)

    if n_terms >= 3:
        # c_2 = sum_{k>0} |V_{k0}|^2 / (e0 - e_k)
        V0k = V @ g
        V0k_in_evec = evecs.T @ V0k            # <k|V|0>
        denom = e0 - evals
        mask = np.abs(evals - e0) > 1e-10
        c2 = float(np.sum(V0k_in_evec[mask]**2 / denom[mask]))
        c.append(c2)

    if n_terms >= 4:
        # c_3 = <0| V (Q/(H0-E0))^2 (V - V00) |0>
        # use numerical inversion on Q-projector
        Q = np.eye(len(evals)) - np.outer(g, g)
        R = np.zeros_like(H0)
        for k in range(1, len(evals)):
            vk = evecs[:, k]
            R += np.outer(vk, vk) / (e0 - evals[k])
        # standard 3rd-order formula:
        #   c_3 = <0|V R (V - V00) R V |0>
        W = V - V00 * np.eye(len(evals))
        c3 = float(g @ V @ R @ W @ R @ V @ g)
        c.append(c3)

    if n_terms >= 5:
        # c_4 terms: from RS PT
        #   c_4 = <0|V R W R W R V|0> - V00 <0|V R^2 W R V|0> - c_2 <0|V R^2 V|0>
        c4 = float(g @ V @ R @ W @ R @ W @ R @ V @ g
                   - c[2] * (g @ V @ R @ R @ V @ g))
        c.append(c4)

    return c


def rational_guess(x, max_denom=64):
    """Try to represent a float as a small rational; None if not."""
    f = Fraction(x).limit_denominator(max_denom)
    if abs(float(f) - x) < 1e-6:
        return f
    return None


def main():
    print('=' * 72)
    print('Hubbard PT series  E_0(U) = sum c_k U^k   for cyclic 6e pi-rings')
    print('=' * 72)

    results = {}
    for L in (4, 5, 6):
        t0 = time.time()
        print(f'\n--- L = {L} ---')
        H0, V, dets = ring_basis(L)
        print(f'  full Sz=0 dim = {len(dets)}')

        # Analytic PT (sum-over-states, exact to machine precision)
        c = pt_coefficients_analytic(H0, V, n_terms=5)
        print(f'  wall time (exact PT): {time.time() - t0:.2f}s')

        results[L] = c
        print(f'  c_0 = {c[0]:+.10f}   [Huckel E_0]')
        print(f'  c_1 = {c[1]:+.10f}   [avg doubly-occ in GS_0]')
        print(f'  c_2 = {c[2]:+.10f}')
        print(f'  c_3 = {c[3]:+.10f}')
        if len(c) >= 5:
            print(f'  c_4 = {c[4]:+.10f}')

    # ------------------------------------------------------------------
    # Side-by-side table
    # ------------------------------------------------------------------
    print('\n' + '=' * 72)
    print('SUMMARY: PT series E_0(U; h=-1) = c_0 + c_1 U + c_2 U^2 + ...')
    print('=' * 72)
    hdr = f'{"L":>3}' + ''.join(f'  {"c_" + str(k):>14}' for k in range(5))
    print(hdr)
    print('-' * len(hdr))
    for L in (4, 5, 6):
        c = results[L]
        row = f'{L:>3}' + ''.join(f'  {c[k]:>+14.8f}' for k in range(5))
        print(row)

    # ------------------------------------------------------------------
    # Rational / simple-radical recognition
    # ------------------------------------------------------------------
    print('\nRational/radical recognition of the coefficients:')
    for L in (4, 5, 6):
        print(f'  L = {L}:')
        for k, val in enumerate(results[L]):
            r = rational_guess(val)
            tag = f'  =  {r}' if r is not None else ''
            # try sqrt(5) recognition for L=5
            if L == 5:
                # Is val of form a + b sqrt(5)?
                s5 = np.sqrt(5)
                # solve 2 samples via rational fit -- just guess coefficients
                # with small denominators
                for denom in range(1, 32):
                    a = val * denom
                    # try val*denom = a + b sqrt(5) with a, b integers?
                    for b in range(-50, 51):
                        a_fit = val - b * s5 / denom
                        if abs(round(a_fit * denom) - a_fit * denom) < 1e-6:
                            a_int = round(a_fit * denom)
                            check = (a_int + b * s5) / denom
                            if abs(check - val) < 1e-9:
                                tag = f'  =  ({a_int} + {b} sqrt(5)) / {denom}'
                                break
                    if tag:
                        break
            print(f'    c_{k} = {val:+.10f}{tag}')

    # ------------------------------------------------------------------
    # Cross-check via numerical polynomial fit
    # ------------------------------------------------------------------
    print('\nCross-check: polynomial fit c_k from E_0(U) for small U ' \
          '(noise-limited):')
    print(f'{"L":>3}' + ''.join(f'  {"c_"+str(k)+" (fit)":>14}' for k in range(5)))
    for L in (4, 5, 6):
        H0, V, _ = ring_basis(L)
        c_fit = pt_coefficients(H0, V, n_terms=6, U_max=0.02, n_samples=96)[:5]
        print(f'{L:>3}' + ''.join(f'  {c_fit[k]:>+14.8f}' for k in range(5)))


if __name__ == '__main__':
    main()
