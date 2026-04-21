"""
Universal MP2 formula for cyclic pi-ring Hubbard at arbitrary L, N.

For a ring of L sites with N electrons (Sz = 0), closed-shell Huckel
reference, and on-site Hubbard U, second-order RSPT gives

                1                              1
    c_2(L)  =  ---  sum_{quadruples}  -------------------------
               L^2                     eps_i + eps_j - eps_a - eps_b

where
  - the sum is over (k_i, k_j, k_a, k_b) with k_i, k_j in the occupied
    Huckel momentum set, k_a, k_b in the virtual set, and
    k_i + k_j == k_a + k_b  (mod L),
  - eps_k = 2 h cos(2 pi k / L)  is the Huckel one-electron energy.

This is the Hubbard MP2 formula: the vertex  U/L^2 delta_momentum  comes
from the plane-wave expansion of the on-site interaction, and the 1/L^2
in front is (1/L)^2 squared momentum-conserving contractions.

Spin structure: on-site Hubbard couples only alpha-beta, so each quadruple
appears exactly once (not twice), and same-spin MP2 contributions vanish.

The script evaluates the sum symbolically for L in {4, 5, 6, 7, 8, 10}
at N = 6, reproducing the three cases treated in the manuscript and
extrapolating to larger rings.
"""
import sympy as sp
from math import floor


def huckel_occ_vir(L, N):
    """Momentum labels (as integers in 0..L-1) of occupied and virtual Huckel
    MOs for a ring of L sites and N electrons (N even), closed-shell
    reference (N/2 lowest energies doubly occupied).  For degenerate
    HOMOs (open-shell) this function insists on a closed-shell fill."""
    if N % 2:
        raise ValueError('N must be even for closed-shell reference')
    ks = list(range(L))
    # energies
    h = sp.Rational(-1)      # convention h = -1
    eps = {k: 2 * h * sp.cos(2 * sp.pi * k / L) for k in ks}
    # sort by energy; simplify so that ties cluster
    order = sorted(ks, key=lambda k: float(eps[k]))

    Nhalf = N // 2
    occ = order[:Nhalf]
    vir = order[Nhalf:]

    # Sanity: HOMO and LUMO must not be degenerate
    if sp.simplify(eps[occ[-1]] - eps[vir[0]]) == 0:
        raise ValueError(
            f'L={L}, N={N}: open-shell HOMO degeneracy, closed-shell '
            'reference not uniquely defined')
    return occ, vir, eps


def mp2_sum(L, N, simplify=True):
    """Symbolic MP2 sum  Sigma 1/Delta  over momentum-conserving quadruples."""
    occ, vir, eps = huckel_occ_vir(L, N)

    total = sp.Rational(0)
    count = 0
    for ki in occ:
        for kj in occ:
            for ka in vir:
                for kb in vir:
                    if (ki + kj - ka - kb) % L != 0:
                        continue
                    denom = eps[ki] + eps[kj] - eps[ka] - eps[kb]
                    total += 1 / denom
                    count += 1
    if simplify:
        total = sp.nsimplify(total, [sp.sqrt(5)], rational=True)
        # fall back to simplify for L = 6 integer case
        total = sp.simplify(total)
    c2 = total / sp.Integer(L) ** 2
    c2 = sp.simplify(c2)
    return c2, count


def main():
    print('=' * 72)
    print('Hubbard MP2 c_2 coefficient as Huckel-MO sum over ring rings')
    print('=' * 72)
    print(f'{"L":>3} {"N":>3} {"# quads":>8} {"c_2 (closed form)":>32} '
          f'{"decimal":>18}')
    print('-' * 72)
    for L in (4, 5, 6, 7, 8, 10, 12):
        try:
            c2, cnt = mp2_sum(L, 6)
        except ValueError as e:
            print(f'{L:>3} {6:>3}   --       {"n/a: " + str(e)[:35]:>32}')
            continue
        print(f'{L:>3} {6:>3} {cnt:>8}   {str(c2):>30}   {float(c2):>+14.10f}')

    print('\nCross-check against earlier vbt3 FCI PT extraction:')
    print(f'  L = 4:  expected -5/128          got {float(mp2_sum(4,6)[0]):+.10f}')
    print(f'  L = 5:  expected 3(1-sqrt(5))/50 got {float(mp2_sum(5,6)[0]):+.10f}')
    print(f'  L = 6:  expected -29/288         got {float(mp2_sum(6,6)[0]):+.10f}')

    # Dense limit
    print('\nLarge-L limit of c_2:')
    for L in (16, 20, 30, 50, 100):
        c2, _ = mp2_sum(L, 6, simplify=False)
        c2_f = float(sp.N(c2, 20))
        print(f'  L = {L:>3}:  c_2 = {c2_f:+.10f}   (c_2 * L = {c2_f*L:+.6f})')


if __name__ == '__main__':
    main()
