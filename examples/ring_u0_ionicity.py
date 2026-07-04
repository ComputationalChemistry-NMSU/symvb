"""Ionicity-class weights for cyclic 6pi-electron rings at L = 4, 5, 6.

Extends the benzene ionicity-class decomposition (manuscript Table 3 /
benzene_weights_validation.py) to the cyclobutadiene dianion (6 electrons
in 4 orbitals) and the cyclopentadienyl anion (6 in 5), via PySCF FCI on
the L-site Hubbard ring at s = 0, where the AO determinant basis is
orthonormal and the class weight of ionicity n_d is the plain sum of
squared CI coefficients over determinants with n_d doubly occupied sites.

Counting alone empties the low-ionicity classes above half filling:
6 electrons in 4 orbitals force n_d >= 2; in 5 orbitals n_d >= 1. The
benzene "covalent at strong coupling" inversion therefore appears in the
only form available to each ring: weight migrates to the minimal-n_d
class as U grows.

Run from the repo root: PYTHONPATH=. python3 examples/ring_u0_ionicity.py
"""
from fractions import Fraction

import numpy as np
from pyscf import fci

RINGS = {  # name: (L sites, n_alpha, n_beta)
    'C4H4^2- (L=4)': (4, 3, 3),
    'Cp^-     (L=5)': (5, 3, 3),
    'benzene  (L=6)': (6, 3, 3),
}
US = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)


def class_weights(L, na, nb, U):
    h1 = np.zeros((L, L))
    for i in range(L):
        h1[i, (i + 1) % L] = h1[(i + 1) % L, i] = -1.0
    eri = np.zeros((L, L, L, L))
    for i in range(L):
        eri[i, i, i, i] = U
    e, ci = fci.direct_spin1.kernel(h1, eri, L, (na, nb))
    sa = fci.cistring.make_strings(range(L), na)
    sb = fci.cistring.make_strings(range(L), nb)
    w = np.zeros(L + 1)
    for ia, a in enumerate(sa):
        for ib, b in enumerate(sb):
            w[bin(a & b).count('1')] += ci[ia, ib] ** 2
    return e, w


for name, (L, na, nb) in RINGS.items():
    print(f'\n=== {name}: {na + nb} electrons in {L} orbitals, '
          f'n_d in [{max(0, na + nb - L)}, {min(na, nb)}] ===')
    nd_lo, nd_hi = max(0, na + nb - L), min(na, nb)
    hdr = '  '.join(f'w_{k}' + ' ' * 4 for k in range(nd_lo, nd_hi + 1))
    print(f'{"U/|h|":>6} {"E_FCI":>10}  ' + hdr)
    for U in US:
        e, w = class_weights(L, na, nb, U)
        row = '  '.join(f'{w[k]:.5f}' for k in range(nd_lo, nd_hi + 1))
        print(f'{U:>6} {e:>10.4f}  ' + row)
        if U == 0.0:
            fr = [Fraction(w[k]).limit_denominator(100000)
                  for k in range(nd_lo, nd_hi + 1)]
            dev = max(abs(float(f) - w[k])
                      for f, k in zip(fr, range(nd_lo, nd_hi + 1)))
            print(f'{"":>6} {"exact":>10}  '
                  + '  '.join(str(f) for f in fr)
                  + f'   (rationalisation residual {dev:.1e})')

# cross-check: benzene must reproduce (5, 31, 31, 5)/72
_, w6 = class_weights(6, 3, 3, 0.0)
assert np.allclose(w6[:4], np.array([5, 31, 31, 5]) / 72, atol=1e-12)
print('\nbenzene U = 0 check vs (5, 31, 31, 5)/72: OK')
