"""Eigenvector-level cross-check of the benzene ionicity-class weights.

The energy-level validation (pyscf_benzene_validation.py) checks only
eigenvalues. This script validates the paper's central quantity, the
Chirgwin-Coulson ionicity-class weights of manuscript Table 3, against an
independent code: PySCF FCI on the 6-site Hubbard ring at s = 0, where the
AO determinant basis is orthonormal and the class weight is the plain sum
of squared CI coefficients over determinants with n_d doubly occupied
sites.

Compares w_0..w_3 at U/|h| in {0, 1, 2, 4, 8, 16} against the manuscript
Table 3 values (3 decimals) and against the exact U = 0 fractions
(5, 31, 31, 5)/72 of Eq. (15).

Run from the repo root: PYTHONPATH=. python3 examples/benzene_weights_validation.py
"""
import numpy as np
from pyscf import fci

L = 6           # ring sites
NA = NB = 3     # half filling, Sz = 0
T3 = {          # manuscript Table 3 (s = 0, h = -1)
    0:  (0.069, 0.431, 0.431, 0.069),
    1:  (0.123, 0.495, 0.344, 0.038),
    2:  (0.206, 0.522, 0.252, 0.019),
    4:  (0.447, 0.443, 0.105, 0.004),
    8:  (0.781, 0.204, 0.015, 0.000),
    16: (0.937, 0.062, 0.001, 0.000),
}

h1 = np.zeros((L, L))
for i in range(L):
    h1[i, (i + 1) % L] = h1[(i + 1) % L, i] = -1.0

strs = fci.cistring.make_strings(range(L), NA)
ndet = len(strs)

def class_weights(U):
    eri = np.zeros((L, L, L, L))
    for i in range(L):
        eri[i, i, i, i] = U
    e, ci = fci.direct_spin1.kernel(h1, eri, L, (NA, NB))
    w = np.zeros(4)
    for ia, sa in enumerate(strs):
        for ib, sb in enumerate(strs):
            nd = bin(sa & sb).count('1')
            w[nd] += ci[ia, ib] ** 2
    return e, w

print(f'{"U/|h|":>6} {"E_FCI":>10}  w_0      w_1      w_2      w_3    max|dev| vs Table 3')
ok = True
for U, ref in T3.items():
    e, w = class_weights(float(U))
    dev = max(abs(w[k] - ref[k]) for k in range(4))
    ok &= dev < 5e-4          # Table 3 carries 3 decimals
    print(f'{U:>6} {e:>10.4f}  ' + '  '.join(f'{x:.5f}' for x in w)
          + f'   {dev:.1e}')

w0_exact = np.array([5, 31, 31, 5]) / 72
_, w_u0 = class_weights(0.0)
assert np.allclose(w_u0, w0_exact, atol=1e-13), (w_u0, w0_exact)
print('U = 0 weights equal (5, 31, 31, 5)/72 to machine precision: OK')
assert ok, 'a class weight deviates from Table 3 beyond rounding'
print('all Table 3 rows reproduced within 3-decimal rounding: OK')
