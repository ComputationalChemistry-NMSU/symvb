"""Compare CC weights vs Löwdin weights for the 22 singlet-A_1g basis vectors.

CC weight:       w_k^CC = c_k * (G c)_k / (c^T G c)         can be < 0
Löwdin weight:   w_k^L  = |(G^{1/2} c)_k|^2 / (c^T G c)     always >= 0
                 = c_k * sum_j (G^{1/2})_{kj} c_j ...
                 implemented as |G^{1/2} c|^2 element-wise / norm.

We want to see whether row 16 (w_16^CC = -1/54) is a true minor contributor
(small w_16^L) or a numerically large contributor whose CC sign is just an
artifact of the metric.

Depends on benzene_nd_group_table (build_22_block) and the cached
/tmp/benzene_hubbard_matrices.pkl. Run from the repo root:
PYTHONPATH=. python3 examples/benzene_22row_loo_weights.py
"""
import os, sys, pickle, numpy as np, sympy as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from examples.benzene_nd_group_table import (
    build_22_block, closed_shell_in_AO_sp, double_occ
)

(H22_sym, S22_sym, US_sp, U_a_int, orbits,
 det_strings, h_sym, s_sym) = build_22_block()

nd_per_row = []
for k in range(22):
    col_abs = [abs(float(US_sp[i, k])) for i in range(38)]
    i_dom = int(np.argmax(col_abs))
    rep = det_strings[orbits[i_dom][0]]
    nd_per_row.append(double_occ(rep))

v0_sp = closed_shell_in_AO_sp(det_strings)
orbit_sizes = (U_a_int.T @ U_a_int).diagonal()
D_sp = sp.diag(*[sp.Integer(int(d)) for d in orbit_sizes])
G_sp = US_sp.T * D_sp * US_sp                          # singlet-basis Gram at s=0
U_a_sp = sp.Matrix(U_a_int.tolist())
v22_naive_sp = US_sp.T * (U_a_sp.T * v0_sp)
v22_sp = G_sp.solve(v22_naive_sp)
v0_norm2_sp = sum(v0_sp[i, 0] ** 2 for i in range(len(det_strings)))

# CC weights (rational)
w_cc = [v22_sp[k, 0] * v22_naive_sp[k, 0] / v0_norm2_sp for k in range(22)]
w_cc = [sp.nsimplify(sp.together(w)) for w in w_cc]

# Löwdin weights: build G as float (s=0), take G^{1/2}, transform c, square
G_num = np.array([[float(G_sp[i, j]) for j in range(22)] for i in range(22)])
c_num = np.array([float(v22_sp[k, 0]) for k in range(22)])
norm2 = float(c_num @ G_num @ c_num)

# G^{1/2} via eigendecomposition (G is symmetric positive-definite)
evals, evecs = np.linalg.eigh(G_num)
assert (evals > 0).all(), "G not SPD"
G_half = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
c_lowdin = G_half @ c_num
w_lowdin = c_lowdin ** 2 / norm2

# Also: inverse-overlap (Gallup-Norbeck-Chirgwin) weight
# w_k^GNC = |c_k|^2 / (G^{-1})_{kk} / norm2     (one common convention)
# This is well-defined and non-negative but doesn't sum to 1 in general.
G_inv = np.linalg.inv(G_num)
w_gnc = (c_num ** 2) / np.diag(G_inv) / norm2

print(f"  norm2 = c^T G c = {norm2:.6f}")
print(f"  sum w_CC = {float(sum(w_cc)):.6f}    sum w_Löwdin = {w_lowdin.sum():.6f}    sum w_GNC = {w_gnc.sum():.6f}")
print()
print(f"  {'k':>2}  {'n_d':>3}  {'c_k':>12}  {'w_CC':>14}  {'w_CC dec':>10}  "
      f"{'w_Löwdin':>10}  {'w_GNC':>10}")
print('  ' + '-' * 80)
for k in range(22):
    flag = ''
    if float(w_cc[k]) < 0:
        flag = '  <- NEGATIVE CC'
    elif float(w_cc[k]) == 0:
        flag = '  <- ZERO CC'
    print(f"  {k+1:>2}  {nd_per_row[k]:>3d}  {c_num[k]:>12.6f}  "
          f"{sp.sstr(w_cc[k]):>14}  {float(w_cc[k]):>10.6f}  "
          f"{w_lowdin[k]:>10.6f}  {w_gnc[k]:>10.6f}{flag}")

print()
print('Focus on rows 14 (w_CC = 0) and 16 (w_CC = -1/54):')
for k in (13, 15):
    print(f'  k={k+1}:  c_k = {c_num[k]:+.6f},  '
          f'w_CC = {float(w_cc[k]):+.6f},  '
          f'w_Löwdin = {w_lowdin[k]:.6f},  '
          f'w_GNC = {w_gnc[k]:.6f}')

print()
print('Verdict on negative w_CC:')
print('  - Coefficient |c_16| =', f'{abs(c_num[15]):.4f},'
      ' Löwdin weight =', f'{w_lowdin[15]:.4f}', '(non-trivial, NOT zero)')
print('  - The negative CC sign comes from the (Sc)_16 component, not from c_16 itself.')
print('  - All Löwdin weights are non-negative by construction; row 16 has w_L =',
      f'{w_lowdin[15]:.4f}')
print('  - Compare: row 1 (w_CC = 1/864 ≈ 0.0012) has Löwdin weight =',
      f'{w_lowdin[0]:.4f}  -- a true minor contributor')
