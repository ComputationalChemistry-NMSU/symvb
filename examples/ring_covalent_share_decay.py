"""Exponential decay of the covalent share with ring size (manuscript support).

Backs the benzene-section statement that the covalent weight of the closed-shell
Hueckel determinant "decreases exponentially" with system size. For a cyclic ring
of L sites at half filling with orthogonal AOs (s = 0), the U = 0 ground state is
the Slater determinant Phi of the L/2 lowest Hueckel MOs (closed shell only for
L = 2 mod 4). Its covalent share

    w_cov(L) = sum over L/2-subsets S of  |det C[S]|^2 |det C[S^c]|^2
             = sum over S of  1 / prod_{i in S, j in S^c} |z_i - z_j|^2 ,

with z_j = exp(2 pi i j / L) and C the occupied-MO coefficient matrix, is the total
squared amplitude of Phi on covalent AO determinants (every site singly occupied).
Equivalently w_cov = sum_S det P[S,S] det P[S^c,S^c] with P the rank-(L/2) occupied
band projector: the free-fermion probability of zero double occupancy, i.e. the
squared norm of the fully Gutzwiller-projected half-filled Fermi sea.

The exact values 1/2 (L=2, "H2"), 5/72 (benzene, L=6), 189/20000 (L=10), ... are
integers over the closed denominator 2^(L/2) (L/2)^(L/2 - 1). They fall off as

    w_cov(L)  ~  sqrt(2) * e^(-L/2)        (numerical fit, not a proof),

so the per-site decay constant is e^(-1/2) ~ 0.607: the covalent share drops by a
factor e^(-2) ~ 0.14 per four-site closed-shell increment. This is a monocyclic-ring
result; polycyclic systems (e.g. naphthalene) have a different MO matrix and are not
covered here. The sqrt(2) e^(-L/2) law and mu = e^(-1/2) are numerical support only;
the finite-L integers are exact.

Run from the repo root: PYTHONPATH=. python3 examples/ring_covalent_share_decay.py
"""
import math
from fractions import Fraction
from itertools import combinations

import numpy as np

# exact covalent shares as numerator / (2^M * M^(M-1)), M = L/2, coprime form
EXACT = {
    2:  Fraction(1, 2),
    6:  Fraction(5, 72),
    10: Fraction(189, 20000),
    14: Fraction(19305, 15059072),
}
NUMERIC_L = (18, 22)          # computed numerically only (numerators reduce)
SQRT2 = math.sqrt(2.0)


def cross_log_matrix(L):
    """LM[i, j] = ln|z_i - z_j|^2 = ln(2 - 2 cos(2 pi (i - j) / L)); 0 on diagonal."""
    ld = np.array([0.0] + [math.log(2.0 - 2.0 * math.cos(2.0 * math.pi * d / L))
                           for d in range(1, L)])
    idx = np.arange(L)
    LM = ld[(idx[:, None] - idx[None, :]) % L]
    np.fill_diagonal(LM, 0.0)
    return LM


def w_cov(L):
    """Covalent share of the half-filled closed-shell Hueckel determinant.

    Uses the cross-product identity w_cov = sum_S 1 / prod_{i in S, j in S^c}
    |z_i - z_j|^2 in the log domain; math.fsum keeps the subset sum (terms span
    many orders of magnitude) accurate to near machine precision.
    """
    assert L % 4 == 2, 'closed shell requires L = 2 (mod 4)'
    M = L // 2
    LM = cross_log_matrix(L)
    rowsum = LM.sum(axis=1)
    terms = []
    for S in combinations(range(L), M):
        s = np.fromiter(S, dtype=np.intp, count=M)
        # ln prod_cross = sum_{i in S} rowsum[i] - sum_{i, j in S} LM[i, j]
        logcross = rowsum[s].sum() - LM[s[:, None], s].sum()
        terms.append(math.exp(-logcross))
    return math.fsum(terms)


def denominator(L):
    """The exact denominator 2^(L/2) (L/2)^(L/2 - 1)."""
    M = L // 2
    return 2 ** M * M ** (M - 1)


# ---- compute and tabulate -------------------------------------------------
Ls = sorted(EXACT) + list(NUMERIC_L)
w = {L: w_cov(L) for L in Ls}
c = {L: w[L] * math.exp(L / 2.0) for L in Ls}      # c(L) = w * e^(L/2) -> sqrt(2)

print(f'{"L":>3} {"M":>3}  {"w_cov":>14}  {"numerator":>13}  {"denom 2^M M^(M-1)":>18}'
      f'  {"c=w e^(L/2)":>11}  {"-ln w / L":>9}')
for L in Ls:
    M = L // 2
    num = round(w[L] * denominator(L))
    exact_tag = f'  = {EXACT[L]}' if L in EXACT else ''
    print(f'{L:>3} {M:>3}  {w[L]:.8e}  {num:>13}  {denominator(L):>18}'
          f'  {c[L]:>11.7f}  {-math.log(w[L]) / L:>9.6f}{exact_tag}')

print(f'\nsqrt(2) = {SQRT2:.7f};  e^(-1/2) = {math.exp(-0.5):.7f} (per-site decay mu)')
print('law: w_cov(L) ~ sqrt(2) e^(-L/2); c(L) rises monotonically toward sqrt(2).')

# ---- assertions -----------------------------------------------------------
# (1) near-exact numeric reproduces the exact rationals, tight relative tol
for L, frac in EXACT.items():
    rel = abs(w[L] - float(frac)) / float(frac)
    assert rel <= 1e-12, f'L={L}: w_cov {w[L]} != {frac} (rel err {rel:.2e})'

# (2) closed denominator law  den = 2^M M^(M-1)  for the exact L
for L, frac in EXACT.items():
    assert frac.denominator == denominator(L), \
        f'L={L}: denominator {frac.denominator} != 2^M M^(M-1) = {denominator(L)}'

# (3) numerator over that denominator is an integer for every computed L
#     (verifies the denominator law at the numeric points L=18, 22 too)
for L in Ls:
    x = w[L] * denominator(L)
    assert abs(x - round(x)) < 1e-2, \
        f'L={L}: w_cov * 2^M M^(M-1) = {x} not integral'

# (4) c(L) = w e^(L/2) increases monotonically and stays within (1.35, sqrt(2)),
#     consistent with w_cov ~ sqrt(2) e^(-L/2) (mu = e^(-1/2) exactly)
cs = [c[L] for L in Ls]
assert all(b > a for a, b in zip(cs, cs[1:])), f'c(L) not monotonic increasing: {cs}'
for L in Ls:
    assert 1.35 < c[L] < SQRT2, f'L={L}: c={c[L]} outside (1.35, sqrt(2))'

print('\nasserts OK:')
print('  - w_cov = 1/2, 5/72, 189/20000, 19305/15059072 for L = 2, 6, 10, 14'
      ' (rel err <= 1e-12)')
print('  - denominator = 2^(L/2) (L/2)^(L/2-1) at those L')
print('  - w_cov * 2^(L/2) (L/2)^(L/2-1) integral for all L (2..22)')
print('  - c(L) = w e^(L/2) monotone increasing, in (1.35, sqrt(2)) for all L')
