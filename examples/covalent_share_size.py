"""Covalent (n_d = 0) Chirgwin-Coulson share of the closed-shell Hueckel
determinant, at the orthogonal-AO point (s = 0), versus system size.

The covalent share of a single closed-shell determinant needs no FCI: expand
the determinant of the k = N/2 occupied Hueckel MOs in the localized AO basis;
a one-electron-per-site (covalent) configuration assigns 5... k AOs to alpha and
the complementary k AOs to beta, so its coefficient is det(C_alpha) * det(C_beta)
with C the N x k occupied-MO coefficient matrix. At s = 0 the covalent share is

    w_cov = sum_{A: k-subset} [ det C[A,:] * det C[complement,:] ]^2 ,

a sum of squared minors (Cauchy-Binet makes the full norm 1). This reproduces
benzene's exact 5/72 and extends the trend the benzene section quotes.

Run from the repo root:  PYTHONPATH=. python3 examples/covalent_share_size.py
"""
import numpy as np
from itertools import combinations


def covalent_share(adjacency, n_electrons):
    A = np.array(adjacency, float)
    n = A.shape[0]
    k = n_electrons // 2                       # doubly-occupied MOs at half filling
    w, V = np.linalg.eigh(A)                    # Hueckel H = -A  ->  occupied = largest-A eigvecs
    C = V[:, np.argsort(w)[::-1][:k]]          # n x k occupied MO coefficients (orthonormal)
    cov = 0.0
    for Aset in combinations(range(n), k):
        Bset = [s for s in range(n) if s not in Aset]   # complement = beta sites, one e- per site
        c = np.linalg.det(C[list(Aset), :]) * np.linalg.det(C[Bset, :])
        cov += c * c
    return cov


def ring(n):
    return [[1 if (abs(i - j) % n) in (1, n - 1) else 0 for j in range(n)] for i in range(n)]


def from_edges(n, edges):
    M = [[0] * n for _ in range(n)]
    for a, b in edges:
        M[a][b] = M[b][a] = 1
    return M


# H2 (one bond): exactly 1/2
h2 = covalent_share([[0, 1], [1, 0]], 2)
assert abs(h2 - 0.5) < 1e-12, h2

# benzene (6-ring): exactly 5/72
benz = covalent_share(ring(6), 6)
assert abs(benz - 5 / 72) < 1e-10, (benz, 5 / 72)

# naphthalene (10 C, 11 bonds; the two fusion carbons have degree 3)
nap_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
             (6, 7), (7, 8), (8, 9), (9, 0), (4, 9)]
nap = covalent_share(from_edges(10, nap_edges), 10)

print("Covalent (n_d=0) share of the closed-shell Hueckel determinant, s = 0:")
print(f"  H2           (2 centers)  : {h2:.6f}   (= 1/2, exact)")
print(f"  benzene      (6 centers)  : {benz:.6f}   (= 5/72 = {5/72:.6f}, exact)")
print(f"  naphthalene  (10 centers) : {nap:.6f}   (= {nap*100:.2f}%)")
print("  fullerene C60 (60 centers): not enumerable (C(60,30) ~ 1.2e17 covalent")
print("                              partitions); vanishingly small by the trend.")
