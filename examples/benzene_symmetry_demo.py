"""
Automatic symmetry handling for the benzene FCI, two strategies.

(A) Numerical eigenanalysis (symvb.symmetry.degenerate_block_basis).
    One diagonalisation exposes the irrep structure through degeneracy
    multiplicities. No group theory, no external dependency.

(B) Orbital permutation -> induced basis permutation -> trivial-irrep basis
    (symvb.symmetry.apply_orbital_permutation / generate_group /
    totally_symmetric_basis). Supply the molecular symmetry at orbital
    level, the module lifts it to the determinant basis, closes the group,
    and returns the orbit-sum subspace that contains the ground state.
    The reduced subspace is ~10x smaller than the full FCI for benzene and
    diagonalises in milliseconds.

Also available (not exercised here): symvb.symmetry.detect_permutation_group,
which uses pynauty to discover the symmetry group from H and S alone. Works
well for small matrices (<~50 dimensions); the aux-vertex edge-colouring
encoding makes it impractical for the 400-dim benzene CI.
"""
import time
import numpy as np
import sympy as sp
from collections import Counter
from scipy.linalg import eigh

from symvb import Molecule, SlaterDet, symmetry


# --- Build benzene FCI (3 alpha + 3 beta electrons, 6 orbitals) ---------
m = Molecule(
    zero_ii=True,
    subst={'s': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af'),
           'h': ('H_ab', 'H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af')},
    interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'],
)
m.generate_basis(3, 3, 6)
det_strings = [fp.dets[0].det_string for fp in m.basis]

print("Building symbolic 400x400 H, S ...")
t0 = time.time()
H_sym = m.build_matrix(m.basis, op='H')
S_sym = m.build_matrix(m.basis, op='S')
print(f"  done in {time.time()-t0:.1f}s")

h, s = sp.symbols('h s')
H_num = np.array(H_sym.subs({h: -1, s: 0.2}).tolist(), dtype=float)
S_num = np.array(S_sym.subs({h: -1, s: 0.2}).tolist(), dtype=float)


# =====  (A) Numerical eigenanalysis  ======================================
print("\n(A) Degenerate-block analysis at (h=-1, s=0.2)")
t0 = time.time()
evals, evecs, blocks = symmetry.degenerate_block_basis(H_num, S_num, tol=1e-6)
print(f"    {time.time()-t0:.2f}s   {len(blocks)} distinct eigenvalues")
hist = Counter(len(b[1]) for b in blocks)
print("    degeneracy histogram (dim : count): "
      + ", ".join(f"{d}x:{c}" for d, c in sorted(hist.items())))
print(f"    ground-state energy = {evals[0]:.6f}   (deg {len(blocks[0][1])})")


# =====  (B) Orbital-permutation -> induced basis -> reduced block  ========
print("\n(B) Symmetry from orbital generators (C_6 and sigma_v)")

def canon(det_string):
    fp = SlaterDet(det_string).get_sorted()
    return fp.dets[0].det_string, fp.coefs[0]

C6     = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'e', 'e': 'f', 'f': 'a'}
sigma  = {'a': 'a', 'b': 'f', 'c': 'e', 'd': 'd', 'e': 'c', 'f': 'b'}

perms = []
for name, om in [('C_6', C6), ('sigma_v', sigma)]:
    perm, signs = symmetry.apply_orbital_permutation(om, det_strings, canon)
    assert perm is not None, f"{name} does not act on this basis"
    perms.append(perm)
    P = np.zeros_like(H_num)
    for a, b in enumerate(perm):
        P[a, b] = 1.0
    err = np.max(np.abs(P @ H_num @ P.T - H_num))
    print(f"    {name}: induced basis permutation verified "
          f"(||P H P.T - H||_inf = {err:.1e})")

group = symmetry.generate_group(perms, N=400)
print(f"    group order: {len(group)}")

U, orbits = symmetry.totally_symmetric_basis(perms, 400)
print(f"    trivial-irrep dim: {U.shape[1]} (orbit sizes: "
      f"{dict(Counter(len(o) for o in orbits))})")

t0 = time.time()
H_red = U.T @ H_num @ U
S_red = U.T @ S_num @ U
E_red = eigh(H_red, S_red)[0][0]
dt = time.time() - t0
print(f"    reduced eigh: {E_red:.6f}   ({dt*1000:.1f} ms)")
print(f"    matches full FCI within {abs(E_red - evals[0]):.1e}")
