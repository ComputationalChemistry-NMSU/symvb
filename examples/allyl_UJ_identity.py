"""Verify whether |psi_Huckel> = |psi_1^2 psi_2^2> is literally an
eigenstate of the 9-dim CI Hamiltonian along U=J (K=M=0, s=0, h=-1).

Strategy:  at several (U=J) values, diagonalize H numerically and check:
(1) what the actual numerical GS vector is (compare against the
    expansion of |psi_1^2 psi_2^2>);
(2) whether the GS energy as a function of U=J is linear (which would be
    the signature of a single-config eigenstate with no correlation
    admixture).
"""
import os, sys
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, hamiltonian
from symvb.fixed_psi import generate_dets
from symvb.huckel import solve
from symvb.mo_projection import mo_determinant_in_ao

m = Molecule(
    zero_ii=True, interacting_orbs=['ab', 'bc'],
    subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
    subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
              'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)
P = generate_dets(2, 2, 3)
det_strings = [p.dets[0].det_string for p in P]
H_full, _ = hamiltonian(m, P)
H_full = sp.Matrix(H_full)
h, s, U, J, K, M = sp.symbols('h s U J K M')
H_full = H_full.subs({s: 0, h: -1, K: 0, M: 0})
Hline_fn = sp.lambdify((U, J), H_full, 'numpy')

# Strategy: diagonalize H(U=J) numerically at several values and see if
# GS energy is exactly linear in U.  If yes -> Huckel eigenstate.
# If no -> there's correlation admixture.
print("=== Numerical diagonalization of full 9x9 H at U=J (K=M=0) ===")
print(f"  {'U':>5s}  {'E_GS':>11s}  {'E_1st':>11s}  {'gap':>8s}  "
      f"{'E_GS - (aU+b)':>14s}")

Us = np.array([0, 0.5, 1, 2, 4, 8, 16])
E_gs = []
for Uv in Us:
    Hn = np.array(Hline_fn(Uv, Uv), dtype=float)
    Hn = 0.5*(Hn+Hn.T)
    ev, _ = np.linalg.eigh(Hn)
    E_gs.append(ev[0])
    print(f"  {Uv:>5.1f}  {ev[0]:>11.6f}  {ev[1]:>11.6f}  {ev[1]-ev[0]:>8.4f}")

E_gs = np.array(E_gs)
# Fit E_GS = a*U + b
a, b = np.polyfit(Us, E_gs, 1)
print(f"\n  Linear fit:  E_GS(U=J, U) = {a:.6f} * U + {b:.6f}")
print(f"  Residuals (E_GS - a*U - b):")
for Uv, E in zip(Us, E_gs):
    resid = E - a*Uv - b
    print(f"    U={Uv:>5.1f}: {resid:+.2e}")

# Theoretical Huckel energy: -2*sqrt(2) + (11/8 + ?)*U at 1st order
# If E_GS is linear -> Huckel IS an eigenstate
# If not -> NOT an eigenstate

# Second test: project the GS vector onto the Huckel det expansion
# using the explicit Huckel vector computed the right way.

# Build |psi_Huckel> = |psi_1^2 psi_2^2> in the 9-det AO basis. mo_determinant_in_ao
# expands the closed-shell MO determinant (doubly-occupied MOs 0, 1 of the 3-chain
# Huckel solution) into AO determinants with the correct fermion signs; normalise it.
hr = solve(sp.Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), site_labels='abc')
psi_symvb = np.array(mo_determinant_in_ao(
    hr.coefficients, ([0, 1], [0, 1]), det_strings, site_labels='abc'),
    dtype=float).ravel()
psi_symvb = psi_symvb / np.linalg.norm(psi_symvb)

print(f"\n  |psi_Huckel| = {np.linalg.norm(psi_symvb):.6f} (should be 1)")

# Check at U=0
Hn = np.array(Hline_fn(0, 0), dtype=float); Hn = 0.5*(Hn+Hn.T)
E_expect = psi_symvb @ Hn @ psi_symvb
print(f"  <psi_Huckel|H_0|psi_Huckel> = {E_expect:.6f}  "
      f"(should be -2*sqrt(2) = {-2*np.sqrt(2):.6f})")

# Now overlap with GS at various U=J
print(f"\n  Overlap of numerical GS with |psi_Huckel> along U=J axis:")
print(f"  {'U':>5s}  {'<GS|psi_H>':>12s}  {'|<GS|psi_H>|^2':>14s}")
for Uv in [0, 0.5, 1, 2, 4, 8, 16]:
    Hn = np.array(Hline_fn(Uv, Uv), dtype=float); Hn = 0.5*(Hn+Hn.T)
    ev, vec = np.linalg.eigh(Hn)
    c_gs = vec[:, 0]
    ovlp = psi_symvb @ c_gs
    print(f"  {Uv:>5.1f}  {ovlp:>12.8f}  {ovlp**2:>14.10f}")
