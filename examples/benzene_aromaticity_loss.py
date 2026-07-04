"""
Benzene + O3 [3+2] cycloaddition: a conceptual VB picture of aromaticity loss.

Covalent Rumer basis for the benzene 6-pi/6-electron system:
    2 Kekule  + 3 Dewar  = 5 singlet-coupled structures.

Reaction coordinate: lambda in [0, 1] scales the pi-coupling H_ab between
the two carbons that become sp3 in the primary ozonide.
    lambda = 1  <->  aromatic reactant (all six ring couplings equal)
    lambda = 0  <->  pi-bond across a-b fully broken

At each lambda we report:
  E_full         lowest eigenvalue of the 5x5 generalized VB problem
  E(structure)   <Psi_i|H|Psi_i> / <Psi_i|Psi_i>  for each isolated structure
  RE_Kek         E_bestKekule - E_full   (classic Kekule resonance energy)
  RE_cov         E_bestSingle - E_full   (covalent vertical resonance energy)
  w_i            Chirgwin-Coulson weights of each structure in the ground state
"""
import numpy as np
import sympy as sp
from scipy.linalg import eigh

from symvb import Molecule, FixedPsi


# ---------------------------------------------------------------------------
# Benzene pi system.
# Keep H_ab as its own free symbol so we can rescale it along the RC.
# All other ring couplings share the common symbol 'h'; overlaps share 's'.
# ---------------------------------------------------------------------------
m = Molecule(
    zero_ii=True,
    subst={
        's': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af'),
        'h': ('H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af'),
    },
    interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'],
)

# ---------------------------------------------------------------------------
# The 5 non-crossing Rumer pairings of 6 ring atoms a,b,c,d,e,f
# (positions 0..5 in the determinant string).
#
# For each structure we choose a parent determinant string with opposite
# spins at every position pair we intend to couple; couple_orbitals then
# expands it into the 8 (= 2^3) determinants of the singlet-coupled VB
# wavefunction.
# ---------------------------------------------------------------------------
# IMPORTANT: use a SINGLE parent string for every structure. symvb's op_det
# only evaluates matrix elements between Slater determinants that share the
# same spin-pattern string (is_compatible), and couple_orbitals preserves
# the parent's pattern at each position -- mixing parents with different
# patterns produces silently-disconnected basis blocks.
PARENT = 'aBcDeF'  # spin pattern '+-+-+-'
STRUCTURES = [
    ('Kek1', PARENT, [(0, 1), (2, 3), (4, 5)]),  # (a-b)(c-d)(e-f)
    ('Kek2', PARENT, [(0, 5), (1, 2), (3, 4)]),  # (a-f)(b-c)(d-e)
    ('Dew1', PARENT, [(0, 1), (2, 5), (3, 4)]),  # (a-b)(c-f)(d-e)
    ('Dew2', PARENT, [(0, 3), (1, 2), (4, 5)]),  # (a-d)(b-c)(e-f)
    ('Dew3', PARENT, [(0, 5), (1, 4), (2, 3)]),  # (a-f)(b-e)(c-d)
]
names  = [t[0] for t in STRUCTURES]
basis  = [FixedPsi(s, coupled_pairs=list(p)) for _, s, p in STRUCTURES]

print("Rumer basis (5 covalent structures):")
for nm, fp in zip(names, basis):
    print(f"  {nm}: {len(fp.dets)} dets — e.g. {fp.dets[0].det_string}")

# ---------------------------------------------------------------------------
# Build the 5x5 H and S matrices once, symbolically.
# ---------------------------------------------------------------------------
print("\nBuilding symbolic 5x5 H and S ...")
H_sym = m.build_matrix(basis, op='H')
S_sym = m.build_matrix(basis, op='S')

h_sym    = sp.Symbol('h')
s_sym    = sp.Symbol('s')
Hab_sym  = sp.Symbol('H_ab')

# ---------------------------------------------------------------------------
# Numerical parameters (standard Huckel-ish conventions).
#   h  = beta  (negative)
#   s  = overlap between neighbouring pi AOs (positive, small)
# ---------------------------------------------------------------------------
H_VAL = -1.0
S_VAL = 0.2

lambdas = np.linspace(1.0, 0.0, 11)

header = ("  lam    E_full   E_Kek1  E_Kek2   E_Dew1  E_Dew2  E_Dew3  "
          "RE_Kek  RE_cov   w_Kek1  w_Kek2  w_Dew1  w_Dew2  w_Dew3")
print("\n" + header)
print("-" * len(header))

for lam in lambdas:
    subs = {h_sym: H_VAL, s_sym: S_VAL, Hab_sym: lam * H_VAL}
    H = np.array(H_sym.subs(subs).tolist(), dtype=float)
    S = np.array(S_sym.subs(subs).tolist(), dtype=float)

    # Generalized eigenproblem  H c = E S c  (eigh normalizes c^T S c = 1)
    evals, evecs = eigh(H, S)
    E_full = evals[0]
    c0 = evecs[:, 0]

    # Isolated-structure energies (Rayleigh quotient of each basis function)
    E_single = np.array([H[i, i] / S[i, i] for i in range(5)])
    E_Kek_best = E_single[:2].min()
    E_any_best = E_single.min()

    RE_Kek = E_Kek_best - E_full          # > 0 => aromatic stabilization
    RE_cov = E_any_best - E_full

    # Chirgwin-Coulson weights in the non-orthogonal basis:  w_i = c_i * (S c)_i
    w = c0 * (S @ c0)

    print(f"  {lam:4.2f}  {E_full:+6.3f}   {E_single[0]:+5.2f}  {E_single[1]:+5.2f}   "
          f"{E_single[2]:+5.2f}  {E_single[3]:+5.2f}  {E_single[4]:+5.2f}   "
          f"{RE_Kek:+5.3f}  {RE_cov:+5.3f}   "
          + "  ".join(f"{wi:+.3f}" for wi in w))

print("\nConventions: h = beta = -1.0,  s = 0.2.")
print("RE > 0 means the delocalized ground state lies below the best "
      "localized structure (= aromatic stabilization).")
