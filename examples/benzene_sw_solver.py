"""4th-order Schrieffer-Wolff for benzene Hubbard, A_1g singlet sector.

Loads the partitioned 14-dim block (P + Q_1 + Q_2 + Q_3) built by
examples/benzene_sw_blocks.py and computes the Bloch effective
Hamiltonian H_eff = P V Omega P, with Omega the wave operator solved
order by order from the Bloch equation

    [Omega, H_0] P = (V Omega - Omega P V Omega) P

H_0 = U * N_d (scalar on each Q_k block, vanishes on P) gives a scalar
resolvent G = -Q / (U * N_d), so each order is a clean rational in
(h, U).  Numerically, we set h = 1 and U = 1 and read off

    H_eff^(n) (h, U) = (h^n / U^{n-1}) * H_eff^(n) (1, 1)

so the *eigenvalues* of the n-th order H_eff (computed at h=U=1) are the
rational SW coefficients c_alpha^{(n)} that multiply h^n / U^{n-1}.

Reports H_eff^(n) eigenvalues on P_A_1g for n = 2, 3, 4 and compares
n=2 to the textbook J_1 = 4 t^2 / U  (matched against the value
extracted from a 400-dim FCI fit in benzene_heisenberg_mapping.py).
"""
import os
import sys
import pickle

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


CACHE = '/tmp/benzene_sw_blocks.pkl'
if not os.path.exists(CACHE):
    raise RuntimeError(f"Missing {CACHE}; "
                       f"run examples/benzene_sw_blocks.py first.")

with open(CACHE, 'rb') as f:
    blk = pickle.load(f)

K = blk['K14']                         # 14x14 numpy float, V = h * K
Nd_eigs = blk['Nd_eigs']               # length-14 int array
slices = blk['slices']                 # {0: P_idx, 1: Q1_idx, 2: Q2_idx, 3: Q3_idx}

DIM = K.shape[0]
P_idx = slices[0]
P_size = len(P_idx)

print(f"Loaded SW blocks: P={len(slices[0])}, Q_1={len(slices[1])}, "
      f"Q_2={len(slices[2])}, Q_3={len(slices[3])}, total={DIM}")


# ------------------------------------------------------------------
# Bloch wave operator and H_eff at h = U = 1.
#   V = K
#   G = -Q diag(1/N_d)  (zero on P)
#   Omega^(n) P = G V Omega^(n-1) P  -  G * sum_{k=1}^{n-1} Omega^(k) (P V Omega^(n-k-1) P)
#   H_eff^(n) = P V Omega^(n-1) P   (note: index convention; V counts as "+1 V")
# ------------------------------------------------------------------
V = K.copy()
G = np.zeros((DIM, DIM))
for i in range(DIM):
    if Nd_eigs[i] > 0:
        G[i, i] = -1.0 / Nd_eigs[i]
P_proj = np.diag([1.0 if Nd_eigs[i] == 0 else 0.0 for i in range(DIM)])

PVP = P_proj @ V @ P_proj
assert np.allclose(PVP, 0, atol=1e-10), "P V P must vanish (V flips N_d on P)"

# Omega^(0)*P = P, Omega^(n)*P stored as 14x14 with image in Q.
W = [P_proj.copy()]
H_eff_PA1g = {}
H_eff_full = {}                               # 2x2 P_A_1g blocks

# Order 2: Omega^(1) = G V P;   H_eff^(2) = P V Omega^(1) P
W.append(G @ V @ W[0])
H_eff_full[2] = P_proj @ V @ W[1] @ P_proj

# Order 3: Omega^(2) = G V Omega^(1)   (subtraction term has factor PVP = 0)
W.append(G @ V @ W[1])
H_eff_full[3] = P_proj @ V @ W[2] @ P_proj

# Order 4: Omega^(3) = G V Omega^(2)  -  G Omega^(1) (P V Omega^(1) P)
PVO1 = P_proj @ V @ W[1] @ P_proj             # = H_eff^(2)
W.append(G @ V @ W[2] - G @ W[1] @ PVO1)
H_eff_full[4] = P_proj @ V @ W[3] @ P_proj

# Restrict to P (rows = cols = P_idx); these are 2x2.
def restrict_PA1g(M):
    return M[np.ix_(P_idx, P_idx)]

for n in (2, 3, 4):
    H_eff_PA1g[n] = restrict_PA1g(H_eff_full[n])


# ------------------------------------------------------------------
# Report eigenvalues, traces, dets - rationalised
# ------------------------------------------------------------------
def nsimp(x, tol=1e-8):
    return sp.nsimplify(float(x), rational=True, tolerance=tol)


print("\n" + "=" * 72)
print("H_eff^(n) on P_A_1g  (n=2,3,4):  values shown are coefficients c_alpha")
print("                                  that multiply  h^n / U^(n-1)")
print("=" * 72)

for n in (2, 3, 4):
    M = H_eff_PA1g[n]
    M = 0.5 * (M + M.T)                 # symmetrise (Bloch H_eff is non-Hermitian
                                        # in general; on the 2-dim P_A_1g block it
                                        # is symmetric here -- verify)
    asym = np.max(np.abs(H_eff_PA1g[n] - H_eff_PA1g[n].T))
    eigvals = np.linalg.eigvalsh(M)
    tr = np.trace(M)
    det = np.linalg.det(M)
    print(f"\nH_eff^({n}):  scaling  h^{n}/U^{n-1}    "
          f"|H - H^T|_inf = {asym:.2e}")
    print(f"  raw eigenvalues   : {eigvals[0]:.10f},  {eigvals[1]:.10f}")
    print(f"  rationalised      : {nsimp(eigvals[0])},  {nsimp(eigvals[1])}")
    print(f"  trace             : {tr:.10f}      (rational: {nsimp(tr)})")
    print(f"  det               : {det:.10f}      (rational: {nsimp(det)})")


# ------------------------------------------------------------------
# Validation: ground-state H_eff^(2) eigenvalue against textbook J_1
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("Validation: H_eff^(2) ground state vs Heisenberg benchmark")
print("=" * 72)
eigs2 = np.linalg.eigvalsh(0.5 * (H_eff_PA1g[2] + H_eff_PA1g[2].T))
gs_coef = eigs2[0]                        # at h=U=1, this is the c_0 / U coefficient
print(f"  H_eff^(2) ground-state coefficient : {gs_coef:+.6f}  *  h^2/U")
print(f"  rational                            : {nsimp(gs_coef)}  *  h^2/U")
print(f"  benzene_heisenberg_mapping.py       : -17.20990  *  h^2/U  "
      f"(extracted from 400-dim FCI)")
print(f"  ratio                               : "
      f"{gs_coef / -17.2099:.6f}   (must be ~1)")

L = 6
sx = np.array([[0, 1], [1, 0]]) / 2
sy = np.array([[0, -1j], [1j, 0]]) / 2
sz = np.array([[1, 0], [0, -1]]) / 2

def s_op(op, site, L=L):
    a = np.array([[1.0]])
    for i in range(L):
        a = np.kron(a, op if i == site else np.eye(2))
    return a

H_hex = sum(np.real(s_op(sz, i) @ s_op(sz, (i+1) % L)
            + 0.5 * (s_op(sx + 1j*sy, i) @ s_op(sx - 1j*sy, (i+1) % L)
                     + s_op(sx - 1j*sy, i) @ s_op(sx + 1j*sy, (i+1) % L)))
            for i in range(L))
E0_hex = np.linalg.eigvalsh(H_hex)[0]
E0_SW  = E0_hex - L / 4.0                 # Schrieffer-Wolff -1/4 per bond
print(f"\n  Heisenberg hexagon (J_1 = 1) ground state : "
      f"{E0_hex:+.6f}    (bare Sigma S.S)")
print(f"  with SW shift -L/4                         : "
      f"{E0_SW:+.6f}    (full effective Hamiltonian)")
print(f"  predicted gs(2) coefficient                : "
      f"{4 * E0_SW:+.6f}   (= 4 * E_SW / J_1, for J_1 = 4 h^2/U)")


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
out = {
    'H_eff_PA1g': H_eff_PA1g,           # numpy 2x2 floats per order
    'H_eff_full': H_eff_full,           # 14x14 with PA1g block populated
    'P_idx': P_idx, 'slices': slices, 'Nd_eigs': Nd_eigs,
    'V': V, 'G': G, 'P_proj': P_proj, 'W': W,
}
with open('/tmp/benzene_sw_heff.pkl', 'wb') as f:
    pickle.dump(out, f)
print("\nSaved H_eff to /tmp/benzene_sw_heff.pkl")
