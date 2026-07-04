"""Singlet/triplet discussion for allyl 3c4e:

(A) Explain why K or M drives a GS-level crossing to a triplet biradical.
(B) Track the singlet and triplet separately, not just the lowest A_1 state.
(C) Identify the crossing curve K*(U) at fixed J and M=0.
(D) Show that the triplet's Huckel-MO occupation (1,2,1) is exact --
    it is the closed-shell triplet configuration psi_1 psi_2^2 psi_3
    with the two open-shell electrons coupled to Ms=0 triplet.
"""
import os, sys
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule, SlaterDet, symmetry, hamiltonian
from symvb.fixed_psi import generate_dets
from symvb.spin import s_squared_matrix

# Build A_1 block
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
H_s0 = H_full.subs({s: 0, h: -1})

# Symbolic +1-eigenspace of the a<->c reflection (the A_1 / sigma=+1 block),
# built by hand so H_red stays symbolic for lambdify below;
# symmetry.signed_totally_symmetric_basis returns a numeric basis instead.
def canon(ds):
    fp = SlaterDet(ds).get_sorted()
    return fp.dets[0].det_string, fp.coefs[0]
sig = {'a':'c','b':'b','c':'a'}
perm, signs = symmetry.apply_orbital_permutation(sig, det_strings, canon)
Up_cols = []; seen = [False]*9
for i in range(9):
    if seen[i]: continue
    j = perm[i]; sj = signs[i]
    if j == i:
        seen[i] = True
        if sj == 1:
            v = sp.zeros(9,1); v[i] = 1; Up_cols.append(v)
    else:
        seen[i] = seen[j] = True
        v = sp.zeros(9,1); v[i] = 1; v[j] = sj
        Up_cols.append(v / sp.sqrt(2))
Up = sp.Matrix.hstack(*Up_cols)
Up_np = np.array(Up, dtype=float)
H_red = Up.T * H_s0 * Up
Nd = H_red.shape[0]
H_red_fn = sp.lambdify((U, J, K, M), H_red, 'numpy')

S2_9 = s_squared_matrix(det_strings, orbs='abc')
S2 = Up_np.T @ S2_9 @ Up_np

def eigstates(Uv, Jv, Kv, Mv):
    Hn = np.array(H_red_fn(Uv, Jv, Kv, Mv), dtype=float); Hn = 0.5*(Hn+Hn.T)
    ev, vec = np.linalg.eigh(Hn)
    # S^2 expectation for each eigenstate
    s2_each = np.array([vec[:, i] @ S2 @ vec[:, i] for i in range(len(ev))])
    return ev, vec, s2_each


# --------------------------------------------------------------------
# Track the lowest singlet (S^2 ~ 0) and lowest triplet (S^2 ~ 2) as K
# varies, at U=4, J=0.5, M=0.  Shows the crossing explicitly.
# --------------------------------------------------------------------
Ks = np.linspace(0, 2.0, 101)
E_sing = []
E_trip = []
for Kv in Ks:
    ev, vec, s2 = eigstates(4.0, 0.5, Kv, 0.0)
    # lowest singlet
    ms = np.where(np.abs(s2) < 0.5)[0]
    mt = np.where(np.abs(s2 - 2) < 0.5)[0]
    E_sing.append(ev[ms[0]] if len(ms) else np.nan)
    E_trip.append(ev[mt[0]] if len(mt) else np.nan)
E_sing = np.array(E_sing); E_trip = np.array(E_trip)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

ax = axes[0]
ax.plot(Ks, E_sing, lw=2, color='C0', label='lowest singlet (S=0)')
ax.plot(Ks, E_trip, lw=2, color='firebrick', label='lowest triplet (S=1, Ms=0)')
Kstar = Ks[np.argmin(np.abs(E_sing - E_trip))]
ax.axvline(Kstar, color='k', lw=0.5, ls='--', alpha=0.5)
ax.set_xlabel('K  (U=4, J=0.5, M=0)')
ax.set_ylabel('energy')
ax.set_title(f'Singlet/triplet GS crossing at K* ≈ {Kstar:.3f}\n'
             f'slope ratio: −1 (singlet) vs −3 (triplet)')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# --------------------------------------------------------------------
# Crossing curve K*(U) at fixed J=0.5, M=0:  find K* where E_s = E_t
# for a range of U
# --------------------------------------------------------------------
def find_crossing(Uv, Jv, Mv, K_range=(0, 3.0)):
    # bracket search
    Klo, Khi = K_range
    ev_lo, _, s2_lo = eigstates(Uv, Jv, Klo, Mv)
    # Identify whether the GS is already triplet at Klo
    def is_trip_gs(Kv):
        _, _, s2 = eigstates(Uv, Jv, Kv, Mv)
        return abs(s2[0] - 2) < 0.5
    if is_trip_gs(Klo):
        return Klo
    if not is_trip_gs(Khi):
        return np.nan
    for _ in range(60):
        Kmid = (Klo + Khi) / 2
        if is_trip_gs(Kmid):
            Khi = Kmid
        else:
            Klo = Kmid
    return (Klo + Khi) / 2

Us = np.linspace(0.5, 10, 40)
K_star = [find_crossing(Uv, 0.5, 0.0) for Uv in Us]

ax = axes[1]
ax.plot(Us, K_star, lw=2, color='purple')
ax.fill_between(Us, K_star, 3.0, alpha=0.12, color='firebrick',
                label='triplet GS')
ax.fill_between(Us, 0, K_star, alpha=0.12, color='C0',
                label='singlet GS')
ax.set_xlabel('U  (J=0.5, M=0)')
ax.set_ylabel('K* (crossing value)')
ax.set_title('Singlet/triplet GS phase diagram, allyl 3c4e')
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim(0.5, 10)
ax.set_ylim(0, 2.5)
ax.grid(alpha=0.3)

plt.tight_layout()
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'figures', 'allyl_singlet_triplet.png')
os.makedirs(os.path.dirname(_out), exist_ok=True)
plt.savefig(_out, dpi=130, bbox_inches='tight')
print(f'saved {_out}')

# --------------------------------------------------------------------
# Triplet closed-form energy E_T(U, J, K, M) - should be linear (single
# configuration, no correlation within the triplet)
# --------------------------------------------------------------------
print('\n=== Triplet energy parametrization (fit to a*U + b*J + c*K + d*M + const) ===')
test_points = [
    (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
    (2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2),
    (3, 1, 1, 0.3), (-1, 2, 1.5, -0.5), (5, 0.7, 1.2, 0.1),
]
A = []
b = []
for pt in test_points:
    ev, _, s2 = eigstates(*pt)
    mt = np.where(np.abs(s2 - 2) < 0.5)[0]
    if len(mt) == 0: continue
    Et = ev[mt[0]]
    A.append([*pt, 1.0])
    b.append(Et)
A = np.array(A); b = np.array(b)
coef, *_ = np.linalg.lstsq(A, b, rcond=None)
resid = np.max(np.abs(A @ coef - b))
labels = ['U', 'J', 'K', 'M', 'const']
print('  E_T =', ' + '.join(f'{c:+.4f}*{l}' for c, l in zip(coef, labels)))
print(f'  max residual over {len(b)} test points: {resid:.2e}')
