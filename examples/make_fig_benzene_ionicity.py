"""Benzene ionicity-class figure for manuscript_v2 Figure 3.

Chirgwin-Coulson weights of the four ionicity classes n_d in {0,1,2,3} in
the benzene FCI ground state versus U/|h| (log axis), at s = 0, h = -1.
Validates against the closed-shell result (5, 31, 31, 5)/72 at U = 0
(manuscript Eq. 12) before writing the figure.

Output: ../vbt-3/figures/fig_benzene_ionicity.{png,pdf}, drawn at ACS
single-column width (3.33 in) with all text in 7-9 pt.

Run from the repo root: PYTHONPATH=. python3 examples/make_fig_benzene_ionicity.py
"""
import os
import pickle
import sys

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb.fixed_psi import generate_dets

CACHE = '/tmp/benzene_hubbard_matrices.pkl'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', '..', 'vbt-3', 'figures')

h_, s_, U_ = sp.symbols('h s U')

with open(CACHE, 'rb') as fh:
    H1_sym, S_sym, H2_sym = pickle.load(fh)

# numeric pieces at h = -1, s = 0; H2 is linear in U
H0 = np.array(H1_sym.subs({h_: -1, s_: 0}).tolist(), dtype=float)
MU = np.array(sp.diff(H2_sym, U_).subs({s_: 0}).tolist(), dtype=float)

# ionicity class of each determinant (number of doubly occupied sites)
dets = [p.dets[0].det_string for p in generate_dets(3, 3, 6)]
def n_d(ds):
    low = {c for c in ds if c.islower()}
    return len([c for c in low if c.upper() in ds])
classes = np.array([n_d(ds) for ds in dets])
assert len(dets) == 400 and sorted(set(classes)) == [0, 1, 2, 3]

def class_weights(U):
    E, V = np.linalg.eigh((H0 + U * MU + (H0 + U * MU).T) / 2)
    c = V[:, 0]
    return np.array([np.sum(c[classes == k] ** 2) for k in range(4)])

# validation: U = 0 must reproduce (5, 31, 31, 5)/72
w0 = class_weights(0.0)
ref = np.array([5, 31, 31, 5]) / 72
assert np.allclose(w0, ref, atol=1e-10), (w0, ref)
print('U = 0 validation against Eq. (12): OK', w0)

Us = np.logspace(-2, 4, 121)
W = np.array([class_weights(u) for u in Us])

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'font.family': 'sans-serif',
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, ax = plt.subplots(figsize=(3.33, 2.5))
labels = [r'$n_d = 0$ (covalent)', r'$n_d = 1$', r'$n_d = 2$',
          r'$n_d = 3$ (max. ionic)']
styles = ['-', '--', '-.', ':']
for k in range(4):
    ax.semilogx(Us, W[:, k], styles[k], lw=1.2, label=labels[k])
for x, lab in [(4.4, 'PPP')]:
    ax.axvline(x, color='0.6', lw=0.7, ls=(0, (2, 2)))
    ax.text(x * 1.15, 0.93, lab, fontsize=7, color='0.35')
ax.set_xlabel(r'$U/|h|$')
ax.set_ylabel('Chirgwin–Coulson class weight')
ax.set_ylim(0, 1.0)
ax.set_xlim(Us[0], Us[-1])
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.97))
fig.tight_layout(pad=0.3)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'fig_benzene_ionicity.{ext}'),
                dpi=600 if ext == 'png' else None)
print('written:', os.path.join(OUT, 'fig_benzene_ionicity.png'))
# headline values for the caption/prose cross-check
for u in (1.0, 2.0, 4.0, 8.0, 16.0):
    print(f'U/|h| = {u:>4}: ' + '  '.join(f'w{k}={v:.3f}'
          for k, v in enumerate(class_weights(u))))
