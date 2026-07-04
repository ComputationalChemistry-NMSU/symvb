"""
Bond character (covalent <-> ionic) of a 2-electron pi-bond
in a HOMONUCLEAR X-X bond, X varying across the first row C, N, O, F.

Companion piece to ``bond_character_CNOF.py`` (heteronuclear C-X).  The
contrast is the point of the analysis:

  * In the heteronuclear C-X case, the dial is the on-site asymmetry
    Delta_alpha = alpha_X - alpha_C; ionic character grows monotonically
    with |Delta_alpha|/|beta| as the EN heteroatom localises both
    electrons on its own site.

  * In the homonuclear X-X case Delta_alpha = 0 by symmetry; the only
    knob that varies as we go C -> N -> O -> F is the ratio U/|beta|.
    Larger U/|beta| (more correlation, weaker hopping) penalises double
    occupation, so COVALENT character grows monotonically C -> F -- the
    OPPOSITE direction of the heteronuclear story.

The symmetric 2-orbital, 2-electron model (Sz=0 sector, 4 dets):

        |aA>   double occupation on atom 1  (ionic)
        |aB>   alpha on 1, beta on 2        (covalent)
        |bA>   alpha on 2, beta on 1        (covalent)
        |bB>   double occupation on atom 2  (ionic)

is built symbolically by symvb with ``zero_ii=False`` so we can later
substitute alpha_a = alpha_b = 0 (any constant works -- the ground-state
weights of a symmetric Hamiltonian are translation-invariant in the
on-site energy).  We sweep U/|beta| and place dots for C-C, N1-N1,
O1-O1, F-F at their literature parameter ratios.
"""
import os
import sys
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule
from symvb.fixed_psi import generate_dets

# Pull literature U and beta values from vbt3-lab/atom_params.py.
LAB = '/home/talipovm/dev-python/vbt3-lab'
sys.path.insert(0, LAB)
import atom_params as _ap  # type: ignore  # noqa: E402


# ------------------------------------------------------------------------
# 1.  Build the symbolic 4x4 (H1, S, H2) once via symvb.
#     zero_ii=False keeps H_aa, H_bb as free symbols (we set them equal
#     when we substitute -- alpha_a = alpha_b for the symmetric bond).
#     max_2e_centers=1 keeps only the on-site Coulomb integrals
#     T_aaaa = U on atom 1 and T_bbbb = U on atom 2.
# ------------------------------------------------------------------------
def build_symbolic_HS():
    """Build (H, S) symbolically.  Returns (H_sym, S_sym, basis_strs)."""
    m = Molecule(
        zero_ii=False,
        interacting_orbs=['ab'],
        subst={'h_ab': ('H_ab',), 's': ('S_ab',)},
        max_2e_centers=1,
    )
    P = generate_dets(1, 1, 2)                       # 1 alpha, 1 beta, 2 AOs
    H1 = m.build_matrix(P, op='H')
    S = m.build_matrix(P, op='S')
    H2 = m.o2_matrix(P)
    H_sym = sp.simplify(H1 + H2)
    S_sym = sp.simplify(S)
    basis_strs = [p.dets[0].det_string for p in P]
    return H_sym, S_sym, basis_strs


def make_evaluators(H_sym, S_sym):
    """Return (H_fn, S_fn, idx_aA, idx_bB, cov_idx, ion_idx, basis_strs).

    H_fn / S_fn take (H_aa, H_bb, h_ab, s, U_a, U_b) and return numpy
    arrays.  All other helpers are derived from the basis ordering.
    """
    H_aa, H_bb, h_ab, s = sp.symbols('H_aa H_bb h_ab s')
    U_a, U_b = sp.symbols('T_aaaa T_bbbb')
    syms = (H_aa, H_bb, h_ab, s, U_a, U_b)
    H_fn = sp.lambdify(syms, H_sym, modules='numpy')
    S_fn = sp.lambdify(syms, S_sym, modules='numpy')
    return H_fn, S_fn


def ground_state(H_fn, S_fn, basis_strs, *,
                 alpha=0.0, beta_eV=-2.4, U_eV=10.0, s_val=0.0):
    """Ground state of the HOMONUCLEAR X-X 2e model.

    For a symmetric bond alpha_a = alpha_b = alpha (free constant; weights
    do not depend on its value).  Returns (E_gs, weights_array).
    """
    Hn = np.array(H_fn(alpha, alpha, beta_eV, s_val, U_eV, U_eV),
                  dtype=float)
    Sn = np.array(S_fn(alpha, alpha, beta_eV, s_val, U_eV, U_eV),
                  dtype=float)
    w, V = eigh(Hn, Sn)
    c = V[:, 0]
    Sc = Sn @ c
    weights = c * Sc
    weights = weights / weights.sum()        # numerical safety
    return w[0], weights


def split_weights(weights, basis_strs):
    """Aggregate covalent and ionic Chirgwin-Coulson weights."""
    cov_idx = [i for i, b in enumerate(basis_strs)
               if b in ('aB', 'bA', 'Ab', 'Ba')]
    ion_idx = [i for i, b in enumerate(basis_strs)
               if b in ('aA', 'bB', 'Aa', 'Bb')]
    w_cov = sum(weights[i] for i in cov_idx)
    w_ion = sum(weights[i] for i in ion_idx)
    return w_cov, w_ion


# ------------------------------------------------------------------------
# 2.  Build the model.
# ------------------------------------------------------------------------
H_sym, S_sym, basis_strs = build_symbolic_HS()
print("Basis (symvb det strings):", basis_strs)
print("\nS =")
sp.pprint(S_sym)
print("\nH (1e + 2e) =")
sp.pprint(H_sym)

H_fn, S_fn = make_evaluators(H_sym, S_sym)


# ------------------------------------------------------------------------
# 3.  Pick the homonuclear pairs to mark on the figure.  Use the
#     "singly-occupied pi" flavour for each element (apples-to-apples
#     trend across C, N, O, F): C, N1 (pyridine), O1 (carbonyl), F.
# ------------------------------------------------------------------------
PAIRS = [
    # (atom_label, BOND_BETA_key)
    ('C',  ('C',  'C')),
    ('N1', ('N1', 'N1')),
    ('O1', ('O1', 'O1')),
    ('F',  ('F',  'F')),
]

# Sanity-check that all required BOND_BETA keys are present (and flag
# missing ones).  We also keep the un-flavoured fallback for diagnostics.
literature = []
for atom, key in PAIRS:
    U_eV = _ap.ATOM_PARAMS[atom]['U']
    if key in _ap.BOND_BETA:
        beta_val = _ap.BOND_BETA[key]
    else:
        # Fallback: should never trigger after the atom_params update,
        # but guard against environments where the key is still absent.
        fallback = (atom.rstrip('12'), atom.rstrip('12'))
        beta_val = _ap.BOND_BETA.get(fallback, -2.0)
        print(f"  [warn] BOND_BETA[{key}] missing; "
              f"falling back to {fallback} = {beta_val:.2f} eV")
    ratio = U_eV / abs(beta_val)
    literature.append((atom, key, U_eV, beta_val, ratio))


# ------------------------------------------------------------------------
# 4.  Sweep U/|beta| from 0 to 20.  We hold |beta| fixed at the C-C
#     value (2.4 eV) and walk U from 0 to 20 * |beta|.  Because the
#     ground-state weights of a symmetric two-orbital Hubbard dimer
#     depend only on the dimensionless ratio U/|beta| (and on s), the
#     curve is identical to one obtained at any other fixed |beta|.
# ------------------------------------------------------------------------
beta_ref = -2.4   # only the magnitude matters for the trend
ratios = np.linspace(0.0, 20.0, 401)
cov_curve = np.empty_like(ratios)
ion_curve = np.empty_like(ratios)
for k, r in enumerate(ratios):
    U_val = r * abs(beta_ref)
    _, w = ground_state(H_fn, S_fn, basis_strs,
                        alpha=0.0, beta_eV=beta_ref, U_eV=U_val)
    cov_curve[k], ion_curve[k] = split_weights(w, basis_strs)


# ------------------------------------------------------------------------
# 5.  Compute the literature dots.
# ------------------------------------------------------------------------
dot_rows = []
for atom, key, U_eV, beta_val, ratio in literature:
    _, w = ground_state(H_fn, S_fn, basis_strs,
                        alpha=0.0, beta_eV=beta_val, U_eV=U_eV)
    wcov, wion = split_weights(w, basis_strs)
    dot_rows.append((atom, key, U_eV, beta_val, ratio, wcov, wion))


# ------------------------------------------------------------------------
# 6.  Plot.
# ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.plot(ratios, cov_curve, lw=2.0, color='#1f77b4',
        label=r'covalent $w(aB)+w(bA)$')
ax.plot(ratios, ion_curve, lw=2.0, color='#d62728',
        label=r'ionic $w(aA)+w(bB)$')

markers = {'C': 'o', 'N1': 's', 'O1': '^', 'F': 'D'}
elem_label = {'C': 'C', 'N1': 'N', 'O1': 'O', 'F': 'F'}
for atom, key, U_eV, beta_val, ratio, wcov, wion in dot_rows:
    mk = markers.get(atom, 'x')
    ax.plot(ratio, wcov, marker=mk, color='#1f77b4',
            ms=9, mew=1.1, mec='k', zorder=5)
    ax.plot(ratio, wion, marker=mk, color='#d62728',
            ms=9, mew=1.1, mec='k', zorder=5)
    elem = elem_label.get(atom, atom)
    ax.annotate(f'{elem}-{elem}', xy=(ratio, wcov), xytext=(0, 8),
                textcoords='offset points', ha='center', fontsize=9)

ax.set_xlabel(r'$U / |\beta|$  (correlation strength, homonuclear $X$-$X$)')
ax.set_ylabel('Chirgwin-Coulson weight in ground state')
ax.set_title(r'Homonuclear $X$-$X$ 2-electron $\pi$-bond character'
             r'   ($\Delta\alpha = 0$, $s = 0$)')
ax.set_ylim(-0.02, 1.02)
ax.set_xlim(0.0, 20.0)
ax.grid(True, alpha=0.3)
ax.legend(loc='center right', fontsize=9)
fig.tight_layout()

OUTDIR = '/home/talipovm/dev-python/vbt-3/figures'
os.makedirs(OUTDIR, exist_ok=True)
fig.savefig(os.path.join(OUTDIR, 'fig_bond_character_homonuclear.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig_bond_character_homonuclear.png'), dpi=160)
print(f"\nSaved figures to {OUTDIR}/fig_bond_character_homonuclear.{{pdf,png}}")


# ------------------------------------------------------------------------
# 7.  Reference table.
# ------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Homonuclear X-X 2-electron pi-bond  (Delta_alpha = 0, s = 0)")
print("=" * 78)
print(f"{'pair':>6}  {'U (eV)':>7}  {'|beta| (eV)':>11}  {'U/|beta|':>9}  "
      f"{'w_cov':>7}  {'w_ion':>7}")
for atom, key, U_eV, beta_val, ratio, wcov, wion in dot_rows:
    elem = elem_label.get(atom, atom)
    pair = f"{elem}-{elem}"
    chk = wcov + wion
    print(f"  {pair:>4}  {U_eV:>7.2f}  {abs(beta_val):>11.2f}  {ratio:>9.3f}  "
          f"{wcov:>7.4f}  {wion:>7.4f}    sum={chk:.4f}")

# Sanity checks.
assert H_sym.shape == (4, 4), "expected 4x4 H"
assert S_sym.shape == (4, 4), "expected 4x4 S"
for atom, key, U_eV, beta_val, ratio, wcov, wion in dot_rows:
    assert abs(wcov + wion - 1.0) < 1e-10, (atom, wcov + wion)
# Monotonicity check across the literature pairs (C, N, O, F).
wcovs = [r[5] for r in dot_rows]
assert all(wcovs[i] < wcovs[i + 1] for i in range(len(wcovs) - 1)), \
    f"covalent weight should increase C -> N -> O -> F, got {wcovs}"
print("\nAll sanity checks passed: 4x4 H/S, weights sum to 1, "
      "covalent weight monotone-increasing C -> N -> O -> F.")
