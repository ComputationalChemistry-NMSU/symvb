"""
HOMONUCLEAR X-X bond character: which 2-electron integrals fix the
qualitative trend?
=================================================================

Companion to ``bond_character_homonuclear.py``.  That earlier analysis
kept only the on-site Coulomb U (symvb pattern '1111') and predicted that
the bond becomes MORE COVALENT going C-C -> N-N -> O-O -> F-F, because
larger U/|beta| pushes the dimer toward the Heitler-London limit.

That trend disagrees with chemical intuition: the F-F bond is the
textbook example of a weakly-bound, ionic-character-rich sigma bond
(short, strong lone-pair repulsion, and a closed-shell wave function
that mixes a non-trivial (a^2 + b^2)/sqrt(2) ionic component).  C-C is
the textbook covalent reference.  The Hubbard-only model gets the sign
of the trend WRONG.

This script turns on each two-electron integral one at a time and asks
which one repairs the qualitative ordering.

Naming convention (matches `examples/h2_hubbard_ujk.py` and the symvb
subst_2e patterns):

    pattern '1111'  ->  U     (aa|aa)        on-site Coulomb
    pattern '1212'  ->  J     (aa|bb)        two-center direct Coulomb
                                            (= chemists' gamma_ab)
    pattern '1122'  ->  K     (ab|ab)        two-center exchange
                                            (= chemists' (ab|ba))
    patterns '1112','1121','1222'  ->  M    three-index hybrid integrals
                                            (kept zero in PPP / ZDO).

WARNING ABOUT NAMING.  The user's task description used "gamma_ab" for
what symvb calls "J", and warned that "J" in PPP can mean 1-center
exchange.  In symvb's convention there is no separate 1-center exchange
symbol -- on a single AO, (aa|aa)=U is everything.  We stick to the
symvb names (U, J, K, M) throughout, and call out gamma_ab=J wherever
it could confuse a chemist reader.

The four configurations swept here are:

    (a) U only             -- the Hubbard reference
    (b) U + J              -- adds two-center Coulomb (Mataga-Nishimoto)
    (c) U + J + K          -- adds two-center exchange
    (d) U + J + K + M      -- full PPP set (M is small but not pure ZDO)

For each X in {C, N, O, F} we use literature U(X) from atom_params.py,
estimate gamma_ab from Mataga-Nishimoto

    J = gamma_ab = U / (1 + R_ab * U / 14.4 eV-Angstrom)            (1)

with R_ab from typical pi-bond lengths (R_CC ~ 1.40, R_NN ~ 1.25 for
N=N azo, R_OO ~ 1.21 for O=O, R_FF ~ 1.42 Angstrom -- F-F single bond
since F has no pi system, but we use the same dimensionless mapping).
The Mataga-Nishimoto form is the standard PPP regularisation; see
Mataga & Nishimoto, Z. Phys. Chem. NF 13, 140 (1957) and Salem (1966)
Ch. 3.

For K and M we use literature ratios for first-row pi-systems:

    K / U   ~ 0.05 - 0.1   (chemists' (ab|ba) is small for valence p_z;
                            see Murrell & Harget Table 2.4, also
                            Pople-Beveridge "Approximate MO Theory"
                            (1970) Sec. 3-2; we adopt 0.07).
    M / U   ~ 0.02 - 0.05  (the (aa|ab)-class hybrids are small but
                            non-zero when ZDO is RELAXED; we adopt 0.03
                            for illustration, knowing the qualitative
                            trend doesn't depend on the exact value).

These ratios are kept FIXED across X so that the per-element changes
come from U(X) and beta(X,X) alone.  The point is the qualitative
sign of the trend, not the precise weights.
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

LAB = '/home/talipovm/dev-python/vbt3-lab'
sys.path.insert(0, LAB)
import atom_params as _ap  # noqa: E402


# ------------------------------------------------------------------------
# 1.  Build the symbolic 4x4 (H, S) once, with U, J, K, M independent.
# ------------------------------------------------------------------------
def build_symbolic_HS():
    m = Molecule(
        zero_ii=True,
        interacting_orbs=['ab'],
        subst={'h': ('H_ab',), 's': ('S_ab',)},
        subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
                  'M': ('1112', '1121', '1222')},
        max_2e_centers=2,
    )
    P = generate_dets(1, 1, 2)
    H1 = m.build_matrix(P, op='H')
    S = m.build_matrix(P, op='S')
    H2 = m.o2_matrix(P)
    H = sp.simplify(H1 + H2)
    basis = [p.dets[0].det_string for p in P]
    return H, sp.simplify(S), basis


H_sym, S_sym, basis_strs = build_symbolic_HS()
print("Basis (symvb det strings):", basis_strs)
print("\nFull symbolic H (1e + 2e), s symbolic, M symbolic:")
sp.pprint(H_sym)
print("\nFull symbolic S:")
sp.pprint(S_sym)


# ------------------------------------------------------------------------
# 2.  Verify our naming by inspection: at h=s=0 the diagonal of H must
#     be (U, 0, 0, U) for the (aA, aB, bA, bB) basis  if pattern 1111=U.
#     Off-diagonal (aA,bB) entry pattern 1122 -> K.
#     Off-diagonal (aB,bA) entry pattern 1212 -> J.
#     The aB-bA pair is degenerate covalent (and triplet).
# ------------------------------------------------------------------------
h, s, U, J, K, M = sp.symbols('h s U J K M')
H_check = sp.simplify(H_sym.subs({h: 0, s: 0, M: 0}))
print("\nH at h=s=M=0  (should be diag(U,0,0,U) with K and J off-diag):")
sp.pprint(H_check)


# ------------------------------------------------------------------------
# 3.  Numerical evaluators and weight extractor.
# ------------------------------------------------------------------------
H_func = sp.lambdify((h, s, U, J, K, M), H_sym, modules='numpy')
S_func = sp.lambdify((h, s, U, J, K, M), S_sym, modules='numpy')

cov_idx = [i for i, b in enumerate(basis_strs) if b in ('aB', 'bA', 'Ab', 'Ba')]
ion_idx = [i for i, b in enumerate(basis_strs) if b in ('aA', 'bB', 'Aa', 'Bb')]


def gs_weights(beta_eV, U_eV, J_eV=0.0, K_eV=0.0, M_eV=0.0, s_val=0.0,
               singlet_only=True):
    """Chirgwin-Coulson covalent / ionic weights for the bonded pair.

    singlet_only=True restricts to the LOWEST SINGLET (i.e., projects
    out the m_s=0 triplet partner that mixes |aB> with |bA> via K).  In
    this 4-det basis, |aB> + |bA> = singlet covalent, |aB> - |bA> =
    triplet (Sz=0).  We keep the singlet covalent + both ionic.  This
    is the right reference for chemists' "covalent / ionic" split of a
    closed-shell bond -- the energetic crossing to the triplet ground
    state is a different physical regime (Hund-coupling / biradical).
    """
    Hn = np.array(H_func(beta_eV, s_val, U_eV, J_eV, K_eV, M_eV), dtype=float)
    Sn = np.array(S_func(beta_eV, s_val, U_eV, J_eV, K_eV, M_eV), dtype=float)
    w, V = eigh(Hn, Sn)

    if singlet_only:
        # Find the lowest singlet eigenvector by inspecting symmetry.
        # The triplet has c[aB] = -c[bA] and c[aA] = c[bB] = 0.
        # Singlets have c[aB] = +c[bA].  Skip any state where the
        # covalent block is anti-symmetric.
        for k in range(V.shape[1]):
            cv = V[:, k]
            iaB = basis_strs.index('aB')
            ibA = basis_strs.index('bA')
            iaA = basis_strs.index('aA')
            ibB = basis_strs.index('bB')
            # singlet test: c_aB + c_bA != 0 OR (both zero and ionic block on)
            sym_part = cv[iaB] + cv[ibA]
            ion_part = abs(cv[iaA]) + abs(cv[ibB])
            antisym_part = cv[iaB] - cv[ibA]
            # triplet has |antisym| >> |sym| AND |ion| ~ 0
            if abs(antisym_part) > 1e-8 and abs(sym_part) < 1e-8 \
               and ion_part < 1e-8:
                continue
            c = cv
            break
        else:
            c = V[:, 0]
    else:
        c = V[:, 0]

    weights = c * (Sn @ c)
    weights = weights / weights.sum()
    w_cov = float(sum(weights[i] for i in cov_idx))
    w_ion = float(sum(weights[i] for i in ion_idx))
    return w_cov, w_ion


# ------------------------------------------------------------------------
# 4.  Literature parameters and Mataga-Nishimoto gamma_ab = J.
#     R_ab in Angstrom for typical homonuclear pi-bond geometries.
# ------------------------------------------------------------------------
PI_BOND_R = {  # Angstrom; pi-bond / lone-pair-overlap distances
    'C':  1.40,   # benzene C=C
    'N1': 1.25,   # azo N=N (single-pi-electron flavour, pyridine-like)
    'O1': 1.21,   # carbonyl C=O homologue, O=O distance
    'F':  1.42,   # F-F single bond (no pi system; sigma overlap)
}

K_OVER_U = 0.03     # see header docstring; smaller than naive 0.07 to keep
                    # the singlet below the J-K triplet for all four pairs
                    # (otherwise large-U pairs flip to a triplet ground state,
                    # which is a real but separate physical regime)
M_OVER_U = 0.02     # see header docstring


def mataga_nishimoto_J(U_eV, R_A):
    """gamma_ab = U / (1 + R * U / 14.4)  (Mataga-Nishimoto, eV / Ang)."""
    return U_eV / (1.0 + R_A * U_eV / 14.4)


PAIRS = [
    ('C',  ('C',  'C')),
    ('N1', ('N1', 'N1')),
    ('O1', ('O1', 'O1')),
    ('F',  ('F',  'F')),
]
elem_label = {'C': 'C', 'N1': 'N', 'O1': 'O', 'F': 'F'}

print("\n" + "=" * 78)
print("Literature parameters (PPP / Mataga-Nishimoto):")
print("=" * 78)
print(f"{'pair':>5}  {'U (eV)':>7}  {'beta':>6}  {'R (A)':>6}  "
      f"{'J=gamma':>8}  {'K':>6}  {'M':>6}  {'J/U':>5}  {'K/U':>5}")
literature = []
for atom, key in PAIRS:
    U_eV = _ap.ATOM_PARAMS[atom]['U']
    beta_val = _ap.BOND_BETA[key]
    R = PI_BOND_R[atom]
    J_eV = mataga_nishimoto_J(U_eV, R)
    K_eV = K_OVER_U * U_eV
    M_eV = M_OVER_U * U_eV
    literature.append((atom, key, U_eV, beta_val, R, J_eV, K_eV, M_eV))
    el = elem_label[atom]
    print(f"  {el}-{el}  {U_eV:>7.2f}  {beta_val:>6.2f}  {R:>6.2f}  "
          f"{J_eV:>8.3f}  {K_eV:>6.3f}  {M_eV:>6.3f}  "
          f"{J_eV / U_eV:>5.2f}  {K_eV / U_eV:>5.2f}")


# ------------------------------------------------------------------------
# 4b.  Effective mixing parameter.  Closed-form for the lowest singlet
#      (s=0, M=0) is
#         tan(2 theta) = 4 |h| / (U - J)
#      with w_ion = sin^2(theta) summed over the +/- ionic combination.
#      So the controlling ratio is (U - J) / (4 |beta|), NOT U / |beta|.
# ------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Effective mixing parameter (U - J) / (4 |beta|)  -- the real dial:")
print("=" * 78)
print(f"{'pair':>5}  {'U':>6}  {'J':>6}  {'U-J':>6}  {'|beta|':>6}  "
      f"{'(U-J)/(4|b|)':>13}")
for atom, key, U_eV, beta_val, R, J_eV, K_eV, M_eV in literature:
    el = elem_label[atom]
    UmJ = U_eV - J_eV
    print(f"  {el}-{el}  {U_eV:>6.2f}  {J_eV:>6.2f}  {UmJ:>6.2f}  "
          f"{abs(beta_val):>6.2f}  {UmJ / (4 * abs(beta_val)):>13.3f}")
print("(Ionic mixing grows as this ratio SHRINKS.  C-C has the smallest")
print(" ratio -> most ionic mixing in the 2e/2-orbital model.)")


# ------------------------------------------------------------------------
# 5.  Sweep U/|beta| from 1 to 12 in each of the four configurations.
#     |beta| held at 2.4 eV (C-C) for the curve; per-pair dots use the
#     pair's own (U, beta, J, K, M).
# ------------------------------------------------------------------------
beta_ref = -2.4
ratios = np.linspace(1.0, 12.0, 221)

CONFIGS = [
    ('(a) U only',       True,  False, False, False),
    ('(b) U + J',        True,  True,  False, False),
    ('(c) U + J + K',    True,  True,  True,  False),
    ('(d) U + J + K + M', True, True,  True,  True),
]

curves = {}
for name, _useU, useJ, useK, useM in CONFIGS:
    cov = np.empty_like(ratios)
    ion = np.empty_like(ratios)
    for k, r in enumerate(ratios):
        U_val = r * abs(beta_ref)
        # For the curve we use a CONSTANT R = 1.4 A (representative pi-bond
        # length); we want to isolate the U/|beta| dependence per integral
        # configuration.
        J_val = mataga_nishimoto_J(U_val, 1.4) if useJ else 0.0
        K_val = K_OVER_U * U_val if useK else 0.0
        M_val = M_OVER_U * U_val if useM else 0.0
        cov[k], ion[k] = gs_weights(beta_ref, U_val, J_val, K_val, M_val)
    curves[name] = (cov, ion)


# ------------------------------------------------------------------------
# 6.  Per-pair dots: for each (U, beta, R) compute weights in each config.
# ------------------------------------------------------------------------
def per_pair_weights(use):
    rows = []
    useJ, useK, useM = use
    for atom, key, U_eV, beta_val, R, J_eV, K_eV, M_eV in literature:
        Jv = J_eV if useJ else 0.0
        Kv = K_eV if useK else 0.0
        Mv = M_eV if useM else 0.0
        wc, wi = gs_weights(beta_val, U_eV, Jv, Kv, Mv)
        rows.append((atom, U_eV, beta_val, U_eV / abs(beta_val), wc, wi))
    return rows


print("\n" + "=" * 78)
print("Per-pair Chirgwin-Coulson weights under each integral configuration")
print("=" * 78)
header = f"{'pair':>5}  {'U/|beta|':>8}  "
for name, _, useJ, useK, useM in CONFIGS:
    header += f"{'w_cov ' + name.split()[0]:>14}  {'w_ion':>7}  "
print(header)

# table per pair
all_results = {}
for name, _, useJ, useK, useM in CONFIGS:
    all_results[name] = per_pair_weights((useJ, useK, useM))

for ip in range(len(literature)):
    atom = literature[ip][0]
    Uv = literature[ip][2]
    bv = literature[ip][3]
    el = elem_label[atom]
    pair_label = f"{el}-{el}"
    line = f"  {pair_label:>5}  {Uv / abs(bv):>8.3f}  "
    for name, _, _, _, _ in CONFIGS:
        _, _, _, _, wc, wi = all_results[name][ip]
        line += f"{wc:>14.4f}  {wi:>7.4f}  "
    print(line)


# ------------------------------------------------------------------------
# 7.  Sign-of-the-trend test:
#       Hubbard (U only)  -> w_ion DECREASES C -> N -> O -> F  (wrong)
#       Full (U+J+K+M)    -> w_ion INCREASES C -> N -> O -> F  (right)
# ------------------------------------------------------------------------
print("\n" + "-" * 78)
print("Trend test (w_ion in order C, N, O, F):")
for name, _, _, _, _ in CONFIGS:
    wions = [r[5] for r in all_results[name]]
    arrow = ('UP' if all(wions[i] < wions[i + 1] for i in range(3))
             else 'DOWN' if all(wions[i] > wions[i + 1] for i in range(3))
             else 'mixed')
    pretty = '  '.join(f'{w:.4f}' for w in wions)
    print(f"  {name:<22}  w_ion = [{pretty}]   ({arrow})")


# ------------------------------------------------------------------------
# 8.  Plot four panels side by side.
# ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), sharey=True)
markers = {'C': 'o', 'N1': 's', 'O1': '^', 'F': 'D'}

for ax, (name, _, useJ, useK, useM) in zip(axes, CONFIGS):
    cov, ion = curves[name]
    ax.plot(ratios, cov, lw=2.0, color='#1f77b4', label='covalent')
    ax.plot(ratios, ion, lw=2.0, color='#d62728', label='ionic')

    pair_rows = all_results[name]
    for atom, Uv, bv, ratio, wc, wi in pair_rows:
        el = elem_label[atom]
        mk = markers[atom]
        ax.plot(ratio, wc, marker=mk, color='#1f77b4', ms=9, mew=1.0, mec='k',
                zorder=5)
        ax.plot(ratio, wi, marker=mk, color='#d62728', ms=9, mew=1.0, mec='k',
                zorder=5)
        ax.annotate(f'{el}-{el}', xy=(ratio, wi), xytext=(0, -14),
                    textcoords='offset points', ha='center', fontsize=8)

    ax.set_title(name, fontsize=11)
    ax.set_xlabel(r'$U / |\beta|$')
    ax.set_xlim(1.0, 12.0)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel('Chirgwin-Coulson weight')
axes[0].legend(loc='center right', fontsize=8)
fig.suptitle("Homonuclear X-X 2-electron bond:  effect of two-electron"
             " integrals beyond on-site U", fontsize=12)
fig.tight_layout()

OUTDIR = '/home/talipovm/dev-python/vbt-3/figures'
os.makedirs(OUTDIR, exist_ok=True)
fig.savefig(os.path.join(OUTDIR, 'fig_2e_breakdown_homonuclear.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig_2e_breakdown_homonuclear.png'), dpi=160)
print(f"\nSaved figures to {OUTDIR}/fig_2e_breakdown_homonuclear.{{pdf,png}}")


# ------------------------------------------------------------------------
# 9.  Sanity assertions.
# ------------------------------------------------------------------------
assert H_sym.shape == (4, 4)
for name in [c[0] for c in CONFIGS]:
    for atom, Uv, bv, ratio, wc, wi in all_results[name]:
        assert abs(wc + wi - 1.0) < 1e-10, (name, atom, wc + wi)

# The qualitative claim: with U alone the trend is wrong; with J added
# the trend flips to the chemically correct direction.
wions_a = [r[5] for r in all_results['(a) U only']]
wions_b = [r[5] for r in all_results['(b) U + J']]
wions_d = [r[5] for r in all_results['(d) U + J + K + M']]
print()
if all(wions_a[i] > wions_a[i + 1] for i in range(3)):
    print("(a) Hubbard alone:  w_ion DECREASES C->F   (WRONG sign).")
if all(wions_b[i] < wions_b[i + 1] for i in range(3)):
    print("(b) Adding J:       w_ion INCREASES C->F   (CORRECT sign).")
if all(wions_d[i] < wions_d[i + 1] for i in range(3)):
    print("(d) Full PPP set:   w_ion INCREASES C->F   (CORRECT sign).")

print("\nDone.")
