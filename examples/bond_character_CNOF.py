"""
Bond character (covalent <-> ionic) of a 2-electron pi-bond
as the heteroatom moves across the first row: C - {C, N, O, F}.

Two-orbital, two-electron Heitler-London-style model
-----------------------------------------------------
- orbitals a (C, alpha_a = 0) and b (heteroatom X, alpha_b = -|Delta_alpha|)
- hopping h_ab = -|beta|       (beta < 0, t = |beta|)
- on-site Coulomb U on each centre (equal here; trivial to differentiate)
- intra-pair overlap s (kept symbolic; numerical sweep at s = 0)

Sz = 0 sector -> 4 dets:
        |aA>   double occupation on a   (ionic, on C)
        |aB>   alpha on a, beta on b    (covalent)
        |bA>   alpha on b, beta on a    (covalent)
        |bB>   double occupation on b   (ionic, on X)

The model is built symbolically with symvb (Molecule + FixedPsi), then
diagonalised numerically as a generalized eigenvalue problem H c = E S c.

For the ground state we report Chirgwin-Coulson weights
        w_i = c_i * sum_j c_j S_ij,
which at s = 0 reduce to |c_i|^2.  We aggregate
        w_cov = w(aB) + w(bA),   w_ion = w(aA) + w(bB),
and additionally split the ionic part into a "C-loaded" piece w(aA)
and an "X-loaded" piece w(bB) so that the sign of the on-site
asymmetry (which atom is more electronegative) is visible.
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


# ------------------------------------------------------------------------
# 1.  Build the 4x4 H, S, H2 symbolically via symvb
#     zero_ii=False keeps H_aa and H_bb as independent symbols, so we can
#     impose alpha_a != alpha_b for the heteroatom case.
# ------------------------------------------------------------------------
m = Molecule(
    zero_ii=False,
    interacting_orbs=['ab'],
    subst={'h_ab': ('H_ab',), 's': ('S_ab',)},   # H_aa, H_bb left symbolic
    # subst_2e uses dense-rank keys, so 'aaaa' and 'bbbb' both map to '1111'
    # and would receive a single symbol. We instead leave the 2e integrals
    # as their default names T_aaaa and T_bbbb and substitute numerically.
    max_2e_centers=1,                            # on-site U only
)
P = generate_dets(1, 1, 2)                       # 1 alpha, 1 beta, 2 AOs
print("Basis (symvb det strings):", [p.dets[0].det_string for p in P])

H1 = m.build_matrix(P, op='H')
S  = m.build_matrix(P, op='S')
H2 = m.o2_matrix(P)
H_sym = sp.simplify(H1 + H2)
S_sym = sp.simplify(S)

print("\nS =")
sp.pprint(S_sym)
print("\nH (1e + 2e) =")
sp.pprint(H_sym)


# ------------------------------------------------------------------------
# 2.  Symbolic insight: at s = 0 the covalent {|aB>, |bA>} block is 2x2
# ------------------------------------------------------------------------
H_aa, H_bb, h_ab, s = sp.symbols('H_aa H_bb h_ab s')
U_a, U_b = sp.symbols('T_aaaa T_bbbb')           # default symvb names
H_s0 = H_sym.subs(s, 0)
print("\nH at s = 0 (rows/cols ordered as basis above) =")
sp.pprint(H_s0)

# Identify the covalent rows/cols in the ordered basis
basis_strs = [p.dets[0].det_string for p in P]
cov_idx = [i for i, b in enumerate(basis_strs) if b in ('aB', 'bA', 'Ab', 'Ba')]
ion_idx = [i for i, b in enumerate(basis_strs) if b in ('aA', 'bB', 'Aa', 'Bb')]
print(f"covalent indices = {cov_idx}, ionic indices = {ion_idx}")

# 2x2 covalent (HL) block:
H_cov = sp.Matrix([[H_s0[i, j] for j in cov_idx] for i in cov_idx])
print("\nCovalent 2x2 block (s = 0) =")
sp.pprint(sp.simplify(H_cov))


# ------------------------------------------------------------------------
# 3.  Convert symbolic (H, S) to fast numerical evaluators
# ------------------------------------------------------------------------
syms = (H_aa, H_bb, h_ab, s, U_a, U_b)
H_fn = sp.lambdify(syms, H_sym, modules='numpy')
S_fn = sp.lambdify(syms, S_sym, modules='numpy')


def ground_state(d_alpha, beta_eV=-2.4, U_eV=10.0, s_val=0.0):
    """Diagonalise (H, S) for a given heteroatom on-site offset.

    Inputs in eV.  d_alpha = alpha_b - alpha_a (Delta < 0 if X is more
    electronegative than carbon, by the standard chemists' sign
    convention).
    """
    Hn = np.array(H_fn(0.0, d_alpha, beta_eV, s_val, U_eV, U_eV),
                  dtype=float)
    Sn = np.array(S_fn(0.0, d_alpha, beta_eV, s_val, U_eV, U_eV),
                  dtype=float)
    w, V = eigh(Hn, Sn)
    c = V[:, 0]
    # Chirgwin-Coulson weights:  w_i = c_i * sum_j (S_ij c_j)
    Sc = Sn @ c
    weights = c * Sc
    # numerical safety: weights should sum to 1
    weights = weights / weights.sum()
    return w[0], weights, basis_strs


# ------------------------------------------------------------------------
# 4.  Atom-pair parameters
#     Try to load /home/talipovm/dev-python/vbt3-lab/atom_params.py if a
#     sister agent created it; otherwise use literature-style defaults
#     in units of |beta|.
# ------------------------------------------------------------------------
LAB = '/home/talipovm/dev-python/vbt3-lab'
sys.path.insert(0, LAB)
ATOM_H = None
try:
    import atom_params as _ap                                 # type: ignore
    # vbt3-lab/atom_params.py exposes HUCKEL_H (the canonical Streitwieser
    # h_X dict: {'C':0, 'N1':0.5, 'O1':1.0, 'F':3.0, ...}). Older agents
    # may have used different names — fall back through the variants.
    for cand in ('HUCKEL_H', 'h_X', 'H_X', 'atoms', 'atom_h', 'H_PARAMS'):
        if hasattr(_ap, cand):
            ATOM_H = getattr(_ap, cand)
            break
except Exception:
    ATOM_H = None

if not isinstance(ATOM_H, dict):
    # Hueckel-style heteroatom parameters in units of |beta|
    # (Streitwieser / classic VB conventions: alpha_X = alpha_C + h_X * |beta|,
    # but with the sign that more EN atoms have a *more negative* alpha.)
    ATOM_H = {'C': 0.0, 'N': 0.5, 'O': 1.0, 'F': 3.0}

beta_eV = -2.4                                                 # |beta| ~ 2.4 eV
U_eV    = 10.0                                                 # generic on-site U
print(f"\nAtom h_X (units of |beta|): {ATOM_H}")
print(f"|beta| = {abs(beta_eV)} eV;  U = {U_eV} eV")


# ------------------------------------------------------------------------
# 5.  Sweep d_alpha / |beta| from 0 to 6, and report the four atom pairs
# ------------------------------------------------------------------------
ratios = np.linspace(0.0, 6.0, 121)
res = []
for r in ratios:
    d_alpha = -r * abs(beta_eV)              # X more EN -> alpha_b lower
    E, w, _ = ground_state(d_alpha, beta_eV=beta_eV, U_eV=U_eV)
    res.append((r, E, w))
res_arr = np.array([(r, E, *w) for (r, E, w) in res])

idx_aA = basis_strs.index('aA') if 'aA' in basis_strs else 0
idx_bB = basis_strs.index('bB') if 'bB' in basis_strs else 3
cov_w = res_arr[:, 2 + cov_idx[0]] + res_arr[:, 2 + cov_idx[1]]
ion_w = res_arr[:, 2 + ion_idx[0]] + res_arr[:, 2 + ion_idx[1]]
ionA  = res_arr[:, 2 + idx_aA]              # double-occ on C
ionB  = res_arr[:, 2 + idx_bB]              # double-occ on X (the EN atom)


# ------------------------------------------------------------------------
# 6.  Plot
# ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.plot(res_arr[:, 0], cov_w, lw=2.0, color='#1f77b4',
        label=r'covalent $w(aB)+w(bA)$')
ax.plot(res_arr[:, 0], ion_w, lw=2.0, color='#d62728',
        label=r'ionic $w(aA)+w(bB)$')
ax.plot(res_arr[:, 0], ionB,  lw=1.2, ls='--', color='#d62728',
        label=r'ionic on X $w(bB)$')
ax.plot(res_arr[:, 0], ionA,  lw=1.2, ls=':',  color='#d62728',
        label=r'ionic on C $w(aA)$')

# Mark the atom-pair values. atom_params.py uses flavored keys
# ('N1' = pyridine-N, 'N2' = pyrrole-N, 'O1' = carbonyl-O, 'O2' = furan-O);
# strip trailing digits to fall back into the simple {C,N,O,F} marker map.
markers = {'C': 'o', 'N': 's', 'O': '^', 'F': 'D'}
def _marker(atom):
    base = ''.join(ch for ch in atom if ch.isalpha())
    return markers.get(base, 'x')
def _label(atom):
    return atom  # keep e.g. 'N1' / 'O2' on the figure annotation
table_rows = []
for atom, hX in ATOM_H.items():
    d_alpha = -hX * abs(beta_eV)
    E, w, _ = ground_state(d_alpha, beta_eV=beta_eV, U_eV=U_eV)
    wcov = w[cov_idx[0]] + w[cov_idx[1]]
    wion = w[ion_idx[0]] + w[ion_idx[1]]
    wA   = w[idx_aA]
    wB   = w[idx_bB]
    mk = _marker(atom)
    ax.plot(hX, wcov, marker=mk, color='#1f77b4',
            ms=9, mew=1.1, mec='k', zorder=5)
    ax.plot(hX, wion, marker=mk, color='#d62728',
            ms=9, mew=1.1, mec='k', zorder=5)
    ax.annotate(f'C-{_label(atom)}', xy=(hX, wcov), xytext=(0, 8),
                textcoords='offset points', ha='center', fontsize=9)
    table_rows.append((atom, hX, d_alpha, E, wcov, wion, wA, wB))

ax.set_xlabel(r'$|\Delta\alpha| / |\beta|$  (heteroatom electronegativity)')
ax.set_ylabel('Chirgwin-Coulson weight in ground state')
ax.set_title(rf'C-X 2-electron $\pi$-bond character'
             rf'   ($|\beta|={abs(beta_eV)}$ eV, $U={U_eV}$ eV, $s=0$)')
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)
ax.legend(loc='center right', fontsize=9)
fig.tight_layout()

OUTDIR = '/home/talipovm/dev-python/vbt-3/figures'
os.makedirs(OUTDIR, exist_ok=True)
fig.savefig(os.path.join(OUTDIR, 'fig_bond_character_CNOF.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig_bond_character_CNOF.png'), dpi=160)
print(f"\nSaved figures to {OUTDIR}/fig_bond_character_CNOF.{{pdf,png}}")


# ------------------------------------------------------------------------
# 7.  Effect of U: redo the C-O point at several U values for the snippet
# ------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Effect of on-site U on bond character (Delta_alpha / |beta| = h_X)")
print("=" * 78)
print(f"{'atom':>4}  {'h_X':>5}  {'U(eV)':>6}  {'E_gs':>10}  "
      f"{'w_cov':>7}  {'w_ion':>7}  {'w(aA)':>7}  {'w(bB)':>7}")
# Pick one representative flavor per element (atom_params keys may be 'N1', 'N2', etc.)
def _first_match(prefixes):
    for atom in ATOM_H:
        for p in prefixes:
            if atom == p or atom.startswith(p):
                return atom
    return None
canonical = [_first_match([p]) for p in ['C', 'N', 'O', 'F']]
canonical = [a for a in canonical if a is not None]
for atom in canonical:
    hX = ATOM_H[atom]
    for Uval in [0.0, 5.0, 10.0, 20.0]:
        d_alpha = -hX * abs(beta_eV)
        E, w, _ = ground_state(d_alpha, beta_eV=beta_eV, U_eV=Uval)
        wcov = w[cov_idx[0]] + w[cov_idx[1]]
        wion = w[ion_idx[0]] + w[ion_idx[1]]
        wA   = w[idx_aA]
        wB   = w[idx_bB]
        print(f"  {atom:>2}  {hX:>5.2f}  {Uval:>6.2f}  {E:>10.4f}  "
              f"{wcov:>7.4f}  {wion:>7.4f}  {wA:>7.4f}  {wB:>7.4f}")


# ------------------------------------------------------------------------
# 8.  Sanity-check + final table
# ------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"Reference table (|beta| = {abs(beta_eV)} eV, U = {U_eV} eV, s = 0)")
print("=" * 78)
print(f"{'pair':>6}  {'h_X':>5}  {'Delta_alpha (eV)':>18}  {'E_gs (eV)':>10}  "
      f"{'w_cov':>7}  {'w_ion':>7}  {'w(aA)':>7}  {'w(bB)':>7}")
for atom, hX, d_alpha, E, wcov, wion, wA, wB in table_rows:
    pair = f"C-{atom}"
    chk = wcov + wion
    print(f"  {pair:>4}  {hX:>5.2f}  {d_alpha:>18.4f}  {E:>10.4f}  "
          f"{wcov:>7.4f}  {wion:>7.4f}  {wA:>7.4f}  {wB:>7.4f}"
          f"   sum={chk:.4f}")

# Sanity: matrix is 4x4, weights sum to 1
assert H_sym.shape == (4, 4),  "expected 4x4 H"
assert S_sym.shape == (4, 4),  "expected 4x4 S"
for atom, hX, d_alpha, E, wcov, wion, wA, wB in table_rows:
    assert abs(wcov + wion - 1.0) < 1e-10, (atom, wcov + wion)
print("\nAll sanity checks passed: 4x4 H/S, weights sum to 1.")
