"""
CSB manuscript support: the explicit VB Hamiltonian and overlap matrices for a
homonuclear 2-orbital / 2-electron sigma bond, built with symvb (NOT hand-derived
-- the s != 0 Loewdin cofactor structure is exactly where hand Slater-Condon
shortcuts fail).

Outputs, in terms of the fixed-atomic-orbital integrals
    h_aa = <a|h|a>,  h_ab = <a|h|b>,  s = <a|b>,
    U = (aa|aa),  J = (aa|bb),  K = (ab|ab),  M = (aa|ab):

  (i)  the 4x4 determinant-basis H (electronic) and S over {aA, aB, bA, bB}
       (lowercase = alpha, uppercase = beta in symvb notation);
  (ii) the 2x2 structure-basis H and S over {covalent singlet, gerade ionic},
       cov = (|aB> + |bA>)/N_cov,  ion = (|aA> + |bB>)/N_ion;
  (iii) verification that the 2x2 reproduces the closed form used in
       decomposition_*.py / recs_pyscf_*.py:
           E_cc = (2 h_aa + 2 h_ab s + J + K)/(1+s^2)
           E_ii = (2 h_aa + 2 h_ab s + U + K)/(1+s^2)
           H_ci = 2 (h_ab + h_aa s + M)/(1+s^2)
           sigma_ci = 2 s/(1+s^2)
  (iv) numerical matrices for H2, He2(2+), F2 (fixed-orbital integrals,
       aug-cc-pVTZ), and the effective orthogonalised coupling
           t_eff = H_ci - sigma_ci * E_cc
       which is the true integral-level resonance discriminant (NOT sigma_ci).

Run:  PYTHONPATH=. python3 examples/csb_vb_matrices.py
"""
import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from symvb import Molecule
from symvb.fixed_psi import generate_dets

haa, hab, s, U, J, K, M = sp.symbols('h_aa h_ab s U J K M', real=True)


def build_4x4():
    """Symbolic 4x4 determinant H (1e+2e, electronic) and S, on-site h kept."""
    m = Molecule(
        zero_ii=False,                       # KEEP the on-site one-electron h_aa
        interacting_orbs=['ab'],
        subst={'h_aa': ('H_aa', 'H_bb'),     # homonuclear: h_aa = h_bb
               'h_ab': ('H_ab',), 's': ('S_ab',)},
        subst_2e={'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
                  'M': ('1112', '1121', '1222')},
        max_2e_centers=2,
    )
    P = generate_dets(1, 1, 2)
    H = sp.simplify(m.build_matrix(P, op='H') + m.o2_matrix(P))
    S = sp.simplify(m.build_matrix(P, op='S'))
    basis = [p.dets[0].det_string for p in P]
    return H, S, basis


def reduce_to_2x2(H4, S4, basis):
    """Project the 4x4 onto {cov singlet, gerade ionic}, normalise to S_ii=1."""
    idx = {b: i for i, b in enumerate(basis)}
    cov = sp.zeros(4, 1); cov[idx['aB']] = 1; cov[idx['bA']] = 1   # |aB>+|bA>
    ion = sp.zeros(4, 1); ion[idx['aA']] = 1; ion[idx['bB']] = 1   # |aA>+|bB>
    C = cov.row_join(ion)
    Hs = sp.simplify(C.T * H4 * C)
    Ss = sp.simplify(C.T * S4 * C)
    # normalise each structure so the overlap diagonal is 1
    n_cov = sp.sqrt(Ss[0, 0]); n_ion = sp.sqrt(Ss[1, 1])
    D = sp.diag(1 / n_cov, 1 / n_ion)
    return sp.simplify(D * Hs * D), sp.simplify(D * Ss * D)


def closed_form():
    den = 1 + s ** 2
    E_cc = (2 * haa + 2 * hab * s + J + K) / den
    E_ii = (2 * haa + 2 * hab * s + U + K) / den
    H_ci = 2 * (hab + haa * s + M) / den
    sig = 2 * s / den
    H = sp.Matrix([[E_cc, H_ci], [H_ci, E_ii]])
    S = sp.Matrix([[1, sig], [sig, 1]])
    return H, S


INTS = {  # fixed-orbital integrals, aug-cc-pVTZ (Hartree)
    'H2':      dict(h_aa=-1.10939, h_ab=-0.96760, s=0.75285,
                    U=0.62458, J=0.50310, K=0.32289, M=0.42546),
    'He2(2+)': dict(h_aa=-3.47598, h_ab=-1.87474, s=0.42354,
                    U=1.24573, J=0.70333, K=0.17915, M=0.40805),
    'F2':      dict(h_aa=-1.23489, h_ab=-0.47341, s=0.22958,
                    U=0.91647, J=0.43710, K=0.03913, M=0.13376),
}


if __name__ == '__main__':
    H4, S4, basis = build_4x4()
    print("det basis:", basis)
    print("\n=== 4x4 determinant H (electronic) ===")
    sp.pprint(H4)
    print("\n=== 4x4 determinant S ===")
    sp.pprint(S4)

    H2x2, S2x2 = reduce_to_2x2(H4, S4, basis)
    print("\n=== 2x2 structure H {cov, ion} ===")
    sp.pprint(H2x2)
    print("\n=== 2x2 structure S {cov, ion} ===")
    sp.pprint(S2x2)

    Hcf, Scf = closed_form()
    dH = sp.simplify(H2x2 - Hcf); dS = sp.simplify(S2x2 - Scf)
    print("\nVERIFY vs closed form: H diff =")
    sp.pprint(dH)
    print("S diff =")
    sp.pprint(dS)
    print("match:", dH.is_zero_matrix and dS.is_zero_matrix)

    subs_syms = (haa, hab, s, U, J, K, M)
    H2f = sp.lambdify(subs_syms, H2x2, 'numpy')
    S2f = sp.lambdify(subs_syms, S2x2, 'numpy')
    H4f = sp.lambdify(subs_syms, H4, 'numpy')
    print("\n=== numerical structure matrices (aug-cc-pVTZ, electronic, Hartree) ===")
    for name, d in INTS.items():
        vals = (d['h_aa'], d['h_ab'], d['s'], d['U'], d['J'], d['K'], d['M'])
        Hn = np.array(H2f(*vals), float); Sn = np.array(S2f(*vals), float)
        sig = Sn[0, 1]; t_eff = Hn[0, 1] - sig * Hn[0, 0]
        print(f"\n{name}:")
        print(f"  H_2x2 = [[{Hn[0,0]:+.4f}, {Hn[0,1]:+.4f}], "
              f"[{Hn[1,0]:+.4f}, {Hn[1,1]:+.4f}]]   (E_cc, H_ci; E_ii)")
        print(f"  S_2x2 = [[1, {Sn[0,1]:.4f}], [{Sn[1,0]:.4f}, 1]]   sigma_ci={sig:.4f}")
        print(f"  effective coupling t_eff = H_ci - sigma_ci*E_cc = {t_eff:+.4f} Ha "
              f"({t_eff*627.509:+.1f} kcal/mol)")
