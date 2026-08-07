"""Stage 2 (Python with sympy/numpy/scipy): symvb symbolic 3-structure H2 model, numerically
substituted with the ab initio STO-6G integrals from stage 1, compared against
the XMVB run in this directory (job.xmo) and the PySCF FCI reference.

The XMVB run uses fragment-pinned pure-AO orbitals (one basis function per VB
orbital), so its VBSCF energy is the nonorthogonal CI over the three classical
structures {cov = aB+bA, ion_a = aA, ion_b = bB} with bare-AO integrals:
exactly what the symbolic matrices give after substitution. Energies and
Chirgwin-Coulson weights must agree to output precision.

Two-electron pattern convention (manuscript): physicist-ordered digits, so
pattern '1212' = <12|r|12> = (11|22) chemist = two-center Coulomb, and
'1122' = <11|r|22> = (12|12) chemist = exchange. The FCI cross-check would
expose a swap.

Run from this directory: PYTHONPATH=../../.. python3 stage2_symvb_vs_xmvb.py
"""
import json
import re

import numpy as np
import sympy as sp
from scipy.linalg import eigh

from symvb import FixedPsi, Molecule, System

ints = json.load(open('integrals.json'))

m = Molecule(
    zero_ii=False,
    interacting_orbs=['ab'],
    subst={'e': ('H_aa', 'H_bb'), 'h': ('H_ab',), 's': ('S_ab',)},
    subst_2e={'U': ('1111',), 'CB': ('1212',), 'EX': ('1122',),
              'M': ('1112', '1121', '1222')},
    max_2e_centers=2,
)
cov = FixedPsi('aB'); cov.add_str_det('bA', coef=1)
ion_a = FixedPsi('aA')
ion_b = FixedPsi('bB')
H, S = System.from_structures(m, [cov, ion_a, ion_b]).hamiltonian()

sub = {sp.Symbol('e'): ints['h_aa'], sp.Symbol('h'): ints['h_ab'],
       sp.Symbol('s'): ints['s_ab'], sp.Symbol('U'): ints['U'],
       sp.Symbol('CB'): ints['coul'], sp.Symbol('EX'): ints['exch'],
       sp.Symbol('M'): ints['M']}
Hn = np.array(sp.Matrix(H).subs(sub), dtype=float)
Sn = np.array(sp.Matrix(S).subs(sub), dtype=float)

w, v = eigh(Hn, Sn)
c = v[:, 0]
e_total = w[0] + ints['e_nuc']

# Chirgwin-Coulson weights, normalized c^T S c = 1
c = c / np.sqrt(c @ Sn @ c)
cc = c * (Sn @ c)

# reference values from the XMVB output in this directory
xmo = open('job.xmo').read()
e_xmvb = float(re.search(r'Total Energy:\s+(-?\d+\.\d+)', xmo).group(1))
wt_block = xmo.split('WEIGHTS OF STRUCTURES')[1]
w_xmvb = [float(x) for x in re.findall(r'^\s+\d+\s+(-?\d\.\d+)', wt_block,
                                       re.M)[:3]]

print(f'E(symvb+integrals) = {e_total:.8f} Ha')
print(f'E(XMVB)            = {e_xmvb:.8f} Ha   diff = {e_total - e_xmvb:+.2e}')
print(f'E(PySCF FCI)       = {ints["e_fci"]:.8f} Ha   diff = '
      f'{e_total - ints["e_fci"]:+.2e}')
print(f'CC weights symvb   = {cc[0]:.8f} {cc[1]:.8f} {cc[2]:.8f}')
print(f'CC weights XMVB    = {w_xmvb[0]:.8f} {w_xmvb[1]:.8f} {w_xmvb[2]:.8f}')

assert abs(e_total - ints['e_fci']) < 1e-9, 'symvb vs PySCF FCI mismatch'
assert abs(e_total - e_xmvb) < 5e-7, 'symvb vs XMVB energy mismatch'
assert max(abs(cc[i] - w_xmvb[i]) for i in range(3)) < 5e-7, 'weights mismatch'
print('ALL CHECKS PASSED')
