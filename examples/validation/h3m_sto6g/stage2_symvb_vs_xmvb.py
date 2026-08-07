"""Stage 2: symvb symbolic 6-structure linear H3- model vs XMVB and PySCF FCI.

The six singlet structures (complete singlet space for 4e/3 orbitals) match
the XMVB $str list in job.xmi: a2[b.c], b2[a.c], c2[a.b], a2b2, a2c2, b2c2,
with a, b, c the AOs on H(-1 A), H(0), H(+1 A).

Two-electron integrals stay as per-tuple symbolic names (subst_2e=None ->
T_<wxyz>); each is substituted from the PySCF ERI tensor. The slot convention
of T_<wxyz> (physicist <wx|yz> vs chemist (wx|yz)) is auto-detected by which
choice reproduces the FCI energy; exactly one must.

Run from this directory: PYTHONPATH=../../.. python3 stage2_symvb_vs_xmvb.py
"""
import json
import re

import numpy as np
import sympy as sp
from scipy.linalg import eigh

from symvb import FixedPsi, Molecule, System

ints = json.load(open('integrals.json'))
S_ao = np.array(ints['S'])
h_ao = np.array(ints['h'])
eri = np.array(ints['eri'])          # chemist (ij|kl)
idx = {'a': 0, 'b': 1, 'c': 2}

m = Molecule(
    zero_ii=False,
    interacting_orbs=['ab', 'ac', 'bc'],
    subst={'e1': ('H_aa', 'H_cc'), 'e2': ('H_bb',),
           'h1': ('H_ab', 'H_bc'), 'h2': ('H_ac',),
           's1': ('S_ab', 'S_bc'), 's2': ('S_ac',)},
    subst_2e=None,
    max_2e_centers=4,
)

def pair(p, q, closed):
    f = FixedPsi(closed + p + q.upper())
    f.add_str_det(closed + q + p.upper(), coef=1)
    return f

structs = [pair('b', 'c', 'aA'), pair('a', 'c', 'bB'), pair('a', 'b', 'cC'),
           FixedPsi('aAbB'), FixedPsi('aAcC'), FixedPsi('bBcC')]
H, S = System.from_structures(m, structs).hamiltonian()

sub1e = {sp.Symbol('e1'): h_ao[0, 0], sp.Symbol('e2'): h_ao[1, 1],
         sp.Symbol('h1'): h_ao[0, 1], sp.Symbol('h2'): h_ao[0, 2],
         sp.Symbol('s1'): S_ao[0, 1], sp.Symbol('s2'): S_ao[0, 2]}
assert abs(h_ao[0, 1] - h_ao[1, 2]) < 1e-12 and abs(S_ao[0, 1] - S_ao[1, 2]) < 1e-12

tsyms = sorted({s for s in sp.Matrix(H).free_symbols if s.name.startswith('T_')},
               key=lambda s: s.name)
print(f'{len(tsyms)} distinct two-electron integral names:',
      ' '.join(s.name for s in tsyms))

def solve(conv):
    sub = dict(sub1e)
    for t in tsyms:
        w, x, y, z = (idx[ch] for ch in t.name[2:])
        val = eri[w, y, x, z] if conv == 'physicist' else eri[w, x, y, z]
        sub[t] = val
    Hn = np.array(sp.Matrix(H).subs(sub), dtype=float)
    Sn = np.array(sp.Matrix(S).subs(sub), dtype=float)
    wv, vv = eigh(Hn, Sn)
    c = vv[:, 0]
    c = c / np.sqrt(c @ Sn @ c)
    return wv[0] + ints['e_nuc'], c * (Sn @ c)

results = {conv: solve(conv) for conv in ('physicist', 'chemist')}
ok = {conv: abs(r[0] - ints['e_fci']) < 1e-9 for conv, r in results.items()}
assert sum(ok.values()) == 1, f'convention detect failed: {ok}'
conv = [c for c, v in ok.items() if v][0]
e_total, cc = results[conv]
print(f'T_<wxyz> slot convention detected: {conv}')

xmo = open('job.xmo').read()
e_xmvb = float(re.search(r'Total Energy:\s+(-?\d+\.\d+)', xmo).group(1))
w_xmvb = [float(x) for x in re.findall(
    r'^\s+\d+\s+(-?\d\.\d+)', xmo.split('WEIGHTS OF STRUCTURES')[1], re.M)[:6]]

print(f'E(symvb+integrals) = {e_total:.8f} Ha')
print(f'E(XMVB)            = {e_xmvb:.8f} Ha   diff = {e_total - e_xmvb:+.2e}')
print(f'E(PySCF FCI)       = {ints["e_fci"]:.8f} Ha   diff = '
      f'{e_total - ints["e_fci"]:+.2e}')
print('CC weights symvb   =', ' '.join(f'{w:.8f}' for w in cc))
print('CC weights XMVB    =', ' '.join(f'{w:.8f}' for w in w_xmvb))

assert abs(e_total - e_xmvb) < 5e-7
assert max(abs(cc[i] - w_xmvb[i]) for i in range(6)) < 5e-7
print('ALL CHECKS PASSED')
