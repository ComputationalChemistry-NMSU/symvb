"""Stage 1 (any Python with PySCF): AO integrals + FCI reference for H2/STO-6G.

Geometry matches the XMVB run in this directory (job.inp): R = 0.70 A, charge +2.
Writes integrals.json consumed by stage2_symvb_vs_xmvb.py.
"""
import json

import numpy as np
from pyscf import fci, gto, scf

mol = gto.M(atom='He 0 0 -0.35; He 0 0 0.35', basis='sto-6g', charge=2,
            unit='Angstrom')
S = mol.intor('int1e_ovlp')
h = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
eri = mol.intor('int2e')          # chemist (ij|kl)

mf = scf.RHF(mol).run()
e_fci = fci.FCI(mf).kernel()[0]   # total energy incl. nuclear repulsion

out = dict(
    e_nuc=mol.energy_nuc(),
    e_fci=e_fci,
    s_ab=S[0, 1],
    h_aa=h[0, 0],
    h_ab=h[0, 1],
    U=eri[0, 0, 0, 0],            # (aa|aa)
    coul=eri[0, 0, 1, 1],         # (aa|bb)  two-center Coulomb
    exch=eri[0, 1, 0, 1],         # (ab|ab)  two-center exchange
    M=eri[0, 0, 0, 1],            # (aa|ab)  three-index hybrid
)
json.dump(out, open('integrals.json', 'w'), indent=1)
print({k: round(v, 8) for k, v in out.items()})
