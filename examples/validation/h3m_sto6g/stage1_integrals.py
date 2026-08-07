"""Stage 1 (any Python with PySCF): AO integrals + FCI for linear H3-/STO-6G.

Geometry matches job.inp: H at z = -1.0, 0.0, +1.0 A; charge -1, singlet.
Dumps the full 3-orbital integral set for stage2.
"""
import json

import numpy as np
from pyscf import fci, gto, scf

mol = gto.M(atom='H 0 0 -1.0; H 0 0 0.0; H 0 0 1.0', basis='sto-6g',
            unit='Angstrom', charge=-1)
S = mol.intor('int1e_ovlp')
h = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
eri = mol.intor('int2e')

mf = scf.RHF(mol).run()
e_fci = fci.FCI(mf).kernel()[0]

json.dump(dict(e_nuc=mol.energy_nuc(), e_fci=e_fci, S=S.tolist(),
               h=h.tolist(), eri=eri.tolist()),
          open('integrals.json', 'w'))
print(f'e_nuc={mol.energy_nuc():.6f}  e_fci={e_fci:.8f}')
