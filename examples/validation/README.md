# Validation of symvb against XMVB and PySCF

Cross-checks of the symvb symbolic matrix construction against two
independent programs, as reported in the paper's Supporting Information
(section S1): the ab initio valence bond package XMVB (v3.2.1 used) and
PySCF full CI (v2.11.0 used).

Design: every VB orbital is pinned to a single atomic basis function of
a minimal STO-6G basis, supplied to XMVB as an explicit fragment-
localized guess ($gus section in job.xmi). XMVB's VBSCF then has no
orbital freedom, and both programs diagonalize the SAME non-orthogonal
structure basis, so energies and Chirgwin-Coulson weights must agree to
output precision.

| case | system | class | structures |
|---|---|---|---|
| h2_sto6g | H2, R = 0.74272 A | 2c2e | cov + 2 ionic |
| he2pp_sto6g | He2^2+, R = 0.70 A | 2c2e | cov + 2 ionic |
| h3m_sto6g | linear H3^-, R(H-H) = 1.0 A | 3c4e | complete singlet set (6) |

Results (energies in hartree): E(symvb) = E(XMVB) to < 5e-9 and
E(symvb) = E(PySCF FCI) to machine precision in every case
(H2 -1.14590177, He2^2+ -3.37562236, H3^- -1.36614464); all
Chirgwin-Coulson weights match XMVB to the eight printed decimals.
The H3^- case exercises the three-center two-electron pathway (21
distinct integrals, kept as per-tuple T_<wxyz> symbols via
subst_2e=None; the T-name slots read in physicist order <wx|yz>).

Each case directory contains:
- job.inp / job.xmi : PREINT and XMVB inputs (reference output job.xmo
  included, so stage 2 runs without an XMVB installation)
- stage1_integrals.py : AO integrals + FCI reference via PySCF ->
  integrals.json (a pre-computed integrals.json is included)
- stage2_symvb_vs_xmvb.py : builds the symbolic matrices, substitutes
  the integrals, solves the generalized eigenvalue problem, and asserts
  agreement of energies and weights with job.xmo and the FCI reference.

To rerun stage 2 only (no XMVB or PySCF needed), from a case directory:

    PYTHONPATH=../../.. python3 stage2_symvb_vs_xmvb.py
