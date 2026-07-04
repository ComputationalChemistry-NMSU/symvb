"""Determinant-convention consistency between standardize_det and the
generate_dets/generate_det_strings basis.

standardize_det fixes the spin *pattern* (uLuL... + majority-spin tail) but
its pairwise flips can permute the alphabetical order within the alpha/beta
blocks, so on unequal-filling (and beta-block-leading) determinants its output
string need not be a member of the generated basis (e.g. 'abcAB' -> 'aAcBb',
not 'aAbBc'). Its output convention is load-bearing in matrix construction and
must not change; the *lookup* side (symvb.system.structure_vector) therefore
standardizes both the basis strings and the structure determinants through the
full canonical map (spin pattern + within-block sort, fermion signs folded).
These tests pin the census of the mismatch, the fixed lookup at every affected
filling, sign correctness against an independent inversion-parity reference,
and exact no-regression on previously-working columns.
"""
import string
import unittest
from itertools import combinations

import numpy as np
import sympy as sp

from symvb import FixedPsi, Molecule
from symvb.fixed_psi import generate_dets
from symvb.functions import generate_det_strings, standardize_det
from symvb.system import _standardize_full, hamiltonian, structure_vector

FILLINGS = [(2, 1, 4), (2, 2, 3), (2, 2, 4), (3, 2, 4), (3, 2, 6), (3, 3, 6)]

# dets (with alphabetically sorted alpha and beta blocks, i.e. exactly the
# strings FixedPsi.canonicalize can emit) whose bare standardize_det output
# is NOT in generate_det_strings(Na, Nb, Norb)
KNOWN_MISS_COUNTS = {(2, 1, 4): 0, (2, 2, 3): 9, (2, 2, 4): 36,
                     (3, 2, 4): 48, (3, 2, 6): 600, (3, 3, 6): 2800}


def _canonicalized_space(Na, Nb, Norb):
    """All det strings with sorted alpha block and sorted beta block, over all
    C(Na+Nb, Na) spin patterns: the input space structure_vector feeds into
    the standardization after FixedPsi.canonicalize()."""
    lo = string.ascii_lowercase[:Norb]
    up = string.ascii_uppercase[:Norb]
    out = []
    for a in combinations(lo, Na):
        for b in combinations(up, Nb):
            for pos in combinations(range(Na + Nb), Na):
                s = [''] * (Na + Nb)
                ia = ib = 0
                for i in range(Na + Nb):
                    if i in pos:
                        s[i] = a[ia]
                        ia += 1
                    else:
                        s[i] = b[ib]
                        ib += 1
                out.append(''.join(s))
    return out


def _ref_canonical(det_string, basis_by_soset):
    """Independent reference: the basis string over the same spin-orbital set
    and the inversion parity of the permutation relating the two creation
    orders."""
    so = [(c.lower(), 0 if c.islower() else 1) for c in det_string]
    tgt = basis_by_soset[frozenset(so)]
    tso = [(c.lower(), 0 if c.islower() else 1) for c in tgt]
    pos = {v: i for i, v in enumerate(tso)}
    idx = [pos[v] for v in so]
    inv = sum(1 for i in range(len(idx)) for j in range(i + 1, len(idx))
              if idx[i] > idx[j])
    return tgt, (-1) ** inv


def _soset_index(basis):
    return {frozenset((c.lower(), 0 if c.islower() else 1) for c in b): b
            for b in basis}


def _old_structure_vector(structure, basis_dets):
    """The pre-fix structure_vector algorithm (bare standardize_det lookup),
    kept here as the no-regression reference on fillings where it worked."""
    fp = FixedPsi(structure)
    fp.canonicalize()
    idx = {d: i for i, d in enumerate(basis_dets)}
    v = sp.zeros(len(basis_dets), 1)
    for d, c in fp:
        std, flips = standardize_det(d.det_string)
        if std not in idx:
            raise ValueError(std)
        v[idx[std]] += (-1) ** flips * c
    return v


class TestFillingCensus(unittest.TestCase):
    def test_standardize_det_miss_counts(self):
        # pin the characterization: standardize_det's output leaves the
        # generated basis exactly this often, per filling, over the
        # canonicalized (sorted-block) input space; it is a no-op on the
        # basis strings themselves.
        for Na, Nb, Norb in FILLINGS:
            basis = set(generate_det_strings(Na, Nb, Norb))
            for b in basis:
                self.assertEqual(standardize_det(b), (b, 0))
            miss = sum(1 for d in _canonicalized_space(Na, Nb, Norb)
                       if standardize_det(d)[0] not in basis)
            self.assertEqual(miss, KNOWN_MISS_COUNTS[(Na, Nb, Norb)],
                             msg=f"filling ({Na},{Nb},{Norb})")

    def test_full_standardization_lands_in_basis_with_correct_sign(self):
        # the fixed lookup map: every canonicalized-space det lands ON the
        # basis, with the sign of the independent inversion-parity reference
        for Na, Nb, Norb in FILLINGS:
            basis = generate_det_strings(Na, Nb, Norb)
            bset = set(basis)
            ref_idx = _soset_index(basis)
            for d in _canonicalized_space(Na, Nb, Norb):
                key, sgn = _standardize_full(d)
                self.assertIn(key, bset,
                              msg=f"({Na},{Nb},{Norb}): {d!r} -> {key!r}")
                self.assertEqual((key, sgn), _ref_canonical(d, ref_idx),
                                 msg=f"({Na},{Nb},{Norb}): {d!r}")

    def test_structure_vector_lookup_succeeds_at_all_fillings(self):
        # end-to-end: structure_vector places every sampled det (including
        # the ones bare standardize_det mishandles) at the right position
        # with the right sign, and never raises
        for Na, Nb, Norb in FILLINGS:
            basis = generate_det_strings(Na, Nb, Norb)
            ref_idx = _soset_index(basis)
            space = _canonicalized_space(Na, Nb, Norb)
            bset = set(basis)
            missing = [d for d in space if standardize_det(d)[0] not in bset]
            step = max(1, len(space) // 20)
            sample = space[::step] + missing[::max(1, len(missing) // 20)]
            for d in sample:
                v = structure_vector(FixedPsi(d), basis)
                key, sgn = _ref_canonical(d, ref_idx)
                nz = {basis[i]: v[i] for i in range(len(basis)) if v[i] != 0}
                self.assertEqual(nz, {key: sgn},
                                 msg=f"({Na},{Nb},{Norb}): {d!r}")

    def test_outside_basis_raises_named_valueerror(self):
        basis = generate_det_strings(2, 1, 4)      # orbitals a..d only
        with self.assertRaises(ValueError) as cm:
            structure_vector(FixedPsi('aEb'), basis)
        self.assertIn('aEb', str(cm.exception))

    def test_duplicate_basis_det_raises(self):
        # 'abA' is the same determinant as 'aAb' in a different creation order
        with self.assertRaises(ValueError):
            structure_vector(FixedPsi('aAb'), ['aAb', 'abA'])


class TestHoleBasisRoundTrip326(unittest.TestCase):
    """(H2)3+ sigma-hole diabatics on the (3,2,6) basis: structure_vector must
    reproduce the low-level to_standard mapping that
    examples/h2h2h2_plus_diabatic.py hand-rolls (positions AND signs)."""

    ORBS = 'abcdef'
    PAIRS = [('a', 'b'), ('c', 'd'), ('e', 'f')]

    def _to_standard(self, det_string):
        # replicated from examples/h2h2h2_plus_diabatic.py
        so_list = [2 * self.ORBS.index(c.lower()) + (0 if c.islower() else 1)
                   for c in det_string]
        if len(set(so_list)) != len(so_list):
            return None, 0
        alphas = sorted(c for c in det_string if c.islower())
        betas = sorted(c for c in det_string if c.isupper())
        std = ''
        na, nb = len(alphas), len(betas)
        for i in range(min(na, nb)):
            std += alphas[i] + betas[i]
        std += ''.join(alphas[nb:]) + ''.join(betas[na:])
        target = [2 * self.ORBS.index(c.lower()) + (0 if c.islower() else 1)
                  for c in std]
        pos = {v: i for i, v in enumerate(target)}
        idx = [pos[v] for v in so_list]
        inv = sum(1 for i in range(len(idx)) for j in range(i + 1, len(idx))
                  if idx[i] > idx[j])
        return std, (-1 if inv % 2 else 1)

    def _hole_raws(self, hole_pair):
        hole_atoms = self.PAIRS[hole_pair]
        full_idx = [i for i in range(3) if i != hole_pair]
        f0, f1 = self.PAIRS[full_idx[0]], self.PAIRS[full_idx[1]]
        return [h_a + f0a + f0b.upper() + f1a + f1b.upper()
                for h_a in hole_atoms
                for f0a in f0 for f0b in f0
                for f1a in f1 for f1b in f1]

    def test_hole_diabatics_match_to_standard_route(self):
        basis = generate_det_strings(3, 2, 6)
        ds_to_idx = {d: i for i, d in enumerate(basis)}
        cols = []
        n_outside = 0
        for hp in range(3):
            raws = self._hole_raws(hp)
            self.assertEqual(len(raws), 32)
            n_outside += sum(1 for r in raws
                             if standardize_det(r)[0] not in ds_to_idx)
            v_ref = np.zeros(len(basis))
            for raw in raws:
                std, sgn = self._to_standard(raw)
                self.assertIsNotNone(std)
                v_ref[ds_to_idx[std]] += sgn
            fp = FixedPsi()
            for raw in raws:
                fp.add_str_det(raw, coef=1)
            v = np.array(structure_vector(fp, basis), float).ravel()
            self.assertTrue(np.array_equal(v, v_ref), msg=f"hole {hp}")
            cols.append(v)
        # the audited mismatch: 64 of the 96 raw hole dets are exactly the
        # strings the pre-fix lookup raised on
        self.assertEqual(n_outside, 64)
        # the three diabatics are orthonormal at s = 0 (norm^2 = 32 raw dets
        # collapsing pairwise onto 32 basis dets of unit coefficient)
        Phi = np.column_stack(cols)
        self.assertTrue(np.allclose(Phi.T @ Phi / 32.0, np.eye(3),
                                    atol=1e-14))


class TestNoRegression(unittest.TestCase):
    """Columns the pre-fix code produced correctly must be reproduced exactly."""

    def test_223_longbond_column_exact(self):
        # the (2,2,3) long-bond column pinned by TestStructureVector in
        # test_system.py, and the full old-vs-new comparison on all three
        # allyl Rumer structures
        basis = [p.dets[0].det_string for p in generate_dets(2, 2, 3)]
        structs = [FixedPsi('aBcC', coupled_pairs=[(0, 1)]),
                   FixedPsi('aAbC', coupled_pairs=[(2, 3)]),
                   FixedPsi('abBC', coupled_pairs=[(0, 3)])]
        for st in structs:
            self.assertEqual(structure_vector(st, basis),
                             _old_structure_vector(st, basis))
        v = structure_vector(structs[2], basis)
        nz = {basis[i]: v[i] for i in range(len(basis)) if v[i] != 0}
        self.assertEqual(nz, {'aBbC': -1, 'bAcB': -1})

    def test_214_hole_columns_exact(self):
        # (2,1,4) is a filling where bare standardize_det never leaves the
        # basis (census count 0), so the old algorithm was fully correct
        # there; the new lookup must agree determinant-for-determinant
        basis = [p.dets[0].det_string for p in generate_dets(2, 1, 4)]
        hole0 = FixedPsi()
        for ha in 'ab':
            for fa in 'cd':
                for fb in 'CD':
                    hole0.add_str_det(ha + fa + fb, coef=1)
        hole1 = FixedPsi()
        for ha in 'cd':
            for fa in 'ab':
                for fb in 'AB':
                    hole1.add_str_det(fa + fb + ha, coef=1)
        for st in [hole0, hole1, FixedPsi('acC'), FixedPsi('aCc')]:
            self.assertEqual(structure_vector(st, basis),
                             _old_structure_vector(st, basis))


class TestProjectionConsistency(unittest.TestCase):
    def test_vthv_matches_direct_hamiltonian_unequal_filling(self):
        # (H2)2+ dimer, Na=2/Nb=1/4 orbitals: projecting the FCI (H, S)
        # through structure_vector columns must equal (H, S) built directly
        # over the structures
        m = Molecule(zero_ii=True, interacting_orbs=['ab', 'cd', 'bc'],
                     subst={'h': ('H_ab', 'H_cd'), 't': ('H_bc',),
                            's': ('S_ab', 'S_cd'), 'sg': ('S_bc',)},
                     subst_2e={'U': ('1111',)}, max_2e_centers=1)
        dets = generate_dets(2, 1, 4)
        strs = [p.dets[0].det_string for p in dets]
        hole0 = FixedPsi()
        for ha in 'ab':
            for fa in 'cd':
                for fb in 'CD':
                    hole0.add_str_det(ha + fa + fb, coef=1)
        hole1 = FixedPsi()
        for ha in 'cd':
            for fa in 'ab':
                for fb in 'AB':
                    hole1.add_str_det(fa + fb + ha, coef=1)
        structs = [hole0, hole1]

        Hfci, Sfci = hamiltonian(m, dets)
        V = sp.Matrix.hstack(*[structure_vector(st, strs) for st in structs])
        H_dir, S_dir = hamiltonian(m, structs)
        self.assertTrue(sp.simplify(V.T * Hfci * V - H_dir).is_zero_matrix)
        self.assertTrue(sp.simplify(V.T * Sfci * V - S_dir).is_zero_matrix)


if __name__ == '__main__':
    unittest.main()
