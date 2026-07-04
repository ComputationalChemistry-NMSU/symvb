"""Robustness tests for non-default Molecule flags.

Almost every example runs with `zero_ii=True` and normalized AOs; these tests
exercise the less-trodden paths (site energies on, AOs not normalized,
non-symmetric, whitelisted orbitals, one electron, empty/None interacting_orbs)
and pin the invariants that connect them back to the default behaviour.
"""
import unittest

import sympy as sp

from symvb import Molecule, FixedPsi, System
from symvb.fixed_psi import generate_dets

H_aa, H_bb, S_aa, S_bb = sp.symbols('H_aa H_bb S_aa S_bb')


def _base(**over):
    kw = dict(interacting_orbs=['ab'], subst={'h': ('H_ab',), 's': ('S_ab',)},
              subst_2e={'U': ('1111',)}, max_2e_centers=1)
    kw.update(over)
    return Molecule(**kw)


class TestZeroII(unittest.TestCase):
    def setUp(self):
        self.P = generate_dets(1, 1, 2)

    def test_builds_with_site_energies(self):
        H = _base(zero_ii=False).build_matrix(self.P, 'H')
        self.assertIn(H_aa, H.free_symbols)          # site energy is present

    def test_reduces_to_zero_ii_true(self):
        # the whole point: zero_ii=False with all H_ii -> 0 is exactly zero_ii=True
        Hf = _base(zero_ii=False).build_matrix(self.P, 'H')
        Hz = _base(zero_ii=True).build_matrix(self.P, 'H')
        self.assertTrue(sp.simplify(Hf.subs({H_aa: 0, H_bb: 0}) - Hz).is_zero_matrix)

    def test_facade_ground_state_with_site_energies(self):
        m = _base(zero_ii=False)
        cov = FixedPsi('aB'); cov.add_str_det('bA', coef=1)
        ion = FixedPsi('aA'); ion.add_str_det('bB', coef=1)
        E, c = System.from_structures(m, [cov, ion]).ground_state()
        self.assertIn(H_aa, E.free_symbols)          # solved with the shift present

    def test_o2_unaffected_and_methods_agree(self):
        Hd = _base(zero_ii=False, o2_method='direct').o2_matrix(self.P)
        Hb = _base(zero_ii=False, o2_method='blocked').o2_matrix(self.P)
        self.assertTrue(sp.simplify(Hd - Hb).is_zero_matrix)


class TestNormalizedBasisOrbs(unittest.TestCase):
    def test_reduces_when_overlaps_unit(self):
        P = generate_dets(1, 1, 2)
        Sf = _base(normalized_basis_orbs=False).build_matrix(P, 'S')
        Sd = _base().build_matrix(P, 'S')
        self.assertIn(S_aa, Sf.free_symbols)
        self.assertTrue(sp.simplify(Sf.subs({S_aa: 1, S_bb: 1}) - Sd).is_zero_matrix)


class TestOtherFlags(unittest.TestCase):
    def test_symm_offdiagonal_false_builds(self):
        H = _base(symm_offdiagonal=False).build_matrix(generate_dets(1, 1, 2), 'H')
        self.assertEqual(H.shape, (4, 4))

    def test_orbitals_whitelist_rejects(self):
        m = Molecule(orbitals='ab', interacting_orbs=['ab'], subst={'h': ('H_ab',)})
        with self.assertRaises(ValueError):
            m.build_matrix([FixedPsi('aC')], 'H')      # 'c' is outside the whitelist

    def test_one_electron_system(self):
        H = _base().build_matrix(generate_dets(1, 0, 2), 'H')   # H2+
        self.assertEqual(H.shape, (2, 2))

    def test_empty_interacting_orbs_decouples(self):
        H = Molecule(interacting_orbs=[], subst={'h': ('H_ab',), 's': ('S_ab',)}
                     ).build_matrix(generate_dets(1, 1, 2), 'H')
        self.assertEqual(H[0, 1], 0)                  # nothing couples

    def test_none_interacting_orbs_couples_all(self):
        H = Molecule(subst={'h': ('H_ab',), 's': ('S_ab',)}, zero_ii=True
                     ).build_matrix(generate_dets(1, 1, 2), 'H')
        self.assertNotEqual(H[0, 1], 0)


class TestRingAndMixedSubst(unittest.TestCase):
    def test_ring_zero_ii_false_invariant(self):
        # site energies flow through Molecule.ring and vanish back to zero_ii=True
        P = generate_dets(2, 2, 4)
        Hf = Molecule.ring(4, zero_ii=False).build_matrix(P, 'H')
        Hz = Molecule.ring(4, zero_ii=True).build_matrix(P, 'H')
        subs0 = {sp.Symbol('H_%s%s' % (c, c)): 0 for c in 'abcd'}
        self.assertIn(sp.Symbol('H_aa'), Hf.free_symbols)
        self.assertTrue(sp.simplify(Hf.subs(subs0) - Hz).is_zero_matrix)

    def test_mixed_subst_partial_site_energies(self):
        # unify two site energies via subst, leave the third symbolic
        m = Molecule(zero_ii=False, interacting_orbs=['ab', 'bc'],
                     subst={'h': ('H_ab', 'H_bc'), 'eps': ('H_aa', 'H_bb'),
                            's': ('S_ab', 'S_bc')},
                     subst_2e={'U': ('1111',)}, max_2e_centers=1)
        syms = set(map(str, m.build_matrix(generate_dets(2, 2, 3), 'H').free_symbols))
        self.assertIn('eps', syms)          # H_aa, H_bb unified to eps
        self.assertIn('H_cc', syms)         # H_cc left symbolic
        self.assertNotIn('H_aa', syms)

    def test_heterogeneous_edge_params(self):
        m = Molecule(zero_ii=True, interacting_orbs=['ab', 'bc'],
                     subst={'h': ('H_ab',), 'h2': ('H_bc',), 's': ('S_ab', 'S_bc')})
        syms = set(map(str, m.build_matrix(generate_dets(2, 2, 3), 'H').free_symbols))
        self.assertEqual(syms, {'h', 'h2', 's'})

    def test_facade_ring_site_energy_shift(self):
        # a uniform site energy eps shifts the total energy by n_electrons * eps
        h, s, U, Haa, Hbb, Hcc = sp.symbols('h s U H_aa H_bb H_cc')
        num = {h: -1, s: 0, U: 2}
        E0, _ = System.ring(3, zero_ii=True).ground_state(subs=num)
        E1, _ = System.ring(3, zero_ii=False).ground_state(
            subs={**num, Haa: 0.5, Hbb: 0.5, Hcc: 0.5})
        self.assertAlmostEqual(E1 - E0, 2 * 0.5, places=6)   # 2 electrons x 0.5


if __name__ == '__main__':
    unittest.main()
