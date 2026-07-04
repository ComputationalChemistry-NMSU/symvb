"""Tests for the high-level facade in symvb.system."""
import unittest

import sympy as sp

from symvb import Molecule, FixedPsi, System
from symvb.system import hamiltonian, ground_state, chirgwin_coulson, structure_vector
from symvb.fixed_psi import generate_dets
from symvb.functions import standardize_det

h, s, U = sp.symbols('h s U')


def _canonical_copy(fp):
    """An independent hand-canonicalization of ``fp``: every determinant is
    rewritten into standard (interleaved) creation order via
    :func:`standardize_det` with the ``(-1)**flips`` sign folded into the
    coefficient. Used as the ground truth the internal canonicalization must
    reproduce."""
    out = FixedPsi()
    for d, c in fp:
        std, flips = standardize_det(d.det_string)
        out.add_str_det(std, coef=c * (-1) ** flips)
    return out


def _h2_molecule():
    return Molecule(zero_ii=True, interacting_orbs=['ab'],
                    subst={'h': ('H_ab',), 's': ('S_ab',)},
                    subst_2e={'U': ('1111',)}, max_2e_centers=1)


def _cov_ion():
    cov = FixedPsi('aB'); cov.add_str_det('bA', coef=1)
    ion = FixedPsi('aA'); ion.add_str_det('bB', coef=1)
    return cov, ion


class TestTopologyConstructors(unittest.TestCase):
    def test_ring_matches_benzene_config(self):
        m = Molecule.ring(6)
        self.assertEqual(m.interacting_orbs, ['ab', 'bc', 'cd', 'de', 'ef', 'af'])
        # every edge gets an H_ and S_ name mapped to h / s
        self.assertEqual(sorted(k for k, v in m.subst.items() if v == 'h'),
                         ['H_ab', 'H_af', 'H_bc', 'H_cd', 'H_de', 'H_ef'])
        self.assertEqual(sorted(k for k, v in m.subst.items() if v == 's'),
                         ['S_ab', 'S_af', 'S_bc', 'S_cd', 'S_de', 'S_ef'])
        self.assertEqual(m.subst_2e, {'1111': 'U'})

    def test_chain_has_no_wraparound(self):
        m = Molecule.chain(3)
        self.assertEqual(m.interacting_orbs, ['ab', 'bc'])

    def test_hubbard_off_drops_2e(self):
        m = Molecule.ring(4, hubbard=False)
        self.assertEqual(m.subst_2e, {})

    def test_ring_too_large_raises(self):
        with self.assertRaises(ValueError):
            Molecule.ring(27)

    def test_ring2_deduped(self):
        # the wrap-around edge of a 2-ring coincides with the forward edge
        self.assertEqual(Molecule.ring(2).interacting_orbs, ['ab'])


class TestHamiltonian(unittest.TestCase):
    def test_matches_manual_build(self):
        m = _h2_molecule()
        P = generate_dets(1, 1, 2)
        H, S = hamiltonian(m, P)
        H_manual = m.build_matrix(P, op='H') + m.o2_matrix(P)
        S_manual = m.build_matrix(P, op='S')
        self.assertTrue(sp.simplify(H - H_manual).is_zero_matrix)
        self.assertTrue(sp.simplify(S - S_manual).is_zero_matrix)

    def test_two_electron_folded_once_not_squared(self):
        # the o2 block carries U; folding it in must keep H linear in U
        m = _h2_molecule()
        cov, ion = _cov_ion()
        H, _ = hamiltonian(m, [cov, ion])
        for i in range(2):
            for j in range(2):
                self.assertEqual(sp.diff(H[i, j], U, 2), 0,
                                 "H must be first-order in U (no accidental U^2)")
        # the ionic structure carries U, the covalent does not
        self.assertNotEqual(sp.diff(H[1, 1], U), 0)
        self.assertEqual(sp.diff(H[0, 0], U), 0)

    def test_default_T_names_still_folded(self):
        # regression: a molecule whose 2e integrals keep the default T_<abcd>
        # names (subst_2e absent or empty) used to get its 2e block silently
        # dropped; the fold must not depend on subst_2e being set
        m = Molecule.chain(3, hubbard=False)
        P = generate_dets(2, 2, 3)
        H, S = hamiltonian(m, P)
        H_manual = m.build_matrix(P, op='H') + m.o2_matrix(P)
        S_manual = m.build_matrix(P, op='S')
        self.assertTrue(sp.simplify(H - H_manual).is_zero_matrix)
        self.assertTrue(sp.simplify(S - S_manual).is_zero_matrix)
        self.assertTrue(any(str(x).startswith('T_') for x in H.free_symbols))

    def test_two_electron_opt_out(self):
        # two_electron=False returns the bare one-electron H (no 2e symbols)
        m = _h2_molecule()
        P = generate_dets(1, 1, 2)
        H, _ = hamiltonian(m, P, two_electron=False)
        H1 = m.build_matrix(P, op='H')
        self.assertTrue(sp.simplify(H - H1).is_zero_matrix)
        self.assertEqual(H.free_symbols & {U}, set())

    def test_system_passes_two_electron_flag(self):
        m = _h2_molecule()
        P = generate_dets(1, 1, 2)
        H1 = m.build_matrix(P, op='H')
        sys1 = System(m, P, two_electron=False)
        self.assertTrue(sp.simplify(sys1.H - H1).is_zero_matrix)
        # the topology constructors forward the flag too
        sysc = System.chain(3, hubbard=False, two_electron=False)
        self.assertFalse(any(str(x).startswith('T_')
                             for x in sysc.H.free_symbols))


class TestGroundStateAndWeights(unittest.TestCase):
    def test_h2_ground_state_closed_form(self):
        m = _h2_molecule()
        sysm = System.from_structures(m, list(_cov_ion()))
        E, c = sysm.ground_state()
        E0 = sp.simplify(E.subs(s, 0))
        ref = U / 2 - sp.sqrt((U / 2) ** 2 + 4 * h ** 2)
        self.assertEqual(sp.simplify(E0 - ref), 0)

    def test_h2_weights_50_50_at_U0(self):
        m = _h2_molecule()
        sysm = System.from_structures(m, list(_cov_ion()))
        w = sysm.weights()
        self.assertEqual([sp.simplify(x.subs(U, 0)) for x in w],
                         [sp.Rational(1, 2), sp.Rational(1, 2)])

    def test_chirgwin_coulson_sums_to_one(self):
        S = sp.eye(3)
        c = sp.Matrix([1, 2, 2])
        w = chirgwin_coulson(c, S)
        self.assertEqual(sp.simplify(sum(w)), 1)
        # grouped
        wg = chirgwin_coulson(c, S, groups=[[0], [1, 2]])
        self.assertEqual(sp.simplify(wg[0]), sp.Rational(1, 9))
        self.assertEqual(sp.simplify(wg[1]), sp.Rational(8, 9))

    def test_chirgwin_coulson_numpy_path(self):
        import numpy as np
        S = np.eye(3)
        c = np.array([1.0, 2.0, 2.0])
        w = chirgwin_coulson(c, S)
        self.assertIsInstance(w, np.ndarray)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)

    def test_chirgwin_coulson_mixed_metric(self):
        # symbolic coefficients with a NumPy metric must not raise
        import numpy as np
        w = chirgwin_coulson(sp.Matrix([1, h]), np.eye(2))
        self.assertEqual(sp.simplify(sum(w)), 1)

    def test_ground_state_numeric_subs(self):
        # the symbolic solve is infeasible on a 9-determinant FCI; subs= is not
        import numpy as np
        mA = Molecule(zero_ii=True, interacting_orbs=['ab', 'bc'],
                      subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
                      subst_2e={'U': ('1111',)}, max_2e_centers=1)
        sysA = System(mA, generate_dets(2, 2, 3))
        E, c = sysA.ground_state(subs={h: -1, s: 0, U: 0})
        self.assertAlmostEqual(E, -2 * np.sqrt(2), places=6)
        self.assertEqual(c.shape, (9,))

    def test_weights_structures_numeric(self):
        # project an FCI ground state onto VB structures (numerically)
        mA = Molecule(zero_ii=True, interacting_orbs=['ab', 'bc'],
                      subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
                      subst_2e={'U': ('1111',)}, max_2e_centers=1)
        sysA = System(mA, generate_dets(2, 2, 3))
        rumer = [FixedPsi('aBcC', coupled_pairs=[(0, 1)]),   # Kekule a-b
                 FixedPsi('aAbC', coupled_pairs=[(2, 3)]),   # Kekule b-c
                 FixedPsi('abBC', coupled_pairs=[(0, 3)])]   # long bond a-c
        w = sysA.weights(structures=rumer, subs={h: -1, s: 0, U: 0})
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)   # normalized over structures
        self.assertAlmostEqual(w[0], w[1], places=9)            # Kekule symmetry
        self.assertAlmostEqual(w[2], 0.2, places=6)             # long-bond share at U=0


class TestStructureVector(unittest.TestCase):
    def test_longbond_vector_fermion_sign(self):
        # the (0,3) coupling produces an alpha-alpha-beta-beta pattern that
        # build_matrix cannot take; structure_vector must still place it.
        dets = [p.dets[0].det_string for p in generate_dets(2, 2, 3)]
        v = structure_vector(FixedPsi('abBC', coupled_pairs=[(0, 3)]), dets)
        nz = {dets[i]: v[i] for i in range(len(dets)) if v[i] != 0}
        self.assertEqual(nz, {'aBbC': -1, 'bAcB': -1})
        self.assertEqual(sp.simplify((v.T * v)[0]), 2)   # norm^2, orthonormal dets

    def test_kekule_vector_is_singlet(self):
        from symvb.spin import s_squared_matrix
        import numpy as np
        dets = [p.dets[0].det_string for p in generate_dets(2, 2, 3)]
        v = structure_vector(FixedPsi('aBcC', coupled_pairs=[(0, 1)]), dets)
        S2 = np.array(s_squared_matrix(dets, orbs='abc'), float)
        vn = np.array(v, float).ravel()
        self.assertAlmostEqual(float(vn @ S2 @ vn) / float(vn @ vn), 0.0, places=12)


class TestNonCanonicalStructures(unittest.TestCase):
    """Hand-built VB structures whose determinants are in a natural (non-
    canonical) creation order must still give the correct (H, S). Before the
    internal canonicalization these silently lost couplings: build_matrix
    matches determinants by their raw spin-position pattern, so an
    alpha-alpha-beta-beta long bond from coupled_pairs did not couple to an
    interleaved Kekule structure."""

    def _allyl_molecule(self):
        return Molecule(zero_ii=True, interacting_orbs=['ab', 'bc'],
                        subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
                        subst_2e={'U': ('1111',)}, max_2e_centers=1)

    def test_allyl_longbond_couplings_recovered(self):
        m = self._allyl_molecule()
        kek_ab = FixedPsi('aBcC', coupled_pairs=[(0, 1)])   # interleaved  +-+-
        kek_bc = FixedPsi('aAbC', coupled_pairs=[(2, 3)])   # interleaved  +-+-
        lb_ac = FixedPsi('abBC', coupled_pairs=[(0, 3)])    # long bond    ++--
        structs = [kek_ab, kek_bc, lb_ac]

        H, S = System.from_structures(m, structs).hamiltonian()
        H0 = sp.simplify(H.subs(s, 0))

        # ground truth: project each structure into the FCI determinant basis
        # (structure_vector tracks the fermion reorder sign) and form V^T H V
        dets = [p.dets[0].det_string for p in generate_dets(2, 2, 3)]
        Hfci, _ = hamiltonian(m, generate_dets(2, 2, 3))
        V = sp.Matrix.hstack(*[structure_vector(st, dets) for st in structs])
        Hgt = sp.simplify((V.T * Hfci * V).subs(s, 0))
        self.assertTrue(sp.simplify(H0 - Hgt).is_zero_matrix)

        # the covalent 3x3 is 2 * [[U,0,-h],[0,U,-h],[-h,-h,U]] (norm^2 = 2)
        expected = 2 * sp.Matrix([[U, 0, -h], [0, U, -h], [-h, -h, U]])
        self.assertTrue(sp.simplify(H0 - expected).is_zero_matrix)

        # the couplings that used to vanish are the long-bond rows/columns
        self.assertEqual(sp.simplify(H0[0, 2]), -2 * h)
        self.assertEqual(sp.simplify(H0[2, 1]), -2 * h)
        self.assertEqual(sp.simplify(H0[0, 1]), 0)   # the two Kekule are uncoupled

    def test_disphenoid_natural_order_matches_canonical(self):
        # a 4-orbital 2+2 system with a canonical covalent structure and a
        # crossed long-bond structure written in alpha-alpha-beta-beta order
        m = Molecule.ring(4)
        cov = FixedPsi('aBcD', coupled_pairs=[(0, 1), (2, 3)])   # HL(a,b) x HL(c,d), interleaved
        lb = FixedPsi('abCD', coupled_pairs=[(0, 3), (1, 2)])    # HL(a,d) x HL(b,c), uuLL parent
        natural = [cov, lb]
        canon = [_canonical_copy(cov), _canonical_copy(lb)]

        Hn, Sn = hamiltonian(m, natural)
        Hc, Sc = hamiltonian(m, canon)
        self.assertTrue(sp.simplify(Hn - Hc).is_zero_matrix)
        self.assertTrue(sp.simplify(Sn - Sc).is_zero_matrix)

        # and both agree with the structure_vector ground truth
        dets = [p.dets[0].det_string for p in generate_dets(2, 2, 4)]
        Hfci, Sfci = hamiltonian(m, generate_dets(2, 2, 4))
        V = sp.Matrix.hstack(*[structure_vector(st, dets) for st in natural])
        self.assertTrue(sp.simplify(Hn - V.T * Hfci * V).is_zero_matrix)
        self.assertTrue(sp.simplify(Sn - V.T * Sfci * V).is_zero_matrix)
        # the crossed long bond does couple to the canonical structure (the bug
        # dropped this to zero)
        self.assertFalse(sp.simplify(Hn[0, 1]).is_zero)

    def test_canonical_input_is_unchanged(self):
        # standardize() is a no-op on an already-canonical determinant basis
        for p in generate_dets(2, 2, 3):
            before = [(d.det_string, c) for d, c in p]
            FixedPsi(p).standardize()          # copy; original must be intact
            fp = FixedPsi(p); fp.standardize()
            after = [(d.det_string, c) for d, c in fp]
            self.assertEqual(before, after)

        # a non-canonical determinant standardizes with the expected sign
        fp = FixedPsi('abBC'); fp.standardize()      # -> -1 * |aBbC|
        self.assertEqual([(d.det_string, c) for d, c in fp], [('aBbC', -1)])

        # and the facade over a canonical (generate_dets) basis is unchanged:
        # equal to the plain manual build_matrix + o2 route
        m = self._allyl_molecule()
        P = generate_dets(2, 2, 3)
        H, S = hamiltonian(m, P)
        H_manual = m.build_matrix(P, op='H') + m.o2_matrix(P)
        self.assertTrue(sp.simplify(H - H_manual).is_zero_matrix)


if __name__ == '__main__':
    unittest.main()
