"""Tests for symmetry.signed_totally_symmetric_basis_exact.

Covers the three correctness gates of the exact signed A_1 builder:
(a) span equality with the numeric `signed_totally_symmetric_basis` on the
    C4H4-dianion filling and on the sign-carrying half-filled 4-ring;
(b) reduction to the unsigned `totally_symmetric_basis` span when every
    group action carries sign +1 (half-filled rings);
(c) reproduction of the hand-rolled symbolic sigma=+1 projector used by
    the allyl PT example scripts.
"""
import unittest

import numpy as np
import sympy as sp

from symvb import Molecule, SlaterDet, symmetry
from symvb.fixed_psi import generate_dets


def _canon(ds):
    fp = SlaterDet(ds).get_sorted()
    return fp.dets[0].det_string, fp.coefs[0]


def _proj(B):
    """Numerical projector onto the column span of B."""
    B = np.array(B, dtype=float)
    if B.shape[1] == 0:
        return np.zeros((B.shape[0], B.shape[0]))
    return B @ np.linalg.inv(B.T @ B) @ B.T


class TestSignedExactVsNumeric(unittest.TestCase):
    """Gate (a): span equality with the numeric signed variant.

    Two cases: the C4H4 dianion of the closed-form example (under the
    current canonical det ordering its C4/sigma_v action happens to carry
    all +1 signs), and the half-filled 4-ring, whose action genuinely
    carries -1 signs and whose signed subspace (dim 6) differs from the
    unsigned orbit basis (dim 7).
    """

    def test_c4h4_dianion_span_matches_numeric(self):
        m = Molecule(
            zero_ii=True,
            subst={'s': ('S_ab', 'S_bc', 'S_cd', 'S_ad'),
                   'h': ('H_ab', 'H_bc', 'H_cd', 'H_ad')},
            interacting_orbs=['ab', 'bc', 'cd', 'ad'],
        )
        m.generate_basis(3, 3, 4)
        dets = [fp.dets[0].det_string for fp in m.basis]
        C4 = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'a'}
        sv = {'a': 'a', 'b': 'd', 'c': 'c', 'd': 'b'}
        signed = [symmetry.apply_orbital_permutation(om, dets, _canon)
                  for om in (C4, sv)]

        U_num, order_num = symmetry.signed_totally_symmetric_basis(
            signed, len(dets))
        U_ex, order_ex = symmetry.signed_totally_symmetric_basis_exact(
            signed, len(dets))

        self.assertEqual(order_ex, order_num)
        self.assertEqual(U_ex.shape[1], U_num.shape[1])
        self.assertEqual(U_ex.shape[1], 3)  # closed-form 3x3 block
        # exact orthonormality (no tolerance)
        self.assertEqual(sp.simplify(U_ex.T * U_ex), sp.eye(3))
        # span equality via projectors
        d = np.max(np.abs(_proj(U_num) - _proj(U_ex.evalf())))
        self.assertLess(d, 1e-10)

    def test_half_filled_4ring_signs_matter(self):
        # (2,2,4) on D_4: the canonical det ordering produces genuine -1
        # signs (18 per generator), and the signed subspace (dim 6) is
        # strictly smaller than the unsigned orbit count (dim 7)
        dets = [p.dets[0].det_string for p in generate_dets(2, 2, 4)]
        C4 = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'a'}
        sv = {'a': 'a', 'b': 'd', 'c': 'c', 'd': 'b'}
        signed = [symmetry.apply_orbital_permutation(om, dets, _canon)
                  for om in (C4, sv)]
        self.assertTrue(any((s == -1).any() for _, s in signed))

        U_num, order_num = symmetry.signed_totally_symmetric_basis(
            signed, len(dets))
        U_ex, order_ex = symmetry.signed_totally_symmetric_basis_exact(
            signed, len(dets))
        U_uns, _ = symmetry.totally_symmetric_basis(
            [p for p, _ in signed], len(dets))

        self.assertEqual(order_ex, order_num)
        self.assertEqual(U_ex.shape[1], U_num.shape[1])
        self.assertLess(U_ex.shape[1], U_uns.shape[1])
        d = np.max(np.abs(_proj(U_num) - _proj(U_ex.evalf())))
        self.assertLess(d, 1e-10)

        # exact invariance: rho(g) U == U symbolically for each generator
        for p, s in signed:
            M = sp.zeros(len(dets), len(dets))
            for i in range(len(dets)):
                M[int(p[i]), i] = int(s[i])
            self.assertEqual(sp.simplify(M * U_ex - U_ex),
                             sp.zeros(len(dets), U_ex.shape[1]))


class TestSignedExactReducesToUnsigned(unittest.TestCase):
    """Gate (b): all-signs-+1 case coincides with the unsigned orbit sums."""

    def test_sub_half_filled_4ring_d4(self):
        dets = [p.dets[0].det_string for p in generate_dets(1, 1, 4)]
        C4 = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'a'}
        sv = {'a': 'a', 'b': 'd', 'c': 'c', 'd': 'b'}
        signed = [symmetry.apply_orbital_permutation(om, dets, _canon)
                  for om in (C4, sv)]
        for _, s in signed:
            self.assertTrue((s == 1).all())

        U_uns, orbits = symmetry.totally_symmetric_basis(
            [p for p, _ in signed], len(dets))
        U_ex, order = symmetry.signed_totally_symmetric_basis_exact(
            signed, len(dets))
        self.assertEqual(order, 8)
        self.assertEqual(U_ex.shape[1], len(orbits))
        # identical columns, not just identical span (same deterministic
        # orbit ordering, all signs +1)
        d = np.max(np.abs(np.array(U_ex.evalf(), dtype=float) - U_uns))
        self.assertLess(d, 1e-12)

    def test_benzene_d6_half_filling(self):
        dets = [p.dets[0].det_string for p in generate_dets(3, 3, 6)]
        C6 = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'e', 'e': 'f', 'f': 'a'}
        sv = {'a': 'a', 'b': 'f', 'c': 'e', 'd': 'd', 'e': 'c', 'f': 'b'}
        signed = [symmetry.apply_orbital_permutation(om, dets, _canon)
                  for om in (C6, sv)]
        for _, s in signed:
            self.assertTrue((s == 1).all())

        U_uns, orbits = symmetry.totally_symmetric_basis(
            [p for p, _ in signed], len(dets))
        U_ex, order = symmetry.signed_totally_symmetric_basis_exact(
            signed, len(dets))
        self.assertEqual(order, 12)
        self.assertEqual(U_ex.shape[1], 38)   # the documented A_1g count
        self.assertEqual(U_ex.shape[1], len(orbits))
        d = np.max(np.abs(np.array(U_ex.evalf(), dtype=float) - U_uns))
        self.assertLess(d, 1e-12)


class TestSignedExactVsHandRolled(unittest.TestCase):
    """Gate (c): the sigma=+1 projector of examples/allyl_hubbard_pt.py."""

    def test_allyl_sigma_plus_subspace(self):
        dets = [p.dets[0].det_string for p in generate_dets(2, 2, 3)]
        sig = {'a': 'c', 'b': 'b', 'c': 'a'}
        perm, signs = symmetry.apply_orbital_permutation(sig, dets, _canon)

        # hand-rolled symbolic projector, verbatim from the example
        U_plus = []
        seen = [False] * 9
        for i in range(9):
            if seen[i]:
                continue
            j = perm[i]
            sj = signs[i]
            if j == i:
                seen[i] = True
                if sj == 1:
                    v = sp.zeros(9, 1)
                    v[i] = 1
                    U_plus.append(v)
            else:
                seen[i] = seen[j] = True
                v = sp.zeros(9, 1)
                v[i] = 1
                v[j] = sj
                U_plus.append(v / sp.sqrt(2))
        Up = sp.Matrix.hstack(*U_plus)

        U_ex, order = symmetry.signed_totally_symmetric_basis_exact(
            [(perm, signs)], 9)
        self.assertEqual(order, 2)
        self.assertEqual(U_ex.shape[1], Up.shape[1])
        # exact span equality: every hand-rolled column lies in the exact
        # subspace (symbolically zero residual), and ranks agree
        R = sp.simplify(Up - U_ex * (U_ex.T * Up))
        self.assertEqual(R, sp.zeros(*R.shape))


class TestSignedExactBehaviour(unittest.TestCase):
    def test_empty_generators_gives_identity(self):
        U, order = symmetry.signed_totally_symmetric_basis_exact([], 4)
        self.assertEqual(order, 1)
        self.assertEqual(U, sp.eye(4))

    def test_frustrated_orbit_drops_out(self):
        # an element FIXING index 0 with sign -1 frustrates that orbit:
        # (1 + (-1))/2 = 0, so index 0 contributes no column
        perm = np.array([0, 1, 2])
        signs = np.array([-1, 1, 1])
        U, order = symmetry.signed_totally_symmetric_basis_exact(
            [(perm, signs)], 3)
        self.assertEqual(order, 2)
        self.assertEqual(U.shape, (3, 2))
        self.assertEqual(U[0, 0], 0)
        self.assertEqual(U[0, 1], 0)

    def test_signed_pair_orbit_survives_with_sign(self):
        # 0<->1 swap where both directions carry sign -1: the invariant
        # combination is (e_0 - e_1)/sqrt(2), not a cancellation
        perm = np.array([1, 0, 2])
        signs = np.array([-1, -1, 1])
        U, order = symmetry.signed_totally_symmetric_basis_exact(
            [(perm, signs)], 3)
        self.assertEqual(order, 2)
        self.assertEqual(U.shape, (3, 2))
        self.assertEqual(sp.simplify(U.T * U), sp.eye(2))
        self.assertEqual(U[0, 0] * U[1, 0], sp.Rational(-1, 2))

    def test_bad_generator_raises(self):
        with self.assertRaises(ValueError):
            symmetry.signed_totally_symmetric_basis_exact(
                [(np.array([0, 0, 1]), np.array([1, 1, 1]))], 3)
        with self.assertRaises(ValueError):
            symmetry.signed_totally_symmetric_basis_exact(
                [(np.array([1, 0, 2]), np.array([2, 1, 1]))], 3)

    def test_max_order_raises(self):
        perm = np.array([1, 2, 3, 0])
        signs = np.array([1, 1, 1, 1])
        with self.assertRaises(RuntimeError):
            symmetry.signed_totally_symmetric_basis_exact(
                [(perm, signs)], 4, max_order=2)


if __name__ == '__main__':
    unittest.main()
