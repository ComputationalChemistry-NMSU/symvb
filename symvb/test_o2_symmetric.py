"""
Agreement tests for the symmetry-projected o2 matrix builder.

For each test system, builds H_2 in the totally-symmetric SALC basis two
ways:

    (a) Full det-basis build, projected via U^T @ H @ U.
    (b) Direct symmetric build via o2_matrix_symmetric.

The two results must agree element-wise (after chemist-symmetry symbol
normalisation, same as the rest of the agreement suite). Any mismatch
indicates a bug in the formula or its sign accounting.
"""
import unittest

import numpy as np
import sympy as sp

from symvb import Molecule, SlaterDet
from symvb import symmetry
from symvb._o2_symmetric import o2_matrix_symmetric
from symvb.test_o2_agreement import assert_symbolic_equal


_PPP_2E = {'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
           'M': ('1112', '1121', '1222')}


def _project_full(H_full, U):
    """Sympy matrix projection: U^T @ H @ U."""
    return U.T * H_full * U


def _basis_perms_from_orbital_map(basis, orbital_map):
    """Translate an orbital-label permutation into a basis-index permutation
    using symvb.symmetry.apply_orbital_permutation."""
    det_strings = [fp.dets[0].det_string for fp in basis]

    def canon(s):
        fp = SlaterDet(s).get_sorted()
        if not fp.dets:
            return None, 1
        return fp.dets[0].det_string, int(fp.coefs[0])

    perm, signs = symmetry.apply_orbital_permutation(orbital_map, det_strings, canon)
    if perm is None:
        return None, None
    return perm, signs


def _U_from_orbits(orbits, N):
    """Build the unsigned orbit-sum projector as a sympy matrix to match
    the symbolic H_full. Uses Rational(1, sqrt(size)) — that is,
    1/sqrt(size) -- as uniform weights."""
    k = len(orbits)
    U = sp.zeros(N, k)
    for col, orb in enumerate(orbits):
        weight = sp.Rational(1, 1) / sp.sqrt(len(orb))
        for idx in orb:
            U[idx, col] = weight
    return U


class TestSymmetricO2Agreement(unittest.TestCase):
    """For each (Molecule config, orbital-permutation generators) pair,
    confirm o2_matrix_symmetric == U^T @ o2_matrix(basis) @ U."""

    def _run(self, *, Na, Nb, Norbs, kwargs, generators_orbital):
        """Helper: build H_full both ways and compare element-wise."""
        m = Molecule(o2_method='blocked', **kwargs)
        m.generate_basis(Na, Nb, Norbs)
        N = len(m.basis)

        # Translate orbital-permutation generators -> basis-index perms.
        basis_perms = []
        for orb_map in generators_orbital:
            perm, signs = _basis_perms_from_orbital_map(m.basis, orb_map)
            self.assertIsNotNone(perm, "orbital permutation must preserve the basis")
            self.assertTrue(np.all(signs == 1),
                            "totally_symmetric_basis only valid for unsigned representations")
            basis_perms.append(perm)

        U_np, orbits = symmetry.totally_symmetric_basis(basis_perms, N)
        # Promote U to sympy for symbolic projection.
        U_sym = _U_from_orbits(orbits, N)

        # Path (a): full build + projection.
        H_full = m.o2_matrix(m.basis)
        H_proj = _project_full(H_full, U_sym)

        # Path (b): direct symmetric build.
        H_sym = o2_matrix_symmetric(m, m.basis, orbits)

        self.assertEqual(H_proj.shape, H_sym.shape,
                         "shape mismatch between projected and direct builds")
        for i in range(H_proj.rows):
            for j in range(H_proj.cols):
                with self.subTest(i=i, j=j):
                    assert_symbolic_equal(H_proj[i, j], H_sym[i, j])

    def test_h2_c2(self):
        """H_2 with a <-> b reflection. 4 dets, 2 orbits.
        S_ab is its own orbit so subst is unnecessary here, but we add
        zero_ii=True to silence H_aa terms."""
        self._run(
            Na=1, Nb=1, Norbs=2,
            kwargs=dict(subst_2e=_PPP_2E, interacting_orbs=['ab'],
                        zero_ii=True),
            generators_orbital=[{'a': 'b', 'b': 'a'}],
        )

    def test_allyl_c2_anion(self):
        """Allyl 2alpha+2beta with a <-> c reflection. 9 dets.
        Uses subst to assert S_ab = S_bc and H_ab = H_bc, otherwise the
        Hamiltonian is not invariant under the swap and the symmetry
        projection is invalid (G-invariance precondition violated)."""
        self._run(
            Na=2, Nb=2, Norbs=3,
            kwargs=dict(
                subst_2e=_PPP_2E,
                subst={'s': ('S_ab', 'S_bc'), 'h': ('H_ab', 'H_bc')},
                interacting_orbs=['ab', 'bc'],
                max_2e_centers=2,
                zero_ii=True,
            ),
            generators_orbital=[{'a': 'c', 'b': 'b', 'c': 'a'}],
        )

    def test_allyl_c2_cation(self):
        """Allyl 1alpha+1beta with a <-> c reflection."""
        self._run(
            Na=1, Nb=1, Norbs=3,
            kwargs=dict(
                subst_2e=_PPP_2E,
                subst={'s': ('S_ab', 'S_bc'), 'h': ('H_ab', 'H_bc')},
                interacting_orbs=['ab', 'bc'],
                max_2e_centers=2,
                zero_ii=True,
            ),
            generators_orbital=[{'a': 'c', 'b': 'b', 'c': 'a'}],
        )

    def test_4ring_c4_sub_half(self):
        """4-orbital ring with C_4 rotation, sub-half-filled (1alpha+1beta,
        16 dets). Sub-half-filling keeps the cyclic-permutation fermion
        signs at +1 so the unsigned projector is valid; the 4-ring at
        half-filling needs signed_totally_symmetric_basis instead and is
        not exercised here."""
        self._run(
            Na=1, Nb=1, Norbs=4,
            kwargs=dict(
                subst_2e=_PPP_2E,
                subst={'s': ('S_ab', 'S_bc', 'S_cd', 'S_ad'),
                       'h': ('H_ab', 'H_bc', 'H_cd', 'H_ad')},
                interacting_orbs=['ab', 'bc', 'cd', 'ad'],
                max_2e_centers=2,
                zero_ii=True,
            ),
            generators_orbital=[{'a': 'b', 'b': 'c', 'c': 'd', 'd': 'a'}],
        )


class TestSymmetricO2Methods(unittest.TestCase):
    """Sanity checks on the API surface."""

    def test_returns_sympy_matrix(self):
        m = Molecule(o2_method='blocked', subst_2e=_PPP_2E,
                     interacting_orbs=['ab'])
        m.generate_basis(1, 1, 2)
        perm, _ = _basis_perms_from_orbital_map(m.basis, {'a': 'b', 'b': 'a'})
        _, orbits = symmetry.totally_symmetric_basis([perm], len(m.basis))
        H = o2_matrix_symmetric(m, m.basis, orbits)
        self.assertIsInstance(H, sp.Matrix)
        self.assertEqual(H.shape, (len(orbits), len(orbits)))

    def test_trivial_group(self):
        """No generators -> every det is its own orbit -> result equals the
        full o2 matrix."""
        m = Molecule(o2_method='blocked', subst_2e=_PPP_2E,
                     interacting_orbs=['ab', 'bc'], max_2e_centers=2)
        m.generate_basis(1, 1, 3)
        N = len(m.basis)
        _, orbits = symmetry.totally_symmetric_basis([], N)
        self.assertEqual(orbits, [[i] for i in range(N)])
        H_full = m.o2_matrix(m.basis)
        H_sym = o2_matrix_symmetric(m, m.basis, orbits)
        for i in range(N):
            for j in range(N):
                assert_symbolic_equal(H_full[i, j], H_sym[i, j])


if __name__ == '__main__':
    unittest.main()
