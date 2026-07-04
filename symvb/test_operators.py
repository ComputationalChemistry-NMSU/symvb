"""Tests for symvb.operators."""
import unittest

import numpy as np
import sympy as sp

import symvb
from symvb import operators as op
from symvb.functions import generate_det_strings
from symvb.spin import s_squared_matrix, eta_squared_matrix


def _to_np(M):
    return np.array(M, dtype=float)


def _psi_dict(psi):
    """Convert a FixedPsi to {det_string: coef} for assertion convenience."""
    return {d.det_string: c for d, c in psi}


class TestPrimitives(unittest.TestCase):
    def test_n_alpha(self):
        self.assertEqual(_psi_dict(op.number('a', 'alpha').apply('aB')), {'aB': 1})
        self.assertEqual(_psi_dict(op.number('b', 'alpha').apply('aB')), {})

    def test_n_total(self):
        self.assertEqual(_psi_dict(op.number('a').apply('aB')), {'aB': 1})
        self.assertEqual(_psi_dict(op.number('b').apply('aB')), {'aB': 1})
        N = op.number('a') + op.number('b')
        self.assertEqual(_psi_dict(N.apply('aB')), {'aB': 2})

    def test_double_occ(self):
        self.assertEqual(_psi_dict(op.double_occ('a').apply('aA')), {'aA': 1})
        self.assertEqual(_psi_dict(op.double_occ('a').apply('aB')), {})
        self.assertEqual(_psi_dict(op.double_occ('a').apply('bB')), {})

    def test_hop_one_way(self):
        # c†_aα c_bα |bA> = |aA>
        self.assertEqual(
            _psi_dict(op.hop('a', 'b', 'alpha', hermitian=False).apply('bA')),
            {'aA': 1},
        )

    def test_hop_zero_when_target_occupied(self):
        # On |aA>, c_bα|aA> = 0, so c†_aα c_bα|aA> = 0
        self.assertEqual(
            _psi_dict(op.hop('a', 'b', 'alpha', hermitian=False).apply('aA')),
            {},
        )


class TestSpin(unittest.TestCase):
    def test_sz_diag_sz_zero(self):
        b = generate_det_strings(2, 2, 4)
        Sz = op.s_z(['a', 'b', 'c', 'd']).matrix(b)
        for i in range(len(b)):
            self.assertEqual(Sz[i, i], 0)

    def test_sz_diag_sz_one(self):
        b = generate_det_strings(3, 1, 4)
        Sz = op.s_z(['a', 'b', 'c', 'd']).matrix(b)
        for i in range(len(b)):
            self.assertEqual(Sz[i, i], 1)

    def test_s_plus_raises_sz(self):
        # S+ |aB> creates |ab> (both alpha, M_S = +1)
        out = op.s_plus(['a', 'b']).apply('aB')
        self.assertEqual(_psi_dict(out), {'ab': 1})

    def test_s_squared_h2_matches_spinpy(self):
        basis = ['aB', 'bA', 'aA', 'bB']
        M_new = _to_np(op.s_squared(['a', 'b']).matrix(basis))
        M_old = s_squared_matrix(basis, orbs=['a', 'b'])
        np.testing.assert_allclose(M_new, M_old, atol=1e-12)

    def test_s_squared_h4_matches_spinpy(self):
        basis = generate_det_strings(2, 2, 4)
        M_new = _to_np(op.s_squared(['a', 'b', 'c', 'd']).matrix(basis))
        M_old = s_squared_matrix(basis, orbs=['a', 'b', 'c', 'd'])
        np.testing.assert_allclose(M_new, M_old, atol=1e-12)

    def test_s_squared_h2_eigenvalues(self):
        # In ['aB','bA','aA','bB','ab','AB']: 3 singlets + 3 triplets
        basis = ['aB', 'bA', 'aA', 'bB', 'ab', 'AB']
        M = _to_np(op.s_squared(['a', 'b']).matrix(basis))
        eigs = sorted(np.linalg.eigvalsh((M + M.T) / 2).round(8).tolist())
        self.assertEqual(eigs.count(0), 3)
        self.assertEqual(eigs.count(2), 3)

    def test_s_dot_singlet_triplet(self):
        # |aB>, |bA>: S_a·S_b eigenvalues -3/4 (singlet) and +1/4 (triplet)
        basis = ['aB', 'bA']
        M = _to_np(op.s_dot('a', 'b').matrix(basis))
        eigs = sorted(np.linalg.eigvalsh((M + M.T) / 2).round(8).tolist())
        self.assertEqual(eigs, [-0.75, 0.25])


class TestEtaPairing(unittest.TestCase):
    def test_eta_squared_h4_matches_spinpy(self):
        basis = generate_det_strings(2, 2, 4)
        site_signs = {'a': 1, 'b': -1, 'c': 1, 'd': -1}
        M_new = _to_np(op.eta_squared(site_signs).matrix(basis))
        M_old = eta_squared_matrix(basis, site_signs, orbs=['a', 'b', 'c', 'd'])
        np.testing.assert_allclose(M_new, M_old, atol=1e-12)

    def test_eta_plus_creates_pair(self):
        # On |aA>: a is doubly occupied, only the b-site term contributes,
        # producing |aAbB> (interleaved canonical for α={a,b}, β={a,b})
        # with a fermion sign whose magnitude must be 1.
        site_signs = {'a': 1, 'b': 1}
        out = op.eta_plus(site_signs).apply('aA')
        d = _psi_dict(out)
        self.assertEqual(set(d.keys()), {'aAbB'})
        self.assertIn(d['aAbB'], (1, -1))

    def test_eta_z_at_half_filling(self):
        # η_z = (N̂ − L)/2; at N = L = 4 it is the zero matrix.
        b = generate_det_strings(2, 2, 4)
        Ez = op.eta_z(['a', 'b', 'c', 'd']).matrix(b)
        self.assertEqual(Ez, sp.zeros(len(b), len(b)))


class TestSymmetry(unittest.TestCase):
    def test_transposition_swaps_dets(self):
        # P_ab |aB>: in interleaved JW the swap a↔b carries a fermion
        # sign; the user-string output sign reflects that.
        out = _psi_dict(op.transposition('a', 'b').apply('aB'))
        self.assertEqual(set(out.keys()), {'bA'})
        self.assertIn(out['bA'], (1, -1))

        out2 = _psi_dict(op.transposition('a', 'b').apply('aA'))
        self.assertEqual(set(out2.keys()), {'bB'})
        self.assertIn(out2['bB'], (1, -1))

    def test_reynolds_z2(self):
        # Z_2 (a↔b) Reynolds projector: ½(I + P_ab).
        # On |aB>: produces ½|aB> + (sign/2)|bA>.  Sign depends on JW
        # convention; in interleaved + user-string this comes out
        # consistent with the s_squared sign pattern: ½|aB> + ½|bA>.
        R = op.reynolds_projector([{'a': 'b', 'b': 'a'}])
        out = _psi_dict(R.apply('aB'))
        self.assertEqual(set(out.keys()), {'aB', 'bA'})
        self.assertEqual(out['aB'], sp.Rational(1, 2))
        self.assertEqual(out['bA'], sp.Rational(1, 2))

    def test_reynolds_idempotent(self):
        R = op.reynolds_projector([{'a': 'b', 'b': 'a'}])
        basis = ['aB', 'bA', 'aA', 'bB']
        M = R.matrix(basis)
        M2 = (M * M).applyfunc(sp.simplify)
        self.assertEqual(M, M2)

    def test_orbital_perm_h4_ring(self):
        basis = generate_det_strings(2, 2, 4)
        C4 = op.orbital_perm({'a': 'b', 'b': 'c', 'c': 'd', 'd': 'a'})
        M = _to_np(C4.matrix(basis))
        for j in range(M.shape[1]):
            col = M[:, j]
            self.assertEqual(int(np.sum(np.abs(col) > 1e-9)), 1)
            self.assertAlmostEqual(np.max(np.abs(col)), 1.0)


class TestVB(unittest.TestCase):
    def test_singlet_proj_singly_occupied_sector(self):
        basis = ['aB', 'bA']
        M = _to_np(op.singlet_proj('a', 'b').matrix(basis))
        eigs = sorted(np.linalg.eigvalsh((M + M.T) / 2).round(8).tolist())
        self.assertEqual(eigs, [0.0, 1.0])

    def test_singlet_state_has_zero_s_squared(self):
        # The singlet (|aB> + |bA>)/sqrt(2) is annihilated by S^2.
        psi = symvb.FixedPsi('aB') + symvb.FixedPsi('bA')
        self.assertEqual(op.s_squared(['a', 'b']).expectation(psi), 0)

    def test_triplet_state_has_s_squared_two(self):
        # Triplet T_0 = (|aB> - |bA>)/sqrt(2): S² eigenvalue 2.
        psi = symvb.FixedPsi('aB') - symvb.FixedPsi('bA')
        # ⟨ψ|ψ⟩ = 2, so ⟨ψ|S²|ψ⟩ = 2·2 = 4.
        self.assertEqual(op.s_squared(['a', 'b']).expectation(psi), 4)


class TestAlgebra(unittest.TestCase):
    def test_sum_distributes(self):
        self.assertEqual(_psi_dict((op.number('a') + op.number('b')).apply('aB')),
                         {'aB': 2})

    def test_scalar_multiply(self):
        self.assertEqual(_psi_dict((3 * op.number('a')).apply('aB')), {'aB': 3})
        self.assertEqual(_psi_dict((op.number('a') * sp.Rational(1, 2)).apply('aB')),
                         {'aB': sp.Rational(1, 2)})

    def test_matmul_is_composition(self):
        nn = op.cdag('a', 'alpha') @ op.c('a', 'alpha')
        self.assertEqual(_psi_dict(nn.apply('aB')), {'aB': 1})

    def test_canonicalize_round_trip(self):
        for s in ['aB', 'Ab', 'aAbB', 'BabA']:
            cs1, sg1 = op.canonicalize(s)
            cs2, sg2 = op.canonicalize(cs1)
            self.assertEqual(cs1, cs2)
            # cs1 fed back through canonicalize should give the same canonical string.

    def test_apply_accepts_fixedpsi(self):
        psi = symvb.FixedPsi('aB') + symvb.FixedPsi('bA')
        out = op.s_z(['a', 'b']).apply(psi)
        # Sz = 0 on every Sz=0 det.
        self.assertEqual(_psi_dict(out), {})

    def test_matrix_accepts_fixedpsi_basis(self):
        # Build A_1g symmetric / antisymmetric basis as FixedPsis,
        # and check S² is block diagonal w.r.t. that basis.
        sym  = symvb.FixedPsi('aB') + symvb.FixedPsi('bA')
        anti = symvb.FixedPsi('aB') - symvb.FixedPsi('bA')
        M = op.s_squared(['a', 'b']).matrix([sym, anti])
        # symmetric is the singlet (eigenvalue 0); antisymmetric is the
        # triplet T_0 (eigenvalue 2).  Off-diagonal entries should
        # vanish; diagonal entries are ⟨ψ|S²|ψ⟩ = norm² × eigenvalue.
        self.assertEqual(M[0, 1], 0)
        self.assertEqual(M[1, 0], 0)
        self.assertEqual(M[0, 0], 0)
        # ⟨anti|anti⟩ = 2, eigenvalue 2 → ⟨anti|S²|anti⟩ = 4.
        self.assertEqual(M[1, 1], 4)


if __name__ == '__main__':
    unittest.main()
