import unittest
import sympy as sp

from symvb import FixedPsi, Molecule, SlaterDet
from symvb.fixed_psi import generate_dets


class TestMolecule(unittest.TestCase):

    def test_subst_2e_1(self):
        m = Molecule(subst_2e={'J': ('T_abab', 'T_bcbc')})
        self.assertEqual(
            m.subst_2e['T_abab'],
            'J'
        )

    def test_subst_2e_2(self):
        m = Molecule(subst_2e={'J': ('1221',), 'K': ('1212',)})
        d1 = SlaterDet('aB')
        d2 = SlaterDet('aB')
        s = m.o2_det(d1, d2)
        self.assertEqual(
            str(sp.simplify(s)),
            'K'
        )

    def test_op_hartree_product_1(self):
        self.assertEqual(
            str(Molecule().Op_Hartree_product('AaBb', 'CcDd')),
            '2*H_ac*S_ac*S_bd**2 + 2*H_bd*S_ac**2*S_bd'
        )

    def test_op_hartree_product_2(self):
        self.assertEqual(
            str(Molecule().Op_Hartree_product('aAbBcC', 'aAbBcC', op='H')),
            '0'
        )

    def test_op_hartree_product_3(self):
        self.assertEqual(
            str(Molecule().Op_Hartree_product('aA', 'aB', op='H')),
            'H_ab'
        )

    def test_op_hartree_product_4(self):
        self.assertEqual(
            str(Molecule().Op_Hartree_product('AaBb', 'CcDd', op='S')),
            'S_ac**2*S_bd**2'
        )

    def test_op_1(self):
        self.assertEqual(
            str(Molecule(zero_ii=False).Ops('AbCd', 'AbCd')),
            '-H_aa*S_bd**2 + H_aa + 2*H_ac*S_ac*S_bd**2 - 2*H_ac*S_ac - H_bb*S_ac**2 + H_bb + 2*H_bd*S_ac**2*S_bd - 2*H_bd*S_bd - H_cc*S_bd**2 + H_cc - H_dd*S_ac**2 + H_dd'
        )

    def test_op_2(self):
        self.assertEqual(
            str(Molecule().Ops('', '')),
            '0'
        )

    def test_op_3(self):
        self.assertEqual(
            str(Molecule().Ops('', '', op='S')),
            '1'
        )

    def test_op_4(self):
        self.assertEqual(
            str(Molecule(zero_ii=False).Ops('A', 'A')),
            'H_aa'
        )

    def test_op_5(self):
        self.assertEqual(
            str(Molecule(zero_ii=False).Ops('b', 'b')),
            'H_bb'
        )

    def test_op_6(self):
        self.assertEqual(
            str(Molecule(zero_ii=False).Ops('aA', 'bB')),
            '2*H_ab*S_ab'
        )

    def test_op_7(self):
        self.assertEqual(
            str(Molecule(zero_ii=False, interacting_orbs=['ab']).Ops('AaBb', 'CcDd')),
            '0'
        )

    def test_op_8(self):
        self.assertEqual(
            str(Molecule(zero_ii=False, interacting_orbs=['ab', 'bc', 'cd']).Ops('ABbCcd', 'ABbCcd', op='S')),
            '(S_ab**2 + S_bc**2 - 1)*(S_bc**2 + S_cd**2 - 1)'
        )

    def test_op_det_fast_1(self):
        m = Molecule(zero_ii=True, subst={'s': ('S_ab', 'S_ac', 'S_bc'), 'h': ('H_ab', 'H_ac', 'H_bc')})
        m.generate_basis(2, 2, 3)
        sd = m.basis[0].dets[0]
        m = m.op_det(sd, sd)
        self.assertEqual(
            str(m),
            '-4*h*s*(1 - s**2)'
        )

    def test_op_det_fast_2(self):
        m = Molecule(zero_ii=True, subst={'s': ('S_ab', 'S_ac', 'S_bc'), 'h': ('H_ab', 'H_ac', 'H_bc')})
        m.generate_basis(2, 2, 3)
        sd = m.basis[0].dets[0]
        m = m.op_det(sd, sd, op='S')
        self.assertEqual(
            str(m),
            '(1 - s**2)**2'
        )

    def test_op_det_fast_3(self):
        m = Molecule(zero_ii=True,
                     subst={
                         's': ('S_ab', 'S_bc', 'S_cd'),
                         'h': ('H_ab', 'H_bc', 'H_cd')},
                     interacting_orbs=['ab', 'bc', 'cd']
                     )
        m.generate_basis(2, 2, 4)
        M2 = m.build_matrix(m.basis, op='S')
        self.assertEqual(
            str(M2[0, 1]),
            's*(1 - s**2)'
        )

    def test_build_matrix(self):
        m = Molecule(zero_ii=True, subst={'s': ('S_ab', 'S_ac', 'S_bc'), 'h': ('H_ab', 'H_ac', 'H_bc')})
        P = [0, ] * 3
        P[0] = FixedPsi('Ab', coupled_pairs=[(0, 1)])
        P[1] = FixedPsi('Bc', coupled_pairs=[(0, 1)])
        P[2] = FixedPsi('Ac', coupled_pairs=[(0, 1)])
        m = m.build_matrix(P, op='S')
        self.assertEqual(
            str(m),
            'Matrix([[2*s**2 + 2, 2*s**2 + 2*s, 2*s**2 + 2*s], [2*s**2 + 2*s, 2*s**2 + 2, 2*s**2 + 2*s], [2*s**2 + 2*s, 2*s**2 + 2*s, 2*s**2 + 2]])'
        )

    def test_normalized_self_energy(self):
        m = Molecule(zero_ii=False)
        P = FixedPsi('AB')
        self.assertEqual(
            str(m.energy(P)),
            '-(H_aa - 2*H_ab*S_ab + H_bb)/((S_ab - 1)*(S_ab + 1))'
        )

    def test_couple(self):
        m = Molecule(zero_ii=True, subst={'s': ('S_ab', 'S_ac', 'S_bc'), 'h': ('H_ab', 'H_ac', 'H_bc')})
        P = generate_dets(1, 1, 3)
        result = m.couple(P)
        self.assertEqual(
            str(result),
            '[|aA|+|aB|+|aC|+|bA|+|bB|+|bC|+|cA|+|cB|+|cC|]'
        )

    def test_couple2(self):
        m = Molecule(zero_ii=True, subst={'s': ('S_ab', 'S_bc'), 'h': ('H_ab', 'H_bc')}, interacting_orbs=['ab', 'bc'])
        P = generate_dets(1, 1, 3)
        result = m.couple(P, N_tries=5, ranges={'s':(.5,.6), 'h':(-.8,-.7)})
        self.assertEqual(
            str(result),
            '[|aA|+1.414213562373|aB|+|aC|+1.414213562373|bA|+2|bB|+1.414213562373|bC|+|cA|+1.414213562373|cB|+|cC|]'
        )

    def test_couple3(self):
        m = Molecule(zero_ii=False,
                     interacting_orbs=['ab'],
                     subst={'H_aa': ['H_aa', 'H_bb']},
                     subst_2e={'R': ('1111'), 'J': ('1212'), 'K': ('1122'), 'M': ('1112', '1121', '1222')},
                     max_2e_centers=1
                     )
        #  K, J confirmed
        nums = {'R': 0.77460594, 'M': 0.30930897, 'K': 0.15786578, 'J': 0.47804137,
                'H_aa': -0.97949638, 'H_ab': -0.68286493,
                'S_ab': 0.49648469}

        P0 = generate_dets(1, 1, 2)
        o2_1 = m.o2_matrix(P0)

        S = m.build_matrix(P0, op='S')
        H = m.build_matrix(P0, op='H')

        P1 = m.couple(P0, mS=S, mH=H + o2_1,
                      ranges={'H_aa': (-1.0, -0.8), 'H_ab': (-0.7, -0.5), 'S_ab': (0.05, 0.2), 'R': (.7, .8)})

        self.assertEqual(
            str(P1),
            '[|aA|+|bB|, |aB|+|bA|]'
        )

    def test_o2_det_1(self):
        m = Molecule(zero_ii=True, subst={'s': ('S_ab',), 'h': ('H_ab',)}, interacting_orbs=['ab', ])
        d1 = SlaterDet('aB')
        d2 = SlaterDet('aB')
        s = m.o2_det(d1, d2)
        self.assertEqual(
            str(sp.simplify(s)),
            'T_abab'
        )

    def test_o2_det_2(self):
        m = Molecule(zero_ii=True, subst={'s': ('S_ab',), 'h': ('H_ab',)}, interacting_orbs=['ab', ])
        d1 = SlaterDet('ab')
        d2 = SlaterDet('ab')
        s = m.o2_det(d1, d2)
        self.assertEqual(
            str(sp.simplify(s)),
            '-T_aabb + T_abab'
        )

    def test_o2_det_3(self):
        m = Molecule(zero_ii=True,
                     subst={'s': ('S_ab', 'S_bc', 'S_cd'),
                            'h': ('H_ab', 'H_bc', "H_cd")},
                     interacting_orbs=['ab', 'bc', 'cd'])
        d1 = SlaterDet('aBc')
        d2 = SlaterDet('aBd')
        s = m.o2_det(d1, d2)
        self.assertEqual(
            str(sp.simplify(s)),
            '-T_aacd + T_abab*s + T_acad + T_bcbd'
        )

    def test_o2_det_4(self):
        m = Molecule(zero_ii=False,
                     interacting_orbs=['ab', 'bc', 'ac'],
                     subst={'h': ['H_aa', 'H_bb', 'H_cc'],
                            'H_ab': ['H_ab', 'H_bc', 'H_ac'],
                            's': ['S_ab', 'S_bc', 'S_ac']},
                     subst_2e={'R': ('1111'), 'J': ('1212'), 'K': ('1122'), 'M': ('1112', '1121', '1222')},
                     max_2e_centers=2,
                     )
        P = generate_dets(1, 1, 3)
        # z = m.op_fixed_psi(P[0], P[5])
        o2 = m.o2_matrix(P)

        self.assertEqual(
            str(o2[0, 0]),
            'R'
        )

    def test_energy_1(self):
        m = Molecule(zero_ii=False,
                     interacting_orbs=['ab'],
                     subst={'h': ['H_aa', 'H_bb']},
                     subst_2e={'R': ('1111'), 'J': ('1212'), 'K': ('1122'), 'M': ('1112', '1121', '1222')}
                     )
        c1 = SlaterDet('a') + SlaterDet('b')
        C1 = SlaterDet('A') + SlaterDet('B')
        self.assertEqual(
            str(m.energy(c1*C1)),
            '2*(H_ab + h)/(S_ab + 1)'
        )


    def test_energy_2(self):
        m = Molecule(zero_ii=False,
                     interacting_orbs=['ab'],
                     subst={'h': ['H_aa', 'H_bb']},
                     subst_2e={'R': ('1111'), 'J': ('1212'), 'K': ('1122'), 'M': ('1112', '1121', '1222')}
                     )
        c1 = SlaterDet('a') + SlaterDet('b')
        C1 = SlaterDet('A') + SlaterDet('B')
        self.assertEqual(
            str(m.energy(c1*C1, o2=True)),
            '2*(H_ab + h)/(S_ab + 1) + (J + 2*K + 4*M + R)/(2*(S_ab + 1)**2)'

        )
    def test_o2_fixed_psi_1(self):
        m = Molecule(zero_ii=True,
                     subst={'s': ('S_ab', 'S_bc', 'S_cd'),
                            'h': ('H_ab', 'H_bc', "H_cd")},
                     interacting_orbs=['ab', 'bc', 'cd'])
        fp1 = FixedPsi('aBc')
        fp2 = FixedPsi('aBd')
        s = m.o2_fixed_psi(fp1, fp2)
        self.assertEqual(
            str(sp.simplify(s)),
            '-T_aacd + T_abab*s + T_acad + T_bcbd'
        )

    def test_o2_matrix(self):
        m = Molecule(subst_2e={'R': ('1111'), 'J': ('1212'), 'K': ('1122'), 'M': ('1112', '1121', '1222')})
        f1 = SlaterDet('aA') + SlaterDet('bB')
        f2 = SlaterDet('aB') + SlaterDet('bA')
        u = [f1, f2]
        r = m.o2_matrix(u)
        self.assertEqual(
            str(r),
            'Matrix([[2*K + 2*R, 4*M], [4*M, 2*J + 2*K]])'
        )

    def test_o2_matrix_2(self):
        m = Molecule(zero_ii=True,
                     subst={
                         's': ('S_ab',),
                         'h': ('H_ab',)},
                     interacting_orbs=['ab'],
                     subst_2e={'R': ('1111'), 'J': ('1212'), 'K': ('1122'), 'M': ('1112', '1121', '1222')}
                     )
        c1 = SlaterDet('a') + SlaterDet('b')
        c2 = SlaterDet('A') + SlaterDet('B')
        c3 = SlaterDet('a') - SlaterDet('b')
        c4 = SlaterDet('A') - SlaterDet('B')
        f1 = c1 * c2
        f2 = c3 * c4
        P = [f1, f2]
        z = m.o2_matrix(P)
        self.assertEqual(
            str(z),
            'Matrix([[2*J + 4*K + 8*M + 2*R, -2*J + 2*R], [-2*J + 2*R, 2*J + 4*K - 8*M + 2*R]])'
        )

    def test_build_matrix_2(self):
        m = Molecule(zero_ii=True,
                     subst={
                         's': ('S_ab',),
                         'h': ('H_ab',)},
                     interacting_orbs=['ab'],
                     subst_2e={'R': ('1111'), 'J': ('1212'), 'K': ('1122'), 'M': ('1112', '1121', '1222')}
                     )
        c1 = SlaterDet('a') + SlaterDet('b')
        c2 = SlaterDet('A') + SlaterDet('B')
        c3 = SlaterDet('a') - SlaterDet('b')
        c4 = SlaterDet('A') - SlaterDet('B')
        f1 = c1 * c2
        f2 = c3 * c4
        P = [f1, f2]
        z = m.build_matrix(P, op='S')
        self.assertEqual(
            str(z),
            'Matrix([[4*s**2 + 8*s + 4, 0], [0, 4*s**2 - 8*s + 4]])'
        )

    def test_get_o2_name_1(self):
        m = Molecule(zero_ii=True,
                     subst={
                         's': ('S_ab',),
                         'h': ('H_ab',)},
                     interacting_orbs=['ab'],
                     subst_2e={'R': ('1111'), 'J': ('1212'), 'K': ('1122'), 'M': ('1112', '1121', '1222')}
                     )
        v = ['a', 'b', 'a', 'b']  # <ab|ab>
        self.assertEqual(
            m.get_o2_name(v),
            'J'
        )

    def test_o2_mo2ao_1(self):
        c1 = SlaterDet('a') + SlaterDet('b')
        m = Molecule(zero_ii=False,
                     interacting_orbs=['ab'],
                     subst={'h': ['H_aa', 'H_bb']},
                     subst_2e={'R': ('1111'), 'J': ('1212'), 'K': ('1122'), 'M': ('1112', '1121', '1222')}
                     )
        rs = sp.simplify(m.o2_mo2ao(c1, c1, c1, c1))
        self.assertEqual(
            str(rs),
            '2*J + 4*K + 8*M + 2*R'
        )

    def test_get_mo_norm_1(self):
        c1 = SlaterDet('a') + SlaterDet('b')
        m = Molecule(zero_ii=False,
                     interacting_orbs=['ab'],
                     subst={'h': ['H_aa', 'H_bb']},
                     subst_2e={'R': ('1111'), 'J': ('1212'), 'K': ('1122'), 'M': ('1112', '1121', '1222')}
                     )
        rs = sp.simplify(m.get_mo_norm([c1, ]))
        self.assertEqual(
            str(rs),
            'Matrix([[sqrt(2)/(2*sqrt(S_ab + 1))]])'
        )

    def test_get_fock_1(self):
        c1 = SlaterDet('a') + SlaterDet('b')
        C1 = SlaterDet('A') + SlaterDet('B')
        c3 = SlaterDet('a') - SlaterDet('b')
        C3 = SlaterDet('A') - SlaterDet('B')
        m = Molecule(zero_ii=False,
                     interacting_orbs=['ab'],
                     subst={'h': ['H_aa', 'H_bb']},
                     subst_2e={'R': ('1111'), 'J': ('1212'), 'K': ('1122'), 'M': ('1112', '1121', '1222')}
                     )
        rs = sp.simplify(m.get_fock([c1, C1, c3, C3], Nel=2))
        self.assertEqual(
            str(rs),
            'Matrix([[(J + 2*K + 4*M + R)/(2*(S_ab + 1)**2), 0, 0, 0], [0, (J + 2*K + 4*M + R)/(2*(S_ab + 1)**2), 0, 0], [0, 0, (-3*J + 4*K - R)/(2*(S_ab**2 - 1)), 0], [0, 0, 0, (-3*J + 4*K - R)/(2*(S_ab**2 - 1))]])'
        )

    def test_get_rhf_fock_1(self):
        c1 = SlaterDet('a') + SlaterDet('b')
        c3 = SlaterDet('a') - SlaterDet('b')
        m = Molecule(zero_ii=False,
                     interacting_orbs=['ab'],
                     subst={'h': ['H_aa', 'H_bb']},
                     subst_2e={'R': ('1111'), 'J': ('1212'), 'K': ('1122'), 'M': ('1112', '1121', '1222')}
                     )
        rs = sp.simplify(m.get_rhf_fock([c1, c3, ], Nel=2))
        self.assertEqual(
            str(rs),
            'Matrix([[(J + 2*K + 4*M + R)/(2*(S_ab + 1)**2), 0], [0, (-3*J + 4*K - R)/(2*(S_ab**2 - 1))]])'
        )

    def test_get_rhf_mo_energies_1(self):
        m = Molecule(zero_ii=False,
                     interacting_orbs=['ab'],
                     subst={'h': ['H_aa', 'H_bb']},
                     subst_2e={'R': ('1111'), 'J': ('1212'), 'K': ('1122'), 'M': ('1112', '1121', '1222')}
                     )
        #  K, J confirmed
        nums = {'R': 0.77460594, 'M': 0.30930897, 'K': 0.15786578, 'J': 0.47804137,
                'h': -0.97949638, 'H_aa': -0.97949638, 'H_bb': -0.97949638, 'H_ab': -0.68286493,
                'S_ab': 0.49648469, 's': 0.49648469}

        c1 = SlaterDet('a') + SlaterDet('b')
        c3 = SlaterDet('a') - SlaterDet('b')
        z = m.get_rhf_mo_energies([c1, c3], Nel = 2)

        self.assertEqual(
            str(z.subs(nums)),
            'Matrix([[-0.484441678333514, 0.457501911989868]])'
        )

    def test_get_rhf_mo_energies_2(self):
        m = Molecule(zero_ii=False,
                     interacting_orbs=['ab', 'bc', 'ac'],
                     subst={'E_0': ['H_aa', 'H_bb', 'H_cc'],
                            'h': ['H_ab', 'H_bc', 'H_ac'],
                            's': ['S_ab', 'S_bc', 'S_ac']},
                     subst_2e={'R': ('1111'), 'J': ('1212'), 'K': ('1122'), 'M': ('1112', '1121', '1222'),
                               'T_1': ('1213', '1323', '1232'),
                               'T_2': ('1123', '1132', '1223', '1233')},
                     max_2e_centers=3,
                     )
        #  K, J confirmed
        nums = {'R': 0.77460594, 'M': 0.30930897, 'K': 0.15786578, 'J': 0.47804137,
                'E_0': -1.4924109, 'H_aa': -1.4924109, 'H_bb': -1.4924109,
                'h': -0.95694583, 'H_ab': -0.95694583,
                'T_1': 0.2484902,
                'T_2': 0.14232937,
                'S_ab': 0.49648469, 's': 0.49648469}

        mo0_a = SlaterDet('a') + SlaterDet('b') + SlaterDet('c')
        mo1_a = -SlaterDet('a') + SlaterDet('c')
        mo2_a = SlaterDet('a') - SlaterDet('b') - SlaterDet('b') + SlaterDet('c')
        # mo2_a = SlaterDet('a') - 2 * SlaterDet('b') + SlaterDet('c') # does not work

        mos = [mo0_a, mo1_a, mo2_a]
        z = m.get_rhf_mo_energies(mos, Nel = 2)

        self.assertEqual(
            str(z.subs(nums)),
            'Matrix([[-1.12428643604810, -0.0696699523024116, -0.0696699523024116]])'
        )


if __name__ == '__main__':
    unittest.main()


