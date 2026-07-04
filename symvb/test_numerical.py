from unittest import TestCase
from symvb import Molecule, FixedPsi
from symvb.numerical import get_coupled, get_combined, validate_solution, get_combined_from_dict
from symvb.fixed_psi import generate_dets


class Test(TestCase):

    def test_get_coupled(self):
        m = Molecule(zero_ii=True, subst={'s': ('S_ab', 'S_ac', 'S_bc'), 'h': ('H_ab', 'H_ac', 'H_bc')})

        P = [0, ] * 6
        P[0] = FixedPsi('Ab', coupled_pairs=[(0, 1)])
        P[1] = FixedPsi('Bc', coupled_pairs=[(0, 1)])
        P[2] = FixedPsi('Ac', coupled_pairs=[(0, 1)])
        P[3] = FixedPsi('Aa')
        P[4] = FixedPsi('Bb')
        P[5] = FixedPsi('Cc')
        mH = m.build_matrix(P, op='H')
        mS = m.build_matrix(P, op='S')
        self.assertEqual(
            str(get_coupled(mH=mH, mS=mS)),
            '{0: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}}'
        )

    def test_get_combined(self):
        P = [0, ] * 6

        P[0] = FixedPsi('Ab', coupled_pairs=[(0, 1)])
        P[1] = FixedPsi('Bc', coupled_pairs=[(0, 1)])
        P[2] = FixedPsi('Ac', coupled_pairs=[(0, 1)])
        P[3] = FixedPsi('Aa')
        P[4] = FixedPsi('Bb')
        P[5] = FixedPsi('Cc')
        P2 = get_combined(P, [0, 1, 2, 3, 4, 5])

        self.assertEqual(
            str(P2),
            '[|Ab|+|Ba|+|Bc|+|Cb|+|Ac|+|Ca|+|Aa|+|Bb|+|Cc|]'
        )

    def test_validate_solution(self):
        m = Molecule(zero_ii=True, subst={'s': ('S_ab', 'S_ac', 'S_bc'), 'h': ('H_ab', 'H_ac', 'H_bc')})

        P = [0, ] * 6
        P[0] = FixedPsi('Ab', coupled_pairs=[(0, 1)])
        P[1] = FixedPsi('Bc', coupled_pairs=[(0, 1)])
        P[2] = FixedPsi('Ac', coupled_pairs=[(0, 1)])
        P[3] = FixedPsi('Aa')
        P[4] = FixedPsi('Bb')
        P[5] = FixedPsi('Cc')

        mS = m.build_matrix(P, op='S')
        mH = m.build_matrix(P, op='H')

        P2 = get_combined(P, [0, 1, 2, 3, 4, 5])
        result = m.energy(P2[0])

        match = validate_solution(result, mH, mS)
        self.assertTrue(match)

    def test_get_combined_from_dict(self):
        m = Molecule(zero_ii=True, subst={'s': ('S_ab', 'S_ac', 'S_bc'), 'h': ('H_ab', 'H_ac', 'H_bc')})
        PP = generate_dets(1, 1, 3)
        mH = m.build_matrix(PP, op='H')
        mS = m.build_matrix(PP, op='S')
        d = get_coupled(mH=mH, mS=mS)
        result = get_combined_from_dict(PP,d)
        self.assertEqual(
            str(result),
            '[|aA|+|aB|+|aC|+|bA|+|bB|+|bC|+|cA|+|cB|+|cC|]'
        )
