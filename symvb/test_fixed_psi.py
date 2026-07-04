import unittest

from symvb import SlaterDet
from symvb.fixed_psi import attempt_int, FixedPsi


class TestMolecule(unittest.TestCase):
    def test_attempt_int_1(self):
        self.assertEqual(
            str(attempt_int(3.5)),
            '3.5'
        )

    def test_attempt_int(self):
        self.assertEqual(
            str(attempt_int(3.0)),
            '3'
        )

    def test_get_item_1(self):
        fp = FixedPsi('abc')
        self.assertEqual(
            fp[0].det_string,
            'abc'
        )

    def test_get_item_2(self):
        fp = FixedPsi('abc')
        fp.add_str_det('bcd')
        self.assertEqual(
            fp[1].det_string,
            'bcd'
        )

    def test_len(self):
        fp = FixedPsi('abc')
        fp.add_str_det('bcd')
        self.assertEqual(
            len(fp),
            2
        )

    def test_add_str_1(self):
        fp2 = FixedPsi('aA')
        fp2.add_str_det('aB', coef=-1)
        fp2.add_str_det('bA', coef=-1)
        fp2.add_str_det('bB')
        self.assertEqual(
            str(fp2),
            '|aA|-|aB|-|bA|+|bB|'
        )

    def test_iter_1(self):
        fp = FixedPsi('aA')
        fp.add_str_det('bB')
        s = ''
        for d,c in fp:
            s += '%s: %s;  ' % (d,c)
        self.assertEqual(
            s,
            '|aA|: 1;  |bB|: 1;  '
        )

    def test_add_1(self):
        fp = FixedPsi('ab')
        self.assertEqual(
            str(fp + 'bc'),
            '|ab|+|bc|'
        )

    def test_mult_1(self):
        fp = FixedPsi('ab')
        fp += 'bc'
        f2 = fp * 2
        self.assertEqual(
            str(f2)
            , '2|ab|+2|bc|'
        )

    def test_mult_2(self):
        fp = FixedPsi('ab')
        fp += 'bc'
        f2 = fp * 2
        f2 *= 3
        self.assertEqual(
            str(f2)
            , '6|ab|+6|bc|'
        )

    def test_mult_3(self):
        fp = FixedPsi('ab') + SlaterDet('bc')
        fp *= 3
        sd = SlaterDet('de')
        r = fp * sd
        self.assertEqual(
            str(r)
            , '3|abde|+3|bcde|'
        )

    def test_mult_4(self):
        fp = FixedPsi('ab') + SlaterDet('bc')
        fp *= 3
        r = SlaterDet('de') * fp
        self.assertEqual(
            str(r)
            , '3|abde|+3|bcde|'
        )

    def test_mult_5(self):
        f1 = SlaterDet('a') + SlaterDet('b')
        self.assertEqual(
            str(f1 * f1)
            , ''
        )

    def test_mult_6(self):
        f1 = SlaterDet('a') - SlaterDet('b')
        self.assertEqual(
            str(f1 * f1)
            , ''
        )

    def test_mult_7(self):
        f1 = SlaterDet('a') - SlaterDet('b')
        r = f1 * f1 - FixedPsi('aA')
        self.assertEqual(
            str(r)
            , '-|aA|'
        )

    def test_mult_8(self):
        f1 = FixedPsi('a') - FixedPsi('b')
        self.assertEqual(
            str(f1)
            , '|a|-|b|'
        )

    def test_sub_9(self):
        r = FixedPsi('a') - 3*FixedPsi('a')
        self.assertEqual(
            str(r),
            '-2|a|'
        )

    def test_add_2(self):
        c1 = SlaterDet('a') + SlaterDet('a')
        self.assertEqual(
            str(c1),
            '2|a|'
        )

    def test_sub_1(self):
        c1 = SlaterDet('a') - SlaterDet('a')
        self.assertEqual(
            str(c1),
            '||'
        )

    def test_mul_8(self):
        c1 = SlaterDet('a') + SlaterDet('b')
        C1 = SlaterDet('A') + SlaterDet('B')
        c3 = SlaterDet('a') - SlaterDet('b')
        C3 = SlaterDet('A') - SlaterDet('B')
        a = c1 * C1
        b = c3 * C3
        res = a*b
        self.assertEqual(
            str(res),
            '4|aAbB|'
        )
