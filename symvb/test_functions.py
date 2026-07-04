from unittest import TestCase
from symvb.functions import generate_det_strings, attempt_int, place_low, place_high, standardize_det, sort_ind, sorti


class TestFixedPsi(TestCase):

    def test_generate_det_strings_1(self):
        self.assertEqual(
            str(generate_det_strings(2, 2, 3)),
            "['aAbB', 'aAbC', 'aBbC', 'aAcB', 'aAcC', 'aBcC', 'bAcB', 'bAcC', 'bBcC']"
        )

    def test_generate_det_strings_2(self):
        self.assertEqual(
            str(generate_det_strings(2, 0, 3)),
            "['ab', 'ac', 'bc']"
        )

    def test_generate_det_strings_3(self):
        self.assertEqual(
            str(generate_det_strings(0, 2, 3)),
            "['AB', 'AC', 'BC']"
        )

    def test_place_low(self):
        self.assertEqual(
            place_low('aBcD', 0),
            ('aBcD', 0)
        )
        self.assertEqual(
            place_low('aBcD', 1),
            ('acBD', 1)
        )
        self.assertEqual(
            place_low('aBcD', 1),
            ('acBD', 1)
        )

    def test_place_high(self):
        self.assertEqual(
            place_high('aBcD', 1),
            ('aBcD', 0)
        )
        self.assertEqual(
            place_high('aBcD', 0),
            ('BacD', 1)
        )
        self.assertEqual(
            place_high('aBcD', 2),
            ('aBDc', 1)
        )

    def test_standardize_det_1(self):
        self.assertEqual(
            standardize_det('Ba'),
            ('aB', 1)
        )

    def test_standardize_det_2(self):
        self.assertEqual(
            standardize_det('BaCd'),
            ('aBdC', 2)
        )

    def test_standardize_det_3(self):
        self.assertEqual(
            standardize_det('BaCdfff'),
            ('aBdCfff', 2)
        )

    def test_standardize_det_4(self):
        self.assertEqual(
            standardize_det('BaCdUUU'),
            ('aBdCUUU', 2)
        )

    def test_sort_ind_1(self):
        v = ['a', 'b', 'b', 'a']
        self.assertEqual(
            sort_ind(v),
            ['a', 'a', 'b', 'b']
        )

    def test_sort_ind_2(self):
        v = ['d', 'c', 'b', 'a']
        self.assertEqual(
            sort_ind(v),
            ['a', 'b', 'c', 'd']
        )

    def test_sorti_1(self):
        self.assertEqual(
            sorti('dsafdksfj'),
            ('addffjkss', 6)
        )
