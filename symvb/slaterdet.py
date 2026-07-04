from symvb.functions import sorti
from symvb.orbital_permutations import OrbitalPermutations
import symvb
import logging

logging.basicConfig(format=('%(levelname)-8s: %(message)s'))


class SlaterDet:
    """
    Parses the Slater determinant string |s| and computes the <det1|O|det2> elements
    """
    def __init__(self, s=""):
        self.det_string = s
        self.alpha_indices = []
        self.alpha_string = ''
        self.beta_indices = []
        self.beta_string = ''
        self.spins = ''
        self.Nel = 0
        if len(self.det_string) > 0:
            self.parse_det()

    def __repr__(self):
        s = '|%s|' % self.det_string
        return s

    # Description of the magic functions: https://docs.python.org/3/reference/datamodel.html
    def __add__(self, other):
        return symvb.FixedPsi(self) + other

    def __sub__(self, other):
        if other.__class__.__name__ == 'SlaterDet' and self.det_string == other.det_string:
            return SlaterDet()
        return symvb.FixedPsi(self) + (-1) * other

    def __rsub__(self, other):
        return symvb.FixedPsi(other) + (-1) * symvb.FixedPsi(self)

    def __neg__(self):
        return (-1) * symvb.FixedPsi(self)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return symvb.FixedPsi(self) * other
        if other.__class__.__name__ == 'SlaterDet':
            # make sure we do not place two electrons on the same orbital
            if len(set(self.alpha_string).intersection(set(other.alpha_string))) > 0:
                return SlaterDet()
            if len(set(self.beta_string).intersection(set(other.beta_string))) > 0:
                return SlaterDet()
            return SlaterDet(self.det_string + other.det_string)
        if other.__class__.__name__ == 'FixedPsi':
            return symvb.FixedPsi(self) * other
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return symvb.FixedPsi(self) * other

    def parse_det(self):
        s = self.det_string
        if len(s) == 0:
            return

        i = 0
        for c in s:
            if c.islower():
                assert c not in self.alpha_string # two electrons cannot occupy the same spinorbital
                self.alpha_indices.append(i)
                self.spins += '+'
                self.alpha_string += c
            else:
                assert c not in self.beta_string # two electrons cannot occupy the same spinorbital
                self.beta_indices.append(i)
                self.spins += '-'
                self.beta_string += c
            i = i + 1
        self.Nel = i

    def get_orbital_permutations(self):
        # gets all spin-restricted permutations of orbital products
        A = OrbitalPermutations(len(self.alpha_indices))
        B = OrbitalPermutations(len(self.beta_indices))

        dets = []
        signs = []
        for a_orbs, a_sign in A:
            for b_orbs, b_sign in B:
                i_a = 0
                i_b = 0
                s = ''
                for i in range(len(self.det_string)):
                    if self.spins[i] == '+':
                        c = self.alpha_string[a_orbs[i_a]]
                        i_a += 1
                    else:
                        c = self.beta_string[b_orbs[i_b]]
                        i_b += 1
                    s = s + c
                dets.append(s)
                signs.append(a_sign * b_sign)

        return [dets, signs]

    def is_compatible(self, R):
        if self.Nel != R.Nel:
            # logging.warning('Different number of electrons: %i vs %i' % (self.Nel, R.Nel))
            return False
        if self.spins != R.spins:
            # logging.warning('The determinant spins are incompatible: %s vs %s' % (self.spins, R.spins))
            return False
        return True

    def get_sorted(self):
        # Sorts orbital labels within alpha and beta blocks into alphabetic order
        # (spin pattern unchanged). Returns FixedPsi with the sign from the
        # required orbital permutation folded into the coefficient.
        sa, ia = sorti(self.alpha_string)
        sb, ib = sorti(self.beta_string)

        s = self.det_string
        for i in range(len(self.alpha_indices)):
            j = self.alpha_indices[i]
            s = s[:j] + sa[i] + s[j + 1:]

        for i in range(len(self.beta_indices)):
            j = self.beta_indices[i]
            s = s[:j] + sb[i] + s[j + 1:]

        # Construct a fresh SlaterDet from the sorted string so internal
        # fields (alpha_indices / alpha_string / etc.) reflect the new ordering.
        d = SlaterDet(s)

        coef = 1 if (ia + ib) % 2 == 0 else -1
        fp = symvb.FixedPsi()
        fp.add_det(d, coef=coef)
        return fp


if __name__ == '__main__':
    print(SlaterDet('AbCd').spins)
