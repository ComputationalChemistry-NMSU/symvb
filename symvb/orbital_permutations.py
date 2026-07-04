import itertools

import sys

sys.setrecursionlimit(30000)


class OrbitalPermutations:
    # will return permutations of vector [0,1,..,n-1] together with parity signs
    def __init__(self, n):
        self.n = n
        self.permutations = []
        self.permutation_signs = []

        self.get_permutation_indices(n)

    def __iter__(self):
        return zip(self.permutations, self.permutation_signs)

    def N_flips(self, v):
        # Computes the number of permutations from the consecutive order
        if len(v) == 0:
            return 0
        v2 = list(v).copy()

        # find 1 and swap elements
        i1 = v2.index(0)
        v2[i1] = v2[0]
        v2[0] = 1

        v2 = [x - 1 for x in v2]
        add_perm = 0 if i1 == 0 else 1

        return self.N_flips(v2[1:]) + add_perm

    def parity_sign(self, v):
        # Returns the parity (+1/-1) of a given permutation vector
        n = self.N_flips(v)
        if (n % 2) == 0:
            return 1
        else:
            return -1

    def get_permutation_indices(self, n):
        # gets the permutation indices and permutation signs
        if n == 0:
            n = 1
        self.permutations = list(itertools.permutations(range(n)))
        self.permutation_signs = [self.parity_sign(v) for v in self.permutations]
