import symvb
from symvb.functions import generate_det_strings, standardize_det, standardize_det_2
from symvb.functions import attempt_int
import copy


class FixedPsi:
    # Linear combination of determinants
    def __init__(self, x=None, coupled_pairs=None):
        """

        """
        self.Nel = 0

        self.dets = []
        self.coefs = []

        if x is None:
            return
        self.__iadd__(x)

        if coupled_pairs is not None:
            for i, j in coupled_pairs:
                self.couple_orbitals(i, j)

    def __iadd__(self, other):
        if other.__class__.__name__ == 'str':
            self.add_str_det(other)
        elif other.__class__.__name__ == 'SlaterDet':
            self.add_det(other)
        elif other.__class__.__name__ == 'FixedPsi':
            self.add_fixedpsi(other)
        return self

    def __add__(self, other):
        result = FixedPsi(self)
        result += other
        return result

    def __sub__(self, other):
        result = FixedPsi(self)
        result += (-1)*other
        return result

    def __rsub__(self, other):
        result = (-1) * FixedPsi(self)
        result += other
        return result

    def __neg__(self):
        return (-1) * FixedPsi(self)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            result = FixedPsi(self)
            for i in range(len(result)):
                result.coefs[i] = attempt_int(result.coefs[i]*other)
            return result

        if other.__class__.__name__ == 'SlaterDet':
            result = FixedPsi()
            for d, c in self:
                result.add_det(d * other, c)
            return result

        if other.__class__.__name__ == 'FixedPsi':
            result = FixedPsi()
            for dS, cS in self:
                for dO, cO in other:
                    d = dS * dO
                    if d.Nel == 0:
                        continue
                    fp1 = d.get_sorted()
                    c1, d1 = fp1.coefs[0], fp1.dets[0]
                    fp2 = standardize_det_2(d1)
                    result = result + (c1*cS*cO)*fp2
                    # result.add_det(d, coef = cS * cO)
            return result
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            result = FixedPsi(self)
            for i in range(len(result)):
                result.coefs[i] = attempt_int(result.coefs[i]*other)
            return result

    def __getitem__(self, item):
        return self.dets[item]

    def __len__(self):
        return len(self.dets)

    def __contains__(self, item):
        for d, c in self:
            if d.det_string == item:
                return True
        return False

    def __iter__(self):
        return zip(self.dets, self.coefs)

    def add_det(self, det, coef=+1):
        assert det.__class__.__name__ == 'SlaterDet'
        if det.Nel == 0:
            return

        # Note: mixed-Nel FixedPsis are allowed (used by N-changing
        # second-quantized operators like η± and bare c/c†). `self.Nel`
        # records the Nel of the first non-zero det added; callers that
        # require a uniform-N state should check this themselves.

        for i in range(len(self)):
            if det.det_string == self.dets[i].det_string:
                self.coefs[i] += coef
                return

        self.dets.append(det)
        cf = attempt_int(coef)
        self.coefs.append(cf)
        if self.Nel == 0:
            self.Nel = det.Nel

    def add_str_det(self, det_string, coef=+1):
        sd = symvb.SlaterDet(det_string)
        self.add_det(sd, coef=coef)

    def add_fixedpsi(self, p, coef=1.0):
        for d, c in p:
            merged = False
            # check if d is already in self
            for i in range(len(self)):
                if self.dets[i].det_string == d.det_string:
                    self.coefs[i] += c
                    # if it turns to 0, shift left the remaining dets
                    if self.coefs[i] == 0:
                        for j in range(i, len(self) - 1):
                            self.dets[j] = self.dets[j + 1]
                            self.coefs[j] = self.coefs[j + 1]
                        self.dets = self.dets[:-1]
                        self.coefs = self.coefs[:-1]
                    merged = True
                    break
            if not merged:
                self.add_det(d, c * coef)

    def canonicalize(self):
        """
        Rewrite self so that every Slater determinant has its alpha- and beta-
        orbital labels in alphabetic order. The permutation sign is folded into
        the coefficient, and dets that collapse to identical canonical strings
        are merged. Determinants whose coefficients cancel to zero are dropped.

        After canonicalize(), every alpha_string matches a key produced by
        generate_det_strings, so the precomputed-half-dets fast path in
        Molecule.op_det works on this FixedPsi.
        """
        new_dets = []
        new_coefs = []
        for d, c in zip(self.dets, self.coefs):
            sorted_fp = d.get_sorted()
            cd = sorted_fp.dets[0]
            cc = sorted_fp.coefs[0]
            signed = attempt_int(c * cc)

            merged = False
            for i, existing in enumerate(new_dets):
                if existing.det_string == cd.det_string:
                    new_coefs[i] = attempt_int(new_coefs[i] + signed)
                    merged = True
                    break
            if not merged:
                new_dets.append(cd)
                new_coefs.append(signed)

        self.dets = [d for d, c in zip(new_dets, new_coefs) if c != 0]
        self.coefs = [c for c in new_coefs if c != 0]
        if not self.dets:
            self.Nel = 0
        return self

    def standardize(self):
        """Rewrite self so that every determinant is in symvb's canonical
        creation order (the interleaved ``uLuL...`` form of
        :func:`symvb.functions.standardize_det`). The permutation sign is
        folded into the coefficient and determinants that collapse to identical
        canonical strings are merged; determinants whose coefficients cancel to
        zero are dropped.

        This is the transformation matrix construction needs. Unlike
        :meth:`canonicalize` (which only sorts orbital labels *within* the alpha
        and beta blocks and leaves the spin pattern alone), this also brings the
        spin pattern to the standard alpha-before-beta interleaving. A structure
        written in a natural creation order, e.g. the alpha-alpha-beta-beta
        pattern produced by ``coupled_pairs`` on a long bond, therefore ends up
        in the same convention as the determinant basis, so its couplings are no
        longer silently dropped by the spin-pattern matching in
        :meth:`Molecule.build_matrix` / :meth:`Molecule.o2_matrix`.

        On a determinant that is already in canonical order this is a no-op
        (zero flips, unchanged string).
        """
        new_dets = []
        new_coefs = []
        for d, c in zip(self.dets, self.coefs):
            std, flips = standardize_det(d.det_string)
            signed = attempt_int(c * ((-1) ** flips))

            merged = False
            for i, existing in enumerate(new_dets):
                if existing.det_string == std:
                    new_coefs[i] = attempt_int(new_coefs[i] + signed)
                    merged = True
                    break
            if not merged:
                new_dets.append(symvb.SlaterDet(std))
                new_coefs.append(signed)

        self.dets = [d for d, c in zip(new_dets, new_coefs) if c != 0]
        self.coefs = [c for c in new_coefs if c != 0]
        if not self.dets:
            self.Nel = 0
        return self

    def couple_orbitals(self, o1, o2):
        # generate determinants that represent a singlet bonding coupling between two orbitals.
        # Orbital numbering starts from 0
        determinants = self.dets.copy()
        coefs = self.coefs.copy()
        for d, coef in zip(determinants, coefs):
            # Flip spins
            ds = d.det_string
            c1, c2 = [c.lower() if c.isupper() else c.upper() for c in [ds[o1], ds[o2]]]
            assert c1.lower() != c2.lower(), 'Cannot couple the same orbital'
            # Flip positions
            ds2 = ds[:o1] + c2 + ds[(o1 + 1):o2] + c1 + ds[(o2 + 1):]
            self.add_str_det(ds2, coef=coef)

    def __repr__(self):
        s = ''
        for d, cf in self:
            dc = attempt_int(cf)

            if dc > 0:
                if dc == 1.0:
                    s += '+'
                else:
                    s += '+%s' % dc
            elif dc < 0:
                if dc == -1.0:
                    s += '-'
                else:
                    s += '%s' % dc
            s += str(d)
        if len(s) > 0 and s[0] == '+':
            s = s[1:]
        return s


def generate_dets(Nela, Nelb, Norb):
    """
    Generate all possible determinants for a given number of electrons and atomic orbitals.
    :param Nela: Number of alpha electrons
    :param Nelb: Number of beta electrons
    :param Norb: Number of atomic orbitals
    :return: List of FixedPsi objects, each containing one determinant
    """
    L = generate_det_strings(Nela, Nelb, Norb)
    PP = [None,]*len(L)
    for i in range(len(L)):
        PP[i] = FixedPsi(L[i])
    return PP
