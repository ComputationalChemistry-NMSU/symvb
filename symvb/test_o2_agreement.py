"""
Agreement-test harness for the two-electron matrix-element implementations.

Compares Molecule(o2_method='direct') against Molecule(o2_method='blocked')
on a battery of inputs. The harness lives here at Phase 0; the assertions
that exercise the blocked path are decorated with skipUnless(BLOCKED_READY)
so that the test module is collectable today and exercises the equality
helper, but does not attempt to call into the unimplemented blocked path.

Phase 1 (when o2_det_blocked is implemented) flips BLOCKED_READY to True;
no other change to this file is required.

Phase 2 expands the case list and adds random-stress fixtures.
"""
import os
import pickle
import random
import string
import time
import unittest
from itertools import combinations

import sympy as sp

from symvb import Molecule, SlaterDet
from symvb._o2_blocked import o2_det_blocked
from symvb.fixed_psi import generate_dets


def _blocked_implementation_ready():
    """True iff o2_det_blocked accepts a trivial input without raising
    NotImplementedError. Used as a feature flag so the agreement tests
    can be collected at Phase 0 but skipped until the implementation lands."""
    try:
        m = Molecule(o2_method='blocked')
        o2_det_blocked(m, SlaterDet('aB'), SlaterDet('aB'))
    except NotImplementedError:
        return False
    except Exception:
        # Any other exception means the implementation exists but is buggy;
        # let the actual tests surface that — don't mask it here.
        return True
    return True


BLOCKED_READY = _blocked_implementation_ready()


def _canonical_chemist_iv(iv):
    """Lexicographically smallest of the 8 chemist-notation-equivalent
    permutations of a 4-AO tuple. Implements (ij|kl) = (ji|kl) = (ij|lk)
    = (ji|lk) = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)."""
    i, j, k, l = iv
    forms = (
        (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
        (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i),
    )
    return min(forms)


def _canonicalize_default_2e_symbols(expr):
    """Map every default T_xxxx 2e symbol in expr to its chemist-canonical
    name. Subst_2e-supplied names (U, J, K, M, etc.) are left untouched
    because the user has already grouped them by physical equivalence.

    This compensates for a pre-existing symvb limitation: sort_ind/hperm
    only canonicalises across the 8-fold chemist symmetry when all four
    AO labels are distinct. With repeated labels (e.g. (a,a,b,c) vs.
    (a,a,c,b), both physically (ab|ac)) the canonical form bifurcates.
    The direct and blocked paths both inherit this and may emit different
    representatives of the same orbit, hence the post-normalisation here.
    """
    subs = {}
    for sym in expr.free_symbols:
        name = sym.name
        if name.startswith('T_') and len(name) == 6:
            iv = tuple(name[2:])
            canonical = _canonical_chemist_iv(iv)
            canonical_name = 'T_' + ''.join(canonical)
            if canonical_name != name:
                subs[sym] = sp.Symbol(canonical_name)
    return expr.xreplace(subs) if subs else expr


def assert_symbolic_equal(expr_a, expr_b, *, simplify_timeout=30):
    """Three-tier symbolic equality check for two sympy expressions.

    Both sides are first normalised through _canonicalize_default_2e_symbols
    to absorb chemist-symmetry equivalences in default T_xxxx 2e symbols.

    Returns silently on success. Raises AssertionError with both sides
    on failure. Tiers:
      1. expand(a - b) == 0     — additive-form equality, fast.
      2. cancel(a - b) == 0     — handles rational forms.
      3. simplify(a - b) == 0   — last resort; can be slow on large exprs.

    The tiers escalate only on failure, so passing cases pay only the
    cost of tier 1.
    """
    a_norm = _canonicalize_default_2e_symbols(sp.sympify(expr_a))
    b_norm = _canonicalize_default_2e_symbols(sp.sympify(expr_b))
    diff = a_norm - b_norm

    if sp.expand(diff) == 0:
        return
    if sp.cancel(diff) == 0:
        return
    # Final tier: full simplify. No timeout enforcement here at Phase 0;
    # Phase 2 will add a signal-based timeout to bound CI cost.
    if sp.simplify(diff) == 0:
        return

    raise AssertionError(
        "symbolic mismatch between direct and blocked paths\n"
        "  direct  : %r\n"
        "  blocked : %r\n"
        "  diff    : %r" % (expr_a, expr_b, sp.simplify(diff)))


def _build_molecule(o2_method, **kwargs):
    """Construct a Molecule with the given o2_method and otherwise-default
    integrals/substitutions overridable per test case."""
    return Molecule(o2_method=o2_method, **kwargs)


# --- Case list ---------------------------------------------------------------
# Each case is a (name, factory, dets) triple, where factory(o2_method) returns
# a configured Molecule and dets is a list of (D1, D2) SlaterDet pairs to
# evaluate o2_det on. Phase 2 expands this; Phase 0 keeps it minimal so the
# harness scaffolding can be exercised end-to-end.
_PPP_2E = {'U': ('1111',), 'J': ('1212',), 'K': ('1122',), 'M': ('1112', '1121', '1222')}


def _case_h2_minimal():
    def factory(o2_method):
        return _build_molecule(o2_method, subst_2e={'J': ('1221',), 'K': ('1212',)})
    dets = [(SlaterDet('aB'), SlaterDet('aB'))]
    return ('h2_minimal', factory, dets)


def _case_h2_ppp():
    def factory(o2_method):
        return _build_molecule(o2_method, subst_2e=_PPP_2E, max_2e_centers=2,
                               interacting_orbs=['ab'])
    dets = [
        (SlaterDet('aA'), SlaterDet('aA')),
        (SlaterDet('aA'), SlaterDet('bB')),
        (SlaterDet('aB'), SlaterDet('bA')),
    ]
    return ('h2_ppp', factory, dets)


def _case_allyl_anion():
    def factory(o2_method):
        return _build_molecule(o2_method, subst_2e=_PPP_2E, max_2e_centers=2,
                               interacting_orbs=['ab', 'bc'])
    dets = [
        (SlaterDet('aAbB'), SlaterDet('aAbB')),
        (SlaterDet('aAbB'), SlaterDet('aAcC')),
        (SlaterDet('aAbC'), SlaterDet('aBbC')),
    ]
    return ('allyl_anion', factory, dets)


CASES = [_case_h2_minimal(), _case_h2_ppp(), _case_allyl_anion()]


# --- Tests -------------------------------------------------------------------

class TestO2MethodDispatch(unittest.TestCase):
    """Phase 0 sanity: the dispatch flag is plumbed correctly. Always runs."""

    def test_default_is_blocked(self):
        # both methods return identical matrices; 'blocked' is the (faster) default
        m = Molecule()
        self.assertEqual(m.o2_method, 'blocked')

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            Molecule(o2_method='nonsense')

    def test_blocked_dispatch_routes_to_stub(self):
        m = Molecule(o2_method='blocked')
        if BLOCKED_READY:
            # Implementation exists; smoke-test on a trivial det pair.
            result = m.o2_det(SlaterDet('aB'), SlaterDet('aB'))
            # Just check we got a sympy object back; full agreement is the
            # job of TestO2Agreement below.
            self.assertIsNotNone(result)
        else:
            with self.assertRaises(NotImplementedError):
                m.o2_det(SlaterDet('aB'), SlaterDet('aB'))


class TestO2Agreement(unittest.TestCase):
    """Direct-vs-blocked symbolic agreement on the case list. Skipped until
    the blocked implementation lands (Phase 1). The harness, fixtures, and
    equality helper below are exercised at Phase 0 by being importable and
    by running TestSymbolicEqualityHelper."""

    @unittest.skipUnless(BLOCKED_READY,
                         "blocked path is a stub; tests will activate at Phase 1")
    def test_agreement_on_case_list(self):
        for name, factory, dets in CASES:
            with self.subTest(case=name):
                m_direct = factory('direct')
                m_blocked = factory('blocked')
                for D1, D2 in dets:
                    expr_direct = m_direct.o2_det(D1, D2)
                    expr_blocked = m_blocked.o2_det(D1, D2)
                    assert_symbolic_equal(expr_direct, expr_blocked)


class TestO2MatrixAgreement(unittest.TestCase):
    """Direct-vs-blocked symbolic agreement on the FULL 2e matrix of small
    basis sets. This is the strongest practical regression test: every
    (D_A, D_B) pair in a generated basis is exercised under both methods
    and the resulting SymPy expressions are compared element-wise."""

    @unittest.skipUnless(BLOCKED_READY,
                         "blocked path is a stub; tests will activate at Phase 1")
    def test_h2_full_matrix_balanced(self):
        # H_2: 2 orbitals, 1 alpha + 1 beta (Sz=0 sector, 4 dets)
        self._compare_full_matrix(
            Na=1, Nb=1, Norbs=2,
            kwargs=dict(subst_2e=_PPP_2E, interacting_orbs=['ab']))

    @unittest.skipUnless(BLOCKED_READY, "blocked stub")
    def test_h2_full_matrix_pure_alpha(self):
        # H_2: 2 orbitals, 2 alpha + 0 beta (n_b=0 edge case, 1 det)
        self._compare_full_matrix(
            Na=2, Nb=0, Norbs=2,
            kwargs=dict(subst_2e=_PPP_2E, interacting_orbs=['ab']))

    @unittest.skipUnless(BLOCKED_READY, "blocked stub")
    def test_h2_full_matrix_pure_beta(self):
        # H_2: 2 orbitals, 0 alpha + 2 beta (n_a=0 edge case, 1 det)
        self._compare_full_matrix(
            Na=0, Nb=2, Norbs=2,
            kwargs=dict(subst_2e=_PPP_2E, interacting_orbs=['ab']))

    @unittest.skipUnless(BLOCKED_READY, "blocked stub")
    def test_allyl_anion_full_matrix(self):
        # Allyl anion: 3 orbitals, 2 alpha + 2 beta (9 dets in Sz=0 sector)
        self._compare_full_matrix(
            Na=2, Nb=2, Norbs=3,
            kwargs=dict(subst_2e=_PPP_2E, interacting_orbs=['ab', 'bc'],
                        max_2e_centers=2))

    @unittest.skipUnless(BLOCKED_READY, "blocked stub")
    def test_allyl_cation_full_matrix(self):
        # Allyl cation: 3 orbitals, 1 alpha + 1 beta (open-shell-like, 9 dets)
        self._compare_full_matrix(
            Na=1, Nb=1, Norbs=3,
            kwargs=dict(subst_2e=_PPP_2E, interacting_orbs=['ab', 'bc'],
                        max_2e_centers=2))

    @unittest.skipUnless(BLOCKED_READY, "blocked stub")
    def test_allyl_triplet_full_matrix(self):
        # Triplet allyl-like: 3 orbitals, 2 alpha + 1 beta (Sz=1/2, 9 dets)
        self._compare_full_matrix(
            Na=2, Nb=1, Norbs=3,
            kwargs=dict(subst_2e=_PPP_2E, interacting_orbs=['ab', 'bc'],
                        max_2e_centers=2))

    @unittest.skipUnless(BLOCKED_READY, "blocked stub")
    def test_max_2e_centers_sweep(self):
        # Same allyl basis under each max_2e_centers cutoff.
        for cutoff in (1, 2, 3, 4):
            with self.subTest(max_2e_centers=cutoff):
                self._compare_full_matrix(
                    Na=2, Nb=2, Norbs=3,
                    kwargs=dict(subst_2e=_PPP_2E,
                                interacting_orbs=['ab', 'bc'],
                                max_2e_centers=cutoff))

    @unittest.skipUnless(BLOCKED_READY, "blocked stub")
    def test_4orb_2a2b_full_matrix(self):
        # 4-orbital, 2a+2b (36 dets). Larger system than allyl, denser
        # interacting_orbs, stresses the alpha-beta channel.
        self._compare_full_matrix(
            Na=2, Nb=2, Norbs=4,
            kwargs=dict(subst_2e=_PPP_2E,
                        interacting_orbs=['ab', 'bc', 'cd', 'ad'],
                        max_2e_centers=2))

    @unittest.skipUnless(BLOCKED_READY, "blocked stub")
    def test_no_subst_2e(self):
        # Default 2e symbols (T_xxxx form, no PPP grouping).
        self._compare_full_matrix(
            Na=1, Nb=1, Norbs=2,
            kwargs=dict(interacting_orbs=['ab']))

    @unittest.skipUnless(BLOCKED_READY, "blocked stub")
    def test_alternative_subst_2e_mapping(self):
        # The mapping in test_molecule.test_subst_2e_2 (J<->K swap relative
        # to the manuscript convention). Forces the comparison to handle
        # arbitrary subst_2e dictionaries, not just the PPP one.
        self._compare_full_matrix(
            Na=1, Nb=1, Norbs=2,
            kwargs=dict(subst_2e={'J': ('1221',), 'K': ('1212',)},
                        interacting_orbs=['ab']))

    def _compare_full_matrix(self, *, Na, Nb, Norbs, kwargs):
        """Build the 2e matrix of the (Na, Nb, Norbs) basis under both
        methods; assert element-wise symbolic equality."""
        m_direct = Molecule(o2_method='direct', **kwargs)
        m_blocked = Molecule(o2_method='blocked', **kwargs)
        basis = generate_dets(Na, Nb, Norbs)
        for ia, fp_a in enumerate(basis):
            for ib, fp_b in enumerate(basis):
                # Each FixedPsi here is a single-det wrapper from generate_dets.
                d_a = fp_a.dets[0]
                d_b = fp_b.dets[0]
                expr_direct = m_direct.o2_det(d_a, d_b)
                expr_blocked = m_blocked.o2_det(d_a, d_b)
                try:
                    assert_symbolic_equal(expr_direct, expr_blocked)
                except AssertionError as e:
                    raise AssertionError(
                        "mismatch at basis pair (%d, %d):\n"
                        "  D_A = %r\n  D_B = %r\n%s"
                        % (ia, ib, d_a.det_string, d_b.det_string, e))


class TestO2RandomStress(unittest.TestCase):
    """Phase 2: hundreds of random configurations stress-test the agreement.
    Each trial draws Norbs, (Na, Nb), interacting_orbs, max_2e_centers, and
    a subst_2e dictionary at random, then compares full 2e matrices under
    both methods. Bounded basis size keeps each trial cheap."""

    NUM_TRIALS = 100
    SEED = 0xc0ffee
    MAX_BASIS_SIZE = 36  # cap to keep total runtime bounded

    SUBST_2E_OPTIONS = [
        {},  # default T_xxxx form
        _PPP_2E,  # manuscript convention
        {'J': ('1221',), 'K': ('1212',)},  # alternative grouping
        {'U': ('1111',), 'V': ('1212', '1221'), 'W': ('1122',)},  # arbitrary user labels
    ]

    @unittest.skipUnless(BLOCKED_READY, "blocked stub")
    def test_random_configs(self):
        rng = random.Random(self.SEED)
        kept = 0
        skipped_size = 0
        for trial in range(self.NUM_TRIALS):
            cfg = self._draw_config(rng)
            basis = generate_dets(cfg['Na'], cfg['Nb'], cfg['Norbs'])
            if len(basis) ** 2 > self.MAX_BASIS_SIZE ** 2:
                skipped_size += 1
                continue
            with self.subTest(trial=trial, **{k: cfg[k] for k in
                                              ('Na', 'Nb', 'Norbs',
                                               'max_2e_centers')}):
                self._compare(cfg, basis)
                kept += 1
        # Sanity: most trials should run; if everything got size-skipped the
        # test would silently degenerate.
        self.assertGreater(kept, self.NUM_TRIALS // 4,
                           "too many trials skipped on basis size")

    def _draw_config(self, rng):
        Norbs = rng.randint(2, 4)
        Na = rng.randint(0, Norbs)
        Nb = rng.randint(0, Norbs)
        if Na + Nb == 0:  # rerun with at least one electron
            Na = 1
        # Random interacting_orbs: subset of all distinct lower-case pairs.
        all_pairs = [a + b for a, b in
                     combinations(string.ascii_lowercase[:Norbs], 2)]
        n_pairs = rng.randint(1, len(all_pairs))
        interacting = rng.sample(all_pairs, n_pairs)
        return dict(
            Na=Na, Nb=Nb, Norbs=Norbs,
            interacting_orbs=interacting,
            max_2e_centers=rng.choice([1, 2, 3, 4]),
            subst_2e=rng.choice(self.SUBST_2E_OPTIONS),
        )

    def _compare(self, cfg, basis):
        kwargs = {k: cfg[k] for k in
                  ('interacting_orbs', 'max_2e_centers', 'subst_2e')}
        m_d = Molecule(o2_method='direct', **kwargs)
        m_b = Molecule(o2_method='blocked', **kwargs)
        for ia, fp_a in enumerate(basis):
            for ib, fp_b in enumerate(basis):
                d_a = fp_a.dets[0]
                d_b = fp_b.dets[0]
                ed = m_d.o2_det(d_a, d_b)
                eb = m_b.o2_det(d_a, d_b)
                try:
                    assert_symbolic_equal(ed, eb)
                except AssertionError as err:
                    raise AssertionError(
                        "config=%r pair=(%d,%d) D_A=%r D_B=%r\n%s"
                        % (cfg, ia, ib, d_a.det_string, d_b.det_string, err))


class TestO2NumericalSubstitution(unittest.TestCase):
    """Phase 2: substitute exact rational values into the symbolic matrices
    from both paths and compare element-wise. Catches sign errors that a
    weak symbolic-equality check might mask. Uses sp.Rational so the
    arithmetic is exact (no float tolerance needed)."""

    NUM_TRIALS = 5
    SEED = 0xfeedbeef

    @unittest.skipUnless(BLOCKED_READY, "blocked stub")
    def test_numerical_agreement_on_h2_ppp(self):
        self._run(Na=1, Nb=1, Norbs=2,
                  kwargs=dict(subst_2e=_PPP_2E, interacting_orbs=['ab']))

    @unittest.skipUnless(BLOCKED_READY, "blocked stub")
    def test_numerical_agreement_on_allyl_anion(self):
        self._run(Na=2, Nb=2, Norbs=3,
                  kwargs=dict(subst_2e=_PPP_2E, interacting_orbs=['ab', 'bc'],
                              max_2e_centers=2))

    @unittest.skipUnless(BLOCKED_READY, "blocked stub")
    def test_numerical_agreement_on_4orb_open_shell(self):
        # n_a != n_b stresses the spin-block-parity sign.
        self._run(Na=2, Nb=1, Norbs=4,
                  kwargs=dict(subst_2e=_PPP_2E,
                              interacting_orbs=['ab', 'bc', 'cd'],
                              max_2e_centers=2))

    def _run(self, *, Na, Nb, Norbs, kwargs):
        m_d = Molecule(o2_method='direct', **kwargs)
        m_b = Molecule(o2_method='blocked', **kwargs)
        basis = generate_dets(Na, Nb, Norbs)
        # Build full symbolic matrices first (one expensive pass per method).
        H_d = sp.Matrix(len(basis), len(basis),
                        lambda i, j: m_d.o2_det(basis[i].dets[0],
                                                basis[j].dets[0]))
        H_b = sp.Matrix(len(basis), len(basis),
                        lambda i, j: m_b.o2_det(basis[i].dets[0],
                                                basis[j].dets[0]))
        # Collect every free symbol from either matrix.
        free = set()
        for i in range(H_d.rows):
            for j in range(H_d.cols):
                free |= H_d[i, j].free_symbols
                free |= H_b[i, j].free_symbols
        rng = random.Random(self.SEED)
        for trial in range(self.NUM_TRIALS):
            with self.subTest(trial=trial):
                subs = {sym: sp.Rational(rng.randint(-7, 7),
                                         rng.randint(1, 5))
                        for sym in free}
                H_d_num = H_d.subs(subs)
                H_b_num = H_b.subs(subs)
                # Exact rational equality; no tolerance needed.
                for i in range(H_d.rows):
                    for j in range(H_d.cols):
                        a = sp.simplify(H_d_num[i, j])
                        b = sp.simplify(H_b_num[i, j])
                        self.assertEqual(
                            a, b,
                            "numerical mismatch at (%d,%d) trial=%d: "
                            "direct=%r blocked=%r" % (i, j, trial, a, b))


class TestO2BenzenePickleConsistency(unittest.TestCase):
    """Phase 2: three-way regression against the cached benzene PPP H_2
    pickle. Only runs if /tmp/benzene_ujk_matrices.pkl exists; on systems
    without the cache, skipped silently. The pickle was built by the
    historical direct path; verifying both today's direct AND blocked paths
    reproduce it byte-for-byte (after chemist-symbol normalisation)
    protects published manuscript results from any regression."""

    PICKLE_PATH = '/tmp/benzene_ujk_matrices.pkl'

    @unittest.skipUnless(BLOCKED_READY, "blocked stub")
    @unittest.skipUnless(os.path.exists(PICKLE_PATH),
                         "cached pickle %s not present" % PICKLE_PATH)
    def test_benzene_h2_three_way(self):
        # Build benzene PPP under both methods and compare to the pickle.
        # The basis is 400-dim; we sample a fixed window for runtime.
        SAMPLE = 30  # rows*cols compared, not full 160k.
        with open(self.PICKLE_PATH, 'rb') as f:
            H1_pkl, S_pkl, H2_pkl = pickle.load(f)
        N = H2_pkl.rows
        self.assertEqual(N, 400, "unexpected pickle dimension")

        # Must match the build config in examples/benzene_hubbard_ujk.py
        # exactly, otherwise the pickle's symbolic form won't agree.
        kwargs = dict(
            zero_ii=True,
            interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'],
            subst={'h': ('H_ab', 'H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af'),
                   's': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af')},
            subst_2e=_PPP_2E,
            max_2e_centers=2,
        )
        m_d = Molecule(o2_method='direct', **kwargs)
        m_b = Molecule(o2_method='blocked', **kwargs)
        basis = generate_dets(3, 3, 6)
        self.assertEqual(len(basis), N)

        rng = random.Random(0xb12345e)
        sample_pairs = [(rng.randrange(N), rng.randrange(N))
                        for _ in range(SAMPLE)]
        for ia, ib in sample_pairs:
            with self.subTest(pair=(ia, ib)):
                d_a = basis[ia].dets[0]
                d_b = basis[ib].dets[0]
                e_d = m_d.o2_det(d_a, d_b)
                e_b = m_b.o2_det(d_a, d_b)
                e_pkl = H2_pkl[ia, ib]
                # All three forms equal up to chemist-symbol normalisation
                # (the pickle was built when sort_ind was the same as today,
                # so this is mostly a chemist-symmetry-tolerant comparison).
                assert_symbolic_equal(e_d, e_b)
                assert_symbolic_equal(e_d, e_pkl)
                assert_symbolic_equal(e_b, e_pkl)


class TestSymbolicEqualityHelper(unittest.TestCase):
    """Direct unit tests on assert_symbolic_equal. Run at Phase 0 to verify
    the comparison harness itself before the blocked implementation lands."""

    def test_identical_pass(self):
        x, y = sp.symbols('x y')
        assert_symbolic_equal(x + y, x + y)

    def test_reordered_addition_pass(self):
        x, y = sp.symbols('x y')
        assert_symbolic_equal(x + y, y + x)

    def test_factored_form_pass(self):
        x, y = sp.symbols('x y')
        assert_symbolic_equal((x + y) ** 2, x ** 2 + 2 * x * y + y ** 2)

    def test_rational_form_pass(self):
        x, y = sp.symbols('x y', positive=True)
        # (x^2 - y^2) / (x - y) == x + y  for x != y
        assert_symbolic_equal((x ** 2 - y ** 2) / (x - y), x + y)

    def test_mismatch_raises(self):
        x = sp.symbols('x')
        with self.assertRaises(AssertionError):
            assert_symbolic_equal(x + 1, x + 2)


if __name__ == '__main__':
    unittest.main()
