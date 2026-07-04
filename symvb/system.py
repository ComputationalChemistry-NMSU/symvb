"""High-level convenience layer over :class:`~symvb.molecule.Molecule`.

This is a thin, *additive* facade. The low-level symbolic core is unchanged
and every object it returns (the SymPy ``H``/``S`` matrices, the ground-state
expression, the structure vectors) stays directly inspectable. The facade
just removes the boilerplate and the two recurring footguns:

* building the Hamiltonian no longer means three calls plus a hand-combine,
  and you never scale the two-electron matrix by ``U`` yourself (it already
  carries the integral name -- doing so silently squares it);
* expressing a VB structure as a basis vector no longer means re-deriving the
  fermion sign of the canonical reordering by hand.

Typical use::

    from symvb import System
    sys = System.ring(6)                       # benzene pi ring, Hubbard U
    H, S = sys.hamiltonian()                   # 400x400 SymPy matrices
    E, c = System.from_structures(mol, [cov, ion]).ground_state()
    w    = sys.weights(groups=...)             # Chirgwin-Coulson weights

Or compose the standalone helpers with a Molecule you built yourself::

    from symvb.system import hamiltonian, ground_state, chirgwin_coulson
    H, S = hamiltonian(mol, basis)
    E, c = ground_state(H, S)
    w    = chirgwin_coulson(c, S)
"""
import sympy as sp

from .molecule import Molecule
from .fixed_psi import FixedPsi, generate_dets
from .functions import standardize_det
from .slaterdet import SlaterDet

__all__ = [
    'System', 'hamiltonian', 'ground_state', 'chirgwin_coulson',
    'structure_vector',
]

# reference point for picking the ground root of a symbolic GHEP
_DEFAULT_REF = {'h': -1, 's': 0, 'U': 1, 'J': 0, 'K': 0, 'M': 0,
                'h_s': -1, 'h_l': sp.Rational(-3, 10)}


def _standardize_full(det_string):
    """Map ``det_string`` to its ``generate_det_strings``-format twin + sign.

    Returns ``(canonical, sign)`` with ``|det_string> = sign * |canonical>``,
    where ``canonical`` is the unique basis-format string for the same set of
    spin-orbitals: alphabetically sorted alpha and beta labels interleaved
    ``uLuL...`` with the majority-spin extras appended, exactly as
    :func:`symvb.functions.generate_det_strings` emits it.

    :func:`symvb.functions.standardize_det` alone is not that map: it fixes
    the spin *pattern* but its pairwise flips can jump a creation operator
    over same-spin neighbours, permuting the alphabetical order within the
    alpha/beta blocks (e.g. ``'abcAB' -> 'aAcBb'``, not ``'aAbBc'``), so its
    output is a valid det string of the same determinant that need not be in
    the generated basis. Composing it with the within-block label sort of
    :meth:`symvb.slaterdet.SlaterDet.get_sorted` (spin pattern unchanged),
    with both fermion signs folded in, gives the true canonical map. A no-op
    (sign ``+1``) on a string that is already in basis format.
    """
    std, flips = standardize_det(det_string)
    fp = SlaterDet(std).get_sorted()
    return fp.dets[0].det_string, ((-1) ** flips) * fp.coefs[0]


def _det_string(b):
    """The canonical determinant string of a basis entry (FixedPsi/SlaterDet/str).

    Returned in symvb's standard (interleaved, block-sorted) creation order so
    that the determinant index this builds agrees with the standardized basis
    that matrix construction and :func:`structure_vector` use. A no-op on an
    entry that is already canonical.
    """
    if isinstance(b, str):
        raw = b
    elif hasattr(b, 'dets'):        # FixedPsi
        raw = b.dets[0].det_string
    else:                           # SlaterDet
        raw = b.det_string
    return _standardize_full(raw)[0]


# --------------------------------------------------------------------------
# standalone helpers (usable without a System)
# --------------------------------------------------------------------------
def hamiltonian(molecule, basis, two_electron=True):
    """Return ``(H, S)`` over ``basis``, with the two-electron block folded in.

    ``basis`` is a list of ``FixedPsi`` / ``SlaterDet`` / determinant strings.
    The two-electron block is **always** folded into ``H`` -- under the names
    declared in ``subst_2e`` when the molecule has them, otherwise under the
    default ``T_<abcd>`` integral names. The integrals are already inside
    ``H``; do **not** multiply the two-electron matrix by ``U`` yourself.

    Pass ``two_electron=False`` for an intentionally one-electron model
    (a Hueckel-level ``H``): this returns the bare ``build_matrix(op='H')``
    and skips the two-electron build, which dominates the cost on large
    bases.
    """
    H = molecule.build_matrix(basis, op='H')
    S = molecule.build_matrix(basis, op='S')
    if two_electron:
        H = H + molecule.o2_matrix(basis)
    return H, S


def structure_vector(structure, basis_dets):
    """Expand a VB structure as a column vector over ``basis_dets``.

    ``structure`` is a ``FixedPsi`` (e.g. a Rumer/Heitler-London structure built
    with ``coupled_pairs``); ``basis_dets`` is the list of determinant strings
    of the target basis (any creation order). Both sides are brought to the
    same canonical form: each basis string and each structure determinant is
    mapped through the full standardization (spin pattern via
    :func:`symvb.functions.standardize_det`, then within-block label sort),
    with both fermion signs folded into the coefficient. Alpha-alpha-beta-beta
    and other non-canonical spin patterns, including the unequal-filling cases
    where ``standardize_det`` alone leaves the generated basis (e.g. 3 alpha /
    2 beta hole determinants), are therefore placed correctly. Use this to
    project a structure onto an explicit determinant basis (e.g. an FCI ground
    state for weights); to build ``(H, S)`` over the structures themselves,
    ``hamiltonian`` / ``build_matrix`` now canonicalize internally.
    """
    fp = FixedPsi(structure)
    fp.canonicalize()
    idx = {}
    for i, b in enumerate(basis_dets):
        key, bsign = _standardize_full(b)
        if key in idx:
            raise ValueError(
                "basis determinants %r and %r are the same determinant "
                "(canonical form %r)" % (basis_dets[idx[key][0]], b, key))
        idx[key] = (i, bsign)
    v = sp.zeros(len(basis_dets), 1)
    for d, c in fp:
        key, dsign = _standardize_full(d.det_string)
        if key not in idx:
            raise ValueError(
                "structure determinant %r (canonical form %r) is not in the "
                "target basis" % (d.det_string, key))
        i, bsign = idx[key]
        v[i] += dsign * bsign * c
    return v


def chirgwin_coulson(c, S, groups=None, simplify=False):
    """Chirgwin-Coulson weights of coefficient vector ``c`` under metric ``S``.

    ``w_i = c_i (S c)_i / (c^T S c)`` (the metric makes them sum to one even for
    non-orthonormal ``c``). If ``groups`` (a list of index lists) is given, the
    summed weight per group is returned instead. Accepts either SymPy
    matrices/vectors or NumPy arrays and returns the matching type. ``simplify``
    is off by default: simplifying raw symbolic weights can be very slow, and
    it is usually cheaper to substitute numeric values first; pass
    ``simplify=True`` only for small closed forms.
    """
    try:
        import numpy as np
        if isinstance(c, np.ndarray):           # numeric path keyed on the coefficient vector
            c = np.asarray(c, float).ravel()
            S = np.asarray(S, float)
            Sc = S @ c
            w = c * Sc / (c @ Sc)
            if groups is None:
                return w
            return np.array([w[list(g)].sum() for g in groups])
    except ImportError:
        pass
    c = sp.Matrix(c)
    S = sp.Matrix(S)                            # accept a numpy metric alongside a symbolic c
    Sc = S * c
    norm = (c.T * Sc)[0]
    w = [c[i] * Sc[i] / norm for i in range(c.rows)]
    if groups is not None:
        w = [sum(w[i] for i in g) for g in groups]
    if simplify:
        w = [sp.simplify(x) for x in w]
    return sp.Matrix(w)


def _ref_subs(H, S, ref):
    syms = H.free_symbols | S.free_symbols
    user = {}
    if ref:
        for k, val in ref.items():
            user[sp.Symbol(k) if isinstance(k, str) else k] = val
    out = {}
    for sym in syms:
        if sym in user:
            out[sym] = user[sym]
        elif str(sym) in _DEFAULT_REF:
            out[sym] = _DEFAULT_REF[str(sym)]
        else:
            out[sym] = sp.Rational(1, 10)     # neutral nonzero default
    return out


def ground_state(H, S, ref=None, subs=None):
    """Ground state of ``H c = E S c``.

    **Symbolic** (``subs=None``, default): solves the characteristic polynomial
    for a SMALL block (2x2, 3x3). Returns ``(E, c)`` as SymPy expressions, ``E``
    simplified, ``c`` the (un-normalized) ground eigenvector (the metric is
    applied later by :func:`chirgwin_coulson`, and simplifying an eigenvector
    full of nested radicals is expensive, so we don't). The ground root is the
    one numerically lowest at the reference point ``ref`` -- a ``{symbol-or-name:
    value}`` dict; defaults are ``h=-1, s=0`` and any electron-repulsion integral
    ``= 1``. The symbolic solve is only practical for a few dimensions.

    **Numeric** (``subs`` given, a substitution dict): the matrices are
    evaluated and the lowest eigenpair is found with ``scipy.linalg.eigh`` --
    use this for anything larger, e.g. a full determinant (FCI) basis. Returns
    ``(E_float, c_ndarray)``.
    """
    if subs is not None:
        import numpy as np
        from scipy.linalg import eigh as _eigh
        Hn = np.array(sp.Matrix(H).subs(subs).tolist(), float)
        Sn = np.array(sp.Matrix(S).subs(subs).tolist(), float)
        w, v = _eigh(Hn, Sn, subset_by_index=[0, 0])
        return float(w[0]), v[:, 0]
    H = sp.Matrix(H)
    S = sp.Matrix(S)
    E = sp.Dummy('E')
    roots = sp.solve((H - E * S).det(), E)
    if not roots:
        raise ValueError("det(H - E S) = 0 has no roots solvable in closed form")
    sub = _ref_subs(H, S, ref)
    E_gs = min(roots, key=lambda r: float(sp.re(r.subs(sub))))
    null = (H - E_gs * S).nullspace()
    if not null:
        raise ValueError("no eigenvector found for the ground root")
    return sp.simplify(E_gs), null[0]


# --------------------------------------------------------------------------
# the System facade
# --------------------------------------------------------------------------
class System:
    """A :class:`Molecule` plus a determinant (or structure) basis.

    Construct directly with ``System(molecule, basis)``, from a structure list
    with ``System.from_structures(molecule, structures)``, or from a topology
    with ``System.ring(L)`` / ``System.chain(n)``.
    """

    def __init__(self, molecule, basis, two_electron=True):
        self.m = molecule
        self.basis = list(basis)
        self.det_strings = [_det_string(b) for b in self.basis]
        self.two_electron = two_electron
        self._H = None
        self._S = None

    # ---- constructors ----
    @classmethod
    def from_structures(cls, molecule, structures, two_electron=True):
        """A System whose basis is an explicit list of VB structures."""
        return cls(molecule, structures, two_electron=two_electron)

    @classmethod
    def ring(cls, L, n_alpha=None, n_beta=None, hubbard=True,
             two_electron=True, **kw):
        """Cyclic ``L``-orbital ring with Hubbard ``U``.

        Defaults to ``floor(L/2)`` electrons of each spin (the Sz=0 reference,
        e.g. 3 + 3 for benzene). For an ion or an odd ring pass ``n_alpha`` /
        ``n_beta`` explicitly (the cyclopentadienyl anion is
        ``System.ring(5, n_alpha=3, n_beta=3)``).
        """
        m = Molecule.ring(L, hubbard=hubbard, **kw)
        na = L // 2 if n_alpha is None else n_alpha
        nb = na if n_beta is None else n_beta
        return cls(m, generate_dets(na, nb, L), two_electron=two_electron)

    @classmethod
    def chain(cls, n, n_alpha=None, n_beta=None, hubbard=True,
              two_electron=True, **kw):
        """Linear ``n``-orbital chain with Hubbard ``U``; same filling convention
        as :meth:`ring` (``floor(n/2)`` electrons of each spin by default)."""
        m = Molecule.chain(n, hubbard=hubbard, **kw)
        na = n // 2 if n_alpha is None else n_alpha
        nb = na if n_beta is None else n_beta
        return cls(m, generate_dets(na, nb, n), two_electron=two_electron)

    # ---- matrices ----
    def hamiltonian(self):
        """``(H, S)`` over the basis (cached). The 2e block is folded into H
        unless the System was built with ``two_electron=False``."""
        if self._H is None:
            self._H, self._S = hamiltonian(self.m, self.basis,
                                           two_electron=self.two_electron)
        return self._H, self._S

    @property
    def H(self):
        return self.hamiltonian()[0]

    @property
    def S(self):
        return self.hamiltonian()[1]

    # ---- VB structures ----
    def structure_vector(self, structure):
        """Column vector of ``structure`` over this System's determinant basis."""
        return structure_vector(structure, self.det_strings)

    # ---- solving ----
    def ground_state(self, ref=None, subs=None):
        """``(E, c)`` of the ground state over this basis.

        Symbolic for small bases; pass ``subs`` (a numeric substitution dict) to
        solve a large/FCI basis numerically with scipy (returns floats/arrays).
        """
        H, S = self.hamiltonian()
        return ground_state(H, S, ref=ref, subs=subs)

    def weights(self, structures=None, groups=None, ref=None, subs=None):
        """Chirgwin-Coulson weights of the ground state.

        With no ``structures``, weights are per basis function (optionally summed
        by ``groups``). With ``structures`` (a list of ``FixedPsi``), the ground
        state is projected onto that (possibly non-orthogonal) structure space
        and weights are returned per structure, normalized over that space (the
        composition of the part of the wavefunction the structures span).

        Symbolic by default and only practical for small bases (the ground-state
        solve is a symbolic characteristic polynomial). For an FCI-sized basis
        pass ``subs`` (a numeric dict): the ground state is found numerically and
        the weights come back as a NumPy array.
        """
        H, S = self.hamiltonian()
        E, c = ground_state(H, S, ref=ref, subs=subs)
        if subs is None:
            S_eff = S
        else:
            import numpy as np
            S_eff = np.array(sp.Matrix(S).subs(subs).tolist(), float)
        if structures is None:
            return chirgwin_coulson(c, S_eff, groups=groups)
        V = sp.Matrix.hstack(*[self.structure_vector(st) for st in structures])
        if subs is None:
            G = V.T * S_eff * V                          # structure-space metric
            a = G.solve(V.T * S_eff * c)                 # ground state in the structure basis
        else:
            import numpy as np
            Vn = np.array(V, float)
            G = Vn.T @ S_eff @ Vn
            a = np.linalg.solve(G, Vn.T @ S_eff @ np.asarray(c, float))
        return chirgwin_coulson(a, G, groups=groups)
