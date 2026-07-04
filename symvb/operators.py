"""
symvb.operators — second-quantized operators acting on Slater determinants.

A small expression-tree framework for building physical operators from
elementary creation/annihilation operators and applying them to symvb
Slater dets.

Scope
-----
This module is for **orthogonal-AO, second-quantized** operators —
spin (S², Sz, S±, Sᵢ·Sⱼ), number / double-occupancy, hopping
(c†_i c_j), η-pairing, orbital-permutation symmetries, and the
projectors / VB structures built from them. The Slater determinants
are assumed orthonormal (i.e. AO overlap s = 0).

For non-orthogonal AOs (s ≠ 0) — i.e. for evaluating ⟨D | H | D'⟩
with one- and two-electron integrals — use the existing
`symvb.Molecule.build_matrix` pipeline, which goes through the Löwdin
cofactor expansion. The two paths are complementary, not redundant.

Quick tour
----------
    from symvb import operators as op

    # primitives
    op.c('a', 'alpha')          # c_{a, α}
    op.cdag('a', 'beta')        # c†_{a, β}

    # composites
    op.number('a')              # n̂_a  =  n̂_aα + n̂_aβ
    op.double_occ('a')          # n̂_aα n̂_aβ
    op.hop('a', 'b', 'alpha')   # c†_{a,α} c_{b,α} + h.c.
    op.s_squared(['a', 'b'])    # total S²
    op.s_dot('a', 'b')          # S_a · S_b

    # algebra
    H = -op.hop('a', 'b') + op.double_occ('a') + op.double_occ('b')

    # action on a state
    H.apply('aB')                       # → FixedPsi
    H.apply(symvb.FixedPsi('aB'))        # also accepted

    # matrix in any basis (det strings, SlaterDets, FixedPsis):
    H.matrix(['aB', 'bA', 'aA', 'bB'])  # → sympy Matrix

Convention
----------
Determinants are symvb strings: lowercase = α, uppercase = β. Internally
operators work in the **interleaved canonical form** that matches
`symvb.functions.generate_det_strings` and `symvb.spin._canon_det` —
sorted α and β are paired up in lex order, with any unmatched extras
appended (e.g. canonical for α={a,c}, β={b,d} is `aBcD`).

Jordan–Wigner ordering is also interleaved: `(a,α) < (a,β) < (b,α) <
(b,β) < …`. Operators emit FixedPsi states keyed by canonical strings,
with coefficients in the *user-string* convention — i.e. they are the
multipliers FixedPsi will reconstruct the physical state from when
parsing the string in left-to-right creation order.

Compose with `+`, `-`, scalar `*`, and `@` (composition); `A @ B`
means "B acts first, then A".
"""
from __future__ import annotations

import numpy as np
import sympy as sp

import symvb


# ---------------------------------------------------------------------------
# Canonical det conversion (interleaved; matches generate_det_strings)
# ---------------------------------------------------------------------------

def _parse_det(det_string):
    """symvb det string → (alpha_set, beta_set, sign).

    `sign` is the ±1 such that |det_string⟩_user = sign · |canonical(α,β)⟩,
    where |canonical(α,β)⟩ has its creation operators in interleaved JW
    order: (a,α) < (a,β) < (b,α) < (b,β) < … . Concretely, parse the
    string left-to-right (creation order) into a list of (orb, spin)
    spin-orbital tuples and return the parity of inversions in that list
    when compared against tuple lex order.
    """
    alpha = set()
    beta = set()
    so_seq = []
    for ch in det_string:
        if ch.islower():
            assert ch not in alpha, f"two electrons in spin-orbital ({ch}, α)"
            alpha.add(ch)
            so_seq.append((ch, 0))
        else:
            lc = ch.lower()
            assert lc not in beta, f"two electrons in spin-orbital ({lc}, β)"
            beta.add(lc)
            so_seq.append((lc, 1))
    return frozenset(alpha), frozenset(beta), _inversion_sign(so_seq)


def _inversion_sign(seq):
    inv = 0
    n = len(seq)
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] > seq[j]:
                inv += 1
    return 1 if inv % 2 == 0 else -1


def _canonical_string(alpha, beta):
    """Interleaved canonical det string for (α, β); matches
    `symvb.functions.generate_det_strings` for any |α| / |β|.

    Pairs sorted α and β letters in lex order, then appends extras of
    whichever block is longer: e.g. ({a,b,c}, {d}) → 'aDbc'.
    """
    a = sorted(alpha)
    b = sorted(beta)
    Na, Nb = len(a), len(b)
    s = ''
    for i in range(min(Na, Nb)):
        s += a[i] + b[i].upper()
    for i in range(Nb, Na):  # extra alphas
        s += a[i]
    for i in range(Na, Nb):  # extra betas
        s += b[i].upper()
    return s


def canonicalize(det_string):
    """Return (canonical_string, sign) for a symvb det string.

    `|det_string⟩_user = sign · |canonical_string⟩_canonical`.
    """
    a, b, s = _parse_det(det_string)
    return _canonical_string(a, b), s


# ---------------------------------------------------------------------------
# Elementary creation/annihilation in interleaved JW order
# ---------------------------------------------------------------------------

def _apply_c(alpha, beta, orb, spin, create):
    """Apply c[†]_{orb,spin} in interleaved canonical ordering.

    Returns (new_alpha, new_beta, sign) or (None, None, 0) if annihilated.
    Spin: 0 = α, 1 = β. Interleaved JW order: (orb, α) < (orb, β) <
    (orb', α) < (orb', β) for orb < orb'.
    """
    occ = (orb in alpha) if spin == 0 else (orb in beta)
    if create == occ:
        return None, None, 0

    # Count occupied modes strictly before (orb, spin) in interleaved order.
    count = 0
    for x in sorted(alpha | beta):
        if x < orb:
            count += int(x in alpha) + int(x in beta)
        elif x == orb:
            # (x, α) < (x, β); only counts if we are creating/annihilating β
            # and the α-mode on the same orbital is occupied.
            if spin == 1 and x in alpha:
                count += 1
    sgn = 1 if count % 2 == 0 else -1

    if spin == 0:
        new_a = (alpha | {orb}) if create else (alpha - {orb})
        new_b = beta
    else:
        new_a = alpha
        new_b = (beta | {orb}) if create else (beta - {orb})
    return frozenset(new_a), frozenset(new_b), sgn


# ---------------------------------------------------------------------------
# Operator base class
# ---------------------------------------------------------------------------

def _is_scalar(x):
    return isinstance(x, (int, float, complex, sp.Expr)) and not isinstance(x, bool)


def _nonzero(x):
    if isinstance(x, sp.Expr):
        return sp.simplify(x) != 0
    return x != 0


def _state_iter(state):
    """Yield (det_string, coef) pairs from any supported state form."""
    if isinstance(state, str):
        yield (state, 1)
        return
    # SlaterDet has a det_string attribute but no dets/coefs.
    if hasattr(state, 'det_string') and not hasattr(state, 'dets'):
        yield (state.det_string, 1)
        return
    if hasattr(state, 'dets') and hasattr(state, 'coefs'):
        for d, c in zip(state.dets, state.coefs):
            yield (d.det_string, c)
        return
    if isinstance(state, dict):
        for k, v in state.items():
            yield (k, v)
        return
    raise TypeError(f"unsupported state type: {type(state).__name__}")


def _state_to_canonical_dict(state):
    """{canonical_string: canonical-basis-coefficient} for any state form.

    The output dict represents the state in the canonical basis (i.e.
    coefficients on |canonical_string⟩_canonical, the interleaved-JW-ordered
    canonical state). Conversion to user-string FixedPsi multiplies by
    canonicalize(canonical_string)[1].
    """
    out = {}
    for det_str, c_user in _state_iter(state):
        a, b, sgn = _parse_det(det_str)        # |det_str⟩_user = sgn · |canon(a,b)⟩
        key = _canonical_string(a, b)
        out[key] = out.get(key, 0) + c_user * sgn
    return {k: v for k, v in out.items() if _nonzero(v)}


def _canonical_dict_to_fixedpsi(canon_dict):
    """{canonical_string: canonical-coef} → FixedPsi (user-basis).

    For each canonical string key, the FixedPsi coefficient is the
    canonical coefficient times the sign relating |canonical_string⟩_user
    (FixedPsi's interpretation) back to |canonical_string⟩_canonical.
    Both ±1, so multiplication and division are the same.
    """
    psi = symvb.FixedPsi()
    for canon_str, c_canon in canon_dict.items():
        if not _nonzero(c_canon):
            continue
        sgn_user = canonicalize(canon_str)[1]
        c_user = c_canon * sgn_user
        if _nonzero(c_user):
            psi.add_str_det(canon_str, coef=c_user)
    return psi


class Operator:
    """Abstract second-quantized operator.

    Subclasses must implement
        _apply_internal((alpha_set, beta_set))
            -> dict {(α', β'): canonical-basis coefficient}
    operating in the canonical (interleaved-JW) representation.

    Public API:
        apply(state)         -> FixedPsi
        matrix(basis)        -> sympy Matrix in user basis
        expectation(state)   -> scalar (assumes orthonormal basis dets)
    Algebra:  +, -, scalar *, @ (composition).
    """

    # ---- subclass hook ----
    def _apply_internal(self, ab):
        raise NotImplementedError

    # ---- internal: canonical-basis dict ----
    def _apply_canonical(self, state):
        """{canonical_string: canonical-basis coefficient} after applying self.

        Goes through the canonical-basis representation; not part of the
        public API. Used by `apply`, `matrix`, and `expectation`.
        """
        in_dict = _state_to_canonical_dict(state)
        out = {}
        for canon_in, c_in in in_dict.items():
            a, b, _ = _parse_det(canon_in)   # canon_in is its own canonical string
            for (aa, bb), coef in self._apply_internal((a, b)).items():
                key = _canonical_string(aa, bb)
                out[key] = out.get(key, 0) + c_in * coef
        return {k: v for k, v in out.items() if _nonzero(v)}

    # ---- public API ----
    def apply(self, state):
        """Action on |state⟩. Returns a FixedPsi (user-basis).

        `state` can be a symvb det string, a SlaterDet, a FixedPsi, or a
        dict {det_string: coef}. The returned FixedPsi has dets keyed by
        canonical (interleaved) strings.
        """
        return _canonical_dict_to_fixedpsi(self._apply_canonical(state))

    def matrix(self, basis):
        """Build M_ij = ⟨bᵢ| O |bⱼ⟩ as a sympy Matrix.

        Each `bᵢ` may be a det string, a SlaterDet, or a FixedPsi. The
        underlying symvb dets are assumed orthonormal (s = 0). Use
        `Molecule.build_matrix` for the s ≠ 0 path.
        """
        N = len(basis)
        basis_canon = [_state_to_canonical_dict(b) for b in basis]
        M = sp.zeros(N, N)
        for j in range(N):
            applied = {}
            for canon_d, c_canon in basis_canon[j].items():
                a, b, _ = _parse_det(canon_d)
                for (aa, bb), coef in self._apply_internal((a, b)).items():
                    key = _canonical_string(aa, bb)
                    applied[key] = applied.get(key, 0) + c_canon * coef
            for i in range(N):
                inner = 0
                for d, c in applied.items():
                    if d in basis_canon[i]:
                        inner = inner + sp.conjugate(basis_canon[i][d]) * c
                M[i, j] = M[i, j] + inner
        return M

    def expectation(self, state):
        """⟨ψ| O |ψ⟩ assuming `state`'s dets are orthonormal."""
        canon = _state_to_canonical_dict(state)
        total = 0
        for d_j, c_j in canon.items():
            a, b, _ = _parse_det(d_j)
            for (aa, bb), coef in self._apply_internal((a, b)).items():
                d_i = _canonical_string(aa, bb)
                c_i = canon.get(d_i, 0)
                total = total + sp.conjugate(c_i) * c_j * coef
        return sp.simplify(total)

    # ---- algebra ----
    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, Operator):
            return NotImplemented
        terms = []
        for op in (self, other):
            if isinstance(op, _Sum):
                terms.extend(op.terms)
            else:
                terms.append((1, op))
        return _Sum(terms)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __neg__(self):
        if isinstance(self, _Sum):
            return _Sum([(-c, op) for c, op in self.terms])
        return _Sum([(-1, self)])

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        # scalar * operator OR operator * operator (composition)
        if isinstance(other, Operator):
            return self.__matmul__(other)
        if _is_scalar(other):
            if isinstance(self, _Sum):
                return _Sum([(c * other, op) for c, op in self.terms])
            return _Sum([(other, self)])
        return NotImplemented

    def __rmul__(self, other):
        if _is_scalar(other):
            return self.__mul__(other)
        return NotImplemented

    def __matmul__(self, other):
        if not isinstance(other, Operator):
            return NotImplemented
        factors = []
        if isinstance(self, _Product):
            factors.extend(self.factors)
        else:
            factors.append(self)
        if isinstance(other, _Product):
            factors.extend(other.factors)
        else:
            factors.append(other)
        return _Product(factors)


# ---------------------------------------------------------------------------
# Internal node types
# ---------------------------------------------------------------------------

class _LadderOp(Operator):
    """Primitive c[†]_{orb,spin}."""

    def __init__(self, orb, spin, dagger):
        self.orb = orb
        self.spin = spin
        self.dagger = dagger

    def _apply_internal(self, ab):
        a, b, s = _apply_c(ab[0], ab[1], self.orb, self.spin, self.dagger)
        if a is None:
            return {}
        return {(a, b): s}

    def __repr__(self):
        sym = 'c†' if self.dagger else 'c'
        sp_label = 'α' if self.spin == 0 else 'β'
        return f"{sym}_{{{self.orb},{sp_label}}}"


class _Sum(Operator):
    """Linear combination Σ_k c_k · O_k."""

    def __init__(self, terms):
        self.terms = list(terms)

    def _apply_internal(self, ab):
        out = {}
        for coef, op in self.terms:
            for k, v in op._apply_internal(ab).items():
                out[k] = out.get(k, 0) + coef * v
        return {k: v for k, v in out.items() if _nonzero(v)}

    def __repr__(self):
        return ' + '.join(f"{c}·{op}" for c, op in self.terms)


class _Product(Operator):
    """Composition F[0] · F[1] · … · F[-1]; rightmost applies first."""

    def __init__(self, factors):
        self.factors = list(factors)

    def _apply_internal(self, ab):
        cur = {ab: 1}
        for op in reversed(self.factors):
            new = {}
            for ab_in, c_in in cur.items():
                for ab_out, c_out in op._apply_internal(ab_in).items():
                    new[ab_out] = new.get(ab_out, 0) + c_in * c_out
            cur = {k: v for k, v in new.items() if _nonzero(v)}
            if not cur:
                return {}
        return cur

    def __repr__(self):
        return ' '.join(repr(f) for f in self.factors)


class _OrbitalPerm(Operator):
    """Relabel orbital labels per `orbital_map`. Fermion sign tracked
    using a unified inversion count over the (orb, spin) tuple sequence,
    which is correct for interleaved JW even when the permutation
    crosses the α/β interleave."""

    def __init__(self, orbital_map):
        self.orbital_map = dict(orbital_map)

    def _apply_internal(self, ab):
        a, b = ab
        # Original creation order in interleaved canonical JW.
        orig_modes = sorted(
            [(x, 0) for x in a] + [(x, 1) for x in b]
        )
        new_modes = [(self.orbital_map.get(o, o), s) for (o, s) in orig_modes]
        if len(set(new_modes)) != len(new_modes):
            return {}
        new_a = frozenset(o for (o, s) in new_modes if s == 0)
        new_b = frozenset(o for (o, s) in new_modes if s == 1)
        return {(new_a, new_b): _inversion_sign(new_modes)}

    def __repr__(self):
        items = sorted(self.orbital_map.items())
        return 'P[' + ','.join(f"{k}→{v}" for k, v in items) + ']'


class _Identity(Operator):
    def _apply_internal(self, ab):
        return {ab: 1}

    def __repr__(self):
        return 'I'


def identity():
    return _Identity()


# ---------------------------------------------------------------------------
# High-level constructors
# ---------------------------------------------------------------------------

def _spin_index(spin):
    if spin in (0, 'a', 'alpha', 'α'):
        return 0
    if spin in (1, 'b', 'beta', 'β'):
        return 1
    raise ValueError(f"unknown spin label: {spin!r}")


def c(orb, spin='alpha'):
    """Annihilation operator c_{orb,spin}."""
    return _LadderOp(orb, _spin_index(spin), dagger=False)


def cdag(orb, spin='alpha'):
    """Creation operator c†_{orb,spin}."""
    return _LadderOp(orb, _spin_index(spin), dagger=True)


def number(orb, spin=None):
    """Occupation operator n̂_{orb,spin}; spin=None gives n̂_orb = n̂_α + n̂_β."""
    if spin is None:
        return number(orb, 'alpha') + number(orb, 'beta')
    return cdag(orb, spin) @ c(orb, spin)


def double_occ(orb):
    """Double-occupancy projector n̂_{orb,α} n̂_{orb,β}."""
    return number(orb, 'alpha') @ number(orb, 'beta')


def hop(i, j, spin=None, hermitian=True):
    """Hopping c†_i c_j (+ h.c. if hermitian); spin=None sums α and β."""
    if spin is None:
        return hop(i, j, 'alpha', hermitian) + hop(i, j, 'beta', hermitian)
    fwd = cdag(i, spin) @ c(j, spin)
    if not hermitian or i == j:
        return fwd
    return fwd + cdag(j, spin) @ c(i, spin)


# ---- spin operators ----

def s_z(orbs):
    """Total Sz = (1/2) Σᵢ (n̂_{i,α} − n̂_{i,β})."""
    half = sp.Rational(1, 2)
    return sum((half * (number(o, 'alpha') - number(o, 'beta')) for o in orbs),
               start=0)


def s_plus(orbs):
    """S₊ = Σᵢ c†_{i,α} c_{i,β}."""
    return sum((cdag(o, 'alpha') @ c(o, 'beta') for o in orbs), start=0)


def s_minus(orbs):
    """S₋ = Σᵢ c†_{i,β} c_{i,α}."""
    return sum((cdag(o, 'beta') @ c(o, 'alpha') for o in orbs), start=0)


def s_squared(orbs):
    """S² = S₊ S₋ + S_z² − S_z."""
    Sz = s_z(orbs)
    return s_plus(orbs) @ s_minus(orbs) + Sz @ Sz - Sz


def s_dot(i, j):
    """Sᵢ · Sⱼ = Sᵢ_z Sⱼ_z + (1/2)(Sᵢ₊ Sⱼ₋ + Sᵢ₋ Sⱼ₊)."""
    half = sp.Rational(1, 2)
    Szi = half * (number(i, 'alpha') - number(i, 'beta'))
    Szj = half * (number(j, 'alpha') - number(j, 'beta'))
    Spi = cdag(i, 'alpha') @ c(i, 'beta')
    Smi = cdag(i, 'beta') @ c(i, 'alpha')
    Spj = cdag(j, 'alpha') @ c(j, 'beta')
    Smj = cdag(j, 'beta') @ c(j, 'alpha')
    return Szi @ Szj + half * (Spi @ Smj + Smi @ Spj)


# ---- η-pairing ----

def eta_plus(site_signs):
    """η₊ = Σᵢ sᵢ c†_{i,α} c†_{i,β}. `site_signs`: dict {orb: ±1}."""
    return sum((s * (cdag(o, 'alpha') @ cdag(o, 'beta'))
                for o, s in site_signs.items()), start=0)


def eta_minus(site_signs):
    """η₋ = Σᵢ sᵢ c_{i,β} c_{i,α}."""
    return sum((s * (c(o, 'beta') @ c(o, 'alpha'))
                for o, s in site_signs.items()), start=0)


def eta_z(orbs):
    """η_z = (1/2) (N̂ − L), with L = len(orbs)."""
    half = sp.Rational(1, 2)
    Nhat = sum((number(o) for o in orbs), start=0)
    L = len(orbs)
    return half * Nhat + (-half * L) * identity()


def eta_squared(site_signs):
    """η² = η₊ η₋ + η_z² − η_z."""
    orbs = list(site_signs.keys())
    Ez = eta_z(orbs)
    return eta_plus(site_signs) @ eta_minus(site_signs) + Ez @ Ez - Ez


# ---- permutation / symmetry ----

def transposition(i, j):
    """Permute orbital labels i ↔ j."""
    return _OrbitalPerm({i: j, j: i})


def orbital_perm(orbital_map):
    """Generic orbital-label permutation; map is a dict {old: new}."""
    return _OrbitalPerm(orbital_map)


def _enumerate_orbital_group(generators):
    """Closure of the group generated by orbital-permutation maps."""

    def compose(p, q):
        keys = set(p.keys()) | set(q.keys())
        return {k: p.get(q.get(k, k), q.get(k, k)) for k in keys}

    def normalize(p):
        return tuple(sorted((k, v) for k, v in p.items() if k != v))

    seen = {(): {}}
    queue = [{}]
    while queue:
        g = queue.pop()
        for gen in generators:
            new = compose(gen, g)
            key = normalize(new)
            if key not in seen:
                seen[key] = new
                queue.append(new)
    return list(seen.values())


def reynolds_projector(generators):
    """P^{(triv)} = (1/|G|) Σ_g g, the trivial-irrep projector.

    `generators` is a list of orbital-permutation dicts. Group closure
    is computed automatically and fermion signs are tracked, so this
    works on over-half-filled bases (e.g. C₄H₄²⁻) too.
    """
    group = _enumerate_orbital_group(generators)
    G = len(group)
    inv_G = sp.Rational(1, G)
    return sum((inv_G * orbital_perm(g) for g in group), start=0)


# ---- VB-flavoured composites ----

def singlet_proj(i, j):
    """Spin-singlet projector for two electrons distributed on (i, j).

    P_S^{ij} = 1/4 − Sᵢ · Sⱼ. On the (1, 1) sector (one electron each on
    i and j) this is the singlet-vs-triplet projector with eigenvalues
    {0, 1}; outside that sector it is *not* a projector — interpret with
    care (e.g. on a closed shell |aA⟩ it returns ¼ |aA⟩).
    """
    return sp.Rational(1, 4) * identity() - s_dot(i, j)


def bond_singlet_creator(i, j):
    """Bond singlet creator: (1/√2)(c†_{i,α} c†_{j,β} − c†_{i,β} c†_{j,α}).

    Acting on a state with i and j unoccupied appends a normalized
    two-electron singlet bond on (i, j). Stacked applications build a
    Rumer structure.
    """
    return (sp.Rational(1, 1) / sp.sqrt(2)) * (
        cdag(i, 'alpha') @ cdag(j, 'beta')
        - cdag(i, 'beta') @ cdag(j, 'alpha')
    )
