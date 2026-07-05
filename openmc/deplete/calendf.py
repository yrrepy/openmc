"""CALENDF probability-table (``.tpe``) reader and Bondarenko material-dilution
fold for GENDF-based unresolved-resonance-range (URR) self-shielding.

This module is the GENDF sibling of the PENDF ``mat_ssf`` correction. Where the
PENDF path bakes PURR equiprobable tables into the HDF5 library, the GENDF path
reads **CALENDF moment/quadrature probability tables** shipped by FISPACT-II as
per-library / per-temperature / per-nuclide ``.tpe`` files, e.g.::

    .../tal2017-n/tp-709-294/W182-294.tpe        (294 K, CCFE-709 structure)

The fold produces a per-group self-shielding factor ``mat_ssf`` that multiplies
the infinite-dilution group cross section in the GENDF collapse. It is the exact
analogue of FISPACT-II ``PROBTABLE multxs=1`` with a homogeneous background
cross section ``sigma0`` from the material composition (no geometry / Dancoff).

Everything the correction needs lives here (parser, sigma0, fold, apply seam);
nothing is imported from :mod:`openmc.deplete.microxs`.

Reverse-engineered ``.tpe`` format (verified against the real TENDL-2017 files)
------------------------------------------------------------------------------
The file is fixed-column ASCII, whitespace-tokenizable (no missing-'E' Fortran
exponents occur — all values carry an explicit ``E``). Two header lines then a
sequence of per-group records::

    line 1:  tables de probabilite pour  74-W -182  IAEA      DIST-      REV1-
    line 2:  ZA= 74182. MAT=7431 TEFF= 293.6  256 gr. de 1.0000E-1 a 1.3183E+4 IP=4

Header line 2 fields:
    ``ZA=``   nuclide ZA (Z*1000 + A), float with trailing dot
    ``MAT=``  ENDF MAT number
    ``TEFF=`` effective (Doppler) temperature in K (293.6 for the 294 K set)
    ``N gr.`` number of *groups carrying tables* (256 here) — NOT the 709-group
              count; the table covers only the range where the nuclide has
              structure (resolved + URR), from the span
    ``de A a B`` energy span in **eV** (1.0e-1 .. 1.3183e4 for W182)
    ``IP=``   nominal partial count (file-level; the per-record ``NPAR`` is
              authoritative for the actual column count — they can differ)

Each per-group record is a header line followed by ``NOR`` band rows::

    IG  161 ENG=1.258925E+4 1.318257E+4 NOR= 9 I= -8 NPAR=3 KP=   2 101   4   0   0
      3.102607E-3  5.301709E-1  4.032892E-1  1.268817E-1  2.00000E-20
      ... (NOR band rows total) ...

Record-header fields:
    ``IG``    CALENDF group id (counts DOWN from high energy). See mapping below.
    ``ENG=``  ascending (Elo, Ehi) eV pair — the group's energy boundaries.
    ``NOR``   number of quadrature bands in this group (VARIES per group: 1..~11).
    ``I``     CALENDF moment/order index (informational; not used by the fold).
    ``NPAR``  number of partial reactions present in THIS record's band rows.
    ``KP``    five slots giving the ENDF MT number of each partial, in column
              order. Only the first ``NPAR`` are active. Observed:
                non-fissile: KP = (2, 101, 4, 0, 0)   -> NPAR=3
                fissile:     KP = (2, 101, 18, 4, 15) -> NPAR up to 5
              ``NPAR`` and ``KP`` change from group to group *within one file*
              (e.g. U-238 has NPAR in {3,4,5} and KP variants including
              (2,101,18,15,0)), so columns MUST be resolved by MT via ``KP`` —
              fixed column positions are unsafe.

Each band row is ``2 + NPAR`` floats, which may WRAP across physical lines
(U-238's 7-value rows print 6 then 1). Read them by streaming ``2 + NPAR``
tokens per band, ignoring line breaks. Columns:
    col 0            band probability  p_b   (Sum_b p_b = 1 per group)
    col 1            band total        sigma_t,b  (barns, absolute)
    col 2 .. 1+NPAR  band partials in KP order (barns, absolute)

Reaction extraction (by MT via KP, robust to varying NPAR):
    elastic  = partial with MT = 2     (always the first partial)
    capture  = partial with MT = 101   (total disappearance; ~ n,gamma in URR)
    fission  = partial with MT = 18    (present only for U-235/238, Pu-239/240)
The library's radiative-capture channel is MT=102; MT=101 disappearance and
MT=102 capture agree to <~1.5% for these nuclides in this energy range.

Zero placeholder: absent partials are written as ``2.00000E-20`` (occasionally
``2.00090E-20`` etc.); any value <= 1e-19 is treated as exactly 0.0.

Verified sum rules (self-checks in :func:`read_tpe`):
    total ~= Sum(partials) : worst 1.1e-5 rel over the 24 flagged nuclides
    Sum_b p_b ~= 1         : worst 2e-7 abs

Group-index mapping (the #1 trap) — PROVEN
------------------------------------------
GENDF group cross sections (``GENDFLibrary.get_xs``) are returned as a
709-length array indexed 0-based in **ascending energy**: element ``i`` is the
group ``[edges[i], edges[i+1]]`` of ``GROUP_STRUCTURES['CCFE-709']`` (edges
ascending, ``edges[0]=1e-5`` eV, ``edges[709]=1e9`` eV).

The CALENDF ``IG`` maps to that 0-based library index ``i`` by matching the
record's ``ENG`` pair to the library edges. Proven against W182 and U238 (exact
at both ends of every table span):

    IG=161  ENG=[1.258925e4, 1.318257e4] eV  <->  edges[455..456]  ->  i = 455
    IG=162  ENG=[1.202264e4, 1.258925e4] eV  <->  edges[454..455]  ->  i = 454
    IG=416  ENG=[1.000000e-1, 1.047129e-1] eV <-> edges[200..201]  ->  i = 200

i.e. for the CCFE-709 structure the relation is exactly ``i = 616 - IG``
(``IG`` counts down from high energy; the +overhang above 20 MeV shifts the
FISPACT numbering off the array top). :func:`read_tpe` does NOT hardcode this;
it maps every record by ENG<->edges matching and stores both ``i`` and the ENG
bounds so the mapping is re-verifiable (gate 1). ``f_g`` factors are therefore
0-based ascending, aligned with ``get_xs`` output for elementwise multiply.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional, Iterable

import numpy as np

from openmc.mgxs import GROUP_STRUCTURES


__all__ = [
    'DEFAULT_FLAGGED',
    'TpeTable',
    'read_tpe',
    'find_tpe',
    'material_dilution_sigma0',
    'mat_ssf_factors_gendf',
    'compute_mat_ssf',
]


# The 24-nuclide default flagged set (same list as the PENDF sibling): nuclides
# that self-shield in the URR in a moderated/epithermal spectrum.
DEFAULT_FLAGGED = frozenset({
    'W180', 'W182', 'W183', 'W184', 'W186',
    'Ta181',
    'Re185', 'Re187',
    'Hf174', 'Hf176', 'Hf177', 'Hf178', 'Hf179', 'Hf180',
    'Os186', 'Os187', 'Os188', 'Os189', 'Os190', 'Os192',
    'U235', 'U238',
    'Pu239', 'Pu240',
})

# ENDF MT numbers as they appear in the .tpe KP array.
_MT_ELASTIC = 2
_MT_CAPTURE = 101   # total disappearance (n,gamma dominated in the URR)
_MT_FISSION = 18

# Values <= this are the .tpe zero placeholder (~2e-20), treated as exactly 0.
_ZERO_PLACEHOLDER = 1e-19

# Tolerances for the read_tpe self-checks (measured worst cases are far tighter).
_SUMRULE_RTOL = 1e-3    # |total - Sum(partials)| / total
_PROB_ATOL = 1e-3       # |Sum_b p_b - 1|
_EDGE_RTOL = 5e-3       # ENG<->library-edge match (library edges are 5 sig figs)

_REACTION_MT = {'capture': _MT_CAPTURE, 'fission': _MT_FISSION}


class _BandGroup:
    """Quadrature bands for one energy group, keyed by library group index.

    Attributes
    ----------
    ig : int
        CALENDF group id from the ``.tpe`` record.
    index : int
        0-based ascending-energy library group index (aligns with
        ``GENDFLibrary.get_xs`` output). This is the dict key in
        :class:`TpeTable`.
    elo, ehi : float
        Group energy boundaries in eV (ascending pair from ``ENG=``).
    prob : numpy.ndarray
        Band probabilities ``p_b`` (sum to 1), length NOR.
    total, elastic, capture, fission : numpy.ndarray
        Absolute band cross sections in barns, length NOR. ``fission`` is all
        zeros when the nuclide has no MT=18 partial.
    """

    __slots__ = ('ig', 'index', 'elo', 'ehi', 'prob',
                 'total', 'elastic', 'capture', 'fission')

    def __init__(self, ig, index, elo, ehi, prob, total,
                 elastic, capture, fission):
        self.ig = ig
        self.index = index
        self.elo = elo
        self.ehi = ehi
        self.prob = prob
        self.total = total
        self.elastic = elastic
        self.capture = capture
        self.fission = fission

    def reaction(self, name):
        """Return the band cross-section array for a reaction name."""
        if name == 'capture':
            return self.capture
        if name == 'fission':
            return self.fission
        if name == 'elastic':
            return self.elastic
        if name == 'total':
            return self.total
        raise ValueError(f"unknown reaction {name!r}")


class TpeTable:
    """Parsed CALENDF ``.tpe`` probability table for one nuclide.

    Bands are keyed by the 0-based ascending-energy **library group index**
    (see the module docstring for the proven ``IG`` -> index mapping).

    Attributes
    ----------
    za : int
        Nuclide ZA (Z*1000 + A).
    mat : int
        ENDF MAT number.
    teff : float
        Effective (Doppler) temperature in K.
    ip : int
        File-level nominal partial count from the ``IP=`` header field.
    group_structure : str
        Name of the group structure used for the IG mapping (e.g. 'CCFE-709').
    span : tuple of float
        (Elow, Ehigh) energy span in eV from the header.
    groups : dict of int to _BandGroup
        Per-group bands keyed by library group index.
    """

    def __init__(self, za, mat, teff, ip, group_structure, span, groups):
        self.za = za
        self.mat = mat
        self.teff = teff
        self.ip = ip
        self.group_structure = group_structure
        self.span = span
        self.groups = groups

    @property
    def indices(self):
        """Sorted list of library group indices carrying bands."""
        return sorted(self.groups)

    def has_fission(self):
        """True if any group carries a nonzero MT=18 fission partial."""
        return any(np.any(g.fission > 0.0) for g in self.groups.values())

    def sigma_inf(self, reaction):
        """Probability-weighted infinite-dilution group XS per library index.

        Returns
        -------
        dict of int to float
            ``{library_index: Sum_b p_b sigma_x,b / Sum_b p_b}`` for the
            reaction (the infinite-dilution group cross section the table
            reproduces). Used by the round-trip gate.
        """
        out = {}
        for i, g in self.groups.items():
            sx = g.reaction(reaction)
            out[i] = float((g.prob * sx).sum() / g.prob.sum())
        return out

    def __repr__(self):
        return (f"TpeTable(za={self.za}, mat={self.mat}, teff={self.teff}, "
                f"n_groups={len(self.groups)}, "
                f"struct={self.group_structure!r})")


_HDR_RE = re.compile(
    r"ZA=\s*([0-9.]+)\s+MAT=\s*(\d+)\s+TEFF=\s*([0-9.+\-eE]+)\s+"
    r"(\d+)\s*gr\.\s*de\s*([0-9.+\-eE]+)\s*a\s*([0-9.+\-eE]+)\s+IP=\s*(\d+)"
)


def _parse_record_header(line):
    """Parse an ``IG ...`` record header line.

    Returns (ig, elo, ehi, nor, npar, kp_list).
    """
    toks = line.replace('=', ' = ').split()
    ig = int(toks[1])
    eidx = toks.index('ENG') + 2
    elo = float(toks[eidx])
    ehi = float(toks[eidx + 1])
    nor = int(toks[toks.index('NOR') + 2])
    npar = int(toks[toks.index('NPAR') + 2])
    kp_pos = toks.index('KP') + 2
    kp = [int(x) for x in toks[kp_pos:kp_pos + 5]]
    return ig, elo, ehi, nor, npar, kp


def read_tpe(path, group_structure='CCFE-709'):
    """Read a CALENDF ``.tpe`` probability table.

    Bands are keyed by the 0-based ascending-energy library group index, mapped
    from the CALENDF ``IG`` by matching each record's ``ENG`` pair to the
    library group edges (no hardcoded offset). Runs sum-rule / probability /
    edge-consistency self-checks and raises ``ValueError`` on violation.

    Parameters
    ----------
    path : path-like
        Path to a ``<Nuclide>-<T>.tpe`` file.
    group_structure : str, optional
        Group-structure name whose edges (from
        :data:`openmc.mgxs.GROUP_STRUCTURES`) the ``IG`` maps onto. Default
        ``'CCFE-709'`` (the tp-709 tables). Pass ``'UKAEA-1102'`` for tp-1102.

    Returns
    -------
    TpeTable
    """
    path = Path(path)
    if group_structure not in GROUP_STRUCTURES:
        raise ValueError(
            f"unknown group_structure {group_structure!r}; "
            f"available: {sorted(GROUP_STRUCTURES)}")
    edges = np.asarray(GROUP_STRUCTURES[group_structure], dtype=float)
    n_bounds = len(edges)

    with open(path) as fh:
        lines = fh.readlines()

    m = _HDR_RE.search(lines[1])
    if m is None:
        raise ValueError(f"cannot parse .tpe header line: {lines[1]!r}")
    za = int(float(m.group(1)))
    mat = int(m.group(2))
    teff = float(m.group(3))
    span = (float(m.group(5)), float(m.group(6)))
    ip = int(m.group(7))

    groups = {}
    worst_sumrule = 0.0
    worst_prob = 0.0
    worst_edge = 0.0

    i = 2
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.lstrip().startswith('IG'):
            i += 1
            continue
        ig, elo, ehi, nor, npar, kp = _parse_record_header(line)

        # Stream 2 + npar floats per band across physical line wrapping.
        need = nor * (2 + npar)
        vals = []
        j = i + 1
        while len(vals) < need:
            vals.extend(float(x) for x in lines[j].split())
            j += 1
        if len(vals) != need:
            raise ValueError(
                f"{path.name} IG={ig}: expected {need} band values, "
                f"got {len(vals)}")
        arr = np.array(vals, dtype=float).reshape(nor, 2 + npar)
        i = j

        prob = arr[:, 0].copy()
        total = arr[:, 1].copy()
        partials = arr[:, 2:2 + npar].copy()
        partials[partials <= _ZERO_PLACEHOLDER] = 0.0

        # Resolve reaction columns by MT via KP (columns move between records).
        def _col(mt):
            for k in range(npar):
                if kp[k] == mt:
                    return partials[:, k]
            return np.zeros(nor)

        elastic = _col(_MT_ELASTIC)
        capture = _col(_MT_CAPTURE)
        fission = _col(_MT_FISSION)

        # --- self-checks ---
        with np.errstate(divide='ignore', invalid='ignore'):
            srule = np.abs(partials.sum(axis=1) - total) / np.where(
                total > 0, total, 1.0)
        worst_sumrule = max(worst_sumrule, float(srule.max()))
        worst_prob = max(worst_prob, abs(float(prob.sum()) - 1.0))

        # --- IG -> library index by ENG<->edge matching ---
        ilo = int(np.argmin(np.abs(edges - elo)))
        # boundary edge, not the last (group index must be < n_bounds-1)
        if ilo >= n_bounds - 1:
            raise ValueError(
                f"{path.name} IG={ig}: Elo {elo:.6e} eV maps past the top "
                f"of the {group_structure} structure")
        d_lo = abs(edges[ilo] - elo) / elo
        d_hi = abs(edges[ilo + 1] - ehi) / ehi
        worst_edge = max(worst_edge, d_lo, d_hi)
        if d_lo > _EDGE_RTOL or d_hi > _EDGE_RTOL:
            raise ValueError(
                f"{path.name} IG={ig}: ENG [{elo:.6e},{ehi:.6e}] does not "
                f"match {group_structure} edges [{edges[ilo]:.6e},"
                f"{edges[ilo + 1]:.6e}] (rel {d_lo:.2e},{d_hi:.2e}); wrong "
                f"group_structure?")

        groups[ilo] = _BandGroup(
            ig=ig, index=ilo, elo=elo, ehi=ehi, prob=prob, total=total,
            elastic=elastic, capture=capture, fission=fission)

    if worst_sumrule > _SUMRULE_RTOL:
        raise ValueError(
            f"{path.name}: total != Sum(partials) worst rel {worst_sumrule:.2e}"
            f" exceeds {_SUMRULE_RTOL}")
    if worst_prob > _PROB_ATOL:
        raise ValueError(
            f"{path.name}: Sum_b p_b != 1 worst abs {worst_prob:.2e} "
            f"exceeds {_PROB_ATOL}")

    return TpeTable(za=za, mat=mat, teff=teff, ip=ip,
                    group_structure=group_structure, span=span, groups=groups)


def find_tpe(calendf_path, nuclide):
    """Locate a nuclide's ``.tpe`` file under a per-temperature directory.

    Globs ``f"{nuclide}-*.tpe"`` (the ``*`` matches the temperature tag, e.g.
    ``294``). The ``nuclide`` name must be in CALENDF/FISPACT form (element +
    mass, metastables as an ``m`` suffix, e.g. ``W182``, ``Ac222m``).

    Parameters
    ----------
    calendf_path : path-like
        Directory of ``.tpe`` files for one library/temperature (e.g.
        ``.../tp-709-294``).
    nuclide : str
        Nuclide name.

    Returns
    -------
    pathlib.Path or None
        The single matching file, or ``None`` if none matches.

    Raises
    ------
    ValueError
        If more than one file matches.
    """
    calendf_path = Path(calendf_path)
    matches = sorted(calendf_path.glob(f"{nuclide}-*.tpe"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"multiple .tpe files match {nuclide!r} in {calendf_path}: "
            f"{[p.name for p in matches]}")
    return matches[0]


def material_dilution_sigma0(densities, resonant, group_totals, n_groups=None):
    """Background cross section ``sigma0_g`` for a resonant nuclide.

    ``sigma0_g = Sum_{j != resonant} N_j * sigma_t,j(g) / N_resonant``

    Only density ratios matter, so ``densities`` may be any consistent
    atom-density unit. The diluters are every nuclide in ``densities`` other
    than ``resonant``; each must have a group-total array in ``group_totals``.

    Parameters
    ----------
    densities : dict of str to float
        Atom densities keyed by nuclide (must include ``resonant``).
    resonant : str
        Name of the resonant (self-shielded) nuclide.
    group_totals : dict of str to numpy.ndarray
        Group total cross sections (barns) for the diluters, keyed by nuclide;
        each array has length ``n_groups`` (as returned by
        ``GENDFLibrary.get_xs(nuc, 1)``).
    n_groups : int, optional
        Group count for the returned array. Inferred from ``group_totals`` when
        diluters are present; required only for the pure-absorber case (no
        diluters), where it cannot be inferred. (Minor addition to the brief's
        3-arg signature so the no-diluter case can return a correctly sized
        zeros array.)

    Returns
    -------
    numpy.ndarray
        ``sigma0_g`` of length ``n_groups`` (zeros if there are no diluters).

    Raises
    ------
    ValueError
        If the resonant nuclide is absent from / has non-positive density in
        ``densities`` (the caller/apply layer is expected to skip those before
        calling), or if a diluter's group totals are missing, or if the
        group count cannot be determined.
    """
    if resonant not in densities:
        raise ValueError(
            f"resonant nuclide {resonant!r} absent from densities")
    n_res = densities[resonant]
    if n_res <= 0.0:
        raise ValueError(
            f"resonant nuclide {resonant!r} has non-positive density {n_res!r}")

    diluters = [j for j in densities if j != resonant]

    if n_groups is None:
        if group_totals:
            n_groups = len(next(iter(group_totals.values())))
        elif diluters:
            n_groups = len(group_totals[diluters[0]])  # triggers KeyError below
        else:
            raise ValueError(
                "cannot infer n_groups: no diluters and n_groups not given")

    sigma0 = np.zeros(n_groups, dtype=float)
    for j in diluters:
        if j not in group_totals:
            raise ValueError(
                f"diluter {j!r} has a density but no entry in group_totals")
        sig = np.asarray(group_totals[j], dtype=float)
        if len(sig) != n_groups:
            raise ValueError(
                f"diluter {j!r} group_totals length {len(sig)} != {n_groups}")
        sigma0 += densities[j] * sig

    sigma0 /= n_res
    return sigma0


def mat_ssf_factors_gendf(tables, sigma0_g, n_groups, reaction):
    """Bondarenko / narrow-resonance self-shielding factors ``f_g``.

    For each group ``g`` that carries bands (absolute barns; no factor-form
    branch)::

        w_b       = p_b / (sigma0_g + sigma_t,b)
        sigma_eff = Sum_b w_b sigma_x,b / Sum_b w_b
        sigma_inf = Sum_b p_b sigma_x,b / Sum_b p_b
        f_g       = sigma_eff / sigma_inf      (f_g = 1 where sigma_inf == 0)

    Groups without bands get ``f_g = 1``.

    Parameters
    ----------
    tables : TpeTable
        Parsed probability tables (bands keyed by library group index).
    sigma0_g : numpy.ndarray or float
        Background cross section per library group (length ``n_groups``), or a
        scalar broadcast to all groups (e.g. ``0.0`` for a pure absorber).
    n_groups : int
        Number of groups in the target library (length of the returned array).
    reaction : {'capture', 'fission'}
        Reaction to shield.

    Returns
    -------
    numpy.ndarray
        ``f_g`` of length ``n_groups``, default 1.0.
    """
    if reaction not in _REACTION_MT:
        raise ValueError(
            f"reaction must be one of {sorted(_REACTION_MT)}, got {reaction!r}")

    sigma0_g = np.asarray(sigma0_g, dtype=float)
    if sigma0_g.ndim == 0:
        sigma0_g = np.full(n_groups, float(sigma0_g))
    elif len(sigma0_g) != n_groups:
        raise ValueError(
            f"sigma0_g length {len(sigma0_g)} != n_groups {n_groups}")

    f = np.ones(n_groups, dtype=float)
    for i, g in tables.groups.items():
        if not (0 <= i < n_groups):
            continue
        sig_x = g.reaction(reaction)
        p = g.prob
        sig_inf = float((p * sig_x).sum() / p.sum())
        if sig_inf <= 0.0:
            continue
        w = p / (sigma0_g[i] + g.total)
        sig_eff = float((w * sig_x).sum() / w.sum())
        f[i] = sig_eff / sig_inf
    return f


def compute_mat_ssf(tables, densities, resonant, group_totals, n_groups,
                    reactions=('capture', 'fission')):
    """Apply-layer seam: fold one nuclide's tables into ``f_g`` per reaction.

    Convenience wrapper the GENDF-collapse hook can import: computes
    ``sigma0_g`` from the material composition and returns self-shielding
    factors for each requested reaction. Fission is skipped (returns all ones)
    for nuclides whose tables carry no MT=18 partial.

    Parameters
    ----------
    tables : TpeTable
        Parsed probability tables for ``resonant``.
    densities : dict of str to float
        Atom densities keyed by nuclide (must include ``resonant``).
    resonant : str
        The resonant nuclide name.
    group_totals : dict of str to numpy.ndarray
        Group totals (barns) for the diluters, length ``n_groups``.
    n_groups : int
        Target library group count.
    reactions : iterable of str, optional
        Subset of ``{'capture', 'fission'}``. Default both.

    Returns
    -------
    dict of str to numpy.ndarray
        ``{reaction: f_g}`` with each ``f_g`` of length ``n_groups``.
    """
    sigma0_g = material_dilution_sigma0(
        densities, resonant, group_totals, n_groups=n_groups)
    out = {}
    has_fis = tables.has_fission()
    for rx in reactions:
        if rx == 'fission' and not has_fis:
            out[rx] = np.ones(n_groups, dtype=float)
            continue
        out[rx] = mat_ssf_factors_gendf(tables, sigma0_g, n_groups, rx)
    return out
