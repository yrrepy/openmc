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

Everything the correction needs lives here (parser, sigma0, fold, row scaler,
apply seam); nothing is imported from :mod:`openmc.deplete.microxs`.

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

from openmc.checkvalue import check_iterable_type
from openmc.data import REACTION_MT
from openmc.mgxs import GROUP_STRUCTURES


__all__ = [
    'DEFAULT_FLAGGED',
    'TpeTable',
    'read_tpe',
    'find_tpe',
    'material_dilution_sigma0',
    'mat_ssf_factors_gendf',
    'mat_ssf_total_factors_gendf',
    'iterate_material_dilution_sigma0_gendf',
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

# Library / GENDF-collapse-side ENDF MT numbers. These differ from the .tpe KP
# MTs above: the .tpe carries capture as MT=101 (total disappearance), while the
# GENDF library and the flux collapse identify radiative capture as MT=102. The
# fold uses the .tpe MT=101 band cross sections; the resulting capture f_g then
# multiplies the collapse's MT=102 group row (the two channels agree to <~1.5%
# for the flagged nuclides in this energy range, see the module docstring).
_LIB_TOTAL_MT = 1        # diluter group totals for the sigma0 background
_LIB_CAPTURE_MT = 102    # (n,gamma) rows shielded with the capture f_g
_LIB_FISSION_MT = 18     # fission rows shielded with the fission f_g

# Values <= this are the .tpe zero placeholder (~2e-20), treated as exactly 0.
_ZERO_PLACEHOLDER = 1e-19

# Tolerances for the read_tpe self-checks (measured worst cases are far tighter).
_SUMRULE_RTOL = 1e-3    # |total - Sum(partials)| / total
_PROB_ATOL = 1e-3       # |Sum_b p_b - 1|
_EDGE_RTOL = 5e-3       # ENG<->library-edge match (library edges are 5 sig figs)

_REACTION_MT = {'capture': _MT_CAPTURE, 'fission': _MT_FISSION}

# Library-side MT of a collapse row -> the reaction whose factor multiplies it:
# MT=102 rows take the capture f_g, MT=18 rows the fission f_g.
_LIB_MT_REACTION = {_LIB_CAPTURE_MT: 'capture', _LIB_FISSION_MT: 'fission'}

# Isomer-qualified reaction-name suffix (``_m1`` ...), stripped once before the
# REACTION_MT lookup. GENDF reaction names never carry it; tolerated anyway.
_ISOMER_SUFFIX = re.compile(r'_m\d+$')

# Fixed-point controls for the mutual-shielding sigma_0 refinement (C3). These
# are policy constants, NOT API parameters: iteration is unconditional when the
# URR flag is on and always uses these bounds. FISPACT reports "a few passes";
# the map d(p) -> (1/f_p) sum_{q!=p} f_q sigma_t,inf(q) R_q(d) is contractive
# (R in (0, 1] -> a monotone-decreasing sequence from d0), so the tolerance is
# reached in a handful of passes: the resolved-range CALENDF tables shield far
# more deeply than URR-only tables (nat-W needs ~10 passes vs the PENDF
# sibling's ~4), which is what sizes the cap. The cap is a safety bound, inert
# once the tolerance breaks the loop. No damping (add 0.5 damping only if a
# gate ever shows oscillation -- it is not expected). Same names and values as
# the PENDF sibling (:mod:`openmc.deplete.mat_ssf`).
SIGMA0_ITER_MAX = 15
SIGMA0_ITER_TOL = 1e-3

# Barn-scale floor for the relative-change denominator max(d, eps): guards groups
# where the background is ~0 (a pure absorber, or groups outside every diluter's
# tabulated span) from a spurious large relative delta. Background changes below
# this floor are physically irrelevant.
_SIGMA0_EPS = 1e-10


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
            raise ValueError(
                f"diluter {diluters[0]!r} has a density but no entry in "
                "group_totals")
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


def mat_ssf_total_factors_gendf(tables, sigma0_g, n_groups):
    r"""Per-group total-cross-section self-shielding factor ``R_tot_g`` (C3).

    The total-XS shielding factor that drives the C3 mutual-shielding ``sigma0``
    iteration (:func:`iterate_material_dilution_sigma0_gendf`). It folds the band
    **total** ``sigma_t,b`` exactly as :func:`mat_ssf_factors_gendf` folds a
    partial reaction, so the numerator's shielded total and the denominator's
    infinite-dilution total use the same ``sigma_t,b`` that already sets the
    partial fold's flux weight ``w_b = p_b / (sigma0_g + sigma_t,b)``::

        R_tot_g   = sigma_t,eff_g(sigma0) / sigma_t,inf_g
        sigma_t,eff = Sum_b w_b sigma_t,b / Sum_b w_b,  w_b = p_b/(sigma0_g+sigma_t,b)
        sigma_t,inf = Sum_b p_b sigma_t,b / Sum_b p_b

    ``R_tot_g`` lies in ``(0, 1]``: exactly 1 in groups the ``.tpe`` table does
    not cover (no bands -> no shielding) and approaching 1 as ``sigma0 ->
    infinity`` (infinite dilution). A diluter whose bands END inside the URR (the
    TENDL-2017 W ceilings at ~13-17 keV) simply has no band records above its
    ceiling, so ``R_tot_g`` there is 1 -- that diluter reverts to its
    infinite-dilute contribution in the uncovered groups (graceful degradation,
    mirroring FISPACT-II's own data limit). A partial URR span is therefore
    handled cleanly: the per-group ``.tpe`` coverage is the only mask needed.

    Band-total source (the "no MT=1 band record" approximation, work-order
    requirement)
    ------------------------------------------------------------------------
    The CALENDF ``.tpe`` format carries the absolute band total ``sigma_t,b``
    directly in column 1 of every band record (parsed into ``_BandGroup.total``
    and validated by :func:`read_tpe`'s sum rule, ``total ~= Sum(partials)`` to
    ``<= 1.1e-5`` rel). ``R_tot`` therefore folds that authoritative total column
    directly -- the "reuse, don't fork" reading of "the SAME total reconstruction
    used for the flux weight" (``mat_ssf_factors_gendf`` also weights with
    ``g.total``). The total column is present in **every** real ``.tpe`` record,
    so the work-order fallback -- approximate ``sigma_t,b ~= MT2 + MT101 + MT18``
    band partials plus a smooth remainder when no total (MT=1) band record is
    present -- is only exercised defensively, if a group's total column were ever
    degenerate (all ``<= 0``, which does not occur in the TENDL-2017 tables). In
    that last-resort case this fold reconstructs the band total from the
    extracted partials ``sigma_t,b ~= elastic (MT2) + capture (MT101) + fission
    (MT18)``; that reconstruction OMITS the inelastic (MT=4) and any other smooth
    channel the authoritative total column includes, so it understates the total
    and is retained only as a guard, never as the working path.

    Parameters
    ----------
    tables : TpeTable
        Parsed CALENDF probability tables (bands keyed by library group index).
    sigma0_g : numpy.ndarray or float
        Background cross section per library group (length ``n_groups``), or a
        scalar broadcast to all groups (e.g. ``0.0`` for a pure absorber).
    n_groups : int
        Number of groups in the target library (length of the returned array).

    Returns
    -------
    numpy.ndarray
        ``R_tot_g`` of length ``n_groups``, default 1.0 (groups without bands).
    """
    sigma0_g = np.asarray(sigma0_g, dtype=float)
    if sigma0_g.ndim == 0:
        sigma0_g = np.full(n_groups, float(sigma0_g))
    elif len(sigma0_g) != n_groups:
        raise ValueError(
            f"sigma0_g length {len(sigma0_g)} != n_groups {n_groups}")

    R = np.ones(n_groups, dtype=float)
    for i, g in tables.groups.items():
        if not (0 <= i < n_groups):
            continue
        sig_t = g.total
        if not np.any(sig_t > 0.0):
            # Defensive no-total fallback (does not occur in real .tpe data):
            # reconstruct the band total from the extracted partials. Omits the
            # inelastic channel, so it understates -- guard only, see docstring.
            sig_t = g.elastic + g.capture + g.fission
        p = g.prob
        sig_inf = float((p * sig_t).sum() / p.sum())
        if sig_inf <= 0.0:
            continue
        w = p / (sigma0_g[i] + sig_t)
        sig_eff = float((w * sig_t).sum() / w.sum())
        R[i] = sig_eff / sig_inf
    return R


def iterate_material_dilution_sigma0_gendf(densities, group_totals, coupling,
                                           n_groups):
    r"""Mutual-shielding refinement of the material-dilution background (C3).

    Starts from the first Bondarenko approximation ``d0`` (each diluter at its
    infinite-dilute total, :func:`material_dilution_sigma0`) and refines it so
    every probability-table-carrying nuclide contributes its own *shielded*
    total, following FISPACT-II (Sublet et al., NDS 139 (2017), ch2 s26)::

        d(i+1)(p,g) = (1/f_p) Sum_{q != p} f_q sigma_t,inf(q,g) R_q(i)(g),
        R_q(i)(g)   = mat_ssf_total_factors_gendf(tables_q, d(i)(q,g)).

    All backgrounds are updated simultaneously from the same iterate (Jacobi, so
    the result is order-independent). Diluters outside ``coupling`` (no ``.tpe``
    probability table, or no library total) keep ``R == 1`` -- their
    infinite-dilute contribution is unchanged, so a resonant-plus-inert mixture
    reduces to the first approximation for the inert part. The map is contractive
    (``R in (0, 1]``), so the sequence decreases monotonically to its fixed
    point; there is no damping. This is the exact GENDF sibling of
    :func:`openmc.deplete.mat_ssf.iterate_material_dilution_sigma0`; it takes
    ``n_groups`` in place of the PENDF ``group_edges`` because the CALENDF factors
    are already indexed by library group.

    Parameters
    ----------
    densities : dict
        Maps nuclide name to number density (or fraction); the composition
        snapshot. Only ratios matter.
    group_totals : dict
        Maps each diluter nuclide (nonzero density) to its infinite-dilution
        group total ``sigma_t,inf(q,g)`` (barns), e.g. ``get_xs(q, MT=1)``. Every
        ``coupling`` key must also appear here.
    coupling : dict
        Maps each probability-table-carrying nuclide ``q`` (the iterated set) to
        its parsed :class:`TpeTable`. Nuclides absent from ``coupling`` keep
        ``R == 1``.
    n_groups : int
        Number of library groups (length of every background/total array).

    Returns
    -------
    sigma0 : dict
        Converged background ``sigma0(p,g)`` (barns) for every ``p`` in
        ``coupling``.
    info : dict
        ``{'n_iter', 'converged', 'trajectory', 'max_rel'}``: the number of
        update passes taken, whether the tolerance was met, the list of
        per-iteration background dicts starting at ``d0``, and the list of
        per-iteration max relative changes.
    """
    coupled = list(coupling)

    # d0: first Bondarenko approximation for every coupled nuclide.
    d = {p: material_dilution_sigma0(densities, p, group_totals,
                                     n_groups=n_groups)
         for p in coupled}

    info = {'n_iter': 0, 'converged': len(coupled) == 0,
            'trajectory': [dict(d)], 'max_rel': []}

    for it in range(SIGMA0_ITER_MAX):
        # R_q at the current background, for every coupled nuclide (Jacobi: all
        # evaluated from the same iterate d before any update).
        R = {q: mat_ssf_total_factors_gendf(coupling[q], d[q], n_groups)
             for q in coupled}

        d_new = {}
        max_rel = 0.0
        for p in coupled:
            f_p = densities[p]
            acc = np.zeros(n_groups)
            for j, n_j in densities.items():
                if j == p or n_j <= 0 or j not in group_totals:
                    continue
                r_j = R[j] if j in R else 1.0
                acc = acc + n_j * np.asarray(group_totals[j], dtype=float) * r_j
            d_new[p] = acc / f_p
            denom = np.maximum(d_new[p], _SIGMA0_EPS)
            max_rel = max(max_rel,
                          float(np.max(np.abs(d_new[p] - d[p]) / denom)))

        d = d_new
        info['n_iter'] = it + 1
        info['trajectory'].append(dict(d))
        info['max_rel'].append(max_rel)
        if max_rel < SIGMA0_ITER_TOL:
            info['converged'] = True
            break

    if not info['converged']:
        warnings.warn(
            f"CALENDF sigma0 mutual-shielding iteration did not converge in "
            f"{info['n_iter']} passes (max relative change "
            f"{info['max_rel'][-1]:.3e} > tol {SIGMA0_ITER_TOL:g}); using the "
            f"last iterate")

    return d, info


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


class _CalendfRowScaler:
    """Per-row applicator of the CALENDF material-dilution self-shielding.

    Holds what one composition needs -- the flagged set, the ``.tpe`` tables,
    the diluter group totals, the iterated background ``sigma0`` and the
    per-``(nuclide, reaction)`` factors -- and applies it to one group cross
    section row at a time through :meth:`scale`. The streaming GENDF collapse
    calls it on each row as it stages it; :func:`_apply_mat_ssf_gendf` calls it
    on every row of a built table. See :func:`_apply_mat_ssf_gendf` for the
    physics and the row-selection rules.

    The background is built lazily, on the first row that needs a factor: a
    composition with no foldable row reads no diluter total and no diluter
    ``.tpe``, and raises no "absent from the GENDF library" error.
    :meth:`check_diluters` applies the same rule ahead of the collapse, so a
    caller can raise that error before an expensive transport solve. The object
    does not depend on the flux, so one scaler serves every collapse of the
    same composition.

    Parameters
    ----------
    gendf_library : GENDF library instance
        Source of ``n_groups``, ``energy_structure`` (the group structure the
        ``.tpe`` records map onto) and the diluter group totals
        (``get_xs(nuclide, 1)``).
    calendf_path : path-like
        Directory of ``<Nuclide>-<T>.tpe`` CALENDF tables (one temperature).
    densities : mapping of str to float
        Atom densities keyed by nuclide (only ratios matter). Non-positive or
        ``None`` entries are dropped.
    mat_ssf_nuclides : iterable of str, optional
        Restrict the correction to this subset of :data:`DEFAULT_FLAGGED`
        (restrict-only). ``None`` uses the full set.
    tpe_cache : dict, optional
        ``{path: TpeTable}`` shared between scalers so each ``.tpe`` file is
        parsed once, however many compositions use it. ``None`` gives the
        scaler a private cache.
    totals_cache : dict, optional
        ``{nuclide: sigma_t,g}`` diluter group totals (MT=1, barns) shared
        between scalers like ``tpe_cache``, so each diluter total is read from
        the library once, however many compositions contain it. The scaler
        keeps references to these arrays and never modifies them. ``None``
        gives the scaler a private cache.

    Raises
    ------
    TypeError
        If ``mat_ssf_nuclides`` is a single string or holds a non-string.
    ValueError
        If ``gendf_library`` has no ``energy_structure``.
    """

    def __init__(self, gendf_library, calendf_path, densities,
                 mat_ssf_nuclides=None, tpe_cache=None, totals_cache=None):
        flagged = set(DEFAULT_FLAGGED)
        if mat_ssf_nuclides is not None:
            # A bare name would restrict to its letters, i.e. to nothing.
            if isinstance(mat_ssf_nuclides, str):
                raise TypeError(
                    "mat_ssf_nuclides must be an iterable of nuclide names, "
                    "not a single string")
            mat_ssf_nuclides = list(mat_ssf_nuclides)
            check_iterable_type('mat_ssf_nuclides', mat_ssf_nuclides, str)
            flagged &= set(mat_ssf_nuclides)
        self.flagged = flagged

        group_structure = getattr(gendf_library, 'energy_structure', None)
        if group_structure is None:
            raise ValueError(
                "urr_material_dilution: the GENDF library has no "
                "energy_structure; cannot map the CALENDF .tpe groups")
        self.group_structure = group_structure
        self.gendf_library = gendf_library
        self.n_groups = gendf_library.n_groups
        self.calendf_path = Path(calendf_path)

        # Positive-density nuclides act as diluters (only ratios matter).
        self.pos_densities = {k: float(v) for k, v in densities.items()
                              if v is not None and float(v) > 0.0}

        self._tpe_cache = {} if tpe_cache is None else tpe_cache
        self._totals_cache = {} if totals_cache is None else totals_cache
        self._tables = {}         # nuc -> TpeTable or None (no .tpe file)
        self._f_cache = {}        # (nuc, reaction) -> f_g, or None (f = 1)
        self._warned = set()
        self._group_totals = None
        self._sigma0_iter = None

    def _table(self, nuc):
        """Parsed ``.tpe`` table for a nuclide (cached), or None if absent."""
        if nuc not in self._tables:
            path = find_tpe(self.calendf_path, nuc)
            if path is None:
                tpe = None
            else:
                tpe = self._tpe_cache.get(path)
                if tpe is None:
                    tpe = read_tpe(path, group_structure=self.group_structure)
                    self._tpe_cache[path] = tpe
            self._tables[nuc] = tpe
        return self._tables[nuc]

    def _raise_absent_diluters(self):
        """Raise if a positive-density diluter has no GENDF data."""
        available = self.gendf_library.available_nuclides_set()
        missing = sorted(j for j in self.pos_densities if j not in available)
        if missing:
            names = ', '.join(repr(j) for j in missing)
            raise ValueError(
                f"diluter {names} has a density but is absent from the "
                f"GENDF library; cannot build sigma0_mat")

    def check_diluters(self, nuclides, mts):
        """Raise the absent-diluter error now if the collapse would raise it.

        The collapse builds the ``sigma0_mat`` background -- and so needs the
        group total of every positive-density diluter -- only when it folds a
        row: a capture (MT=102) or fission (MT=18) row of a flagged nuclide
        with a positive density and a ``.tpe`` table. This check applies the
        same rule before the collapse, so a caller can fail before an
        expensive transport solve. The ``.tpe`` tables it reads stay cached
        for the collapse.

        Parameters
        ----------
        nuclides : iterable of str
            Nuclides whose rows the collapse will stage.
        mts : iterable of int
            MT numbers of the reactions the collapse will stage.

        Raises
        ------
        ValueError
            If a fold will happen and one or more positive-density diluters
            are absent from the GENDF library (all are named).
        """
        if not any(mt in _LIB_MT_REACTION for mt in mts):
            return
        folds = any(self.pos_densities.get(nuc, 0.0) > 0.0
                    and self._table(nuc) is not None
                    for nuc in sorted(self.flagged.intersection(nuclides)))
        if folds:
            self._raise_absent_diluters()

    def _build_background(self):
        """Diluter group totals and the mutual-shielding sigma0 iteration."""
        self._raise_absent_diluters()
        group_totals = {}
        for j in self.pos_densities:
            sigma_t = self._totals_cache.get(j)
            if sigma_t is None:
                sigma_t = np.asarray(
                    self.gendf_library.get_xs(j, _LIB_TOTAL_MT), dtype=float)
                self._totals_cache[j] = sigma_t
            group_totals[j] = sigma_t

        # Coupling set: every positive-density nuclide with a .tpe table. Its
        # R_tot shields its own contribution to the background; the others
        # keep R = 1 (infinite-dilute).
        coupling = {}
        for q in self.pos_densities:
            tpe = self._table(q)
            if tpe is not None:
                coupling[q] = tpe

        self._group_totals = group_totals
        self._sigma0_iter, _info = iterate_material_dilution_sigma0_gendf(
            self.pos_densities, group_totals, coupling, self.n_groups)

    def scale(self, nuc, rxn, row):
        """Multiply one group cross section row by its factor, in place.

        ``nuc`` and ``rxn`` are the row's nuclide and reaction names; ``row``
        is its ``(n_groups,)`` array, scaled with ``*=``. A row outside the
        shielded set -- non-flagged nuclide, reaction other than capture
        (MT=102) or fission (MT=18), non-positive density, or a flagged
        nuclide without a ``.tpe`` table (warned once) -- is left untouched.
        """
        if nuc not in self.flagged:
            return
        mt = REACTION_MT.get(rxn)
        if mt is None:
            mt = REACTION_MT.get(_ISOMER_SUFFIX.sub('', rxn, count=1))
        reaction = _LIB_MT_REACTION.get(mt)
        if reaction is None:
            return
        if self.pos_densities.get(nuc, 0.0) <= 0.0:
            return

        tpe = self._table(nuc)
        if tpe is None:
            if nuc not in self._warned:
                warnings.warn(
                    f"urr_material_dilution: no CALENDF .tpe table for {nuc!r} "
                    f"under {self.calendf_path}; leaving it infinite-dilution "
                    f"(f=1)", UserWarning)
                self._warned.add(nuc)
            return

        # First foldable row: build the background once for the composition.
        if self._sigma0_iter is None:
            self._build_background()

        key = (nuc, reaction)
        if key not in self._f_cache:
            if reaction == 'fission' and not tpe.has_fission():
                # No MT=18 partial in the table: f = 1 (matches compute_mat_ssf).
                self._f_cache[key] = None
            else:
                sigma0_g = self._sigma0_iter.get(nuc)
                if sigma0_g is None:
                    # Outside the coupling: first-approximation background.
                    sigma0_g = material_dilution_sigma0(
                        self.pos_densities, nuc, self._group_totals,
                        n_groups=self.n_groups)
                self._f_cache[key] = mat_ssf_factors_gendf(
                    tpe, sigma0_g, self.n_groups, reaction)
        f_g = self._f_cache[key]
        if f_g is not None:
            row *= f_g


def _apply_mat_ssf_gendf(table, gendf_library, calendf_path, densities,
                         mat_ssf_nuclides=None):
    """Self-shield a sparse GENDF cross-section table in place (material dilution).

    The collapse entry points
    (:meth:`openmc.deplete.MicroXS.from_multigroup_flux_with_gendf` and
    :func:`openmc.deplete.gendf.collapse.get_gendfxs_and_flux`) apply the same
    correction row by row through :class:`_CalendfRowScaler` during the
    streaming collapse; this function is the table-level form, kept for the
    gate scripts and tests. For every flagged, self-shielding nuclide that is
    present in ``table``, has a positive density in ``densities`` and a
    ``.tpe`` file under ``calendf_path``, it folds the nuclide's CALENDF
    probability tables against the material background cross section
    ``sigma0_mat`` and multiplies that nuclide's capture (MT=102) and, when
    fissile, fission (MT=18) group rows in ``table.xs_matrix`` by the per-group
    self-shielding factor ``f_g``. This is the exact analogue of FISPACT-II
    ``PROBTABLE multxs=1``; the CALENDF tables span the resolved range as well
    as the URR, so ``f_g`` deviates from 1 across both (in some resolved-range
    groups capture anti-correlates with the total, giving ``f_g`` > 1 --
    legitimate ``multxs=1`` behaviour, not a bug).

    ``sigma0_mat`` starts from the composition snapshot's first Bondarenko
    approximation (:func:`material_dilution_sigma0`) and is then refined to
    mutual-shielding self-consistency (C3,
    :func:`iterate_material_dilution_sigma0_gendf`): every ``.tpe``-carrying
    diluter contributes its own *shielded* total, not its infinite-dilute one, so
    self-diluted resonant mixtures (nat-W, U metal/oxide) are shielded more
    deeply. Iteration is unconditional when the flag is on -- it is the same
    physics done correctly, not a new feature (the ``urr_material_dilution``
    signature is frozen). Diluters whose bands END inside the URR (the TENDL-2017
    W ceilings at ~13-17 keV) keep ``R = 1`` in the uncovered groups (graceful
    degradation); diluters with no ``.tpe`` table keep ``R = 1`` everywhere, so a
    resonant-plus-inert mixture reduces to the first approximation for the inert
    part.

    The per-row work is done by :class:`_CalendfRowScaler`, the same object the
    streaming collapse applies to each staged row. The table is modified
    **in place**; nothing is returned.

    Parameters
    ----------
    table : openmc.deplete.microxs._SparseXSTable
        Sparse infinite-dilution table (``xs_matrix`` rows keyed by
        ``nuc_indices`` -> ``table.nuclides`` and ``rxn_indices`` ->
        ``table.reactions``; capture / fission rows are identified by the
        reaction name's MT, 102 / 18).
    gendf_library : GENDF library instance
        Source of diluter group totals (``get_xs(diluter, 1)``), of the group
        count and of the energy-structure name used to map the ``.tpe`` groups.
    calendf_path : path-like
        Directory of ``<Nuclide>-<T>.tpe`` CALENDF tables (per temperature).
    densities : dict of str to float
        Atom densities keyed by nuclide (only ratios matter). A resonant nuclide
        with zero / non-positive / absent density is silently skipped (``f=1``).
    mat_ssf_nuclides : sequence of str, optional
        Restrict the correction to this subset. ``None`` (default) uses the full
        :data:`DEFAULT_FLAGGED` set. When given it is **intersected** with
        :data:`DEFAULT_FLAGGED` (restrict-only, matching the PENDF sibling's
        ``mat_ssf_nuclides`` semantics), so it can only narrow -- never widen --
        the flagged set.

    Raises
    ------
    TypeError
        If ``mat_ssf_nuclides`` is a single string or holds a non-string.
    ValueError
        If ``gendf_library`` has no ``energy_structure``, or if a nuclide named
        as a diluter in ``densities`` (positive density) is absent from the
        GENDF library once a fold happens.

    Warns
    -----
    UserWarning
        Once per flagged nuclide that has a positive density and a capture or
        fission row but no ``.tpe`` table under ``calendf_path`` (that nuclide
        is left at ``f=1``).
    """
    scaler = _CalendfRowScaler(gendf_library, calendf_path, densities,
                               mat_ssf_nuclides)
    for k in range(table.xs_matrix.shape[0]):
        scaler.scale(table.nuclides[table.nuc_indices[k]],
                     table.reactions[table.rxn_indices[k]],
                     table.xs_matrix[k])
