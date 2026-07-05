"""URR material-dilution self-shielding fold (``mat_ssf``).

This module computes an optional unresolved-resonance-region (URR)
self-shielding correction for the PENDF multigroup-flux collapse. "Material
dilution" means the background cross section :math:`\\sigma_0` is built from the
homogeneous material composition only; the escape/Dancoff geometry term is
omitted (:math:`\\sigma_e = 0`, infinite-medium limit).

The correction is a per-group self-shielding factor ``mat_ssf`` that multiplies
the infinite-dilution collapsed capture (and fission) reaction rates in
URR-overlapping groups. The physics is the narrow-resonance (NR) Bondarenko
fold of ENDF probability tables at a homogeneous background :math:`\\sigma_0`:

.. math::

    \\sigma_{x,\\mathrm{eff}}(E) = \\frac{\\sum_b w_b\\,\\sigma_{x,b}}
        {\\sum_b w_b},\\qquad
    w_b = \\frac{p_b}{\\sigma_0 + \\sigma_{t,b}}

with the infinite-dilution (flux-flat) limit
:math:`\\sigma_{x,\\infty}(E) = \\sum_b p_b\\,\\sigma_{x,b} / \\sum_b p_b`. The
self-shielding factor is the ratio :math:`f_x = \\sigma_{x,\\mathrm{eff}} /
\\sigma_{x,\\infty}`, which cancels the smooth-background magnitude and drops
cleanly onto a PENDF- or GENDF-collapsed base row (this is FISPACT
``PROBTABLE multxs=1``).

Two band conventions occur in a JEFF-3.3 PENDF tape and both are supported,
switched on the table's ``multiply_smooth`` attribute:

* ``multiply_smooth = False`` (LSSF=0) -- the bands are *absolute* cross
  sections. The total column (band dimension 1) is used directly as
  :math:`\\sigma_{t,b}`; no smooth multiply is applied. (e.g. W182/183/184/186,
  Ta181.)
* ``multiply_smooth = True`` (LSSF=1) -- the bands are *factors* relative to
  the smooth (infinite-dilution pointwise) cross section of each reaction, so
  :math:`\\sum_b p_b\\,\\tau_{x,b} = 1`. The absolute band cross sections are
  recovered as :math:`\\sigma_{t,b} = \\tau_{t,b}\\,S_t(E)` and
  :math:`\\sigma_{x,b} = \\tau_{x,b}\\,S_x(E)`, where :math:`S_t` / :math:`S_x`
  are the smooth total / reaction cross sections at the node (supplied by the
  caller from the library MF=3). The smooth reaction factor cancels in the
  self-shielding ratio, but the smooth *total* is required for the flux weight.
  (e.g. W180, all Re/Hf/Os, and the actinides U235/238, Pu239/240 -- the
  majority of the flagged list.)

**Provenance / build note.** The JEFF-3.3 tape mixes these conventions per
nuclide; the parser (C1) must set ``multiply_smooth`` from the URR LSSF flag
(not hard-code it), and the collapse hook (C4) must supply the smooth total (and
reaction) at the URR nodes for ``multiply_smooth`` nuclides via
``smooth_total`` / ``smooth_rxn``. If a ``multiply_smooth`` table is folded as
absolute the weight background is wrong by orders of magnitude and the
correction silently under-shields (validated: 19/24 flagged nuclides are
factor-form).

Pure Python; consumes :class:`openmc.data.urr.ProbabilityTables` read back from
the PENDF library. No C++ rebuild and no transport change.
"""

import numpy as np


def material_dilution_sigma0(densities, resonant, group_totals):
    r"""Homogeneous background cross section :math:`\sigma_0` per group.

    Builds the material-dilution background seen by the resonant nuclide from
    the composition snapshot,

    .. math::

        \sigma_{0,g} = \frac{1}{N_\mathrm{res}}
            \sum_{j \neq \mathrm{res}} N_j\,\sigma_{t,j,g},

    where the sum runs over the diluter nuclides (everything in ``densities``
    except the resonant nuclide). Only density *ratios* matter, so number
    densities or atom/weight fractions are equally acceptable. The escape term
    is omitted (infinite-medium limit): a mono-isotopic pure resonant material
    has no diluters and yields :math:`\sigma_{0,g} = 0`, the over-shielding
    limit of a pure absorber.

    Diluter build-in over long irradiations is not modelled here: the supplied
    ``densities`` are a single composition snapshot (phase 1). Per-step
    densities are a later phase.

    Parameters
    ----------
    densities : dict
        Maps nuclide name to number density (or fraction). Trace/transmutation
        nuclides that carry probability tables but are absent from the actual
        material composition should simply be omitted (or given zero density);
        the caller skips shielding a nuclide whose own density is zero.
    resonant : str
        Name of the resonant nuclide whose :math:`\sigma_0` is being computed.
    group_totals : dict
        Maps each diluter nuclide name to its group-averaged total cross
        section array :math:`\sigma_{t,j,g}` (barns, on the collapse groups).
        Every diluter in ``densities`` with a nonzero density must be present.

    Returns
    -------
    numpy.ndarray
        Background cross section :math:`\sigma_{0,g}` in barns, length
        ``n_groups``.

    Raises
    ------
    ValueError
        If the resonant nuclide has zero/absent density (the caller must skip
        such a row before calling — a zero-density nuclide is infinitely dilute,
        :math:`\sigma_0 \to \infty`, :math:`f \to 1`), or if any diluter with a
        nonzero density is missing from ``group_totals``.
    """
    n_res = densities.get(resonant, 0.0)
    if n_res <= 0:
        raise ValueError(
            f"resonant nuclide {resonant!r} has zero/absent density; the "
            "caller must skip zero-density rows before computing sigma_0 (a "
            "trace nuclide is infinitely dilute, so f -> 1)")

    # A missing diluter would silently understate sigma_0 and over-shield --
    # worse than failing -- so name every one and refuse (amendment A2).
    missing = [j for j, n_j in densities.items()
               if j != resonant and n_j > 0 and j not in group_totals]
    if missing:
        raise ValueError(
            f"diluter(s) {sorted(missing)} of {resonant!r} have no group "
            "totals in the library; add them to the library or remove them "
            "from densities")

    sigma0_g = None
    for j, n_j in densities.items():
        if j == resonant or n_j <= 0:
            continue
        contrib = n_j * np.asarray(group_totals[j], dtype=float) / n_res
        sigma0_g = contrib if sigma0_g is None else sigma0_g + contrib

    if sigma0_g is None:
        # No diluters: sigma_0 = 0 everywhere (pure absorber). Infer the group
        # count from any available total so the returned shape is correct.
        if not group_totals:
            raise ValueError(
                f"{resonant!r} has no diluters and group_totals is empty; the "
                "group count cannot be inferred -- pass at least one "
                "group-total array")
        n_groups = len(np.atleast_1d(next(iter(group_totals.values()))))
        sigma0_g = np.zeros(n_groups)

    return np.asarray(sigma0_g, dtype=float)


def _fold_node(ptab, i, sigma0, col, smooth_total=None, smooth_rxn=None):
    r"""Narrow-resonance Bondarenko fold at one URR energy node.

    Evaluates the (absolute) self-shielded and infinite-dilution band-averaged
    cross sections for reaction column ``col`` at the ``i``-th tabulated URR
    energy of ``ptab``, at homogeneous background ``sigma0``. This is the kernel
    validated against MT=152 by the Bondarenko oracle gate, kept separate and
    testable.

    For ``ptab.multiply_smooth`` tables the total/reaction bands are factors and
    are absolutized as :math:`\sigma_{t,b} = \tau_{t,b}\,S_t` and
    :math:`\sigma_{x,b} = \tau_{x,b}\,S_x` using the supplied smooth cross
    sections before folding.

    Parameters
    ----------
    ptab : openmc.data.urr.ProbabilityTables
        Probability tables. Band dimension 0 holds the *cumulative* probability,
        1 the total, 3 the fission and 4 the :math:`(n,\gamma)` value (absolute
        cross sections if ``multiply_smooth`` is False, else factors).
    i : int
        Index of the URR energy node.
    sigma0 : float
        Homogeneous background cross section in barns.
    col : int
        Band-dimension index of the shielded reaction (4 for capture, 3 for
        fission).
    smooth_total, smooth_rxn : float, optional
        Absolute smooth total and reaction cross sections (barns) at this node.
        Required (and only used) when ``ptab.multiply_smooth`` is True.

    Returns
    -------
    sx_eff : float
        Self-shielded band-averaged cross section
        :math:`\sum_b w_b\sigma_{x,b}/\sum_b w_b`, with
        :math:`w_b = p_b/(\sigma_0 + \sigma_{t,b})`.
    sx_inf : float
        Infinite-dilution (flux-flat) band-averaged cross section
        :math:`\sum_b p_b\sigma_{x,b}/\sum_b p_b`. The explicit
        :math:`/\sum_b p_b` normalization makes the ratio equal 1 exactly as
        :math:`\sigma_0 \to \infty`, since the ENDF probability column sums to 1
        only to float precision (amendment A5).

    Raises
    ------
    ValueError
        If ``ptab.multiply_smooth`` is True and the smooth cross sections are
        not supplied.
    """
    p_b = np.diff(ptab.table[i, 0, :], prepend=0.0)   # raw band probs (Sigma=1)
    sig_t = ptab.table[i, 1, :]                        # total (authoritative)
    sig_x = ptab.table[i, col, :]                      # capture=4, fission=3
    if ptab.multiply_smooth:
        # Bands are factors relative to the smooth XS; absolutize them. The
        # smooth reaction cancels in the sx_eff/sx_inf ratio, but the smooth
        # total sets the (barns) magnitude of the flux-weight denominator.
        if smooth_total is None or smooth_rxn is None:
            raise ValueError(
                "multiply_smooth probability table requires smooth_total and "
                "smooth_rxn (absolute smooth cross sections at this URR node) "
                "to absolutize the factor bands")
        sig_t = sig_t * smooth_total
        sig_x = sig_x * smooth_rxn
    w = p_b / (sigma0 + sig_t)
    sx_eff = np.sum(w * sig_x) / np.sum(w)
    sx_inf = np.sum(p_b * sig_x) / np.sum(p_b)
    return sx_eff, sx_inf


def mat_ssf_factors(ptab, sigma0_g, group_edges, reaction, flux_g=None,
                    smooth_total=None, smooth_rxn=None):
    r"""Per-group URR self-shielding factors for one reaction.

    Folds the probability tables at each URR node using the background
    :math:`\sigma_0` of the group that node falls in, then group-averages the
    resulting self-shielded and infinite-dilution pointwise cross sections onto
    ``group_edges`` and returns their ratio.

    The NR/homogeneous-:math:`\sigma_0` (material-dilution) approximation is
    used: escape/geometry is omitted. Absolute bands
    (``multiply_smooth = False``) are folded directly; factor bands
    (``multiply_smooth = True``) are absolutized per node with the supplied
    smooth cross sections (see ``smooth_total`` / ``smooth_rxn``).

    Because only ~19 URR nodes exist but the collapse structure may be much
    finer, most URR-span groups contain no node. Rather than setting
    :math:`f_g = 1` for such groups (which would under-shield them), the folded
    :math:`\sigma_{x,\mathrm{eff}}(E_i)` and :math:`\sigma_{x,\infty}(E_i)` are
    treated as pointwise cross sections tabulated at the nodes and each
    group-averaged with :func:`~openmc.deplete.microxs._group_average` (the same
    flat-in-energy kernel that built the base collapsed row). The ratio
    :math:`f_g = \sigma_{x,\mathrm{eff},g} / \sigma_{x,\infty,g}` is then defined
    for every group overlapping the URR span, and equals 1 only where
    :math:`\sigma_{x,\infty,g} = 0` (i.e. wholly outside the span, where
    ``_group_average`` returns 0 with no extrapolation).

    Parameters
    ----------
    ptab : openmc.data.urr.ProbabilityTables
        Probability tables with ascending ``energy`` (eV); absolute or factor
        bands per ``ptab.multiply_smooth``.
    sigma0_g : numpy.ndarray or float
        Background cross section per group (barns), length ``n_groups``. A
        scalar is broadcast to a uniform background.
    group_edges : numpy.ndarray
        Ascending energy group boundaries in eV, length ``n_groups + 1``.
    reaction : str
        Reaction name; ``'(n,gamma)'`` (and its ``'_m*'`` isomeric variants) map
        to the capture column, ``'fission'`` to the fission column.
    flux_g : numpy.ndarray, optional
        Accepted for API symmetry but unused: the collapse flux is constant
        within a group, so it cancels in the per-group ratio and the
        within-group weighting reduces to ``_group_average``'s flat-in-energy
        average (matching the base row's collapse).
    smooth_total, smooth_rxn : numpy.ndarray, optional
        Absolute smooth total and reaction cross sections (barns) evaluated at
        each URR node ``ptab.energy`` (length equal to the number of nodes).
        Required when ``ptab.multiply_smooth`` is True; ignored otherwise.

    Returns
    -------
    numpy.ndarray
        Self-shielding factors :math:`f_g`, length ``n_groups``. Values in
        groups outside the URR span are exactly 1.

    Raises
    ------
    ValueError
        If ``reaction`` is not a capture or fission channel, or if
        ``ptab.multiply_smooth`` is True and the smooth cross sections are not
        supplied.
    """
    # Imported lazily: microxs will import mat_ssf at the Wave-2 collapse hook,
    # and a top-level `from .microxs import ...` here would then form an import
    # cycle. _group_average / _ISOMER_SUFFIX are module-internal helpers.
    from .microxs import _group_average, _ISOMER_SUFFIX

    col_map = {'(n,gamma)': 4, 'fission': 3}
    base = _ISOMER_SUFFIX.sub('', reaction)
    try:
        col = col_map[base]
    except KeyError:
        raise ValueError(
            f"mat_ssf is only defined for capture/fission; got {reaction!r} "
            f"(base {base!r})")

    energy = np.asarray(ptab.energy, dtype=float)
    edges = np.asarray(group_edges, dtype=float)
    n_groups = len(edges) - 1

    sigma0_g = np.asarray(sigma0_g, dtype=float)
    if sigma0_g.ndim == 0:
        sigma0_g = np.full(n_groups, float(sigma0_g))

    n_nodes = len(energy)
    if ptab.multiply_smooth:
        if smooth_total is None or smooth_rxn is None:
            raise ValueError(
                "multiply_smooth probability table requires smooth_total and "
                "smooth_rxn arrays evaluated at ptab.energy")
        smooth_total = np.asarray(smooth_total, dtype=float)
        smooth_rxn = np.asarray(smooth_rxn, dtype=float)

    sx_eff = np.empty(n_nodes)
    sx_inf = np.empty(n_nodes)
    for i in range(n_nodes):
        # side='right' puts an on-edge node in the group above; the clip guards
        # nodes at/below group_edges[0] (which would wrap to -1) and above the
        # last edge (amendment A3).
        g = int(np.clip(
            np.searchsorted(edges, energy[i], side='right') - 1,
            0, n_groups - 1))
        st = smooth_total[i] if ptab.multiply_smooth else None
        sx = smooth_rxn[i] if ptab.multiply_smooth else None
        sx_eff[i], sx_inf[i] = _fold_node(ptab, i, sigma0_g[g], col, st, sx)

    eff_g = _group_average(energy, sx_eff, edges)
    inf_g = _group_average(energy, sx_inf, edges)

    # f_g = 1 only where inf_g == 0 (outside the URR span); np.divide with a
    # `where` mask avoids a divide-by-zero warning on those groups.
    f_g = np.ones(n_groups)
    np.divide(eff_g, inf_g, out=f_g, where=inf_g > 0)
    return f_g
