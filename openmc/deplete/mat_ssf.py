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

import warnings

import numpy as np

# Nuclides that carry URR probability tables in the JEFF-3.3 PENDF library and
# for which the material-dilution self-shielding correction is applied by
# default. All 24 carry MF=2 MT=152/153; the collapse hook intersects this with
# any caller-supplied ``mat_ssf_nuclides`` and with the library's actual ptable
# coverage (a flagged nuclide lacking a ``/urr`` group warns once and is left at
# infinite dilution).
DEFAULT_FLAGGED = frozenset({
    'W180', 'W182', 'W183', 'W184', 'W186', 'Ta181', 'Re185', 'Re187',
    'Hf174', 'Hf176', 'Hf177', 'Hf178', 'Hf179', 'Hf180',
    'Os186', 'Os187', 'Os188', 'Os189', 'Os190', 'Os192',
    'U235', 'U238', 'Pu239', 'Pu240',
})

# Base reaction name -> MF=3 MT of its smooth cross section, used to evaluate the
# resonant nuclide's smooth reaction XS at the URR nodes for factor-form tables.
_BASE_REACTION_MT = {'(n,gamma)': 102, 'fission': 18}

# Fixed-point controls for the mutual-shielding sigma_0 refinement (C2). These
# are policy constants, NOT API parameters: iteration is unconditional when the
# URR flag is on and always uses these bounds. FISPACT reports "a few passes";
# the map d(p) -> (1/f_p) sum_{q!=p} f_q sigma_t,inf(q) R_q(d) is contractive
# (R in (0, 1] -> a monotone-decreasing sequence from d0), so the tolerance is
# reached in a handful of passes: URR-only PENDF tables converge in ~4, and the
# GENDF sibling's resolved-range CALENDF tables need ~10 (nat-W), which is what
# sizes the cap. The cap is a safety bound, inert once the tolerance breaks the
# loop. No damping (add 0.5 damping only if a gate ever shows oscillation -- it
# is not expected).
SIGMA0_ITER_MAX = 15
SIGMA0_ITER_TOL = 1e-3

# Barn-scale floor for the relative-change denominator max(d, eps): guards groups
# where the background is ~0 (a pure absorber, or groups outside every diluter's
# tabulated span) from a spurious large relative delta. Background changes below
# this floor are physically irrelevant.
_SIGMA0_EPS = 1e-10


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
        to the capture column, ``'fission'`` to the fission column, and
        ``'total'`` to the total column (band 1) -- the flux-weight total whose
        self-shielding factor drives the mutual-shielding sigma_0 iteration
        (fold via :func:`mat_ssf_total_factors`).
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

    col_map = {'(n,gamma)': 4, 'fission': 3, 'total': 1}
    base = _ISOMER_SUFFIX.sub('', reaction)
    try:
        col = col_map[base]
    except KeyError:
        raise ValueError(
            f"mat_ssf is only defined for capture/fission/total; got "
            f"{reaction!r} (base {base!r})")

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


def mat_ssf_total_factors(ptab, sigma0_g, group_edges, smooth_total=None):
    r"""Per-group total-cross-section self-shielding factor :math:`R_{tot,g}`.

    The C1 total-XS shielding factor consumed by the C2 mutual-shielding sigma_0
    iteration. It folds the *total* column (band 1) exactly as the partial
    factors fold capture/fission, so the numerator's shielded total and the
    denominator's infinite-dilution total are the same :math:`\sigma_{t,b}` that
    already sets the partial fold's flux weight :math:`w_b = p_b/(\sigma_0 +
    \sigma_{t,b})` (band 1 for absolute tables, :math:`\tau_{t,b}\,S_t` for
    factor tables):

    .. math::

        R_{tot,g} = \frac{\sigma_{t,\mathrm{eff},g}(\sigma_0)}
            {\sigma_{t,\infty,g}},\qquad
        \sigma_{t,\mathrm{eff}} = \frac{\sum_b w_b \sigma_{t,b}}{\sum_b w_b},
        \quad \sigma_{t,\infty} = \frac{\sum_b p_b \sigma_{t,b}}{\sum_b p_b}.

    :math:`R_{tot,g} \in (0, 1]`: exactly 1 in groups outside the URR span and
    approaching 1 as :math:`\sigma_0 \to \infty` (infinite dilution, no
    shielding). It is folded at each URR node at that node's group background and
    group-averaged onto ``group_edges``, identically to the partial factors.

    Parameters
    ----------
    ptab : openmc.data.urr.ProbabilityTables
        Probability tables (band dimension 1 = total).
    sigma0_g : numpy.ndarray or float
        Background cross section per group (barns), length ``n_groups``.
    group_edges : numpy.ndarray
        Ascending energy group boundaries in eV, length ``n_groups + 1``.
    smooth_total : numpy.ndarray, optional
        Absolute smooth total cross section (barns) at each URR node
        ``ptab.energy``. Required when ``ptab.multiply_smooth`` is True (the
        factor bands are absolutized against it); ignored otherwise.

    Returns
    -------
    numpy.ndarray
        Total self-shielding factor :math:`R_{tot,g}`, length ``n_groups``; 1 in
        groups outside the URR span.
    """
    # The total is folded as its own "reaction" (band 1). For a factor table the
    # reaction smooth IS the total smooth, so the absolutized band is
    # sigma_t,b = tau_t,b * S_t -- exactly the flux-weight total. Passing
    # smooth_rxn = smooth_total makes sig_x == sig_t inside _fold_node.
    return mat_ssf_factors(
        ptab, sigma0_g, group_edges, 'total',
        smooth_total=smooth_total, smooth_rxn=smooth_total)


def iterate_material_dilution_sigma0(densities, group_totals, coupling,
                                     group_edges):
    r"""Mutual-shielding refinement of the material-dilution background (C2).

    Starts from the first Bondarenko approximation :math:`d^{(0)}` (each diluter
    at its infinite-dilute total, :func:`material_dilution_sigma0`) and refines
    it so every probability-table-carrying nuclide contributes its own
    *shielded* total, following FISPACT-II (Sublet et al., NDS 139 (2017),
    ch2 s26):

    .. math::

        d^{(i+1)}(p,g) = \frac{1}{f_p} \sum_{q \neq p} f_q\,
            \sigma_{t,\infty}(q,g)\, R_q^{(i)}(g),\qquad
        R_q^{(i)}(g) = R_{tot}\bigl(q, g, d^{(i)}(q,g)\bigr).

    All backgrounds are updated simultaneously from the same iterate (Jacobi, so
    the result is order-independent). Diluters outside ``coupling`` (no
    probability tables, or a factor table with no smooth total to absolutize)
    keep :math:`R \equiv 1` -- their infinite-dilute contribution is unchanged,
    so a resonant-plus-inert mixture reduces to the first approximation for the
    inert part. The map is contractive (:math:`R \in (0,1]`), so the sequence
    decreases monotonically to its fixed point; there is no damping.

    Parameters
    ----------
    densities : dict
        Maps nuclide name to number density (or fraction); the composition
        snapshot.
    group_totals : dict
        Maps each diluter nuclide (nonzero density) to its group-averaged
        infinite-dilution total :math:`\sigma_{t,\infty}(q,g)` (barns).
    coupling : dict
        Maps each probability-table-carrying nuclide ``q`` (the iterated set) to
        a ``(ptab, smooth_total)`` pair: ``ptab`` its
        :class:`~openmc.data.urr.ProbabilityTables`, ``smooth_total`` its smooth
        total at ``ptab.energy`` (or ``None`` for absolute tables). Every key
        must also appear in ``group_totals``.
    group_edges : numpy.ndarray
        Ascending energy group boundaries in eV, length ``n_groups + 1``.

    Returns
    -------
    sigma0 : dict
        Converged background :math:`\sigma_0(p,g)` (barns) for every ``p`` in
        ``coupling``.
    info : dict
        ``{'n_iter', 'converged', 'trajectory', 'max_rel'}``: the number of
        update passes taken, whether the tolerance was met, the list of
        per-iteration background dicts starting at :math:`d^{(0)}`, and the list
        of per-iteration max relative changes.
    """
    edges = np.asarray(group_edges, dtype=float)
    n_groups = len(edges) - 1
    coupled = list(coupling)

    # d^(0): first Bondarenko approximation for every coupled nuclide (and the
    # place a genuinely missing diluter is named, via material_dilution_sigma0).
    d = {p: material_dilution_sigma0(densities, p, group_totals)
         for p in coupled}

    info = {'n_iter': 0, 'converged': len(coupled) == 0,
            'trajectory': [dict(d)], 'max_rel': []}

    for it in range(SIGMA0_ITER_MAX):
        # R_q at the current background, for every coupled nuclide (Jacobi: all
        # evaluated from the same iterate d before any update).
        R = {}
        for q in coupled:
            ptab_q, smooth_total_q = coupling[q]
            R[q] = mat_ssf_total_factors(
                ptab_q, d[q], edges, smooth_total=smooth_total_q)

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
            f"URR sigma0 mutual-shielding iteration did not converge in "
            f"{info['n_iter']} passes (max relative change "
            f"{info['max_rel'][-1]:.3e} > tol {SIGMA0_ITER_TOL:g}); using the "
            f"last iterate")

    return d, info


def _library_group_total(lib, nuc, group_edges, grouped):
    r"""Group-averaged total cross section :math:`\sigma_{t,g}` of one nuclide.

    Reads the diluter's MF=3 MT=1 total from the PENDF library and, for a
    pointwise library, flat-weights it onto ``group_edges`` with
    :func:`~openmc.deplete.microxs._group_average` (a grouped library returns its
    pre-binned ``xs_g``). If MT=1 is absent the total is reconstructed as
    MT=2 (elastic) + MT=102 (capture) [+ MT=18 (fission)] from whichever of those
    are present. Returns ``None`` if the nuclide is absent from the library or
    carries none of those MTs, so the caller can defer the missing-diluter error
    to :func:`material_dilution_sigma0` (which names them, amendment A2).
    """
    from .microxs import _group_average

    if nuc not in lib.nuclides:
        return None
    mts = set(lib.reactions(nuc))
    if 1 in mts:
        order = [1]
    else:
        order = [mt for mt in (2, 102, 18) if mt in mts]
        if not order:
            return None
        if 2 not in mts:
            # Neither the total (MT=1) nor elastic (MT=2) is present, so the
            # reconstructed "total" is capture/fission only -- it omits the
            # dominant elastic channel, understates the diluter total and hence
            # sigma_0, and over-shields silently. (The JEFF-3.3 URR library
            # carries MT=1, so this fires only on a thinned library.)
            warnings.warn(
                f"{nuc}: neither MT=1 (total) nor MT=2 (elastic) is present in "
                f"the library; reconstructing the diluter total from {order} "
                f"only understates sigma_0 for the URR self-shielding background "
                f"and over-shields. Add MT=1 or MT=2 for {nuc} to the library.")
    total = None
    for mt in order:
        part = (np.asarray(lib.xs_g(nuc, mt), dtype=float) if grouped
                else _group_average(*lib.xs(nuc, mt), group_edges))
        total = part if total is None else total + part
    return total


def _smooth_at_nodes(lib, nuc, nodes, mts, grouped, group_edges):
    r"""Sum of a nuclide's smooth MF=3 cross sections at the URR nodes.

    Evaluates :math:`\sum_{mt} \sigma_{mt}(E_i)` over the ``mts`` present for
    ``nuc`` at each URR node energy ``nodes``. For a pointwise library the smooth
    XS is linearly interpolated onto the nodes; for a grouped library it is a
    per-node step-function lookup of the group the node falls in (the URR smooth
    XS is nearly flat, so this is a second-order approximation). Returns ``None``
    if none of ``mts`` are present.
    """
    present = [mt for mt in mts if mt in set(lib.reactions(nuc))]
    if not present:
        return None
    nodes = np.asarray(nodes, dtype=float)
    total = None
    for mt in present:
        if grouped:
            xs_g = np.asarray(lib.xs_g(nuc, mt), dtype=float)
            g = np.clip(np.searchsorted(group_edges, nodes, side='right') - 1,
                        0, len(xs_g) - 1)
            part = xs_g[g]
        else:
            energy, xs = lib.xs(nuc, mt)
            part = np.interp(nodes, energy, xs)
        total = part if total is None else total + part
    return total


def _apply_mat_ssf(table, pendf_library, energies, densities,
                   mat_ssf_nuclides=None):
    r"""Apply the URR material-dilution self-shielding factors to a collapse table.

    Mutates ``table.xs_matrix`` in place: each capture/fission row of a flagged
    resonant nuclide is multiplied by its per-group self-shielding factor
    :math:`f_g` (:func:`mat_ssf_factors`), evaluated at the homogeneous
    background :math:`\sigma_0`. The background starts from the composition
    snapshot's first Bondarenko approximation (:func:`material_dilution_sigma0`)
    and is then refined to mutual-shielding self-consistency
    (:func:`iterate_material_dilution_sigma0`, C2): every probability-table
    diluter contributes its own *shielded* total, not its infinite-dilute one,
    so self-diluted resonant mixtures (nat-W, U metal/oxide) are shielded more
    deeply. Iteration is unconditional when the flag is on. The correction is
    confined to URR-overlapping groups (:math:`f_g = 1` elsewhere). This is the
    C4 collapse hook consumed by :meth:`MicroXS.from_multigroup_flux` when
    ``urr_material_dilution=True``.

    **Limitations.** :math:`\sigma_0` is homogeneous (no escape/Dancoff
    geometry, infinite-medium limit) and is built from the single ``densities``
    snapshot supplied by the caller; diluter build-in over a long irradiation
    (per-step densities) is not modelled here (phase 2).

    Row handling:

    * Non-flagged nuclides and non-capture/fission reactions are left untouched.
    * A flagged nuclide whose own density is zero/absent is infinitely dilute
      (:math:`\sigma_0 \to \infty`, :math:`f \to 1`) and is skipped silently
      (amendment A1) -- transmutation products carry ptables but need no
      shielding at trace density.
    * A flagged nuclide with no ``/urr`` probability tables warns **once** and is
      left at infinite dilution.

    :math:`f_g` is cached per ``(nuclide, base reaction)`` and shared across the
    isomeric MF=10 partial rows (``'(n,gamma)'``, ``'(n,gamma)_m1'``, ...);
    :math:`\sigma_{0,g}` is cached per nuclide; each diluter's group totals are
    computed once per call (amendment A9).

    Parameters
    ----------
    table : openmc.deplete.microxs._SparseXSTable
        Sparse collapse table; ``xs_matrix`` rows are multiplied in place.
    pendf_library : openmc.data.PendfLibrary or openmc.data.GroupedPendfLibrary
        Library supplying probability tables (via ``ptables``) and diluter/
        resonant smooth cross sections.
    energies : numpy.ndarray
        Ascending collapse group edges (eV), length ``n_groups + 1``. For a
        grouped library these must equal the library's own ``group_edges``.
    densities : dict
        Maps nuclide name to number density (or fraction); the composition
        snapshot the :math:`\sigma_0` background is built from.
    mat_ssf_nuclides : iterable of str, optional
        Restricts the shielded set; the effective set is
        ``DEFAULT_FLAGGED`` intersected with this (and, per row, ptable
        coverage). ``None`` uses the full flagged list.

    Raises
    ------
    ValueError
        If a grouped library's ``group_edges`` differ from ``energies``, or if a
        diluter with nonzero density has no group totals in the library
        (propagated from :func:`material_dilution_sigma0`).
    """
    from .microxs import _ISOMER_SUFFIX

    edges = np.asarray(energies, dtype=float)
    grouped = hasattr(pendf_library, 'group_edges')
    if grouped:
        lib_edges = np.asarray(pendf_library.group_edges, dtype=float)
        if not np.array_equal(lib_edges, edges):
            raise ValueError(
                "grouped pendf_library group_edges differ from the collapse "
                "energies; URR self-shielding of a grouped library requires the "
                "collapse to run on the library's own group structure (omit "
                "`energies` so it defaults to group_edges)")

    flagged = set(DEFAULT_FLAGGED)
    if mat_ssf_nuclides is not None:
        flagged &= set(mat_ssf_nuclides)

    # Diluter group totals, once per call (A9). A nuclide the library cannot
    # supply is omitted here; material_dilution_sigma0 then raises a *named*
    # ValueError (A2) for any diluter a shielded nuclide actually needs.
    group_totals = {}
    for j, n_j in densities.items():
        if n_j <= 0:
            continue
        tot = _library_group_total(pendf_library, j, edges, grouped)
        if tot is not None:
            group_totals[j] = tot

    # Coupling set for the mutual-shielding sigma_0 iteration (C2): every
    # nonzero-density nuclide in the composition that carries probability tables
    # AND has a library total. Its smooth total (factor tables) is read once here
    # and reused for the reaction fold below. Nuclides outside this set keep
    # R = 1 -- their infinite-dilute total contributes unchanged (a W+Fe mix
    # reduces to the first approximation for Fe).
    smooth_total_cache = {}  # nuc -> smooth total at that nuc's URR nodes
    coupling = {}            # nuc -> (ptab, smooth_total or None)
    for q, n_q in densities.items():
        if n_q <= 0 or q not in group_totals:
            continue
        ptab_q = pendf_library.ptables(q)
        if ptab_q is None:
            continue
        if ptab_q.multiply_smooth:
            nodes = np.asarray(ptab_q.energy, dtype=float)
            st = _smooth_at_nodes(pendf_library, q, nodes, [1], grouped, edges)
            if st is None:
                st = _smooth_at_nodes(
                    pendf_library, q, nodes, [2, 102, 18], grouped, edges)
            if st is None:
                # Factor table with no smooth total to absolutize against: R
                # cannot be formed, so this nuclide stays at infinite dilution
                # (R = 1) in the coupling. A shielded row for it raises below.
                continue
            smooth_total_cache[q] = st
            coupling[q] = (ptab_q, st)
        else:
            coupling[q] = (ptab_q, None)

    # Refine the background to mutual-shielding self-consistency (unconditional
    # when the flag is on). Nuclides outside the coupling fall back to the
    # first-approximation d0 in the row loop.
    sigma0_iter, _ = iterate_material_dilution_sigma0(
        densities, group_totals, coupling, edges)

    sigma0_cache = {}       # nuc -> sigma0_g (iterated where available)
    f_cache = {}            # (nuc, base reaction) -> f_g
    warned = set()

    for r in range(table.xs_matrix.shape[0]):
        nuc = table.nuclides[table.nuc_indices[r]]
        if nuc not in flagged:
            continue
        rxn = table.reactions[table.rxn_indices[r]]
        base = _ISOMER_SUFFIX.sub('', rxn)
        if base not in ('(n,gamma)', 'fission'):
            continue
        # A1: a zero/absent-density resonant nuclide is infinitely dilute -> f=1.
        if densities.get(nuc, 0.0) == 0:
            continue

        ptab = pendf_library.ptables(nuc)
        if ptab is None:
            if nuc not in warned:
                warnings.warn(
                    f"{nuc} is flagged for URR material-dilution "
                    f"self-shielding but carries no probability tables (no "
                    f"/urr group); leaving its reaction rates at infinite "
                    f"dilution (f=1)")
                warned.add(nuc)
            continue

        key = (nuc, base)
        f_g = f_cache.get(key)
        if f_g is None:
            sigma0_g = sigma0_cache.get(nuc)
            if sigma0_g is None:
                sigma0_g = sigma0_iter.get(nuc)
                if sigma0_g is None:
                    # Not in the coupling (no library total / no smooth total):
                    # fall back to the first-approximation background with
                    # unshielded diluters.
                    sigma0_g = material_dilution_sigma0(
                        densities, nuc, group_totals)
                sigma0_cache[nuc] = sigma0_g
            if ptab.multiply_smooth:
                nodes = np.asarray(ptab.energy, dtype=float)
                smooth_total = smooth_total_cache.get(nuc)
                if smooth_total is None:
                    smooth_total = _smooth_at_nodes(
                        pendf_library, nuc, nodes, [1], grouped, edges)
                    if smooth_total is None:
                        smooth_total = _smooth_at_nodes(
                            pendf_library, nuc, nodes, [2, 102, 18],
                            grouped, edges)
                    smooth_total_cache[nuc] = smooth_total
                smooth_rxn = _smooth_at_nodes(
                    pendf_library, nuc, nodes, [_BASE_REACTION_MT[base]],
                    grouped, edges)
                f_g = mat_ssf_factors(
                    ptab, sigma0_g, edges, base,
                    smooth_total=smooth_total, smooth_rxn=smooth_rxn)
            else:
                f_g = mat_ssf_factors(ptab, sigma0_g, edges, base)
            f_cache[key] = f_g

        table.xs_matrix[r] *= f_g
