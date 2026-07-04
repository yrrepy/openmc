"""MicroXS module

A class for storing microscopic cross section data that can be used with the
IndependentOperator class for depletion.
"""

from __future__ import annotations
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Union, TypeAlias, Self
from warnings import warn

import h5py
import pandas as pd
import numpy as np

from openmc.checkvalue import check_type, check_value, check_iterable_type, PathLike
from openmc import StatePoint
from openmc.mgxs import GROUP_STRUCTURES
from openmc.data import REACTION_MT
import openmc
from .chain import Chain, REACTIONS, _get_chain
from .coupled_operator import _find_cross_sections, _get_nuclides_with_data
from ..utility_funcs import h5py_file_or_group
import openmc.lib
from openmc.mpi import comm

_valid_rxns = list(REACTIONS)
_valid_rxns.append('fission')
_valid_rxns.append('damage-energy')


# TODO: Replace with type statement when support is Python 3.12+
DomainTypes: TypeAlias = Union[
    Sequence[openmc.Material],
    Sequence[openmc.Cell],
    Sequence[openmc.Universe],
    openmc.MeshBase,
    openmc.Filter,
    Sequence[openmc.Filter]
]


def get_microxs_and_flux(
    model: openmc.Model,
    domains: DomainTypes,
    nuclides: Sequence[str] | None = None,
    reactions: Sequence[str] | None = None,
    energies: Sequence[float] | str | None = None,
    reaction_rate_mode: str = 'direct',
    chain_file: PathLike | Chain | None = None,
    path_statepoint: PathLike | None = None,
    path_input: PathLike | None = None,
    run_kwargs=None,
    reaction_rate_opts: dict | None = None,
) -> tuple[list[np.ndarray], list[MicroXS]]:
    """Generate microscopic cross sections and fluxes for multiple domains.

    This function runs a neutron transport solve to obtain the flux and reaction
    rates in the specified domains and computes multigroup microscopic cross
    sections that can be used in depletion calculations with the
    :class:`~openmc.deplete.IndependentOperator` class.

    .. versionadded:: 0.14.0

    .. versionchanged:: 0.15.3
        Added `reaction_rate_mode`, `path_statepoint`, `path_input` arguments.

    Parameters
    ----------
    model : openmc.Model
        OpenMC model object. Must contain geometry, materials, and settings.
    domains : list of openmc.Material or openmc.Cell or openmc.Universe, or openmc.MeshBase, or openmc.Filter, or list of openmc.Filter
        Domains in which to tally reaction rates, or a spatial tally filter.
        A list of filters can be provided to create one set of tallies per
        filter (e.g., one :class:`~openmc.MeshMaterialFilter` per mesh) that
        are all evaluated in a single transport solve. Results are
        concatenated across all filters in order.
    nuclides : list of str
        Nuclides to get cross sections for. If not specified, all burnable
        nuclides from the depletion chain file are used.
    reactions : list of str
        Reactions to get cross sections for. If not specified, all neutron
        reactions listed in the depletion chain file are used.
    energies : iterable of float or str
        Energy group boundaries in [eV] or the name of the group structure.
        If left as None, no energy filter is applied to the flux tally. When
        `reaction_rate_mode` is "direct", these boundaries define the output
        flux and microscopic cross section energy group structure. When
        `reaction_rate_mode` is "flux", these boundaries define the multigroup
        flux tally used to collapse continuous-energy cross sections; returned
        fluxes and microscopic cross sections are one-group.
    reaction_rate_mode : {"direct", "flux"}, optional
        The "direct" method tallies reaction rates directly (per energy
        group). The "flux" method tallies a multigroup flux spectrum and then
        collapses reaction rates after a transport solve. When
        `reaction_rate_opts` is provided with `reaction_rate_mode='flux'`, the
        specified nuclide/reaction pairs are tallied directly and those values
        override the flux-collapsed values.
    chain_file : PathLike or Chain, optional
        Path to the depletion chain XML file or an instance of
        openmc.deplete.Chain. Used to determine cross sections for materials not
        present in the inital composition. Defaults to
        ``openmc.config['chain_file']``.
    path_statepoint : path-like, optional
        Path to write the statepoint file from the neutron transport solve to.
        By default, The statepoint file is written to a temporary directory and
        is not kept.
    path_input : path-like, optional
        Path to write the model XML file from the neutron transport solve to.
        By default, the model XML file is written to a temporary directory and
        not kept.
    run_kwargs : dict, optional
        Keyword arguments passed to :meth:`openmc.Model.run`
    reaction_rate_opts : dict, optional
        When `reaction_rate_mode="flux"`, allows selecting a subset of
        nuclide/reaction pairs to be computed via direct reaction-rate tallies
        over one energy bin spanning the full `energies` range. Supported keys:
        "nuclides", "reactions". If "reactions" are specified without
        "nuclides", all selected nuclides are used.

    Returns
    -------
    list of numpy.ndarray
        Flux in each group in [n-cm/src] for each domain
    list of MicroXS
        Cross section data in [b] for each domain

    See Also
    --------
    openmc.deplete.IndependentOperator

    """
    check_value('reaction_rate_mode', reaction_rate_mode, {'direct', 'flux'})

    # Save any original tallies on the model
    original_tallies = list(model.tallies)

    # Determine what reactions and nuclides are available in chain
    chain = _get_chain(chain_file)
    if reactions is None:
        reactions = chain.reactions
    if not nuclides:
        cross_sections = _find_cross_sections(model)
        nuclides_with_data = _get_nuclides_with_data(cross_sections)
        nuclides = [nuc.name for nuc in chain.nuclides
                    if nuc.name in nuclides_with_data]

    # Set up the reaction rate and flux tallies. When energies are omitted, no
    # energy filter is needed for the transport calculation. A one-group energy
    # range is still needed later if flux collapse is requested.
    collapse_energies = energies
    if energies is None:
        energy_filter = None
        collapse_energies = [0.0, 100.0e6]
    elif isinstance(energies, str):
        energy_filter = openmc.EnergyFilter.from_group_structure(energies)
    else:
        energy_filter = openmc.EnergyFilter(energies)

    # Build list of domain filters
    if isinstance(domains, openmc.Filter):
        domain_filters = [domains]
    elif isinstance(domains, openmc.MeshBase):
        domain_filters = [openmc.MeshFilter(domains)]
    elif isinstance(domains, Sequence) and len(domains) > 0 and \
            isinstance(domains[0], openmc.Filter):
        domain_filters = list(domains)
    elif isinstance(domains[0], openmc.Material):
        domain_filters = [openmc.MaterialFilter(domains)]
    elif isinstance(domains[0], openmc.Cell):
        domain_filters = [openmc.CellFilter(domains)]
    elif isinstance(domains[0], openmc.Universe):
        domain_filters = [openmc.UniverseFilter(domains)]
    else:
        raise ValueError(f"Unsupported domain type: {type(domains[0])}")

    # Prepare reaction-rate nuclides/reactions
    rr_nuclides: list[str] = []
    rr_reactions: list[str] = []
    if reaction_rate_mode == 'direct':
        rr_nuclides = list(nuclides)
        rr_reactions = list(reactions)
    elif reaction_rate_mode == 'flux' and reaction_rate_opts:
        opts = reaction_rate_opts or {}
        rr_reactions = list(opts.get('reactions', []))
        if rr_reactions:
            rr_nuclides = list(opts.get('nuclides', nuclides))
        else:
            rr_nuclides = list(opts.get('nuclides', []))
        # Keep only requested pairs within overall sets
        if rr_nuclides:
            rr_nuclides = [n for n in rr_nuclides if n in set(nuclides)]
        if rr_reactions:
            rr_reactions = [r for r in rr_reactions if r in set(reactions)]

    # Use 1-group energy filter for RR in flux mode
    has_rr = bool(rr_nuclides and rr_reactions)
    if has_rr and reaction_rate_mode == 'flux' and energy_filter is not None:
        rr_energy_filter = openmc.EnergyFilter(
            [energy_filter.values[0], energy_filter.values[-1]])
    else:
        rr_energy_filter = energy_filter

    # Create one flux tally (and optionally one RR tally) per domain filter.
    flux_tallies = []
    rr_tallies = []
    model.tallies = []
    for i, domain_filter in enumerate(domain_filters):
        flux_tally = openmc.Tally(name=f'MicroXS flux {i}')
        flux_tally.filters = [domain_filter]
        if energy_filter is not None:
            flux_tally.filters.append(energy_filter)
        flux_tally.scores = ['flux']
        model.tallies.append(flux_tally)
        flux_tallies.append(flux_tally)

        if has_rr:
            rr_tally = openmc.Tally(name=f'MicroXS RR {i}')
            rr_tally.filters = [domain_filter]
            if rr_energy_filter is not None:
                rr_tally.filters.append(rr_energy_filter)
            rr_tally.nuclides = rr_nuclides
            rr_tally.multiply_density = False
            rr_tally.scores = rr_reactions
            model.tallies.append(rr_tally)
            rr_tallies.append(rr_tally)

    if openmc.lib.is_initialized:
        openmc.lib.finalize()

        if comm.rank == 0:
            model.export_to_model_xml()
        comm.barrier()
        # Reinitialize with tallies
        openmc.lib.init(intracomm=comm)

    with TemporaryDirectory() as temp_dir:
        # Indicate to run in temporary directory unless being executed through
        # openmc.lib, in which case we don't need to specify the cwd
        run_kwargs = dict(run_kwargs) if run_kwargs else {}
        if not openmc.lib.is_initialized:
            run_kwargs.setdefault('cwd', temp_dir)

        # Run transport simulation and synchronize
        statepoint_path = model.run(**run_kwargs)
        comm.barrier()

        if comm.rank == 0:
            # Move the statepoint file if it is being saved to a specific path
            if path_statepoint is not None:
                shutil.move(statepoint_path, path_statepoint)
                statepoint_path = path_statepoint

            # Export the model to path_input if provided
            if path_input is not None:
                model.export_to_model_xml(path_input)

        # Broadcast updated statepoint path to all ranks
        statepoint_path = comm.bcast(statepoint_path)

        # Read in tally results (on all ranks)
        with StatePoint(statepoint_path) as sp:
            for i in range(len(flux_tallies)):
                flux_tallies[i] = sp.tallies[flux_tallies[i].id]
                flux_tallies[i]._read_results()
                if rr_tallies:
                    rr_tallies[i] = sp.tallies[rr_tallies[i].id]
                    rr_tallies[i]._read_results()

    # Concatenate results across all domain filters
    fluxes = []
    all_flux_arrays = []
    for flux_tally in flux_tallies:
        # Get flux values and make energy groups last dimension
        flux = flux_tally.get_reshaped_data()
        if energy_filter is None:
            flux = flux[..., np.newaxis]  # (domains, 1, 1, groups)
        else:
            # (domains, groups, 1, 1) -> (domains, 1, 1, groups)
            flux = np.moveaxis(flux, 1, -1)
        all_flux_arrays.append(flux)
        fluxes.extend(flux.squeeze((1, 2)))

    # If we built reaction-rate tallies, compute microscopic cross sections
    if rr_tallies:
        direct_micros = []
        for flux_arr, rr_tally in zip(all_flux_arrays, rr_tallies):
            flux = flux_arr
            # Get reaction rates and make energy groups last dimension
            reaction_rates = rr_tally.get_reshaped_data()
            if rr_energy_filter is None:
                # (domains, nuclides, reactions) ->
                # (domains, nuclides, reactions, groups)
                reaction_rates = reaction_rates[..., np.newaxis]
            else:
                # (domains, groups, nuclides, reactions) ->
                # (domains, nuclides, reactions, groups)
                reaction_rates = np.moveaxis(reaction_rates, 1, -1)

            # If RR is 1-group, sum flux over groups
            if reaction_rate_mode == "flux":
                flux = flux.sum(axis=-1, keepdims=True)

            xs = np.zeros_like(reaction_rates)
            d, _, _, g = np.nonzero(flux)
            xs[d, ..., g] = reaction_rates[d, ..., g] / flux[d, :, :, g]
            direct_micros.extend(
                MicroXS(xs_i, rr_nuclides, rr_reactions) for xs_i in xs)

    if reaction_rate_mode == 'flux':
        # Resolve the library from the model (from_multigroup_flux defaults to config)
        cross_sections = _find_cross_sections(model)
        # Collapse all domains against one table, built once
        flux_micros = MicroXS.from_multigroup_flux(
            collapse_energies, fluxes, chain_file=chain, nuclides=nuclides,
            reactions=reactions, cross_sections=cross_sections)

        # We need to return one-group fluxes to match the microscopic cross
        # sections, which are always one-group by virtue of the collapse
        fluxes = [flux.sum(keepdims=True) for flux in fluxes]

    # Decide which micros to use and merge if needed
    if reaction_rate_mode == 'flux' and rr_tallies:
        micros = [m1.merge(m2) for m1, m2 in zip(flux_micros, direct_micros)]
    elif rr_tallies:
        micros = direct_micros
    else:
        micros = flux_micros

    # Reset tallies
    model.tallies = original_tallies

    return fluxes, micros


@dataclass
class _SparseXSTable:
    """Sparse group cross sections for vectorized flux collapse.

    Only non-zero ``(nuclide, reaction)`` pairs are stored: ``xs_matrix`` holds
    one ``(n_groups,)`` row per pair and ``nuc_indices``/``rxn_indices`` map each
    row into the dense ``(n_nuclides, n_reactions)`` result. Rows may come from
    any group cross section source (e.g. :func:`_build_xs_table_ce`).
    """
    nuclides: list[str]
    reactions: list[str]
    xs_matrix: np.ndarray
    nuc_indices: np.ndarray
    rxn_indices: np.ndarray

    def collapse(self, phi_norm: np.ndarray) -> np.ndarray:
        """Collapse the table against a group flux with one matrix-vector product.

        A normalized flux (summing to 1) yields one-group cross sections, a raw
        flux yields reaction rates. Returns a dense
        ``(n_nuclides, n_reactions)`` array.
        """
        n_groups = self.xs_matrix.shape[1]
        if len(phi_norm) != n_groups:
            raise ValueError(
                f'Flux has {len(phi_norm)} groups but the cross section table '
                f'expects {n_groups}')
        result = np.zeros((len(self.nuclides), len(self.reactions)))
        result[self.nuc_indices, self.rxn_indices] = self.xs_matrix @ phi_norm
        return result


def _build_xs_table_ce(
    nuclides: Sequence[str],
    reactions: Sequence[str],
    energies: Sequence[float],
    temperature: float,
    nuclides_with_data: set,
    cross_sections=None,
    **init_kwargs,
) -> _SparseXSTable:
    """Build a sparse group cross section table from continuous-energy data.

    Group-averaged cross sections for each requested ``(nuclide, reaction)`` are
    computed once via :meth:`openmc.lib.Nuclide.group_xs` inside a single
    :class:`openmc.lib.TemporarySession`; all-zero rows (reaction absent, or
    threshold above the group structure) are skipped.

    Parameters
    ----------
    nuclides : sequence of str
        Nuclide names defining the result's nuclide axis.
    reactions : sequence of str
        Reaction names defining the result's reaction axis.
    energies : sequence of float
        Ascending energy group boundaries in [eV], length ``n_groups + 1``.
    temperature : float
        Temperature in [K] for cross section evaluation.
    nuclides_with_data : set
        Nuclides available in the cross section library; others are skipped.
    cross_sections : PathLike, optional
        Cross section library for the session, matching the one
        ``nuclides_with_data`` was resolved from. Defaults to ``openmc.config``.
    **init_kwargs : dict
        Keyword arguments passed to :func:`openmc.lib.init`.
    """
    mts = [REACTION_MT[name] for name in reactions]
    energies = np.asarray(energies, dtype=float)
    n_groups = len(energies) - 1

    rows, nuc_idx_list, rxn_idx_list = [], [], []
    # Load against the same library nuclides_with_data was resolved from
    library = (openmc.config.patch('cross_sections', cross_sections)
               if cross_sections is not None else nullcontext())
    with library, openmc.lib.TemporarySession(**init_kwargs):
        for nuc_idx, nuc in enumerate(nuclides):
            if nuc not in nuclides_with_data:
                continue
            lib_nuc = openmc.lib.load_nuclide(nuc)
            # Index by reaction, not MT, so fission/(n,fission) stay separate
            for rxn_idx, mt in enumerate(mts):
                xs_g = lib_nuc.group_xs(mt, temperature, energies)
                if xs_g.any():
                    rows.append(xs_g)
                    nuc_idx_list.append(nuc_idx)
                    rxn_idx_list.append(rxn_idx)

    xs_matrix = np.vstack(rows) if rows else np.empty((0, n_groups))

    return _SparseXSTable(
        list(nuclides), list(reactions), xs_matrix,
        np.array(nuc_idx_list, np.int32), np.array(rxn_idx_list, np.int32))


def _group_average(
    energy: np.ndarray,
    xs: np.ndarray,
    group_edges: np.ndarray,
) -> np.ndarray:
    r"""Flat-in-bin group average of a pointwise cross section.

    Computes :math:`\sigma_g = \int \sigma(E)\,dE / \Delta E_g` for each group,
    treating the tabulated cross section as linear-linear between points. The
    integral is evaluated by the trapezoid rule on the union of the group edges
    and the reaction's own energy grid, so the result equals the exact analytic
    integral of the piecewise-linear cross section. Coincident-energy points
    (step discontinuities, e.g. TENDL's repeated 30 MeV node) split the data
    into strictly increasing segments integrated separately, so both sides of
    a jump contribute exactly. Energy regions outside the tabulated
    ``(energy[0], energy[-1])`` range contribute zero (no extrapolation); a
    group lying entirely outside that range averages to 0.

    This replicates the C++ ``for_each_panel`` flat-weighting kernel, which
    skips the zero-width panel at a coincident point (src/reaction.cpp).

    Parameters
    ----------
    energy : numpy.ndarray
        Ascending tabulated energies in [eV].
    xs : numpy.ndarray
        Cross section values in [b] at each ``energy`` point (linear-linear
        between points).
    group_edges : numpy.ndarray
        Ascending energy group boundaries in [eV], length ``n_groups + 1``.

    Returns
    -------
    numpy.ndarray
        Group-averaged cross sections in [b], length ``n_groups``.
    """
    energy = np.asarray(energy, dtype=float)
    xs = np.asarray(xs, dtype=float)
    edges = np.asarray(group_edges, dtype=float)

    # Reject silently-wrong inputs: a descending grid makes the interpolation
    # and reduceat masking return zeros, and a NaN would propagate through the
    # all-zero keep-guard into the whole table.
    if np.any(np.diff(energy) < 0):
        raise ValueError('Tabulated energy grid must be non-decreasing')
    if np.isnan(xs).any():
        raise ValueError('Cross section data contains NaN values')

    # Coincident energies mark step discontinuities. np.interp would take only
    # the right-hand value there, dropping the sliver left of the jump, so
    # integrate each strictly increasing segment separately instead.
    splits = np.flatnonzero(np.diff(energy) == 0.0) + 1
    group_area = np.zeros(len(edges) - 1)
    for seg_e, seg_xs in zip(np.split(energy, splits), np.split(xs, splits)):
        if len(seg_e) >= 2:
            group_area += _segment_group_area(seg_e, seg_xs, edges)
    return group_area / np.diff(edges)


def _segment_group_area(
    energy: np.ndarray,
    xs: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Per-group trapezoid integral of one strictly increasing segment."""
    e_lo, e_hi = energy[0], energy[-1]

    # Union grid: group edges plus the tabulated points inside the group span
    inside_span = (energy >= edges[0]) & (energy <= edges[-1])
    union = np.unique(np.concatenate((edges, energy[inside_span])))

    # Linear-linear interpolate onto the union grid. Values outside the
    # tabulated range are clamped by np.interp but land in intervals masked out
    # below, so their value is irrelevant.
    xs_u = np.interp(union, energy, xs)

    # Trapezoid area of each union interval, zeroed for intervals outside the
    # tabulated range. When e_lo/e_hi fall inside the group span they are union
    # nodes, so no interval straddles the tabulated boundary and masking whole
    # intervals is exact.
    area = 0.5 * (xs_u[:-1] + xs_u[1:]) * np.diff(union)
    covered = (union[:-1] >= e_lo) & (union[1:] <= e_hi)
    area = np.where(covered, area, 0.0)

    # Sum the intervals within each group. Every
    # group edge is a union node (located exactly by searchsorted) and, because
    # edges are strictly ascending, each group spans at least one interval,
    # which sidesteps the np.add.reduceat empty-slice quirk.
    edge_idx = np.searchsorted(union, edges)
    return np.add.reduceat(area, edge_idx[:-1])


def _liso_from_gnds(name: str) -> int:
    """Return the isomeric state (LISO) parsed from a GNDS name.

    ``'Am242_m1'`` -> ``1``, ``'Am242'`` -> ``0`` (ground). The suffix is the
    product's isomer ordinal, not the MF=10 LFS level index.
    """
    match = re.search(r'_m(\d+)$', name)
    return int(match.group(1)) if match else 0


def _build_xs_table_pendf(
    nuclides: Sequence[str],
    reactions: Sequence[str],
    energies: Sequence[float],
    pendf_library,
    pathways: bool = True,
) -> _SparseXSTable:
    """Build a sparse group cross section table from a pointwise PENDF library.

    Mirrors :func:`_build_xs_table_ce` but sources group cross sections from a
    preprocessed pointwise PENDF library instead of a continuous-energy
    openmc.lib session. Each requested ``(nuclide, reaction)`` present in the
    library has its MF=3 cross section flat-weighted onto the group structure
    via :func:`_group_average`; all-zero MF=3-total rows (nuclide or reaction
    absent, or a threshold above the group structure) are skipped.

    When ``pathways`` is true and the library exposes isomeric pathway data
    (MF=10 partial cross sections, per the ORIGEN-style "Option A" scheme), a
    reaction with mapped MF=10 partials is expanded into one row per product
    isomer instead of the single MF=3 total. The ground product (LISO 0) keeps
    the canonical reaction name (e.g. ``(n,gamma)``) and each metastable product
    is product-qualified (``(n,gamma)_m1``); the suffix is the product's LISO.
    Rows come exclusively from the MF=10 partials -- never from static branching
    ratios. The result's ``reactions`` axis is the expanded list: for each base
    reaction in input order, the base name first then its ``_m{n}`` variants in
    ascending isomer order. Unlike MF=3-total rows, a pathway-expanded partial
    row is staged even when it group-averages to zero (a metastable threshold
    above the group structure), so a product-qualified name in the reaction axis
    reliably marks that the collapse resolved pathways for that reaction.

    Parameters
    ----------
    nuclides : sequence of str
        Nuclide names defining the result's nuclide axis.
    reactions : sequence of str
        Base reaction names. The result's reaction axis contains these plus any
        product-qualified names emitted from MF=10 partials.
    energies : sequence of float
        Ascending energy group boundaries in [eV], length ``n_groups + 1``.
    pendf_library : openmc.data.PendfLibrary
        Pointwise PENDF library, duck-typed with ``nuclides`` (list of GNDS
        names), ``reactions(nuclide)`` (list of MTs with MF=3 data) and
        ``xs(nuclide, mt)`` (returning an ``(energy, xs)`` tuple). Isomeric
        pathway expansion additionally uses ``pathways(nuclide, mt)`` (LFS
        values with MF=10, ``[]`` if none), ``pathway_xs(nuclide, mt, lfs)``
        (``(energy, xs)`` of a partial) and ``product(nuclide, mt, lfs)`` (baked
        GNDS product name, ``None`` if the library was written unmapped).
    pathways : bool, optional
        If true (default), expand reactions with mapped MF=10 partials into
        per-product rows. If false, always emit the single MF=3-total row per
        reaction (reaction axis equals ``reactions``).
    """
    mts = [REACTION_MT[name] for name in reactions]
    energies = np.asarray(energies, dtype=float)
    n_groups = len(energies) - 1

    # Fast path: a grouped PENDF library exposes ``group_edges`` and pre-binned
    # ``xs_g``/``pathway_xs_g`` accessors, so rows are read straight from the
    # file instead of flat-weighting pointwise data at runtime. The library's
    # own edges must equal the requested tally structure exactly -- a grouped
    # library carries no pointwise data to rebin, so a mismatch is a hard error
    # (never a silent fallback).
    lib_edges = getattr(pendf_library, 'group_edges', None)
    grouped = lib_edges is not None
    if grouped:
        lib_edges = np.asarray(lib_edges, dtype=float)
        if not np.array_equal(lib_edges, energies):
            raise ValueError(
                f'Grouped PENDF library has {len(lib_edges)} group edges but '
                f'the requested tally structure has {len(energies)}; a grouped '
                f'library must be collapsed on its own edges (no rebinning). '
                f'Edge arrays differ (counts and/or values).')

    # Pathway expansion needs all three MF=10 accessors; a library lacking them
    # (e.g. an MF=3-only stand-in) transparently falls back to the total row.
    # Grouped libraries expose pre-binned ``pathway_xs_g``; pointwise ones expose
    # ``pathway_xs``.
    pathways_fn = getattr(pendf_library, 'pathways', None) if pathways else None
    pathway_xs_fn = getattr(
        pendf_library, 'pathway_xs_g' if grouped else 'pathway_xs', None)
    product_fn = getattr(pendf_library, 'product', None)
    have_pathways = None not in (pathways_fn, pathway_xs_fn, product_fn)

    # Stage rows as (nuc_idx, base_idx, row_name, xs_g). Metastable isomer
    # ordinals seen per base reaction are collected to build the expanded axis.
    staged: list[tuple[int, int, str, np.ndarray]] = []
    # Position of each (nuc_idx, row_name) in ``staged`` so a duplicate row sums
    # into the existing one instead of appending a second row that would be
    # silently clobbered by the last-wins fancy-index assignment in
    # ``_SparseXSTable.collapse`` (result[nuc, rxn] = ...).
    staged_pos: dict[tuple[int, str], int] = {}
    meta_by_base: dict[int, set[int]] = {i: set() for i in range(len(reactions))}
    available = set(pendf_library.nuclides)

    def stage(nuc_idx, base_idx, row_name, xs_g, keep_zero=False):
        # Pathway-expanded partials (keep_zero) stage even when the whole row
        # group-averages to zero (e.g. a metastable partial whose threshold lies
        # above the group structure), so a product-qualified name in the
        # reaction axis reliably marks that the collapse resolved pathways for
        # this reaction. MF=3-total rows keep the drop-if-zero behaviour, where
        # an absent reaction means "no data". A zero metastable row still
        # registers its isomer in meta_by_base so the axis gains the column.
        if not (keep_zero or xs_g.any()):
            return
        key = (nuc_idx, row_name)
        pos = staged_pos.get(key)
        if pos is None:
            staged_pos[key] = len(staged)
            staged.append((nuc_idx, base_idx, row_name, xs_g))
        else:
            # Two MF=10 levels (LFS is a level index, not the observable final
            # state) can map to the same product isomer; production of that
            # isomer sums over the levels rather than the later partial silently
            # overwriting the earlier one at collapse time.
            n_i, b_i, r_i, prev = staged[pos]
            staged[pos] = (n_i, b_i, r_i, prev + xs_g)
        liso = _liso_from_gnds(row_name)
        if liso > 0:
            meta_by_base[base_idx].add(liso)

    for nuc_idx, nuc in enumerate(nuclides):
        if nuc not in available:
            continue
        mts_present = set(pendf_library.reactions(nuc))
        # Index by reaction, not MT, so fission/(n,fission) stay separate
        for base_idx, (name, mt) in enumerate(zip(reactions, mts)):
            if mt not in mts_present:
                continue
            if grouped:
                total_g = pendf_library.xs_g(nuc, mt)
            else:
                energy, xs = pendf_library.xs(nuc, mt)
                total_g = _group_average(energy, xs, energies)

            lfs_list = list(pathways_fn(nuc, mt)) if have_pathways else []
            if not lfs_list:
                # No MF=10 partials -> single canonical (ground) row
                stage(nuc_idx, base_idx, name, total_g)
                continue

            products = [product_fn(nuc, mt, lfs) for lfs in lfs_list]
            if any(p is None for p in products):
                # Library written with mapping='none': product names are unknown
                # so pathway rows cannot be named. Fall back to the MF=3 total.
                # Chain-carried mapping is the B3 integration TODO (plan §1.7:
                # HDF5-baked mapping supersedes chain mapping; if absent the
                # chain must carry it). warn() fires once per (nuclide, mt) since
                # each pair is visited exactly once here.
                warn(f'PENDF library has MF=10 partials for {nuc} MT={mt} but no '
                     f'baked product names (mapping=none); emitting the MF=3 '
                     f'total instead of pathway rows. Use a product-mapped '
                     f'library to resolve isomeric pathways.')
                stage(nuc_idx, base_idx, name, total_g)
                continue

            # All partials mapped: one row per product, valued from its MF=10
            # partial (never a branching ratio). Consistency-check the partials
            # against the MF=3 total before staging.
            partial_g = []
            for lfs in lfs_list:
                if grouped:
                    partial_g.append(pathway_xs_fn(nuc, mt, lfs))
                else:
                    pe, pxs = pathway_xs_fn(nuc, mt, lfs)
                    partial_g.append(_group_average(pe, pxs, energies))
            part_sum = np.sum(partial_g, axis=0)
            nz = total_g != 0.0
            if nz.any():
                dev = np.abs(part_sum[nz] - total_g[nz]) / np.abs(total_g[nz])
                worst = float(dev.max())
                # Real TENDL-2017 partials deviate from the MF=3 total by up to
                # ~4e-6 per group (genuine data property); 1e-6 would warn on
                # nearly every isomeric nuclide at full-library scale.
                if worst > 1e-5:
                    g = int(np.nonzero(nz)[0][dev.argmax()])
                    warn(f'PENDF MF=10 partials for {nuc} MT={mt} sum to '
                         f'{part_sum[g]:.6e} b but the MF=3 total is '
                         f'{total_g[g]:.6e} b in group {g} (max relative '
                         f'deviation {worst:.3e} > 1e-5).')
            # Emit ground first, then ascending isomer order
            for liso, xs_g in sorted(
                    ((_liso_from_gnds(p), pg)
                     for p, pg in zip(products, partial_g)),
                    key=lambda t: t[0]):
                row_name = name if liso == 0 else f'{name}_m{liso}'
                stage(nuc_idx, base_idx, row_name, xs_g, keep_zero=True)

    # Build the expanded reaction axis: every base name (always present, so the
    # dense result keeps a column for each requested reaction) followed by its
    # emitted metastable variants in ascending isomer order.
    expanded: list[str] = []
    name_to_idx: dict[str, int] = {}
    for base_idx, name in enumerate(reactions):
        name_to_idx[name] = len(expanded)
        expanded.append(name)
        for liso in sorted(meta_by_base[base_idx]):
            qname = f'{name}_m{liso}'
            name_to_idx[qname] = len(expanded)
            expanded.append(qname)

    rows = [xs_g for _, _, _, xs_g in staged]
    nuc_idx_list = [nuc_idx for nuc_idx, _, _, _ in staged]
    rxn_idx_list = [name_to_idx[row_name] for _, _, row_name, _ in staged]

    xs_matrix = np.vstack(rows) if rows else np.empty((0, n_groups))

    return _SparseXSTable(
        list(nuclides), expanded, xs_matrix,
        np.array(nuc_idx_list, np.int32), np.array(rxn_idx_list, np.int32))


def _collapse_fluxes(table: _SparseXSTable, fluxes: Sequence[np.ndarray]) -> list[MicroXS]:
    """Collapse each domain's multigroup flux against a built XS table.

    Each flux is validated (finite, non-negative) and normalized to sum 1 before
    collapse; a zero-sum flux yields an all-zero MicroXS. Returns one
    ``(n_nuclides, n_reactions, 1)`` :class:`MicroXS` per domain.
    """
    micros = []
    for flux in fluxes:
        flux = np.asarray(flux, dtype=float)
        if not np.isfinite(flux).all():
            raise ValueError('Multigroup flux contains non-finite values')
        if (flux < 0).any():
            raise ValueError('Multigroup flux contains negative values')
        flux_sum = flux.sum()
        # Zero-sum flux (all zeros, given the checks above) collapses to zeros
        collapsed = table.collapse(flux / flux_sum if flux_sum else flux)
        micros.append(MicroXS(collapsed[:, :, np.newaxis],
                              table.nuclides, table.reactions))
    return micros


def _check_pathway_consistency(chain: Chain, micro_xs: MicroXS):
    """Fail on chain/MicroXS isomeric-pathway mismatches before depletion.

    :meth:`Chain.form_rxn_matrix` matches reaction rates to chain reactions by
    reaction type, so a product-qualified pathway (e.g. ``(n,gamma)_m1``) that
    exists on only one side is silently dropped or zeroed. For every nuclide
    present in both ``chain`` and ``micro_xs``, this raises ``ValueError``
    listing every offending ``(nuclide, reaction)`` pair when either:

    (a) ``micro_xs`` carries a qualified pathway *with non-zero data* for the
        nuclide whose reaction type the chain nuclide cannot route (its rate
        would be dropped), or
    (b) the chain nuclide carries a qualified pathway whose reaction type is
        entirely *absent from the* ``micro_xs`` *reaction axis* while the
        nuclide's unqualified base row is non-zero (pathway expansion never ran
        for this reaction, so the isomer route silently gets zero rate).

    Rule (b) is axis-level: because pathway expansion stages every partial row
    (even those that group-average to zero), the presence of a qualified name in
    ``micro_xs.reactions`` marks that the collapse resolved pathways for that
    reaction. A per-nuclide zero row under a *present* qualified column is
    therefore legitimate physics (a threshold above the group structure), not a
    mismatch. The residual limitation is that a cross-library chain/MicroXS mix,
    where a nuclide's partials exist in one library but not the other, is not
    detectable per-nuclide once the axis carries the qualified name; the
    axis-level test plus rule (a) is the guarantee. Plain (unqualified) reaction
    differences and nuclides present on only one side are left alone (ordinary
    OpenMC behaviour).
    """
    qualified = re.compile(r'_m\d+$')
    offenders = []
    for nuc in micro_xs.nuclides:
        if nuc not in chain:
            continue
        chain_rxns = {r.type for r in chain[nuc].reactions}
        n_idx = micro_xs._index_nuc[nuc]
        # Reactions this MicroXS actually carries for this nuclide (non-zero row)
        micro_rxns = {rx for rx in micro_xs.reactions
                      if micro_xs.data[n_idx, micro_xs._index_rx[rx]].any()}
        # (a) MicroXS carries a qualified pathway the chain cannot route -> its
        # rate is dropped when the reaction type is not in the chain.
        for rx in micro_rxns:
            if qualified.search(rx) and rx not in chain_rxns:
                offenders.append((nuc, rx, 'in MicroXS but not in chain'))
        # (b) Chain carries a qualified pathway whose reaction type is absent
        # from the MicroXS reaction axis while the unqualified base carries data
        # -> pathway expansion never ran here and the isomer route silently gets
        # zero rate. A qualified column that IS in the axis (even if this
        # nuclide's row is zero) means expansion ran, so it is not a mismatch.
        for rx in chain_rxns:
            if (qualified.search(rx) and rx not in micro_xs.reactions
                    and qualified.sub('', rx) in micro_rxns):
                offenders.append((nuc, rx, 'in chain but missing from MicroXS'))

    if offenders:
        lines = '\n'.join(f'  {nuc} {rx} ({why})' for nuc, rx, why in offenders)
        raise ValueError(
            'Isomeric pathway mismatch between the depletion chain and MicroXS. '
            'These product-qualified reaction rates would be silently dropped or '
            'zeroed when forming the transmutation matrix (rates are matched by '
            f'reaction type):\n{lines}\n'
            'Regenerate the chain and MicroXS from the same PENDF MF=10 product '
            'mapping so their qualified reactions agree.')


class MicroXS:
    """Microscopic cross section data for use in transport-independent depletion.

    .. versionadded:: 0.13.1

    .. versionchanged:: 0.14.0
        Class was heavily refactored and no longer subclasses pandas.DataFrame.

    Parameters
    ----------
    data : numpy.ndarray of floats
        3D array containing microscopic cross section values for each
        nuclide, reaction, and energy group. Cross section values are assumed to
        be in [b], and indexed by [nuclide, reaction, energy group]
    nuclides : list of str
        List of nuclide symbols for that have data for at least one
        reaction.
    reactions : list of str
        List of reactions. Each reaction must match those in
        :data:`openmc.deplete.chain.REACTIONS`, optionally with an isomeric
        product suffix (e.g. ``(n,gamma)_m1``) for pathway-expanded data.

    """
    def __init__(self, data: np.ndarray, nuclides: list[str], reactions: list[str]):
        # Validate inputs
        if len(data.shape) != 3:
            raise ValueError('Data array must be 3D.')
        if data.shape[:2] != (len(nuclides), len(reactions)):
            raise ValueError(
                f'Nuclides list of length {len(nuclides)} and '
                f'reactions array of length {len(reactions)} do not '
                f'match dimensions of data array of shape {data.shape}')
        check_iterable_type('nuclides', nuclides, str)
        check_iterable_type('reactions', reactions, str)
        check_type('data', data, np.ndarray, expected_iter_type=float)
        # Isomeric pathway reactions carry a product-qualified suffix (e.g.
        # '(n,gamma)_m1'); validate the canonical base reaction, ignoring it.
        for reaction in reactions:
            check_value('reactions', re.sub(r'_m\d+$', '', reaction),
                        _valid_rxns)

        self.data = data
        self.nuclides = nuclides
        self.reactions = reactions
        self._index_nuc = {nuc: i for i, nuc in enumerate(nuclides)}
        self._index_rx = {rx: i for i, rx in enumerate(reactions)}

    @classmethod
    def from_multigroup_flux(
        cls,
        energies: Sequence[float] | str | None = None,
        multigroup_flux: Sequence[float] | Sequence[Sequence[float]] | None = None,
        chain_file: PathLike | None = None,
        temperature: float | None = None,
        nuclides: Sequence[str] | None = None,
        reactions: Sequence[str] | None = None,
        *,
        cross_sections: PathLike | None = None,
        pendf_library=None,
        pathways: bool = True,
        **init_kwargs: dict,
    ) -> MicroXS | list[MicroXS]:
        """Generated microscopic cross sections from a known flux.

        The size of the MicroXS matrix depends on the chain file and cross
        sections available. MicroXS entry will be 0 if the nuclide cross section
        is not found.

        Multiple fluxes can be collapsed at once by passing a 2-D array (or a
        list of 1-D arrays); the group cross section table is then built only
        once and reused for every flux.

        It is recommended to make repeated calls to this method within a context
        manager using the :class:`openmc.lib.TemporarySession` class to avoid
        re-initializing OpenMC and loading cross sections each time.

        .. versionadded:: 0.15.0

        .. versionchanged:: 0.15.4
            ``multigroup_flux`` may be 2-D (or a list of 1-D arrays) to collapse
            several fluxes against a single shared cross section table, returning
            a list of :class:`MicroXS`. Added the ``cross_sections`` and
            ``pendf_library`` arguments. When ``pendf_library`` is a grouped
            PENDF library, ``energies`` may be omitted and defaults to the
            library's ``group_edges``.

        Parameters
        ----------
        energies : iterable of float or str or None, optional
            Energy group boundaries in [eV] or the name of a group structure.
            May be omitted (``None``) only when ``pendf_library`` is a grouped
            PENDF library (:class:`~openmc.data.GroupedPendfLibrary`), in which
            case the library's own ``group_edges`` supply the group structure;
            omitting it otherwise raises ``ValueError``.
        multigroup_flux : iterable of float or iterable of iterable of float
            Energy-dependent multigroup flux values. Must be finite and
            non-negative. A 1-D input is a single flux; a 2-D input (or a list of
            1-D arrays) is a batch of fluxes that share the same group structure.
        chain_file : PathLike or Chain, optional
            Path to the depletion chain XML file or an instance of
            openmc.deplete.Chain. Defaults to ``openmc.config['chain_file']``.
        temperature : float, optional
            Temperature for cross section evaluation in [K]. Default 293.6 K.
        nuclides : list of str, optional
            Nuclides to get cross sections for. If not specified, all burnable
            nuclides from the depletion chain file are used.
        reactions : list of str, optional
            Reactions to get cross sections for. If not specified, all neutron
            reactions listed in the depletion chain file are used.
        cross_sections : PathLike, optional
            Cross section library used to resolve nuclide data availability and
            evaluate cross sections. Defaults to ``openmc.config['cross_sections']``.
        pendf_library : openmc.data.PendfLibrary or openmc.data.GroupedPendfLibrary, optional
            PENDF cross section library, duck-typed with ``nuclides``,
            ``reactions(nuclide)`` and ``xs(nuclide, mt)``. When given, group
            cross sections are taken from this library rather than from
            continuous-energy data; the continuous-energy session arguments
            (``cross_sections`` and any :func:`openmc.lib.init` keyword
            arguments) are then invalid and raise ``ValueError``. The
            ``temperature`` argument is likewise rejected; the library's own
            preprocessed temperature is used. A pointwise
            :class:`~openmc.data.PendfLibrary` is flat-weighted onto ``energies``
            at runtime, whereas a pre-binned
            :class:`~openmc.data.GroupedPendfLibrary` (matched to ``energies``)
            is read directly without rebinning.
        pathways : bool, optional
            Only used with ``pendf_library``. If true (default), reactions with
            mapped isomeric MF=10 partials are expanded into per-product rows
            (ground keeps the canonical name; metastable products are qualified,
            e.g. ``(n,gamma)_m1``), so the returned ``reactions`` axis may
            contain product-qualified names. If false, only MF=3-total rows are
            emitted.
        **init_kwargs : dict
            Keyword arguments passed to :func:`openmc.lib.init`

        Returns
        -------
        MicroXS or list of MicroXS
            A single :class:`MicroXS` for a 1-D ``multigroup_flux``; a list with
            one entry per flux for a 2-D input (a 1-row batch returns a
            1-element list, not an unwrapped :class:`MicroXS`).
        """

        check_type("temperature", temperature, (int, float, type(None)))

        # ``multigroup_flux`` is required; it only carries a default so that
        # ``energies`` (which precedes it positionally) can default to None.
        if multigroup_flux is None:
            raise ValueError('multigroup_flux is a required argument')

        # Default the group structure to a grouped PENDF library's own edges
        # when the caller omits ``energies``. A grouped library is duck-detected
        # exactly as in ``_build_xs_table_pendf`` -- by exposing ``group_edges``.
        # A grouped library carries no pointwise data to rebin, so its edges are
        # the only structure it can be collapsed on.
        if energies is None:
            energies = getattr(pendf_library, 'group_edges', None)
            if energies is None:
                raise ValueError(
                    'energies must be provided unless pendf_library is a grouped '
                    'PENDF library (openmc.data.GroupedPendfLibrary), whose '
                    'group_edges then define the group structure')

        # if energy is string then use group structure of that name
        if isinstance(energies, str):
            energies = GROUP_STRUCTURES[energies]
        else:
            # if user inputs energies check they are ascending (low to high) as
            # some depletion codes use high energy to low energy.
            if not np.all(np.diff(energies) > 0):
                raise ValueError('Energy group boundaries must be in ascending order')

        # A 1-D flux is a single domain; 2-D (or a list of 1-D arrays) is a batch
        try:
            single = {1: True, 2: False}[1 + np.ndim(multigroup_flux[0])]
        except (TypeError, IndexError, KeyError):
            raise ValueError('multigroup_flux must be 1-D or 2-D') from None
        fluxes = [np.asarray(multigroup_flux, dtype=float)] if single else multigroup_flux

        # check dimension consistency per flux
        n_groups = len(energies) - 1
        for flux in fluxes:
            if len(flux) != n_groups:
                raise ValueError('Length of flux array should be len(energies)-1')

        # Default nuclides/reactions from the chain only when needed
        if not nuclides or reactions is None:
            chain = _get_chain(chain_file)
            if not nuclides:
                nuclides = [nuc.name for nuc in chain.nuclides]
            if reactions is None:
                reactions = chain.reactions

        # Build the group cross section table once and collapse every flux
        if pendf_library is not None:
            # The pointwise PENDF path is mutually exclusive with the
            # continuous-energy openmc.lib session path
            if cross_sections is not None or init_kwargs:
                raise ValueError(
                    'cross_sections and openmc.lib init arguments configure the '
                    'continuous-energy path and cannot be combined with '
                    'pendf_library')
            # temperature selects a continuous-energy evaluation; the PENDF
            # library carries its own preprocessed temperature
            if temperature is not None:
                raise ValueError(
                    'temperature configures the continuous-energy path and '
                    'cannot be combined with pendf_library')
            table = _build_xs_table_pendf(
                nuclides, reactions, energies, pendf_library, pathways=pathways)
        else:
            # None selects the continuous-energy default (293.6 K); resolve it
            # here, the sole place temperature is consumed (passed to group_xs).
            temperature = 293.6 if temperature is None else temperature
            # Resolve the library once; data availability is derived from it
            if cross_sections is None:
                cross_sections = _find_cross_sections(model=None)
            nuclides_with_data = _get_nuclides_with_data(cross_sections)
            table = _build_xs_table_ce(
                nuclides, reactions, energies, temperature, nuclides_with_data,
                cross_sections=cross_sections, **init_kwargs)

        micros = _collapse_fluxes(table, fluxes)
        return micros[0] if single else micros

    @classmethod
    def from_csv(cls, csv_file, **kwargs):
        """Load data from a comma-separated values (csv) file.

        Parameters
        ----------
        csv_file : str
            Relative path to csv-file containing microscopic cross section
            data. Cross section values are assumed to be in [b]
        **kwargs : dict
            Keyword arguments to pass to :func:`pandas.read_csv()`.

        Returns
        -------
        MicroXS

        """
        kwargs.setdefault('float_precision', 'round_trip')

        df = pd.read_csv(csv_file, **kwargs)
        df.set_index(['nuclides', 'reactions', 'groups'], inplace=True)
        nuclides = list(df.index.unique(level='nuclides'))
        reactions = list(df.index.unique(level='reactions'))
        groups = list(df.index.unique(level='groups'))
        shape = (len(nuclides), len(reactions), len(groups))
        data = df.values.reshape(shape)
        return cls(data, nuclides, reactions)

    def __getitem__(self, index):
        nuc, rx = index
        i_nuc = self._index_nuc[nuc]
        i_rx = self._index_rx[rx]
        return self.data[i_nuc, i_rx]

    def to_csv(self, *args, **kwargs):
        """Write data to a comma-separated values (csv) file

        Parameters
        ----------
        *args
            Positional arguments passed to :meth:`pandas.DataFrame.to_csv`
        **kwargs
            Keyword arguments passed to :meth:`pandas.DataFrame.to_csv`

        """
        groups = self.data.shape[2]
        multi_index = pd.MultiIndex.from_product(
            [self.nuclides, self.reactions, range(1, groups + 1)],
            names=['nuclides', 'reactions', 'groups']
        )
        df = pd.DataFrame({'xs': self.data.flatten()}, index=multi_index)
        df.to_csv(*args, **kwargs)

    def to_hdf5(self, group_or_filename: h5py.Group | PathLike, **kwargs):
        """Export microscopic cross section data to HDF5 format

        Parameters
        ----------
        group_or_filename : h5py.Group or path-like
            HDF5 group or filename to write to
        kwargs : dict, optional
            Keyword arguments to pass to :meth:`h5py.Group.create_dataset`.
            Defaults to {'compression': 'lzf'}.

        """
        kwargs.setdefault('compression', 'lzf')

        with h5py_file_or_group(group_or_filename, 'w') as group:
            # Store cross section data as 3D dataset
            group.create_dataset('data', data=self.data, **kwargs)

            # Store metadata as datasets using string encoding
            group.create_dataset('nuclides', data=np.array(self.nuclides, dtype='S'))
            group.create_dataset('reactions', data=np.array(self.reactions, dtype='S'))

    @classmethod
    def from_hdf5(cls, group_or_filename: h5py.Group | PathLike) -> Self:
        """Load data from an HDF5 file

        Parameters
        ----------
        group_or_filename : h5py.Group or str or PathLike
            HDF5 group or path to HDF5 file. If given as an h5py.Group, the
            data is read from that group. If given as a string, it is assumed
            to be the filename for the HDF5 file.

        Returns
        -------
        MicroXS
        """

        with h5py_file_or_group(group_or_filename, 'r') as group:
            # Read data from HDF5 group
            data = group['data'][:]
            nuclides = [nuc.decode('utf-8') for nuc in group['nuclides'][:]]
            reactions = [rxn.decode('utf-8') for rxn in group['reactions'][:]]

        return cls(data, nuclides, reactions)

    def merge(self, other: Self, prefer: str = 'other') -> Self:
        """Merge two MicroXS objects by taking the union of nuclides/reactions.

        If the two objects contain overlapping nuclide/reaction entries, values
        from `other` will overwrite values from `self` when `prefer='other'`.
        When `prefer='self'`, values from `self` are retained for overlapping
        entries, and values from `other` are used only for non-overlapping
        entries.

        Parameters
        ----------
        other : MicroXS
            Other MicroXS instance to merge with this one.
        prefer : {"other", "self"}
            Which instance's data should take precedence on overlap.

        Returns
        -------
        MicroXS
            New instance containing the merged data.
        """
        check_value('prefer', prefer, {'other', 'self'})

        # Require same number of energy groups
        if self.data.shape[2] != other.data.shape[2]:
            raise ValueError(
                'Cannot merge MicroXS with different number of energy groups: '
                f"{self.data.shape[2]} vs {other.data.shape[2]}. Ensure that "
                'both were generated with consistent group structures and '
                'treatments (e.g., both multigroup or both collapsed).'
            )

        # Build unified axes preserving order (self first, then other's new)
        new_nuclides = list(self.nuclides)
        for nuc in other.nuclides:
            if nuc not in self._index_nuc:
                new_nuclides.append(nuc)
        new_reactions = list(self.reactions)
        for rx in other.reactions:
            if rx not in self._index_rx:
                new_reactions.append(rx)

        # Allocate and fill from self (self's nuclides/reactions map to the
        # first indices of new_nuclides/new_reactions by construction)
        groups = self.data.shape[2]
        data = np.zeros((len(new_nuclides), len(new_reactions), groups))
        idx_n = {nuc: i for i, nuc in enumerate(new_nuclides)}
        idx_r = {rx: i for i, rx in enumerate(new_reactions)}

        n_self = len(self.nuclides)
        r_self = len(self.reactions)
        data[:n_self, :r_self] = self.data

        # Build destination index arrays for other's nuclides/reactions
        dst_n = np.array([idx_n[nuc] for nuc in other.nuclides])
        dst_r = np.array([idx_r[rx] for rx in other.reactions])

        # Copy from other, respecting precedence
        if prefer == 'other':
            data[np.ix_(dst_n, dst_r)] = other.data
        else:
            # Copy only entries where nuc or rx is absent from self
            nuc_is_new = np.array(
                [nuc not in self._index_nuc for nuc in other.nuclides])
            rx_is_new = np.array(
                [rx not in self._index_rx for rx in other.reactions])
            mask = nuc_is_new[:, np.newaxis] | rx_is_new[np.newaxis, :]
            src_i, src_j = np.where(mask)
            if src_i.size:
                data[dst_n[src_i], dst_r[src_j]] = other.data[src_i, src_j]

        return MicroXS(data, new_nuclides, new_reactions)


def write_microxs_hdf5(
    micros: Sequence[MicroXS],
    filename: PathLike,
    names: Sequence[str] | None = None,
    **kwargs
):
    """Write multiple MicroXS objects to an HDF5 file

    Parameters
    ----------
    micros : list of MicroXS
        List of MicroXS objects
    filename : PathLike
        Output HDF5 filename
    names : list of str, optional
        Names for each MicroXS object. If None, uses 'domain_0', 'domain_1',
        etc.
    **kwargs
        Additional keyword arguments passed to :meth:`h5py.Group.create_dataset`
    """
    if names is None:
        names = [f'domain_{i}' for i in range(len(micros))]

    # Open file once and write all domains using group interface
    with h5py.File(filename, 'w') as f:
        for microxs, name in zip(micros, names):
            group = f.create_group(name)
            microxs.to_hdf5(group, **kwargs)


def read_microxs_hdf5(filename: PathLike) -> dict[str, MicroXS]:
    """Read multiple MicroXS objects from an HDF5 file

    Parameters
    ----------
    filename : path-like
        HDF5 filename

    Returns
    -------
    dict
        Dictionary mapping domain names to MicroXS objects
    """
    with h5py.File(filename, 'r') as f:
        return {name: MicroXS.from_hdf5(group) for name, group in f.items()}
