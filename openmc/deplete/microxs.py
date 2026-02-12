"""MicroXS module

A class for storing microscopic cross section data that can be used with the
IndependentOperator class for depletion.
"""

from __future__ import annotations
from collections.abc import Collection, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Union, TypeAlias, Self

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
        """Collapse the table against a single group flux.

        A normalized flux (summing to 1) yields one-group cross sections, a raw
        flux yields reaction rates. Returns a dense
        ``(n_nuclides, n_reactions)`` array. Thin wrapper over
        :meth:`collapse_batch` with a one-flux batch.
        """
        n_groups = self.xs_matrix.shape[1]
        if len(phi_norm) != n_groups:
            raise ValueError(
                f'Flux has {len(phi_norm)} groups but the cross section table '
                f'expects {n_groups}')
        return self.collapse_batch(np.asarray(phi_norm)[np.newaxis])[0]

    def collapse_batch(self, phi_norm: np.ndarray) -> np.ndarray:
        """Collapse the table against a batch of group fluxes with one GEMM.

        ``phi_norm`` is an ``(n_flux, n_groups)`` array; normalized rows (each
        summing to 1) give one-group cross sections, raw rows give reaction
        rates. A single matrix-matrix product ``phi_norm @ xs_matrix.T`` yields
        the ``(n_flux, nnz)`` collapsed values, which are then scattered into a
        dense ``(n_flux, n_nuclides, n_reactions)`` result exactly as
        :meth:`collapse` does per flux. Only the ``(n_flux, nnz)`` product is
        materialized beyond the returned result.
        """
        phi_norm = np.asarray(phi_norm)
        result = np.zeros(
            (phi_norm.shape[0], len(self.nuclides), len(self.reactions)))
        result[:, self.nuc_indices, self.rxn_indices] = phi_norm @ self.xs_matrix.T
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


def _flux_is_single(multigroup_flux):
    """Whether ``multigroup_flux`` is one 1-D flux (True) or a batch of
    1-D fluxes (False).

    Raises ValueError for anything else.
    """
    # A 1-D flux is a single domain; 2-D (or a list of 1-D arrays) is a batch
    try:
        return {1: True, 2: False}[1 + np.ndim(multigroup_flux[0])]
    except (TypeError, IndexError, KeyError):
        raise ValueError('multigroup_flux must be 1-D or 2-D') from None


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


# Trailing metastable qualifier ('_m1') on nuclide names / qualified
# reaction types.
_ISOMER_SUFFIX = re.compile(r'_m\d+$')


# Number of fluxes collapsed per GEMM; bounds working memory to the dense
# scatter buffer of ``chunk * n_nuclides * n_reactions`` floats regardless of
# the total flux count.
_COLLAPSE_CHUNK_SIZE = 1024


def _normalize_flux_batch(
    fluxes: Sequence[np.ndarray],
    start: int,
    n_groups: int,
) -> np.ndarray:
    """Stack, validate and row-normalize one batch of multigroup fluxes.

    The shape / finite / non-negative guards and the sum-to-1 normalization of
    :func:`_collapse_fluxes`, factored out so the block PENDF collapse
    (:mod:`openmc.deplete.pendf.collapse`) normalizes its flux batch exactly the
    same way -- same error messages, same *global* flux index in them.

    Parameters
    ----------
    fluxes : sequence of numpy.ndarray
        The batch itself (already sliced out of the full flux list).
    start : int
        Index of ``fluxes[0]`` in the full flux list, so a validation error
        names the offending flux by its global index.
    n_groups : int
        Expected number of groups per flux.

    Returns
    -------
    numpy.ndarray
        ``(n_batch, n_groups)`` array whose rows sum to 1; an all-zero flux is
        left all-zero (divided by 1, hence no NaN).
    """
    phi = np.asarray(fluxes, dtype=float)
    if phi.ndim != 2 or phi.shape[1] != n_groups:
        raise ValueError(f'Each multigroup flux must have length {n_groups}')

    # Vectorized equivalents of the per-flux finite / non-negative checks,
    # reporting the first offending flux in iteration order. As in the
    # per-flux path, the finite check takes precedence for a flux that
    # fails both.
    not_finite = ~np.isfinite(phi).all(axis=1)
    negative = (phi < 0).any(axis=1)
    bad = not_finite | negative
    if bad.any():
        local = int(np.argmax(bad))
        index = start + local
        if not_finite[local]:
            raise ValueError(
                f'Multigroup flux {index} contains non-finite values')
        raise ValueError(
            f'Multigroup flux {index} contains negative values')

    # Row-normalize to sum 1; zero-sum rows (all zeros) divide by 1 and
    # stay all-zero instead of producing NaN.
    flux_sum = phi.sum(axis=1)
    return phi / np.where(flux_sum > 0, flux_sum, 1.0)[:, np.newaxis]


def _collapse_fluxes(
    table: _SparseXSTable,
    fluxes: Sequence[np.ndarray],
    chunk_size: int = _COLLAPSE_CHUNK_SIZE,
) -> list[MicroXS]:
    """Collapse each domain's multigroup flux against a built XS table.

    Fluxes are processed in chunks of ``chunk_size`` (default
    :data:`_COLLAPSE_CHUNK_SIZE`): each chunk is stacked into an
    ``(n_chunk, n_groups)`` array, validated (finite, non-negative) and
    row-normalized to sum 1, then collapsed with a single GEMM via
    :meth:`_SparseXSTable.collapse_batch`. A zero-sum flux stays all-zero (no
    division, hence no NaN) and yields an all-zero MicroXS. Returns one
    ``(n_nuclides, n_reactions, 1)`` :class:`MicroXS` per domain.

    Peak memory beyond the returned MicroXS is set by the working chunk,
    dominated by the dense scatter buffer that
    :meth:`_SparseXSTable.collapse_batch` allocates: ``chunk_size *
    n_nuclides * n_reactions`` floats (at least ``chunk_size * nnz``),
    independent of the number of fluxes.
    """
    n_groups = table.xs_matrix.shape[1]
    micros = []
    for start in range(0, len(fluxes), chunk_size):
        phi = _normalize_flux_batch(
            fluxes[start:start + chunk_size], start, n_groups)

        collapsed = table.collapse_batch(phi)
        for result in collapsed:
            micros.append(MicroXS(result[:, :, np.newaxis],
                                  table.nuclides, table.reactions))
    return micros


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
        check_type('data', data, np.ndarray, expected_iter_type=np.floating)
        # Isomeric pathway reactions carry a product-qualified suffix (e.g.
        # '(n,gamma)_m1'); validate the canonical base reaction, ignoring it.
        for reaction in reactions:
            check_value('reactions', _ISOMER_SUFFIX.sub('', reaction),
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
        urr_material_dilution: openmc.Material | Mapping[str, float] | bool = False,
        mat_ssf_nuclides=None,
        partial_binding: bool | Collection[tuple[str, str]] = False,
        **init_kwargs: dict,
    ) -> MicroXS | list[MicroXS]:
        """Generated microscopic cross sections from a known flux.

        The size of the MicroXS matrix depends on the chain file and cross
        sections available. MicroXS entry will be 0 if the nuclide cross section
        is not found.

        Multiple fluxes can be collapsed at once by passing a 2-D array (or a
        list of 1-D arrays); the group cross sections are then read from the
        library only once and reused for every flux. Any number of fluxes is
        accepted, processed in chunks of :data:`_COLLAPSE_CHUNK_SIZE` (1024).

        On the continuous-energy path the chunk is collapsed against a
        ``(nnz, n_groups)`` cross section table built once up front. **The PENDF
        path never builds that table**: its rows are staged one nuclide at a time
        and contracted against the normalized fluxes in blocks of 256 rows, so
        the peak row buffer is ``256 x n_groups x 8 B`` (33 MB at 16000 groups)
        no matter how many non-zero rows the library carries. Beyond that a
        chunk holds its flux array, ``n_flux x n_groups x 8 B`` (131 MB at
        1024 x 16000), and the results,
        ``n_flux x n_nuclides x n_reactions x 8 B`` (365 kB per flux at 481
        nuclides x 95 reactions). PENDF rows are re-staged per chunk -- one pass
        over the library per 1024 fluxes -- so the collapse's summary warnings
        may be raised once per chunk; Python's default warning filter shows
        identical messages once.

        Each PENDF flux is contracted per block with the same matrix-vector
        product a single-flux call uses, so a batch equals the corresponding
        single-flux calls exactly, whatever the chunk split. The default 256-row
        block reproduces the former full-matrix contraction bit for bit under
        single-threaded BLAS (the validated configuration); a multithreaded BLAS
        moves the last bit for the former path as well as this one, and other
        block sizes may move it too, because the BLAS kernel picks its summation
        order from the operand shape.

        It is recommended to make repeated calls to this method within a context
        manager using the :class:`openmc.lib.TemporarySession` class to avoid
        re-initializing OpenMC and loading cross sections each time.

        .. versionadded:: 0.15.0

        .. versionchanged:: 0.15.4
            ``multigroup_flux`` may be 2-D (or a list of 1-D arrays) to collapse
            several fluxes against a single shared cross section table, returning
            a list of :class:`MicroXS`. Added the ``cross_sections`` and
            ``pendf_library`` arguments. When
            ``pendf_library`` is a grouped PENDF library, ``energies`` may be
            omitted and defaults to the library's ``group_edges``.

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
            **Required on the PENDF path** (``pendf_library`` given): the chain
            supplies the isomer<->LFS mapping that names the MF=10 pathway rows,
            and a clear error is raised when it cannot be resolved.
        temperature : float, optional
            Temperature for cross section evaluation in [K]. Default 293.6 K.
        nuclides : list of str, optional
            Nuclides to get cross sections for. If not specified, all burnable
            nuclides from the depletion chain file are used.
        reactions : list of str, optional
            Reactions to get cross sections for. If not specified, all neutron
            reactions listed in the depletion chain file are used. Product
            ``_mN`` qualifiers are stripped and the list is deduped (qualified
            names are pathway-expansion outputs, not inputs); on the PENDF path a
            chain-defaulted list additionally drops channels with no MT mapping
            with one summary warning.
        cross_sections : PathLike, optional
            Cross section library used to resolve nuclide data availability and
            evaluate cross sections. Defaults to ``openmc.config['cross_sections']``.
        pendf_library : openmc.data.PendfLibrary or openmc.data.GroupedPendfLibrary or openmc.data.PendfTapeLibrary or path-like, optional
            PENDF cross section library, duck-typed with ``nuclides``,
            ``reactions(nuclide)`` and ``xs(nuclide, mt)``. May also be a path,
            opened via :func:`openmc.data.open_pendf_library`: a grouped or
            pointwise ``.h5`` file, a directory of ``.h5`` files, or a directory
            of raw ASC PENDF tapes -- the last routed to the cross-validation
            :class:`~openmc.data.PendfTapeLibrary`, which collapses
            bit-identically to a pointwise ``.h5`` built from the same tapes but
            (like a pointwise library) still requires the caller's ``energies``
            and does not support ``urr_material_dilution``. A path is opened and
            closed here; a passed object is left to the caller. When given, group
            cross sections are taken from this library rather than from
            continuous-energy data; the continuous-energy session arguments
            (``cross_sections`` and any :func:`openmc.lib.init` keyword
            arguments) are then invalid and raise ``ValueError``. The
            ``temperature`` argument is likewise rejected; the library's own
            preprocessed temperature is used. A pointwise
            :class:`~openmc.data.PendfLibrary` is flat-weighted onto ``energies``
            at runtime, whereas a pre-binned
            :class:`~openmc.data.GroupedPendfLibrary` (matched to ``energies``)
            is read directly without rebinning. Reactions with isomeric MF=10
            partials are always expanded into per-product rows, with the row
            names sourced from ``chain_file`` (each ``LFS`` partial is bound to
            the chain reaction carrying that ``pendf_lfs``): the ground keeps the
            canonical name and metastable products are qualified, e.g.
            ``(n,gamma)_m1``, so the returned ``reactions`` axis may contain
            product-qualified names. A partial with no matching chain reaction
            falls back to the MF=3 total (one summary warning), except when the
            ground (``LFS=0``) is the only demanded pathway the library lacks:
            its row is then served by balance as
            ``max(0, total - Sigma(library metastable partials))`` and reported
            in a separate informational summary. Any negative
            final one-group value (from noisy or internally inconsistent source
            data) is clamped to zero, with one summary warning naming the
            offenders.
        urr_material_dilution : openmc.Material or dict or False, optional
            Only valid with ``pendf_library``. Enables the unresolved resonance
            region (URR) material-dilution self-shielding correction: the
            collapsed capture (and fission) reaction rates of flagged resonant
            nuclides are multiplied, in URR-overlapping groups only, by a
            per-group self-shielding factor computed from the nuclides'
            probability tables at a homogeneous background cross section
            ``sigma_0`` built from the supplied composition. The background is
            infinite-medium (no escape/Dancoff geometry) from a single
            composition snapshot (diluter build-in over an irradiation is not
            modelled). Accepts either an :class:`openmc.Material` (its
            :meth:`~openmc.Material.get_nuclide_atom_densities` supplies the
            composition) or a ``{nuclide: number-density-or-fraction}`` mapping
            (only ratios matter, so number densities or atom/weight fractions are
            equivalent). A flagged nuclide absent from the composition (e.g. a
            trace transmutation product) is treated as infinitely dilute
            (self-shielding factor 1). ``False`` (default) or ``None`` leaves the
            collapse unchanged; bare ``True`` and an empty mapping raise
            ``ValueError``. The composition must be given explicitly because this
            is the transport-free collapse path -- no live session exists, and a
            model has many materials, so one :class:`MicroXS` is built per
            material composition. Raises ``ValueError`` on the continuous-energy
            path. The Bondarenko fold uses the temperature baked into the PENDF
            library's probability tables; no cross-check against the material or
            transport temperature is performed, so ensure the library's
            temperature matches the conditions modelled.
        mat_ssf_nuclides : iterable of str, optional
            Restricts the URR self-shielding to these nuclides (intersected with
            the default flagged list and the library's ptable coverage). ``None``
            (default) uses the full flagged list. Only used when
            ``urr_material_dilution`` is true.
        partial_binding : bool or collection of (str, str), optional
            Opt-in switch (only meaningful on the PENDF path) for chain-**stock**
            reactions whose ELIS-failed metastable would otherwise lump into the
            ground row. ``False`` (default) is today's behaviour exactly: a stock
            reaction emits the single MF=3 total. ``True`` binds every candidate
            that survives the veto; a collection of ``(nuclide, reaction_type)``
            pairs -- the reaction type as it appears in the chain / MicroXS row
            names, e.g. ``{('Np239', '(n,gamma)')}`` -- binds only those. A
            candidate is a stock reaction whose library carries an LFS=0 ground
            partial AND >=1 other partial (an unmapped metastable, or a lumped
            second product); it binds its ground row to the LFS=0 partial and
            drops the unmapped metastables. **Three honest costs:**

            1. *Library-dependence.* Under the toggle the same chain gives the
               MF=3 total on a library without a usable MF=10 ground and the
               MF=10 ground on one with it. ``False`` preserves the
               library-independent "stock = MF=3 total" invariant.
            2. *Convergent-channel degradation.* The MF=3 total is the TRUE
               parent removal rate; binding under-burns the parent by the
               dropped metastable fraction AND under-produces a shared daughter
               that the MF=3 lump delivers correctly. The switch is intended for
               the few divergent-daughter exotics (Bk247, Au176 class) -- that is
               what the collection form scopes; global ``True`` is exploration
               only.
            3. *Veto semantics.* Binding is reason-blind (a stock reaction may be
               band-rejected or carry a thermal-placeholder LFS=0, Bk247 class),
               so a candidate is refused when, at/below the LFS=0 partial's last
               tabulated point, the branching is silent anywhere (placeholder/gap)
               or ``Sigma(all)/total`` spikes above 1.5 (corrupt grid). Vetoed
               candidates silently keep the MF=3 total; one summary warning names
               the bound, vetoed, and kept channels.
        **init_kwargs : dict
            Keyword arguments passed to :func:`openmc.lib.init`

        Returns
        -------
        MicroXS or list of MicroXS
            A single :class:`MicroXS` for a 1-D ``multigroup_flux``; a list with
            one entry per flux for a 2-D input (a 1-row batch returns a
            1-element list, not an unwrapped :class:`MicroXS`).
        """
        from .pendf.collapse import _from_multigroup_flux
        return _from_multigroup_flux(
            energies, multigroup_flux, chain_file, temperature, nuclides,
            reactions, cross_sections, pendf_library, urr_material_dilution,
            mat_ssf_nuclides, partial_binding, init_kwargs)

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


def _write_flux_data(f, fluxes, dtype='float64', compression=None,
                     compression_opts=None):
    """Write flux arrays and optional energy bounds to an open HDF5 file."""
    energy_bounds = None
    flux_arrays = []
    for item in fluxes:
        if isinstance(item, tuple) and len(item) == 2:
            flux_arrays.append(np.asarray(item[0], dtype=dtype))
            if energy_bounds is None:
                energy_bounds = np.asarray(item[1], dtype='float64')
        else:
            flux_arrays.append(np.asarray(item, dtype=dtype))

    f.create_dataset('fluxes', data=np.stack(flux_arrays),
                     compression=compression, compression_opts=compression_opts)
    if energy_bounds is not None:
        f.create_dataset('energy_bounds', data=energy_bounds)


def write_global_microxs_hdf5(
    micros: Sequence[MicroXS],
    filename: PathLike,
    material_ids: Sequence[str],
    fluxes: Sequence | None = None,
    dtype: str = 'float64',
    compression: bool | tuple = True,
) -> None:
    """Write all MicroXS to a single rank-independent HDF5 file.

    Stores cross section data for every burnable material in a stacked 4D
    dataset that can be efficiently subset-read by individual MPI ranks
    using :func:`read_local_microxs_hdf5`.
    This is meant to reduce RAM usage by each individal rank and enable greater MPI scaling.
    
    .. versionadded:: 0.15.4

    Parameters
    ----------
    micros : list of MicroXS
        MicroXS objects, one per burnable material, ordered by
        ``sorted(material_ids, key=int)``.
    filename : path-like
        Output HDF5 file path.
    material_ids : list of str
        Material ID strings in the same order as ``micros``. Must be sorted
        by ``int()`` value.
    fluxes : list, optional
        Flux data for each material. Each element is either a 1D numpy
        array or a ``(flux_array, energy_bounds)`` tuple. MG-flux is needed for isomeric branching
    dtype : str, optional
        NumPy dtype for cross section and flux data. Default ``'float64'``.
        Use ``'float32'`` to halve file size and per-rank RAM.
    compression : bool or tuple, optional
        HDF5 compression. Default ``True`` uses lzf. ``False`` disables
        compression. A tuple ``('gzip', level)`` uses gzip.

    See Also
    --------
    read_local_microxs_hdf5 : Read local material slices from this file.

    """
    if len(micros) == 0:
        raise ValueError("No MicroXS objects to write.")
    if len(micros) != len(material_ids):
        raise ValueError(
            f"Length of micros ({len(micros)}) != length of "
            f"material_ids ({len(material_ids)})")
    if fluxes is not None and len(fluxes) != len(micros):
        raise ValueError(
            f"Length of fluxes ({len(fluxes)}) != length of "
            f"micros ({len(micros)})")

    int_ids = [int(mid) for mid in material_ids]
    if int_ids != sorted(int_ids):
        raise ValueError("material_ids must be sorted by int() value")

    ref_shape = micros[0].data.shape
    for i, m in enumerate(micros):
        if m.data.shape != ref_shape:
            raise ValueError(
                f"MicroXS[{i}] shape {m.data.shape} != "
                f"MicroXS[0] shape {ref_shape}")

    # Resolve compression settings
    if compression is False:
        comp, comp_opts = None, None
    elif compression is True:
        comp, comp_opts = 'lzf', None
    elif isinstance(compression, tuple):
        comp, comp_opts = compression
    else:
        raise ValueError(
            f"compression must be True, False, or a tuple like "
            f"('gzip', 4), got {compression!r}")

    n_mats = len(micros)
    n_nuc, n_rxn, n_grp = ref_shape
    bytes_per_elem = np.dtype(dtype).itemsize
    target_chunk_bytes = 32 * 1024 * 1024
    row_bytes = n_nuc * n_rxn * n_grp * bytes_per_elem
    chunk_mats = max(1, min(n_mats, 256, target_chunk_bytes // row_bytes))

    with h5py.File(filename, 'w') as f:
        f.attrs['version'] = 1
        f.attrs['n_materials'] = n_mats
        f.attrs['n_nuclides'] = n_nuc
        f.attrs['n_reactions'] = n_rxn
        f.attrs['n_groups'] = n_grp

        stacked = np.stack([m.data for m in micros]).astype(dtype)
        f.create_dataset(
            'xs_data',
            data=stacked,
            chunks=(chunk_mats, n_nuc, n_rxn, n_grp),
            compression=comp,
            compression_opts=comp_opts,
        )

        f.create_dataset(
            'nuclides', data=np.array(micros[0].nuclides, dtype='S'))
        f.create_dataset(
            'reactions', data=np.array(micros[0].reactions, dtype='S'))
        f.create_dataset(
            'material_ids', data=np.array(material_ids, dtype='S'))

        if fluxes is not None:
            _write_flux_data(f, fluxes, dtype=dtype,
                             compression=comp, compression_opts=comp_opts)


def read_local_microxs_hdf5(
    filename: PathLike,
    local_mat_ids: Sequence[str],
) -> tuple[list[MicroXS], list[tuple] | None]:
    """Read local material slices from a global MicroXS HDF5 file.

    Reads only the rows corresponding to ``local_mat_ids`` from the stacked
    dataset, using h5py fancy indexing for efficient I/O.
    This is meant to reduce RAM usage by each individal rank and enable greater MPI scaling.

    .. versionadded:: 0.15.4

    Parameters
    ----------
    filename : path-like
        Path to HDF5 file written by :func:`write_global_microxs_hdf5`.
    local_mat_ids : list of str
        Material IDs for this MPI rank.

    Returns
    -------
    micros : list of MicroXS
        MicroXS objects in the same order as ``local_mat_ids``.
    flux_with_energy : list of tuple or None
        If flux data exists in the file, a list of
        ``(flux_array, energy_bounds)`` tuples. Returns ``None`` if no
        flux data in file.

    See Also
    --------
    write_global_microxs_hdf5 : Write the global file.

    """
    if len(local_mat_ids) == 0:
        return [], None

    with h5py.File(filename, 'r') as f:
        version = f.attrs.get('version', None)
        if version != 1:
            raise ValueError(
                f"Unsupported MicroXS HDF5 version: {version}. "
                f"Expected version 1.")

        all_mat_ids = [s.decode() for s in f['material_ids'][:]]
        nuclides = [s.decode() for s in f['nuclides'][:]]
        reactions = [s.decode() for s in f['reactions'][:]]

        mat_id_to_row = {mid: i for i, mid in enumerate(all_mat_ids)}
        local_rows = []
        for mid in local_mat_ids:
            if mid not in mat_id_to_row:
                sample = all_mat_ids[:5]
                raise ValueError(
                    f"Material '{mid}' not found in {filename}. "
                    f"File contains {len(all_mat_ids)} materials: "
                    f"{sample}{'...' if len(all_mat_ids) > 5 else ''}")
            local_rows.append(mat_id_to_row[mid])

        # h5py requires fancy indices to be strictly sorted ascending
        sorted_rows = sorted(local_rows)
        xs_block = f['xs_data'][sorted_rows]

        # Reorder to match local_mat_ids order
        sort_map = {row: pos for pos, row in enumerate(sorted_rows)}
        reorder = [sort_map[r] for r in local_rows]
        xs_block = xs_block[reorder]

        micros = [MicroXS(xs_block[i], nuclides, reactions)
                  for i in range(len(local_mat_ids))]

        flux_with_energy = None
        if 'fluxes' in f:
            energy_bounds = None
            if 'energy_bounds' in f:
                energy_bounds = f['energy_bounds'][:]

            flux_block = f['fluxes'][sorted_rows]
            flux_block = flux_block[reorder]
            flux_with_energy = [
                (flux_block[i], energy_bounds)
                for i in range(len(local_mat_ids))
            ]

    return micros, flux_with_energy
