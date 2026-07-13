"""MicroXS module

A class for storing microscopic cross section data that can be used with the
IndependentOperator class for depletion.
"""

from __future__ import annotations
from collections.abc import Collection, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import os
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
from openmc.data import REACTION_MT, open_pendf_library
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


def _pendf_dilution_material(domain) -> openmc.Material:
    """Return the :class:`openmc.Material` shielding ``domain`` for URR dilution.

    A :class:`~openmc.Material` shields with itself; a :class:`~openmc.Cell`
    shields with its ``fill`` when that fill is a single Material (the tally
    domain stays the Cell, so the flux is still per-cell). Any other domain -- a
    Cell filled with void, a Universe, a Lattice or a distributed material, or a
    non-Material/Cell object -- has no single composition and raises
    ``ValueError``.
    """
    if isinstance(domain, openmc.Material):
        return domain
    if isinstance(domain, openmc.Cell):
        fill = domain.fill
        if isinstance(fill, openmc.Material):
            return fill
        kind = 'void' if fill is None else type(fill).__name__
        raise ValueError(
            'urr_material_dilution=True requires each openmc.Cell domain to be '
            'filled with a single openmc.Material (its composition builds the '
            f'URR self-shielding sigma_0 background); cell {domain.id} has a '
            f'{kind} fill, which has no single composition. Pass a '
            'material-filled cell or a Material, or set '
            'urr_material_dilution=False.')
    raise ValueError(
        'urr_material_dilution=True requires every domain to be an '
        'openmc.Material or an openmc.Cell filled with a single openmc.Material; '
        f'got {type(domain).__name__}. Meshes, tally filters, universes and '
        'lattices have no single composition -- pass materials or '
        'material-filled cells, or set urr_material_dilution=False.')


def _pendf_domain_filters(domains: DomainTypes) -> list:
    """Build the flux-tally domain filters, one flux per domain in input order.

    Mirrors :func:`get_microxs_and_flux` for a single spatial filter, a
    :class:`~openmc.MeshBase`, an explicit list of filters, or a homogeneous
    sequence of materials / cells / universes. A *mixed* Material/Cell sequence
    (allowed under ``urr_material_dilution``) is handled by emitting one filter
    per domain so the per-domain flux ordering matches the input sequence.
    """
    if isinstance(domains, openmc.Filter):
        return [domains]
    if isinstance(domains, openmc.MeshBase):
        return [openmc.MeshFilter(domains)]
    if (isinstance(domains, Sequence) and len(domains) > 0
            and isinstance(domains[0], openmc.Filter)):
        return list(domains)
    if all(isinstance(d, openmc.Material) for d in domains):
        return [openmc.MaterialFilter(domains)]
    if all(isinstance(d, openmc.Cell) for d in domains):
        return [openmc.CellFilter(domains)]
    if all(isinstance(d, openmc.Universe) for d in domains):
        return [openmc.UniverseFilter(domains)]
    if all(isinstance(d, (openmc.Material, openmc.Cell)) for d in domains):
        # Mixed Material/Cell: one filter per domain preserves input order.
        return [openmc.MaterialFilter([d]) if isinstance(d, openmc.Material)
                else openmc.CellFilter([d]) for d in domains]
    raise ValueError(f"Unsupported domain type: {type(domains[0])}")


def get_pendf_microxs_and_flux(
    model: openmc.Model,
    domains: DomainTypes,
    pendf_library,                                   # PendfLibrary | GroupedPendfLibrary object
    nuclides: Sequence[str] | None = None,
    reactions: Sequence[str] | None = None,
    energies: Sequence[float] | str | None = None,   # None -> pendf_library.group_edges
    chain_file: PathLike | Chain | None = None,
    path_statepoint: PathLike | None = None,
    path_input: PathLike | None = None,
    run_kwargs=None,
    *,
    urr_material_dilution: bool = False,
    mat_ssf_nuclides: Sequence[str] | None = None,
    partial_binding: bool | Collection[tuple[str, str]] = False,
) -> tuple[list[np.ndarray], list[MicroXS]]:
    """Generate PENDF microscopic cross sections and fluxes for multiple domains.

    This is the transport-coupled counterpart of
    :meth:`MicroXS.from_multigroup_flux` with a ``pendf_library``. It runs one
    neutron transport solve that tallies **only** the multigroup flux in each
    domain (no reaction-rate tallies), then collapses group cross sections out of
    ``pendf_library`` per domain against that domain's flux. Given identical
    flux, each returned :class:`MicroXS` is exactly the direct
    :meth:`MicroXS.from_multigroup_flux` call for that domain -- the point of the
    design.

    .. versionadded:: 0.15.4

    Parameters
    ----------
    model : openmc.Model
        OpenMC model object. Must contain geometry, materials, and settings. The
        transport solve is continuous-energy; ``pendf_library`` supplies only the
        group cross sections used for the collapse.
    domains : list of openmc.Material or openmc.Cell or openmc.Universe, or openmc.MeshBase, or openmc.Filter, or list of openmc.Filter
        Domains in which to tally flux, or a spatial tally filter. When
        ``urr_material_dilution=True`` every domain must resolve to a single
        composition -- an :class:`openmc.Material`, or an :class:`openmc.Cell`
        filled with a single Material (mixed Material/Cell sequences allowed);
        see that argument.
    pendf_library : openmc.data.PendfLibrary or openmc.data.GroupedPendfLibrary or openmc.data.PendfTapeLibrary or path-like
        PENDF cross section library object (duck-typed with ``nuclides``,
        ``reactions(nuclide)`` and ``xs(nuclide, mt)``; a grouped library is
        detected via ``group_edges``/``xs_g``). May also be a path, opened via
        :func:`openmc.data.open_pendf_library`: a grouped or pointwise ``.h5``
        file, a directory of ``.h5`` files, or a directory of raw ASC PENDF tapes
        (routed to the cross-validation
        :class:`~openmc.data.PendfTapeLibrary`, which requires ``energies`` and
        does not support ``urr_material_dilution``). A path is opened once and
        closed here; a passed object is left to the caller. Required. Group cross
        sections are taken from this library rather than from continuous-energy
        data.
    nuclides : list of str, optional
        Nuclides to get cross sections for. If not specified, all burnable
        nuclides from the depletion chain file are used.
    reactions : list of str, optional
        Reactions to get cross sections for. If not specified, all neutron
        reactions listed in the depletion chain file are used.
    energies : iterable of float or str or None, optional
        Energy group boundaries in [eV] or the name of a group structure. These
        define both the flux tally and the collapse group structure. May be
        omitted (``None``) only when ``pendf_library`` is a grouped PENDF library
        (:class:`~openmc.data.GroupedPendfLibrary`), in which case the library's
        own ``group_edges`` supply the structure; omitting it with a pointwise
        library raises ``ValueError`` (there are no ``group_edges`` to default
        from, and a group structure is needed to build the flux tally).
    chain_file : PathLike or Chain, optional
        Path to the depletion chain XML file or an instance of
        openmc.deplete.Chain. Used to default ``nuclides``/``reactions``.
        Defaults to ``openmc.config['chain_file']``.
    path_statepoint : path-like, optional
        Path to write the statepoint file from the neutron transport solve to.
        By default it is written to a temporary directory and not kept.
    path_input : path-like, optional
        Path to write the model XML file from the neutron transport solve to.
        By default it is written to a temporary directory and not kept.
    run_kwargs : dict, optional
        Keyword arguments passed to :meth:`openmc.Model.run`.
    urr_material_dilution : bool, optional
        Enable the URR material-dilution self-shielding correction on the
        collapse. This wrapper owns the domains, so the toggle is a plain
        **bool**: ``True`` shields each domain with **that domain's own
        composition** automatically (via
        :meth:`~openmc.Material.get_nuclide_atom_densities`), so every domain must
        resolve to a single composition -- an :class:`openmc.Material` (shields
        with itself) or an :class:`openmc.Cell` filled with a single Material
        (shields with its fill; the tally domain stays the Cell, so the flux is
        per-cell). Mixed Material/Cell sequences are allowed. Universes,
        lattices, meshes, tally filters and cells with void/universe/lattice/
        distributed fills have no single composition and raise ``ValueError``.
        ``False`` (default) leaves the collapse unchanged. To supply an
        *explicit* composition
        (an :class:`openmc.Material` or ``{nuclide: density}`` mapping) rather
        than each domain's own, call :meth:`MicroXS.from_multigroup_flux`
        directly -- that collapse-level argument takes the Material/mapping form;
        this wrapper-level argument is bool only. The Bondarenko fold uses the
        temperature baked into the PENDF library's probability tables; no
        cross-check against the material or transport temperature is performed,
        so ensure the library's temperature matches the conditions modelled.
    mat_ssf_nuclides : iterable of str, optional
        Restricts the URR self-shielding to these nuclides (intersected with the
        default flagged list and the library's ptable coverage). ``None``
        (default) uses the full flagged list. Only used when
        ``urr_material_dilution`` is ``True``.
    partial_binding : bool or collection of (str, str), optional
        Opt-in switch, threaded to :meth:`MicroXS.from_multigroup_flux`, for
        chain-**stock** reactions whose ELIS-failed metastable would otherwise
        lump into the ground row. ``False`` (default) is today's behaviour
        exactly (a stock reaction emits the single MF=3 total). ``True`` binds
        every candidate that survives the veto; a collection of
        ``(nuclide, reaction_type)`` pairs (the reaction type as it appears in
        the chain / MicroXS row names, e.g. ``{('Np239', '(n,gamma)')}``) binds
        only those. A candidate -- a stock reaction with an MF=10 LFS=0 ground
        partial AND >=1 other partial -- binds its ground row to the LFS=0
        partial and drops the unmapped metastables. **Three honest costs:**
        (1) *library-dependence* -- the same chain then gives the MF=3 total on a
        library without a usable MF=10 ground and the MF=10 ground on one with
        it (``False`` preserves the library-independent "stock = MF=3 total"
        invariant); (2) *convergent-channel degradation* -- the MF=3 total is the
        true parent removal rate, so binding under-burns the parent by the
        dropped metastable fraction AND under-produces a shared daughter the MF=3
        lump delivers correctly (the switch is intended for the few
        divergent-daughter exotics -- Bk247, Au176 class -- which the collection
        form scopes; global ``True`` is exploration only); (3) *veto semantics*
        -- binding is reason-blind, so a candidate whose LFS=0 is a thermal
        placeholder (Bk247 class) or whose grid over-sums (``Sigma(all)/total >
        1.5``) is refused and silently keeps the MF=3 total, with one summary
        warning naming the bound, vetoed, and kept channels.

    Returns
    -------
    list of numpy.ndarray
        Flux in each group in [n-cm/src] for each domain (raw tallied
        magnitudes, as :func:`get_microxs_and_flux` returns them). Unlike
        :func:`get_gendfxs_and_flux`, which returns ``(flux, energy-bounds)``
        tuples, this wrapper returns bare flux arrays.
    list of MicroXS
        Cross section data in [b] for each domain. Negative final one-group
        collapsed values are clamped to zero with a summary warning (see
        :meth:`MicroXS.from_multigroup_flux`).

    See Also
    --------
    openmc.deplete.get_microxs_and_flux
    openmc.deplete.MicroXS.from_multigroup_flux
    openmc.deplete.IndependentOperator

    """
    # --- Pre-run validation (all before the expensive model.run) -------------

    # This wrapper-level toggle is a plain bool (the wrapper owns the domains).
    # A Material/mapping belongs one level down, at the collapse.
    if not isinstance(urr_material_dilution, bool):
        raise ValueError(
            "urr_material_dilution must be a bool for "
            "get_pendf_microxs_and_flux(); True uses each domain's own "
            "composition automatically. To supply an explicit composition "
            "(openmc.Material or {nuclide: density} mapping), call "
            "MicroXS.from_multigroup_flux directly.")

    # ``pendf_library`` may be an already-opened reader or a path: a
    # grouped/pointwise .h5 file, a directory of .h5 files, or a directory of
    # raw ASC PENDF tapes (the cross-validation tape adapter). Resolve a path to
    # a reader once here and share the object across every domain's collapse
    # (from_multigroup_flux then receives the object, not the path). Only a
    # grouped .h5 is self-describing; the pointwise .h5 and the tape adapter both
    # require the caller's ``energies``.
    _owned_pendf = None
    if isinstance(pendf_library, (str, os.PathLike)):
        pendf_library = open_pendf_library(pendf_library)
        _owned_pendf = pendf_library

    # The URR self-shielding fold reads probability tables the raw ASC tape
    # adapter does not serve; reject the combination loudly rather than silently
    # skipping the correction (build a pointwise .h5 for URR work).
    if urr_material_dilution and getattr(pendf_library, 'is_tape_source', False):
        if _owned_pendf is not None:
            _owned_pendf.close()
        raise ValueError(
            'urr_material_dilution is not supported on the raw ASC PENDF tape '
            'adapter (a deterministic-collapse cross-validation path); build a '
            'pointwise .h5 with PendfLibrary.from_endf_directory for URR '
            'self-shielding')

    # Resolve the group structure. Explicit ``energies`` win; otherwise fall back
    # to a grouped PENDF library's own ``group_edges`` (duck-detected exactly as
    # in the collapse). A pointwise library carries none, so ``energies=None``
    # with a pointwise library is an error -- and we need the edges here anyway
    # to build the flux tally's energy filter.
    if energies is None:
        energies = getattr(pendf_library, 'group_edges', None)
        if energies is None:
            raise ValueError(
                'energies must be provided unless pendf_library is a grouped '
                'PENDF library (openmc.data.GroupedPendfLibrary), whose '
                'group_edges then define the group structure')
    if isinstance(energies, str):
        energies = GROUP_STRUCTURES[energies]

    # ``urr_material_dilution=True`` shields each domain with its own
    # composition. Every domain must therefore resolve to a single Material: a
    # Material shields with itself, an openmc.Cell with its single-Material fill
    # (the tally domain stays the Cell, so the flux is per-cell). Mixed
    # Material/Cell sequences are allowed; meshes, tally filters, universes and
    # lattices (and cells with void/universe/lattice/distributed fills) have no
    # single composition and raise. Resolve the per-domain shielding compositions
    # up front so a bad domain fails before the expensive model.run.
    dilution_materials = None
    if urr_material_dilution:
        if (isinstance(domains, (openmc.MeshBase, openmc.Filter))
                or not isinstance(domains, Sequence) or len(domains) == 0
                or isinstance(domains[0], openmc.Filter)):
            raise ValueError(
                'urr_material_dilution=True requires a sequence of '
                'openmc.Material or material-filled openmc.Cell domains: the URR '
                "self-shielding sigma_0 background is built from each domain's "
                'own composition. Meshes and tally filters have no single '
                'composition -- pass materials or material-filled cells, or set '
                'urr_material_dilution=False.')
        dilution_materials = [_pendf_dilution_material(d) for d in domains]

    # The PENDF collapse always needs the chain (it names the MF=10 pathway
    # rows). Resolve it once, before the expensive model.run and after the
    # cheaper argument/domain validation, so a missing chain fails fast; the
    # resolved Chain is shared across every domain's collapse.
    chain = _get_pendf_chain(chain_file)

    # Save any original tallies on the model
    original_tallies = list(model.tallies)

    # The flux tally's energy filter uses the resolved group structure.
    energy_filter = openmc.EnergyFilter(energies)

    # Build list of domain filters (mirrors get_microxs_and_flux, plus mixed
    # Material/Cell support for the URR material-dilution path)
    domain_filters = _pendf_domain_filters(domains)

    # One flux-only tally per domain filter -- no reaction-rate tallies.
    flux_tallies = []
    model.tallies = []
    for i, domain_filter in enumerate(domain_filters):
        flux_tally = openmc.Tally(name=f'MicroXS flux {i}')
        flux_tally.filters = [domain_filter, energy_filter]
        flux_tally.scores = ['flux']
        model.tallies.append(flux_tally)
        flux_tallies.append(flux_tally)

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

    # Concatenate flux results across all domain filters (raw magnitudes -- the
    # IndependentOperator normalization depends on them, so do not normalize).
    fluxes = []
    for flux_tally in flux_tallies:
        # Get flux values and make energy groups last dimension.
        # (domains, groups, 1, 1) -> (domains, 1, 1, groups)
        flux = np.moveaxis(flux_tally.get_reshaped_data(), 1, -1)
        fluxes.extend(flux.squeeze((1, 2)))

    # Per-domain collapse against the PENDF library. When dilution is on, each
    # domain shields with its own composition (a Material, or a material-filled
    # Cell's fill -- resolved above); off passes False, an exact no-op relative
    # to the plain flux-supplied collapse. The chain resolved up front is shared
    # across every domain (loaded once).
    if urr_material_dilution:
        dilution_per_domain = dilution_materials
    else:
        dilution_per_domain = [False] * len(fluxes)

    micros = [
        MicroXS.from_multigroup_flux(
            energies=energies, multigroup_flux=flux_i, chain_file=chain,
            nuclides=nuclides, reactions=reactions, pendf_library=pendf_library,
            urr_material_dilution=dilution, mat_ssf_nuclides=mat_ssf_nuclides,
            partial_binding=partial_binding)
        for flux_i, dilution in zip(fluxes, dilution_per_domain)
    ]

    # Reset tallies
    model.tallies = original_tallies

    # Close a library we opened from a path (a caller-passed object is left to
    # the caller). The tape adapter's close is a no-op; an h5 reader closes its
    # file handles.
    if _owned_pendf is not None:
        _owned_pendf.close()

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


def _liso_from_gnds(name: str) -> int:
    """Return the isomeric state (LISO) parsed from a GNDS name.

    ``'Am242_m1'`` -> ``1``, ``'Am242'`` -> ``0`` (ground). The suffix is the
    product's isomer ordinal, not the MF=10 LFS level index.
    """
    match = re.search(r'_m(\d+)$', name)
    return int(match.group(1)) if match else 0


# Real TENDL-2017 partials deviate from the MF=3 total by up to ~4e-6
# per group (genuine data property); 1e-6 would warn on nearly every
# isomeric nuclide at full-library scale.
CONSISTENCY_RTOL = 1e-5

# Groups where BOTH the MF=3 total and the summed partials sit below this
# (barns) are evaluator floor placeholders (e.g. the ubiquitous 1e-20 b
# "effective zero" in JEFF-4.0, floored independently per section), not
# physics -- their relative deviation is meaningless. A group is only
# exempt when both sides are dust: a meaningful partial against a dust
# total (or vice versa) is a genuine inconsistency and still warns.
CONSISTENCY_ABS_FLOOR = 1e-15

# Below this Sigma(all MF=10 partials)/total the isomeric branching is a
# bit-identical evaluator placeholder (census: silent <= 3e-5, live >= 0.79)
# while the real cross section lives only in the MF=3 total. On the qualified
# collapse path the ground pathway is then filled by balance
# (total - demanded metastables) inside the ground partial's own energy range;
# see _silence_fill_ground. Hardwired (no per-run knob), mirroring the
# MF=10-always-on precedent.
SILENCE_EPS = 1e-3

# Sigma(all MF=10 partials)/total above this is a corrupt/over-summing grid,
# so the opt-in partial-binding switch refuses to bind such a reaction's LFS=0
# ground (decision B1). Same value as the retired patcher gate's spike cap.
_SPIKE_CAP = 1.5

# Trailing metastable qualifier ('_m1') on nuclide names / qualified
# reaction types.
_ISOMER_SUFFIX = re.compile(r'_m\d+$')


def _partials_total_max_deviation(total_g, part_sum):
    """Max relative deviation of summed MF=10 partials from the MF=3 total.

    Returns (worst, group_idx) over groups with nonzero total, skipping
    groups where both sides are below ``CONSISTENCY_ABS_FLOOR`` (evaluator
    floor dust). Returns (0.0, -1) when no group qualifies.
    """
    nz = (total_g != 0.0) & (
        (np.abs(total_g) >= CONSISTENCY_ABS_FLOOR)
        | (np.abs(part_sum) >= CONSISTENCY_ABS_FLOOR))
    if not nz.any():
        return 0.0, -1
    dev = np.abs(part_sum[nz] - total_g[nz]) / np.abs(total_g[nz])
    worst = float(dev.max())
    return worst, int(np.nonzero(nz)[0][dev.argmax()])


def _dedupe_base_reactions(reactions: Sequence[str]) -> list[str]:
    """Strip ``_mN`` product qualifiers and dedupe, preserving first-seen order.

    Product-qualified names (e.g. ``(n,gamma)_m1``) are pathway-expansion
    *outputs*, not collapse inputs. Reducing a reaction list to its distinct base
    names keeps ``REACTION_MT[name]`` from raising on a qualified name and lets a
    chain-defaulted list (which carries qualified reaction types) feed the
    collapse unchanged.
    """
    seen: set[str] = set()
    out: list[str] = []
    for name in reactions:
        base = _ISOMER_SUFFIX.sub('', name)
        if base not in seen:
            seen.add(base)
            out.append(base)
    return out


def _default_pendf_reactions(chain: Chain) -> list[str]:
    """Base reaction list for the PENDF collapse defaulted from a chain.

    Strips ``_mN``, dedupes (see :func:`_dedupe_base_reactions`), and drops any
    reaction the collapse cannot map to an MT (no ``REACTION_MT`` entry) -- an
    activation chain carries transmutation channels the pointwise collapse does
    not support -- with a single summary warning. Unlike an explicitly passed
    reaction list, a defaulted one must not crash the build.
    """
    base = _dedupe_base_reactions(chain.reactions)
    known = [r for r in base if r in REACTION_MT]
    dropped = [r for r in base if r not in REACTION_MT]
    if dropped:
        warn('PENDF collapse skipping depletion-chain reaction(s) with no '
             f'REACTION_MT mapping: {", ".join(dropped)}.')
    return known


def _get_pendf_chain(chain_file: PathLike | Chain | None) -> Chain:
    """Resolve the depletion chain required by the PENDF collapse.

    The PENDF collapse always needs a chain: it is the authority for isomeric
    row names (each MF=10 ``LFS`` partial is bound to the chain reaction carrying
    that ``pendf_lfs``). Raises a clear error when no chain can be resolved.
    """
    if chain_file is None and 'chain_file' not in openmc.config:
        raise ValueError(
            'PENDF collapse requires chain_file -- the chain carries the '
            'isomer<->LFS mapping; build one with '
            'tools/add_pendf_isomeric_branching_to_chain.py')
    return _get_chain(chain_file)


def _pendf_library_basename(pendf_library) -> str | None:
    """Best-effort basename of a PENDF library's backing HDF5 file, or ``None``.

    :class:`~openmc.data.PendfLibrary` holds its open ``h5py.File`` handles in
    ``_files``; :class:`~openmc.data.GroupedPendfLibrary` records its ``_path``.
    Neither is part of the duck-typed collapse interface, so a library exposing
    neither (a test fake, or a directory-mode :class:`PendfLibrary` spanning
    several files) yields ``None`` and the basename is dropped from the mismatch
    message.
    """
    path = getattr(pendf_library, '_path', None)
    if path is not None:
        return Path(path).name
    files = getattr(pendf_library, '_files', None)
    if files:
        try:
            names = {Path(f.filename).name for f in files}
        except Exception:
            return None
        if len(names) == 1:
            return next(iter(names))
    return None


def _verify_pendf_chain_stamp(chain, pendf_library) -> None:
    """Warn when a stamped chain's PENDF provenance disagrees with the library.

    The chain patcher (``tools/add_pendf_isomeric_branching_to_chain.py``) stamps
    the exported chain's root element with the identity of the PENDF source it was
    built from: ``pendf_library`` (the tape-derived source identity),
    ``pendf_nuclides`` (the nuclide count), an informational ``pendf_source`` (the
    h5 basename / dir last-two components) and the provenance-only ``decay_source``
    / ``decay_library``. The PENDF collapse is chain-driven -- a stock reaction
    silently takes the MF=3 total -- so a wrong/stale chain paired with a library
    produces
    silently degraded physics that no pathway-set comparison can catch. This makes
    such a pairing self-detecting.

    Behavior:

    * Unstamped chain (no ``pendf_*`` root attrs) -> silent (backward compatible
      with chains built before stamping, and with vanilla chains).
    * Library exposing no identity string at all -- neither a tape-derived
      ``source_identity`` nor a user ``library`` label (e.g. a duck-typed test
      fake) -> silent; there is nothing to verify against.
    * The stamp is a tape-derived identity, so the stamped ``pendf_library`` is
      compared against BOTH the library's ``source_identity`` and its ``library``
      label; a mismatch fires only when it matches NEITHER. The nuclide-count
      trigger is unchanged. Either triggering yields one :class:`UserWarning`
      naming the identities compared. A differing ``pendf_source`` alone (a file
      rename) is never a trigger; the ``decay_*`` provenance attrs are never
      verified.
    """
    root_attrs = getattr(chain, 'root_attrs', None) or {}
    stamped_lib = root_attrs.get('pendf_library')
    stamped_n = root_attrs.get('pendf_nuclides')
    if stamped_lib is None and stamped_n is None:
        return  # unstamped chain -- nothing to verify

    # The stamp is a tape-derived identity; a library belongs to the chain if the
    # stamp matches EITHER its tape-derived source_identity OR its user library
    # label. A library carrying neither (test fakes) cannot be verified against,
    # so the check skips entirely.
    source_identity = getattr(pendf_library, 'source_identity', None)
    lib_name = getattr(pendf_library, 'library', None)
    identities = [x for x in (source_identity, lib_name) if x is not None]
    if not identities:
        return
    lib_nuclides = getattr(pendf_library, 'nuclides', None)
    lib_n = len(lib_nuclides) if lib_nuclides is not None else None

    # Trigger on the library string (matches NEITHER identity) OR the nuclide
    # count (compared as strings so the XML-sourced stamp and the int count
    # agree). The source basename is never a trigger.
    lib_mismatch = stamped_lib is not None and stamped_lib not in identities
    n_mismatch = (stamped_n is not None and lib_n is not None
                  and str(stamped_n) != str(lib_n))
    if not (lib_mismatch or n_mismatch):
        return

    source = root_attrs.get('pendf_source', 'unknown source')
    basename = _pendf_library_basename(pendf_library)
    against = f'{basename} ' if basename else ''
    compared = ' / '.join(repr(x) for x in identities)
    warn(
        f'PENDF provenance mismatch: chain built from {source} (library '
        f'{stamped_lib!r}, {stamped_n} nuclides) but collapsing against '
        f'{against}(identity {compared}, {lib_n} nuclides) -- regenerate the '
        f'chain from this library with '
        f'tools/add_pendf_isomeric_branching_to_chain.py')


def _chain_lfs_reactions(chain: Chain, nuc: str, base_reaction: str) -> dict:
    """Map MF=10 ``LFS`` levels to the chain reactions that consume them.

    Returns ``{pendf_lfs: ReactionTuple}`` for ``nuc``'s chain reactions whose
    base type (``_mN`` stripped) equals ``base_reaction`` and that carry a
    ``pendf_lfs``. A nuclide absent from the chain yields an empty map.

    Raises ``ValueError`` if a *qualified* (metastable) reaction for this base
    carries ``pendf_lfs=None``: such a chain was built without LFS recording and
    cannot bind MF=10 partials, so it must be regenerated with the patcher tool.
    An *unqualified* base reaction without an LFS (e.g. a plain total-fallback
    channel) is simply not bindable and is skipped.
    """
    if nuc not in chain:
        return {}
    by_lfs = {}
    for rx in chain[nuc].reactions:
        if _ISOMER_SUFFIX.sub('', rx.type) != base_reaction:
            continue
        if rx.pendf_lfs is None:
            if _ISOMER_SUFFIX.search(rx.type):
                raise ValueError(
                    f'Depletion chain reaction {nuc} {rx.type!r} carries no '
                    'pendf_lfs, so its MF=10 partials cannot be bound by LFS. '
                    'This chain was built without LFS recording; regenerate it '
                    'with tools/add_pendf_isomeric_branching_to_chain.py.')
            continue
        by_lfs[rx.pendf_lfs] = rx
    return by_lfs


@dataclass(eq=False)
class _SilenceFill:
    """Result of the in-domain silence-fill of a reaction's ground pathway.

    ``fired`` is True when at least one in-domain group is silent (the ground
    was filled). ``e_dom`` / ``ground_dom`` are the union grid restricted to the
    LFS=0 partial's own tabulated range and the filled ground on it, ready to
    :func:`_group_average` into the tally structure. The remaining fields are
    full-union-grid diagnostics reused by the partial-binding veto (a follow-on
    feature): ``e`` the union of the MF=3 and every MF=10 partial grid, ``total``
    the MF=3 total on ``e``, ``sum_all`` every library partial (all LFS,
    demanded or not) on ``e``, ``silent`` the ``total > floor`` and
    ``sum_all/total < eps`` mask, and ``ground0_range`` the ``(emin, emax)`` of
    the LFS=0 partial (``None`` when the reaction carries no LFS=0 partial).
    """
    fired: bool
    e_dom: np.ndarray
    ground_dom: np.ndarray
    e: np.ndarray
    total: np.ndarray
    sum_all: np.ndarray
    silent: np.ndarray
    ground0_range: tuple[float, float] | None


def _silence_fill_ground(
    pathways_fn,
    pathway_xs_fn,
    nuc: str,
    mt: int,
    energy3: np.ndarray,
    xs3: np.ndarray,
    demanded_lfs: set,
    eps: float = SILENCE_EPS,
    floor: float = CONSISTENCY_ABS_FLOOR,
) -> _SilenceFill:
    """In-domain silence-fill of a reaction's ground (LFS=0) pathway.

    For a qualified (n,gamma)-style reaction whose MF=10 branching is a thermal
    placeholder (every partial ~1e-20 b while the MF=3 total carries the real
    1/v capture), replace the placeholder ground with ``total - Sigma(demanded
    metastables)`` wherever the branching is silent
    (``Sigma(all partials)/total < eps`` with ``total > floor``), restricted to
    the LFS=0 partial's own tabulated range ("in-domain", so the fill only
    REPLACES stored placeholder values and never extends the evaluation past its
    last tabulated point). Elsewhere the ground stays the source-faithful LFS=0
    partial, so a channel whose branching is live wherever the total is does not
    fire and its ground row is bit-identical to the raw LFS=0 partial average.

    The silence *test* sums EVERY library partial (all LFS -- including
    undemanded ELIS-dropped partials and multi-product ``LFS{l}_ZAP{z}`` entries
    enumerated via ``pathways_fn``); the *fill* subtracts only the demanded
    metastables (``LFS > 0`` in ``demanded_lfs``). Using the full partial set
    keeps an undemanded live partial from having its cross section absorbed into
    ground by the subtraction. Positivity is automatic: a filled group is
    silent, so ``Sigma(demanded m) <= Sigma(all) < eps*total`` and the ground
    stays ``>= (1 - eps)*total > 0`` -- no clamp.

    The union-grid / lin-lin / zero-fill-outside-range arithmetic mirrors
    ``claude/ground_by_balance/merit_probe.load_channel``.

    Parameters
    ----------
    pathways_fn : callable
        ``pathways(nuclide, mt) -> [(lfs, izap), ...]`` (pointwise library).
    pathway_xs_fn : callable
        ``pathway_xs(nuclide, mt, lfs, izap) -> (energy, xs)`` (pointwise).
    nuc : str
        Nuclide GNDS name.
    mt : int
        Reaction MT number.
    energy3, xs3 : numpy.ndarray
        MF=3 total energy grid and cross section.
    demanded_lfs : set of int
        Chain-demanded LFS levels; only the ``LFS > 0`` members are subtracted.
    eps, floor : float
        Silence threshold on ``Sigma(all)/total`` and the significance floor on
        ``total``.
    """
    energy3 = np.asarray(energy3, dtype=float)
    xs3 = np.asarray(xs3, dtype=float)

    grids = [energy3]
    partials = []
    for lfs, izap in pathways_fn(nuc, mt):
        pe, pxs = pathway_xs_fn(nuc, mt, lfs, izap)
        pe = np.asarray(pe, dtype=float)
        pxs = np.asarray(pxs, dtype=float)
        partials.append((lfs, pe, pxs))
        grids.append(pe)

    e = np.unique(np.concatenate(grids))
    total = np.interp(e, energy3, xs3)
    sum_all = np.zeros_like(e)
    sum_demanded_meta = np.zeros_like(e)
    ground0 = np.zeros_like(e)
    g0_lo = g0_hi = None
    for lfs, pe, pxs in partials:
        y = np.interp(e, pe, pxs, left=0.0, right=0.0)
        sum_all = sum_all + y
        if lfs == 0:
            ground0 = ground0 + y
            g0_lo = pe[0] if g0_lo is None else min(g0_lo, pe[0])
            g0_hi = pe[-1] if g0_hi is None else max(g0_hi, pe[-1])
        elif lfs in demanded_lfs:
            sum_demanded_meta = sum_demanded_meta + y

    sig = total > floor
    ratio = np.divide(sum_all, total, out=np.zeros_like(total), where=sig)
    silent = sig & (ratio < eps)

    if g0_lo is None:
        in_domain = np.zeros_like(e, dtype=bool)
        ground0_range = None
    else:
        in_domain = (e >= g0_lo) & (e <= g0_hi)
        ground0_range = (float(g0_lo), float(g0_hi))
    mask = silent & in_domain
    fired = bool(mask.any())

    # Restrict the filled ground to the LFS=0 native range so the terminal
    # interval above the ground partial's last tabulated point stays
    # source-faithful (zero contribution, no np.interp zero-fill down-ramp).
    e_dom = e[in_domain]
    ground_dom = np.where(mask[in_domain],
                          total[in_domain] - sum_demanded_meta[in_domain],
                          ground0[in_domain])
    return _SilenceFill(fired=fired, e_dom=e_dom, ground_dom=ground_dom,
                        e=e, total=total, sum_all=sum_all, silent=silent,
                        ground0_range=ground0_range)


def _normalize_partial_binding(partial_binding):
    """Normalize ``partial_binding`` to ``False``, ``True``, or a set of pairs.

    Accepts ``False`` (off), ``True`` (bind every candidate that survives the
    veto), or a collection of ``(nuclide, reaction_type)`` string pairs (bind
    only those, e.g. ``{('Np239', '(n,gamma)')}``); ``None`` is treated as
    ``False``. Returns ``False``, ``True``, or a ``set`` of ``(str, str)`` pairs.
    """
    if isinstance(partial_binding, bool):
        return partial_binding
    if partial_binding is None:
        return False
    try:
        return {(str(nuc), str(rx)) for nuc, rx in partial_binding}
    except (TypeError, ValueError):
        raise ValueError(
            'partial_binding must be a bool or a collection of '
            f'(nuclide, reaction_type) pairs; got {partial_binding!r}')


def _partial_binding_veto(fill: _SilenceFill) -> str | None:
    """Veto reason for binding a stock ground to the LFS=0 partial, or ``None``.

    The pointwise/tape analogue of decision B1: refuse to bind when, at or below
    the LFS=0 partial's last tabulated point and where the MF=3 total is
    significant, either the branching is silent anywhere (``'silent'`` -- a
    placeholder/gap region exists, so the stored LFS=0 ground cannot be trusted
    as the full ground row; the Bk247 thermal-placeholder class) or
    ``Sigma(all)/total`` exceeds :data:`_SPIKE_CAP` anywhere (``'spike'`` -- a
    corrupt/over-summing grid). Energies above the LFS=0 partial's last point
    (the universal terminal sliver where the MF=10 partials have ended but the
    MF=3 total still tails off) are NOT examined, matching the in-domain
    silence-fill decision (a bound channel there behaves like a qualified one).
    """
    if fill.ground0_range is None:
        return None
    in_dom = fill.e <= fill.ground0_range[1]
    if bool((fill.silent & in_dom).any()):
        return 'silent'
    sig = fill.total > CONSISTENCY_ABS_FLOOR
    ratio = np.divide(fill.sum_all, fill.total,
                      out=np.zeros_like(fill.total), where=sig)
    if bool((sig & in_dom & (ratio > _SPIKE_CAP)).any()):
        return 'spike'
    return None


def _partial_binding_veto_grouped(sum_all_g: np.ndarray,
                                  total_g: np.ndarray) -> str | None:
    """Group-space veto analogue for the grouped-library partial-binding path.

    Conservative relative to :func:`_partial_binding_veto`: a grouped library
    carries no pointwise grid, so the whole group range is examined and a top
    group lying wholly in the terminal sliver (partials ended, MF=3 total still
    live) may over-veto. Silent group = ``total`` significant and
    ``Sigma(all)/total < SILENCE_EPS``; spike group = ratio ``> _SPIKE_CAP``.
    """
    sig_g = total_g > CONSISTENCY_ABS_FLOOR
    ratio_g = np.divide(sum_all_g, total_g,
                        out=np.zeros_like(total_g), where=sig_g)
    if bool((sig_g & (ratio_g < SILENCE_EPS)).any()):
        return 'silent'
    if bool((sig_g & (ratio_g > _SPIKE_CAP)).any()):
        return 'spike'
    return None


def _build_xs_table_pendf(
    nuclides: Sequence[str],
    reactions: Sequence[str],
    energies: Sequence[float],
    pendf_library,
    chain: Chain,
    partial_binding: bool | Collection[tuple[str, str]] = False,
) -> _SparseXSTable:
    """Build a sparse group cross section table from a pointwise PENDF library.

    Mirrors :func:`_build_xs_table_ce` but sources group cross sections from a
    preprocessed pointwise PENDF library instead of a continuous-energy
    openmc.lib session. Each requested ``(nuclide, reaction)`` present in the
    library has its MF=3 cross section flat-weighted onto the group structure
    via :func:`_group_average`; all-zero MF=3-total rows (nuclide or reaction
    absent, or a threshold above the group structure) are skipped.

    When the library exposes isomeric pathway data (MF=10 partial cross
    sections, per the ORIGEN-style "Option A" scheme), a reaction is expanded
    into one row per product isomer instead of the single MF=3 total. **The
    depletion** ``chain`` **is the row-naming authority**: for base reaction
    ``R`` (MT) and each ``(lfs, izap)`` from ``pathways(nuclide, mt)``, the
    partial is bound to ``nuclide``'s chain reaction whose base type is ``R`` and
    whose ``pendf_lfs`` equals ``lfs``; the row name is that reaction's ``type``
    (the ground ``LFS 0`` keeps the canonical name ``R``, a metastable is
    ``R_m{n}``). Rows come exclusively from the MF=10 partials -- never from
    static branching ratios.

    **The chain is the demand side.** For each ``(nuclide, R)`` the chain is
    consulted first: a reaction left *stock* (no ``pendf_lfs`` pathway) emits the
    single MF=3 total row silently, regardless of any MF=10 partials the library
    carries. A *qualified* reaction emits one pathway row per **demanded** ``LFS``,
    bound to its library partial; ``LFS`` the library carries but the chain does
    not demand are ignored silently (the chain is the source of truth). When a
    demanded ``LFS`` is *missing* from the library the reaction falls back to the
    MF=3 total and is collected into one summary warning per build -- except the
    **self-loop ground waiver**: if the only missing demanded ``LFS`` is the ground
    and that ground reaction is a self-loop (target == parent, e.g. In115
    ``(n,n')``), the base row is staged from the MF=3 total and the demanded
    metastable rows from their partials with no warning, since such tapes define no
    LFS=0 partial and the self-loop base is a transmutation-matrix no-op. A chain
    whose *qualified* reactions carry ``pendf_lfs=None`` (built without LFS
    recording) is a hard error (see :func:`_chain_lfs_reactions`).

    The result's ``reactions`` axis is the expanded list: for each base reaction
    in input order, the base name first then its ``_m{n}`` variants in ascending
    isomer order. Unlike MF=3-total rows, a pathway-expanded partial row is
    staged even when it group-averages to zero (a metastable threshold above the
    group structure), so a product-qualified name in the reaction axis reliably
    marks that the collapse resolved pathways for that reaction.

    Parameters
    ----------
    nuclides : sequence of str
        Nuclide names defining the result's nuclide axis.
    reactions : sequence of str
        Reaction names. Product ``_mN`` qualifiers are stripped and the list is
        deduped (qualified names are expansion outputs, not inputs); the result's
        reaction axis contains the base names plus any product-qualified names
        emitted from MF=10 partials.
    energies : sequence of float
        Ascending energy group boundaries in [eV], length ``n_groups + 1``.
    pendf_library : openmc.data.PendfLibrary
        Pointwise PENDF library, duck-typed with ``nuclides`` (list of GNDS
        names), ``reactions(nuclide)`` (list of MTs with MF=3 data) and
        ``xs(nuclide, mt)`` (returning an ``(energy, xs)`` tuple). Isomeric
        pathway expansion additionally uses ``pathways(nuclide, mt)``
        (sorted ``(lfs, izap)`` int pairs, ``[]`` if none) and
        ``pathway_xs(nuclide, mt, lfs, izap=None)`` (``(energy, xs)`` of a
        partial; ``izap`` selects one product of a shared/lumped LFS).
    chain : openmc.deplete.Chain
        Depletion chain supplying isomeric row names (see above). Each MF=10
        ``LFS`` partial is bound to the chain reaction carrying that
        ``pendf_lfs``.
    partial_binding : bool or collection of (str, str), optional
        Opt-in switch (default ``False`` = today's behaviour exactly) that lets a
        chain-**stock** reaction bind its ground row to the library's MF=10 LFS=0
        partial and DROP the unmapped metastables, instead of routing the MF=3
        total into ground. ``True`` binds every surviving candidate; a collection
        of ``(nuclide, base-reaction)`` pairs binds only those. See the veto and
        cost notes on :func:`get_pendf_microxs_and_flux`.
    """
    # Qualified names are expansion outputs, not inputs; reduce to distinct base
    # reactions so a qualified name never reaches ``REACTION_MT`` (KeyError).
    reactions = _dedupe_base_reactions(reactions)
    mts = [REACTION_MT[name] for name in reactions]
    energies = np.asarray(energies, dtype=float)
    n_groups = len(energies) - 1
    partial_binding = _normalize_partial_binding(partial_binding)

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

    # Pathway expansion needs both MF=10 accessors; a library lacking them
    # (e.g. an MF=3-only stand-in) transparently falls back to the total row.
    # Grouped libraries expose pre-binned ``pathway_xs_g``; pointwise ones expose
    # ``pathway_xs``. Product names come from the chain, not the library.
    pathways_fn = getattr(pendf_library, 'pathways', None)
    pathway_xs_fn = getattr(
        pendf_library, 'pathway_xs_g' if grouped else 'pathway_xs', None)
    have_pathways = None not in (pathways_fn, pathway_xs_fn)

    # (nuclide, base reaction, demanded-LFS set, library-LFS set) tuples where the
    # chain demands qualified MF=10 pathways the library does not exactly match, so
    # the MF=3 total was staged instead of pathway rows. Collected across the whole
    # build so the disagreement is reported once, not once per reaction.
    mismatched: list[tuple[str, str, set[int], set[int]]] = []

    # (nuclide, base reaction) for qualified reactions whose GROUPED ground still
    # carries the thermal placeholder (some group silent while the total is
    # significant). Grouped libraries are not silence-filled at collapse time
    # (the fill is a build-time bake), so these are collected for one summary
    # warning. A pointwise build silence-fills instead and never populates this.
    grouped_placeholder: list[tuple[str, str]] = []

    # Partial-binding diagnostic (only emitted when the toggle is enabled),
    # counted over CANDIDATES only (stock reactions with an LFS=0 ground plus
    # >=1 other partial that are in scope): 'nuc rx' strings for the ground rows
    # bound to the MF=10 LFS=0 partial, the vetoed ones (with '(silent)'/
    # '(spike)'), and the ones kept on the MF=3 total for lack of a usable LFS=0.
    pb_bound: list[str] = []
    pb_vetoed: list[str] = []
    pb_kept: list[str] = []

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

            # The depletion CHAIN is the demand side: consult it first for the
            # qualified pathways it expects for this (nuclide, base reaction).
            # ``_chain_lfs_reactions`` returns {pendf_lfs: ReactionTuple} and
            # raises on a legacy qualified-but-lfs-less chain (unbindable).
            lfs_reactions = _chain_lfs_reactions(chain, nuc, name)

            # Chain stock (no qualified pathway for this reaction): emit the single
            # MF=3 total row, SILENTLY -- regardless of any MF=10 partials the
            # library carries. The chain patcher deliberately leaves such reactions
            # stock (rejected / no-decay-data / ELIS-tolerance), and the MF=3 total
            # equals the LFS=0 partial for a ground-only reaction, so a fallback
            # warning here would be pure by-design noise.
            #
            # Opt-in partial-binding (default off, bit-identical): a stock
            # reaction whose library carries an LFS=0 ground partial AND >=1 other
            # partial (an unmapped metastable, or a lumped second product) can
            # instead bind its ground row to that LFS=0 partial and DROP the
            # metastables -- past a silence/spike veto, because stockness is
            # reason-blind (a band-rejected or thermal-placeholder LFS=0 must not
            # replace the live MF=3 ground). See get_pendf_microxs_and_flux.
            if not lfs_reactions:
                ground_g = None
                if partial_binding is not False and have_pathways:
                    pathway_list = list(pathways_fn(nuc, mt))
                    ground_pairs = [(lfs, izap) for lfs, izap in pathway_list
                                    if lfs == 0]
                    # Candidate: a bindable LFS=0 ground plus >=1 other partial to
                    # drop. A plain no-MF=10 stock reaction or an LFS=0-only
                    # channel is never a candidate and is never counted.
                    if ground_pairs and len(pathway_list) > 1 and (
                            partial_binding is True
                            or (nuc, name) in partial_binding):
                        if grouped:
                            sum_all_g = np.zeros(n_groups)
                            for lfs, izap in pathway_list:
                                sum_all_g = sum_all_g + pathway_xs_fn(
                                    nuc, mt, lfs, izap)
                            reason = _partial_binding_veto_grouped(
                                sum_all_g, total_g)
                        else:
                            fill = _silence_fill_ground(
                                pathways_fn, pathway_xs_fn, nuc, mt, energy, xs,
                                set())
                            reason = _partial_binding_veto(fill)
                        if reason is not None:
                            pb_vetoed.append(f'{nuc} {name} ({reason})')
                        else:
                            # Ground row = the LFS=0 partial's group average (a
                            # multi-product ground sums into the single stock
                            # ground row); the metastables are simply not staged.
                            g = np.zeros(n_groups)
                            for lfs, izap in ground_pairs:
                                if grouped:
                                    g = g + pathway_xs_fn(nuc, mt, lfs, izap)
                                else:
                                    pe, pxs = pathway_xs_fn(nuc, mt, lfs, izap)
                                    g = g + _group_average(pe, pxs, energies)
                            if g.any():
                                ground_g = g
                                pb_bound.append(f'{nuc} {name}')
                            else:
                                pb_kept.append(f'{nuc} {name}')
                stage(nuc_idx, base_idx, name,
                      total_g if ground_g is None else ground_g)
                continue

            # Chain qualified: the chain (demand side) lists the LFS levels it
            # expects for this reaction; bind each to the library partial carrying
            # that LFS. Each MF=10 partial is keyed by (LFS, IZAP): an LFS is a
            # level index, and a single LFS may be shared by several product
            # nuclides (a lumped channel), so IZAP disambiguates the product.
            # Extra library LFS the chain does not demand are IGNORED silently --
            # the chain is the source of truth for which pathways to emit.
            pathway_list = list(pathways_fn(nuc, mt)) if have_pathways else []
            demanded_lfs = set(lfs_reactions)
            library_lfs = {lfs for lfs, _izap in pathway_list}
            missing = demanded_lfs - library_lfs

            # Self-loop ground waiver: JEFF In113/In115-style (n,n') tapes carry no
            # LFS=0 MF=10 partial (only the metastable), yet the folded chain always
            # lists a ground member. When the ONLY demanded LFS the library lacks is
            # the ground AND that ground reaction is a self-loop (target == parent),
            # waive it: the self-loop base row is a transmutation-matrix no-op (loss
            # and gain both land on the diagonal), and the tape defines no ground
            # partial, so stage the base row from the MF=3 total and the demanded
            # metastable rows from their partials. This restores In113m/In115m
            # (n,n') production that a blanket fallback-to-total would kill.
            ground_rx = lfs_reactions.get(0)
            self_loop_waiver = (missing == {0} and ground_rx is not None
                                and ground_rx.target == nuc)

            if missing and not self_loop_waiver:
                # A demanded LFS is missing from the library and this is not the
                # self-loop ground case (a metastable is missing, or a non-self-loop
                # ground is missing): fall back to the MF=3 total and collect one
                # honest summary warning naming both LFS sets.
                mismatched.append((nuc, name, demanded_lfs, library_lfs))
                stage(nuc_idx, base_idx, name, total_g)
                continue

            # Emit exactly the demanded pathways: select the library partials whose
            # LFS the chain demands (extras dropped), skipping the ground when the
            # self-loop waiver is in effect (its base row comes from the MF=3 total
            # below, since the library has no LFS=0 partial).
            emit_pairs = [(lfs, izap) for lfs, izap in pathway_list
                          if lfs in demanded_lfs
                          and not (self_loop_waiver and lfs == 0)]
            bound = [lfs_reactions[lfs] for lfs, _izap in emit_pairs]

            # A lumped reaction (e.g. MT=5 (n,misc)) can carry MF=10 partials for
            # several distinct daughter nuclides that share an LFS and therefore
            # bind to one chain reaction. Two LFS levels of the SAME daughter
            # (same IZAP) summing into one isomer row is the designed case, but
            # two DIFFERENT daughters (distinct IZAP) landing on one row would
            # silently sum unrelated cross sections. Refuse a collapse row (chain
            # reaction) claimed by more than one daughter, identified by IZAP
            # (= 1000*Z + A, the product nuclide ignoring its isomeric state).
            row_izap: dict[str, int] = {}
            for (_lfs, izap), rx in zip(emit_pairs, bound):
                claimed = row_izap.setdefault(rx.type, izap)
                if claimed != izap:
                    raise ValueError(
                        f'PENDF reaction {nuc} MT={mt} has MF=10 partials for '
                        f'multiple daughter nuclides (IZAP {claimed}, {izap}) '
                        f'mapping to the same collapse row {rx.type!r}; '
                        f'multi-product lumped channels (e.g. MT=5 (n,misc)) are '
                        f'not supported as collapse rows.')

            if self_loop_waiver:
                # Self-loop base row from the MF=3 total (the tape has no LFS=0
                # partial for it).
                stage(nuc_idx, base_idx, name, total_g)

            # One row per demanded product, valued from its MF=10 partial (never a
            # branching ratio). The build-time patcher audit is the authoritative
            # partials-vs-total consistency diagnosis, so no runtime check here.
            partial_g = []
            for lfs, izap in emit_pairs:
                if grouped:
                    partial_g.append(pathway_xs_fn(nuc, mt, lfs, izap))
                else:
                    pe, pxs = pathway_xs_fn(nuc, mt, lfs, izap)
                    partial_g.append(_group_average(pe, pxs, energies))

            # In-domain silence-fill (pointwise) / placeholder detection
            # (grouped) of the ground pathway. Only on the qualified path with a
            # demanded ground plus >=1 demanded metastable; the self-loop waiver
            # already staged its base row from the MF=3 total (no LFS=0 partial),
            # so it is excluded. missing is empty here (a demanded-LFS-missing
            # reaction fell back to the total above), so a demanded ground means
            # the library carries an LFS=0 partial.
            if (not self_loop_waiver and 0 in demanded_lfs
                    and any(lfs > 0 for lfs in demanded_lfs)):
                if grouped:
                    # Grouped libraries are not filled at collapse time. Detect a
                    # still-placeholder ground (a group where the MF=3 total is
                    # significant but every MF=10 partial is silent) and collect
                    # it for one summary warning.
                    sum_all_g = np.zeros(n_groups)
                    for lfs, izap in pathways_fn(nuc, mt):
                        sum_all_g = sum_all_g + pathway_xs_fn(nuc, mt, lfs, izap)
                    sig_g = total_g > CONSISTENCY_ABS_FLOOR
                    ratio_g = np.divide(sum_all_g, total_g,
                                        out=np.zeros_like(total_g), where=sig_g)
                    if (sig_g & (ratio_g < SILENCE_EPS)).any():
                        grouped_placeholder.append((nuc, name))
                else:
                    fill = _silence_fill_ground(pathways_fn, pathway_xs_fn, nuc,
                                                mt, energy, xs, demanded_lfs)
                    if fill.fired:
                        for i, (lfs, _izap) in enumerate(emit_pairs):
                            if lfs == 0:
                                partial_g[i] = _group_average(
                                    fill.e_dom, fill.ground_dom, energies)
                                break

            # Emit ground first, then ascending isomer order; the row name is the
            # bound chain reaction's type (ground keeps the base name R).
            for rx, xs_g in sorted(
                    zip(bound, partial_g),
                    key=lambda t: _liso_from_gnds(t[0].type)):
                stage(nuc_idx, base_idx, rx.type, xs_g, keep_zero=True)

    # One summary warning when the chain demanded MF=10 pathways the library did
    # not exactly match (emitted once per build, not per reaction/group). The
    # staged values are unaffected (the MF=3 total was used); only the chain <->
    # library disagreement is surfaced.
    if mismatched:
        summary = ', '.join(
            f'{n} {r} (chain LFS '
            f'{{{", ".join(map(str, sorted(dem)))}}} vs library LFS '
            f'{{{", ".join(map(str, sorted(lib)))}}})'
            for n, r, dem, lib in mismatched)
        warn('The depletion chain expects MF=10 pathways the PENDF library does '
             f'not match for: {summary}. The MF=3 total was used for these '
             'reactions. The chain and library disagree -- regenerate the chain '
             'from this library with '
             'tools/add_pendf_isomeric_branching_to_chain.py.')

    # One summary warning when a GROUPED library carries an unfilled placeholder
    # ground for qualified reactions (pointwise builds silence-fill in place; a
    # grouped build must bake the fill at group-binning time). Detection is
    # robust in group space -- the placeholder-vs-total gap is ~20 decades even
    # after group averaging.
    if grouped_placeholder:
        examples = ', '.join(f'{n} {r}' for n, r in grouped_placeholder[:5])
        more = (f' (and {len(grouped_placeholder) - 5} more)'
                if len(grouped_placeholder) > 5 else '')
        warn('Grouped PENDF library carries an unfilled placeholder ground for '
             f'{len(grouped_placeholder)} qualified reaction(s): {examples}'
             f'{more}. The MF=10 branching is silent where the MF=3 total is '
             'significant, so the ground pathway is the raw placeholder (grouped '
             'libraries are not silence-filled at collapse time). Rebuild the '
             'grouped h5 with the baked fill for correct thermal ground '
             'production.')

    # One partial-binding summary per build (only when the toggle is enabled and
    # at least one candidate was in scope), matching the one-summary-per-build
    # style above. Counts and names are over candidates only.
    if partial_binding is not False and (pb_bound or pb_vetoed or pb_kept):
        b = f' [{", ".join(pb_bound)}]' if pb_bound else ''
        v = f' [{", ".join(pb_vetoed)}]' if pb_vetoed else ''
        k = f' [{", ".join(pb_kept)}]' if pb_kept else ''
        warn(f'PENDF partial-binding (opt-in): {len(pb_bound)} bound to MF=10 '
             f'ground{b}; {len(pb_vetoed)} vetoed{v}; {len(pb_kept)} kept '
             f'MF=3 total, no usable LFS=0{k}. Bound stock reactions use the '
             'library MF=10 ground and drop unmapped metastables (a '
             'library-dependent choice; see get_pendf_microxs_and_flux).')

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


# Number of fluxes collapsed per GEMM; bounds working memory to the dense
# scatter buffer of ``chunk * n_nuclides * n_reactions`` floats regardless of
# the total flux count.
_COLLAPSE_CHUNK_SIZE = 1024


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
        phi = np.asarray(fluxes[start:start + chunk_size], dtype=float)
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
        phi = phi / np.where(flux_sum > 0, flux_sum, 1.0)[:, np.newaxis]

        collapsed = table.collapse_batch(phi)
        for result in collapsed:
            micros.append(MicroXS(result[:, :, np.newaxis],
                                  table.nuclides, table.reactions))
    return micros


def _clamp_negative_pendf_microxs(micros: list[MicroXS]) -> None:
    """Clamp negative final PENDF collapsed cross sections to zero, in place.

    A cross section is physically non-negative, but the PENDF collapse can
    occasionally yield a slightly negative FINAL one-group value from noisy or
    internally inconsistent source data (e.g. a group-averaged MF=10 partial
    that dips below zero). This always-on backstop clamps any such negative
    value -- per ``(nuclide, reaction, domain)``, not per energy bin or
    pointwise -- to ``0.0`` across every :class:`MicroXS` of one collapse
    invocation, and emits a single summary :func:`warnings.warn` giving the
    count and up to ten of the most-negative ``(nuclide, reaction[, domain])``
    offenders.

    When no value is negative the ``micros`` are left completely untouched, so
    a clean collapse stays bit-identical: the strict ``< 0`` mask never
    disturbs a ``-0.0`` and no warning fires (the overwhelmingly common case).
    """
    # Fast path: act only when a genuine negative exists. A strict ``< 0`` test
    # excludes ``-0.0``, so a clean collapse is left bit-for-bit unchanged
    # (an unconditional np.maximum would flip -0.0 to +0.0 and perturb it).
    if not any((m.data < 0).any() for m in micros):
        return

    multi = len(micros) > 1
    offenders = []  # (value, nuclide, reaction, domain_index)
    for d, m in enumerate(micros):
        neg = m.data < 0
        if not neg.any():
            continue
        nuc_idx, rxn_idx, _ = np.nonzero(neg)
        for ni, ri in zip(nuc_idx, rxn_idx):
            offenders.append((float(m.data[ni, ri, 0]),
                              m.nuclides[ni], m.reactions[ri], d))
        m.data[neg] = 0.0  # leaves -0.0 and positive values untouched

    # Most-negative first; list up to ten as (nuclide, reaction[, domain]).
    offenders.sort(key=lambda o: o[0])
    parts = []
    for val, nuc, rxn, dom in offenders[:10]:
        if multi:
            parts.append(f'{nuc} {rxn} (domain {dom}): {val:.3e} b')
        else:
            parts.append(f'{nuc} {rxn}: {val:.3e} b')
    warn(f'PENDF collapse clamped {len(offenders)} negative one-group cross '
         f'section value(s) to 0. Most-negative offender(s): {"; ".join(parts)}.')


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
            if _ISOMER_SUFFIX.search(rx) and rx not in chain_rxns:
                offenders.append((nuc, rx, 'in MicroXS but not in chain'))
        # (b) Chain carries a qualified pathway whose reaction type is absent
        # from the MicroXS reaction axis while the unqualified base carries data
        # -> pathway expansion never ran here and the isomer route silently gets
        # zero rate. A qualified column that IS in the axis (even if this
        # nuclide's row is zero) means expansion ran, so it is not a mismatch.
        for rx in chain_rxns:
            if (_ISOMER_SUFFIX.search(rx) and rx not in micro_xs.reactions
                    and _ISOMER_SUFFIX.sub('', rx) in micro_rxns):
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
            falls back to the MF=3 total (one summary warning). Any negative
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

        check_type("temperature", temperature, (int, float, type(None)))

        # ``multigroup_flux`` is required; it only carries a default so that
        # ``energies`` (which precedes it positionally) can default to None.
        if multigroup_flux is None:
            raise ValueError('multigroup_flux is a required argument')

        # ``pendf_library`` may be an already-opened reader or a path: a
        # grouped/pointwise .h5 file, a directory of .h5 files, or a directory
        # of raw ASC PENDF tapes (routed to the cross-validation tape adapter).
        # A path is opened here and closed after the collapse; a passed object
        # is left to the caller. Only a grouped .h5 is self-describing (carries
        # its own group_edges); the pointwise .h5 and the tape adapter both
        # require the caller's ``energies``.
        _owned_pendf = None
        if isinstance(pendf_library, (str, os.PathLike)):
            pendf_library = open_pendf_library(pendf_library)
            _owned_pendf = pendf_library

        # Fuse the URR dilution toggle with its composition: normalize
        # ``urr_material_dilution`` to a local ``densities`` mapping (or None
        # when off) here, before any collapse work, so the impossible "on but no
        # composition" state cannot be represented.
        if urr_material_dilution is False or urr_material_dilution is None:
            densities = None
        elif urr_material_dilution is True:
            raise ValueError(
                'urr_material_dilution=True is under-specified: the URR '
                'self-shielding sigma_0 background needs a composition. Pass '
                'the openmc.Material being depleted, or a {nuclide: '
                'density-or-fraction} mapping, instead of True')
        elif isinstance(urr_material_dilution, openmc.Material):
            densities = urr_material_dilution.get_nuclide_atom_densities()
        elif isinstance(urr_material_dilution, Mapping):
            if not urr_material_dilution:
                raise ValueError(
                    'urr_material_dilution mapping is empty: with no diluters '
                    'the sigma_0 background is zero and every flagged nuclide '
                    'silently degrades to f=1. Pass the depleted composition, '
                    'or omit the argument to disable the correction')
            densities = urr_material_dilution
        else:
            raise ValueError(
                'urr_material_dilution must be an openmc.Material, a {nuclide: '
                'density-or-fraction} mapping, or False; got '
                f'{type(urr_material_dilution).__name__}')

        # The correction is defined only on the PENDF path -- it is built from
        # the library's probability tables.
        if densities is not None and pendf_library is None:
            raise ValueError(
                'urr_material_dilution requires a pendf_library; the URR '
                'self-shielding correction is built from its probability '
                'tables and is not available on the continuous-energy path')

        # The raw ASC tape adapter serves no probability tables; reject URR on it
        # loudly rather than silently skipping the correction.
        if densities is not None and getattr(pendf_library, 'is_tape_source',
                                             False):
            if _owned_pendf is not None:
                _owned_pendf.close()
            raise ValueError(
                'urr_material_dilution is not supported on the raw ASC PENDF '
                'tape adapter (a deterministic-collapse cross-validation path); '
                'build a pointwise .h5 with PendfLibrary.from_endf_directory for '
                'URR self-shielding')

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

        single = _flux_is_single(multigroup_flux)
        fluxes = [np.asarray(multigroup_flux, dtype=float)] if single else multigroup_flux

        # check dimension consistency per flux
        n_groups = len(energies) - 1
        for flux in fluxes:
            if len(flux) != n_groups:
                raise ValueError('Length of flux array should be len(energies)-1')

        # Validate the pendf_library argument combination before loading any
        # data (the chain below), so an invalid CE-vs-PENDF mix raises its own
        # clear error rather than a downstream "requires chain_file".
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

        # Resolve the depletion chain. The PENDF collapse ALWAYS needs it (it is
        # the isomer<->LFS row-naming authority; see _build_xs_table_pendf); the
        # continuous-energy path needs it only to default nuclides/reactions.
        # Load it once here and share it with both defaulting and the table build.
        if pendf_library is not None:
            chain = _get_pendf_chain(chain_file)
            # Verify the chain's PENDF provenance stamp against the library
            # actually in use (no-op for unstamped chains / identity-less
            # libraries); a wrong chain<->library pairing warns once here.
            _verify_pendf_chain_stamp(chain, pendf_library)
        elif not nuclides or reactions is None:
            chain = _get_chain(chain_file)
        else:
            chain = None

        if chain is not None:
            if not nuclides:
                nuclides = [nuc.name for nuc in chain.nuclides]
            if reactions is None:
                # A chain-defaulted reaction list carries qualified reaction
                # types and channels the pointwise collapse cannot map; sanitize
                # it for the PENDF path (strip _mN, dedupe, drop unmappable).
                reactions = (_default_pendf_reactions(chain)
                             if pendf_library is not None else chain.reactions)

        # Build the group cross section table once and collapse every flux
        if pendf_library is not None:
            table = _build_xs_table_pendf(
                nuclides, reactions, energies, pendf_library, chain,
                partial_binding=partial_binding)
            # URR material-dilution self-shielding: multiply the capture/fission
            # rows of flagged resonant nuclides by their per-group factor in the
            # URR-overlapping groups (in place). The =False path is untouched.
            if densities is not None:
                from .mat_ssf import _apply_mat_ssf
                _apply_mat_ssf(table, pendf_library, energies, densities,
                               mat_ssf_nuclides)
            # Close a library we opened from a path; a caller-passed object is
            # left to the caller. The library is no longer used below.
            if _owned_pendf is not None:
                _owned_pendf.close()
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
        # PENDF-only always-on backstop: clamp any negative FINAL collapsed
        # value to zero with one summary warning. Gated on pendf_library so the
        # continuous-energy path (pendf_library is None) stays untouched.
        if pendf_library is not None:
            _clamp_negative_pendf_microxs(micros)
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
