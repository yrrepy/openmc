"""PENDF flux-collapse entry points.

Holds the pointwise-PENDF counterparts extracted from
:mod:`openmc.deplete.microxs`: the transport-plus-collapse driver
:func:`get_pendf_microxs_and_flux`, the PENDF cross section table builder
:func:`_build_xs_table_pendf`, and their domain / dilution / clamp helpers.

This submodule is never imported at ``openmc.deplete`` package-init time, so it
may import from :mod:`openmc.deplete.microxs` at module level.

.. versionadded:: 0.15.4
"""

from __future__ import annotations
from collections.abc import Collection, Sequence
import os
import shutil
from tempfile import TemporaryDirectory
from warnings import warn

import numpy as np

from openmc.checkvalue import PathLike
from openmc import StatePoint
from openmc.mgxs import GROUP_STRUCTURES, _canonical_group_structure_name
from openmc.data import REACTION_MT, open_pendf_library
import openmc
import openmc.lib
from openmc.mpi import comm
from ..chain import Chain
from ..microxs import DomainTypes, MicroXS, _SparseXSTable, _group_average
from .chain_check import (
    CONSISTENCY_ABS_FLOOR,
    _chain_lfs_reactions,
    _dedupe_base_reactions,
    _get_pendf_chain,
    _liso_from_gnds,
)
from .ground import (
    SILENCE_EPS,
    _balance_ground,
    _balance_remainder,
    _normalize_partial_binding,
    _partial_binding_veto,
    _partial_binding_veto_grouped,
    _silence_fill_ground,
)


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
        energies = GROUP_STRUCTURES[_canonical_group_structure_name(energies)]

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
    MF=3 total and is collected into one summary warning per build -- except
    **ground-by-balance**: if the ONLY missing demanded ``LFS`` is the ground
    (``missing == {0}``, the JEFF In113/In115 ``(n,n')`` class and every
    radioactive-products-only evaluation), the ground row is served implicitly as
    ``max(0, MF=3 total - Sigma(ALL library metastable partials))`` (see
    :func:`_balance_ground`) and the demanded metastable rows come from their
    partials as usual. That is a complete, self-consistent emission -- rows sum to
    the total -- so it is not a chain<->library disagreement; it is reported in a
    separate informational summary with the clamp count. A chain whose *qualified*
    reactions carry ``pendf_lfs=None`` (built without LFS recording) is a hard
    error (see :func:`_chain_lfs_reactions`).

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

    # (nuclide, base reaction, clamped points, total points, unit) for reactions
    # whose demanded ground pathway was served by balance (the library carries
    # every demanded metastable partial but no LFS=0 one). Reported once per
    # build as an informational summary -- these emissions are complete, not a
    # mismatch. The unit is 'pts' when the remainder was taken pointwise on the
    # union grid and 'groups' when it was taken group-wise, so the count in the
    # message always says what it counts.
    balanced: list[tuple[str, str, int, int, str]] = []

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

            # Ground-by-balance: the library carries every demanded metastable
            # partial but NO LFS=0 one (JEFF In113/In115-style (n,n') tapes, and
            # any radioactive-products-only evaluation, tabulate only the isomeric
            # levels), while the chain's fold demands a ground member. The ground
            # production is then implied rather than absent -- each event yields
            # exactly one final state -- so serve it as the remainder
            # ``max(0, total - Sigma(ALL library metastable partials))`` below.
            # The trigger is the EXACT match ``missing == {0}``: if a metastable is
            # missing too (``missing`` a strict superset), the chain and library
            # genuinely disagree and the reaction takes the MF=3-total fallback.
            # A synthesized-total MF=10-only reaction balances to an identically
            # zero ground (its total IS the partial sum), which is correct.
            ground_rx = lfs_reactions.get(0)
            balance_served = missing == {0} and ground_rx is not None

            if missing and not balance_served:
                # A demanded LFS is missing from the library and this is not the
                # ground-by-balance case (a metastable is missing, with or without
                # the ground): fall back to the MF=3 total and collect one honest
                # summary warning naming both LFS sets.
                mismatched.append((nuc, name, demanded_lfs, library_lfs))
                stage(nuc_idx, base_idx, name, total_g)
                continue

            # Emit exactly the demanded pathways: select the library partials whose
            # LFS the chain demands (extras dropped). Under ``balance_served`` the
            # library carries no LFS=0 partial by construction, so the ground is
            # never among these; its row is appended from the balance below.
            emit_pairs = [(lfs, izap) for lfs, izap in pathway_list
                          if lfs in demanded_lfs]
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
            # demanded ground plus >=1 demanded metastable; a balance-served
            # ground is EXCLUDED -- it is already a remainder, so filling it
            # would apply the same subtraction twice (and it has no LFS=0 partial
            # to supply the fill's domain anyway). ``missing`` is empty here
            # unless balance-served (a demanded-LFS-missing reaction fell back to
            # the total above), so a demanded ground means the library carries an
            # LFS=0 partial.
            if (not balance_served and 0 in demanded_lfs
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

            # Ground-by-balance row (trigger above): the demanded ground has no
            # MF=10 partial, so its row is the clamped remainder of the MF=3
            # total over ALL library metastable partials -- pointwise on the
            # union grid, group-wise for a grouped library (which carries no
            # pointwise data to rebin). It then joins the demanded metastables as
            # an ordinary pathway row: staged even when identically zero, exactly
            # like a partial.
            if balance_served:
                if not pathway_list:
                    # No MF=10 partials at all: nothing to subtract, so the
                    # remainder IS the MF=3 total, used verbatim (no union-grid
                    # round trip) to keep the row bit-identical to the total.
                    ground_g, n_clamped = total_g, 0
                    n_pts, n_unit = len(total_g), 'groups'
                elif grouped:
                    sum_meta_g = np.zeros(n_groups)
                    for lfs, izap in pathway_list:
                        sum_meta_g = sum_meta_g + pathway_xs_fn(nuc, mt, lfs,
                                                                izap)
                    ground_g, n_clamped = _balance_remainder(total_g, sum_meta_g)
                    n_pts, n_unit = n_groups, 'groups'
                else:
                    e_b, ground_b, n_clamped = _balance_ground(
                        pathways_fn, pathway_xs_fn, nuc, mt, energy, xs)
                    ground_g = _group_average(e_b, ground_b, energies)
                    n_pts, n_unit = len(e_b), 'pts'
                balanced.append((nuc, name, n_clamped, n_pts, n_unit))
                bound.append(ground_rx)
                partial_g.append(ground_g)

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

    # One informational summary per build for the reactions whose demanded ground
    # pathway was served by balance. This is NOT a chain<->library disagreement
    # (the emission is complete: the rows sum to the MF=3 total wherever nothing
    # was clamped), so it is deliberately separate from the mismatch warning
    # above; the clamp count exposes source data whose metastable partials
    # over-sum their total (GENDF's '[clamped N/M pts]' record); the unit is
    # 'pts' for a pointwise union grid, 'groups' for a grouped library.
    if balanced:
        summary = ', '.join(f'{n} {r} [clamped {c}/{t} {u}]'
                            for n, r, c, t, u in balanced)
        warn(f'PENDF ground-by-balance: {len(balanced)} reaction(s) demand a '
             'ground (LFS=0) pathway the library tabulates no MF=10 partial for '
             f"(the JEFF In113/In115 (n,n') class): {summary}. Their ground rows "
             'are the clamped remainder max(0, MF=3 total - Sigma(all library '
             'metastable partials)); the metastable rows are their partials, '
             'unchanged.')

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
