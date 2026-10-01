"""GENDF flux-collapse entry points.

Holds the GENDF counterparts of the continuous-energy collapse path that used
to live in :mod:`openmc.deplete.microxs`: the transport-plus-collapse driver
:func:`get_gendfxs_and_flux`, the MT=4 substitution applied by
:func:`~openmc.deplete.get_microxs_and_flux`, and the implementation behind
:meth:`openmc.deplete.MicroXS.from_multigroup_flux_with_gendf`.

This submodule is never imported at ``openmc.deplete.gendf`` package-init time,
so it may import from :mod:`openmc.deplete.microxs` at module level.

.. versionadded:: 0.15.4
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from openmc.checkvalue import PathLike
from openmc import StatePoint
from openmc.data import REACTION_MT
import openmc
import openmc.lib
from openmc.mpi import comm
from ..chain import Chain, _get_chain
from .. import microxs as _microxs
from ..microxs import (DomainTypes, Flux, MicroXS, _collapse_gendf_streaming,
                       _normalize_flux_batch)
from .library import GENDFLibrary
from .calendf import _CalendfRowScaler


def _gendf_dilution_material(domain) -> openmc.Material:
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


def get_gendfxs_and_flux(
    model: openmc.Model,
    domains: DomainTypes,
    gendf_library: PathLike | 'openmc.deplete.gendf.GENDFLibrary',
    nuclides: Sequence[str] | None = None,
    reactions: Sequence[str] | None = None,
    chain_file: PathLike | Chain | None = None,
    path_statepoint: PathLike | None = None,
    path_input: PathLike | None = None,
    run_kwargs=None,
    *,
    urr_material_dilution: bool = False,
    calendf_path: PathLike | None = None,
    mat_ssf_nuclides: Sequence[str] | None = None,
) -> tuple[list[Flux], list[MicroXS]]:
    """Generate microscopic cross sections and fluxes for multiple domains using GENDF library.

    This function runs a neutron transport solve to obtain the flux in the
    specified domains and computes collapsed one-group microscopic cross sections
    from said multi-group flux and a multi-group GENDF cross section library. 
    It is similar to :func:`get_microxs_and_flux` but uses pre-processed 
    group-averaged cross-sections instead of calculating them from continuous-energy data.

    .. versionadded:: 0.15.4

    Parameters
    ----------
    model : openmc.Model
        OpenMC model object. Must contain geometry, materials, and settings.
    domains : list of openmc.Material or openmc.Cell or openmc.Universe, or openmc.MeshBase, or openmc.Filter
        Domains in which to tally reaction rates, or a spatial tally filter. When
        ``urr_material_dilution=True`` every domain must resolve to a single
        composition -- all :class:`openmc.Material`, or all
        :class:`openmc.Cell` filled with a single Material; see that argument.
    gendf_library : path-like or GENDFLibrary
        Path to GENDF library directory or GENDFLibrary instance. If a path is
        provided, a GENDFLibrary object will be created.
    nuclides : list of str, optional
        Nuclides to get cross sections for. If not specified, all burnable
        nuclides from the depletion chain file that are available in the
        GENDF library are used.
    reactions : list of str, optional
        Reactions to get cross sections for. If not specified, all neutron
        reactions listed in the depletion chain file are used.
    chain_file : PathLike or Chain, optional
        Path to the depletion chain XML file or an instance of
        openmc.deplete.Chain. Defaults to ``openmc.config['chain_file']``.
    path_statepoint : path-like, optional
        Path to write the statepoint file from the neutron transport solve to.
        By default, the statepoint file is written to a temporary directory and
        is not kept.
    path_input : path-like, optional
        Path to write the model XML file from the neutron transport solve to.
        By default, the model XML file is written to a temporary directory and
        not kept.
    run_kwargs : dict, optional
        Keyword arguments passed to :meth:`openmc.Model.run`
    urr_material_dilution : bool, optional
        Apply an unresolved-resonance-range (URR) self-shielding correction by
        **material dilution** before the collapse, per domain. ``True`` uses
        **each domain's own composition** automatically (via
        :meth:`~openmc.Material.get_nuclide_atom_densities`) as the
        ``sigma0_mat`` background, so every domain must resolve to a single
        composition -- an :class:`openmc.Material` (shields with itself) or an
        :class:`openmc.Cell` filled with a single Material (shields with its
        fill; the tally domain stays the Cell, so the flux is per-cell). All
        domains must be of one kind (all Materials or all Cells). Universes,
        lattices, meshes, tally filters and cells with
        void/universe/lattice/distributed fills have no single composition and
        raise ``ValueError``. This is the exact analogue of FISPACT-II
        ``PROBTABLE multxs=1``; the CALENDF tables span the resolved range as
        well as the URR, so the correction acts across both. Requires
        ``calendf_path``. Every input check, including the diluter lookup in
        the GENDF library, happens before the transport solve. ``False``
        (default) leaves the collapse unchanged (infinite dilution,
        byte-identical to prior behaviour). This wrapper takes a plain bool
        because it owns the domains; to supply an explicit composition
        (:class:`openmc.Material` or ``{nuclide: density}`` mapping) call
        :meth:`MicroXS.from_multigroup_flux_with_gendf` directly.
        Keyword-only. The Bondarenko fold uses the temperature baked into the
        chosen CALENDF ``tp-...`` directory (``calendf_path``); no cross-check
        against the material or transport temperature is performed, so ensure the
        CALENDF temperature matches the conditions modelled.
    calendf_path : path-like, optional
        Directory of per-temperature CALENDF ``<Nuclide>-<T>.tpe`` probability
        tables (e.g. ``.../tp-709-294``). Required when
        ``urr_material_dilution`` is enabled (the GENDF path takes an explicit
        CALENDF path); it must be an existing directory. Keyword-only.
    mat_ssf_nuclides : list of str, optional
        Restrict the material-dilution correction to this subset of nuclides.
        ``None`` (default) uses the built-in flagged set (bulk self-shielders:
        W, Ta, Re, Hf, Os isotopes and U/Pu). When given it is intersected with
        the built-in set (restrict-only). Keyword-only.

    Returns
    -------
    list of openmc.deplete.Flux
        Flux in each group in [n-cm/src] for each domain. Each :class:`Flux`
        is a :class:`numpy.ndarray` subclass that also carries the energy group
        boundaries in its ``energy_bounds`` attribute.
    list of MicroXS
        Cross section data in [b] for each domain, retrieved from GENDF library

    See Also
    --------
    get_microxs_and_flux : Similar function using continuous-energy cross-sections
    openmc.deplete.IndependentOperator
    openmc.deplete.gendf.GENDFLibrary

    """
    # Handle GENDF library input
    if isinstance(gendf_library, (str, Path)):
        gendf_library = GENDFLibrary(gendf_library)
    elif not isinstance(gendf_library, _microxs._GENDF_TYPES):
        raise TypeError(
            f"gendf_library must be a path or GENDFLibrary instance, "
            f"not {type(gendf_library)}")

    # --- URR material-dilution validation (all before the expensive
    # transport solve). This wrapper owns the domains, so the toggle is a
    # plain bool -- True means "each domain's own composition". ---
    if not isinstance(urr_material_dilution, bool):
        raise ValueError(
            "urr_material_dilution must be a bool for get_gendfxs_and_flux(); "
            "True uses each domain's own composition automatically. To supply "
            "an explicit composition (openmc.Material or {nuclide: density} "
            "mapping), call MicroXS.from_multigroup_flux_with_gendf directly.")
    # ``urr_material_dilution=True`` shields each domain with its own
    # composition. Every domain must therefore resolve to a single Material: a
    # Material shields with itself, an openmc.Cell with its single-Material fill
    # (the tally domain stays the Cell, so the flux is per-cell). All domains
    # must be of one kind, because the flux tally filter is built from one
    # domain kind. Meshes, tally filters, universes and lattices (and cells
    # with void/universe/lattice/distributed fills) have no single composition
    # and raise. Resolve the per-domain shielding compositions up front so a
    # bad domain fails before the expensive model.run.
    dilution_materials = None
    if urr_material_dilution:
        # The correction needs the CALENDF probability tables; the GENDF path
        # takes an explicit CALENDF path.
        if calendf_path is None:
            raise ValueError(
                "urr_material_dilution requires `calendf_path` (the directory "
                "of CALENDF .tpe probability tables); the GENDF path uses an "
                "explicit CALENDF path")
        if not Path(calendf_path).is_dir():
            raise ValueError(
                f"calendf_path {calendf_path!s} is not a directory of CALENDF "
                f".tpe tables")
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
        dilution_materials = [_gendf_dilution_material(d) for d in domains]
        if not (all(isinstance(d, openmc.Material) for d in domains)
                or all(isinstance(d, openmc.Cell) for d in domains)):
            raise ValueError(
                "urr_material_dilution=True requires all domains to be "
                "openmc.Material or all to be material-filled openmc.Cell; "
                "the flux tally filter is built from one domain kind")

    # Use GENDF library's energy structure
    energies = gendf_library.energy_bounds

    # Save any original tallies on the model
    original_tallies = list(model.tallies)

    # Determine what reactions and nuclides are available
    chain = _get_chain(chain_file)
    if reactions is None:
        reactions = chain.reactions

    # Get nuclides from chain, filtered to those with GENDF data
    available_nuclides = gendf_library.available_nuclides_set()
    requested_nuclides = nuclides
    if not nuclides:
        nuclides = [nuc.name for nuc in chain.nuclides
                    if nuc.name in available_nuclides]
    else:
        nuclides = [nuc for nuc in nuclides if nuc in available_nuclides]

    mts = [REACTION_MT[name] for name in reactions]

    # URR material dilution: one self-shielding scaler per distinct material,
    # built before the transport solve because the factors do not depend on
    # the flux. Any bad input -- a malformed mat_ssf_nuclides, or a diluter
    # absent from the GENDF library in a composition that will be folded --
    # therefore raises before model.run. The parsed .tpe probability tables
    # and the diluter group totals are shared between the scalers of this call.
    scalers = {}
    if urr_material_dilution:
        tpe_cache, totals_cache = {}, {}
        for mat in dilution_materials:
            # Keyed by Material.id, as MaterialFilter identifies materials
            # (the C++ side rejects duplicate material ids).
            if mat.id in scalers:
                continue
            scaler = scalers[mat.id] = _CalendfRowScaler(
                gendf_library, calendf_path,
                mat.get_nuclide_atom_densities(),
                mat_ssf_nuclides, tpe_cache=tpe_cache,
                totals_cache=totals_cache)
            scaler.check_diluters(nuclides, mts)

    # Set up the flux tallies
    energy_filter = openmc.EnergyFilter(energies)

    if isinstance(domains, openmc.Filter):
        domain_filter = domains
    elif isinstance(domains, openmc.MeshBase):
        domain_filter = openmc.MeshFilter(domains)
    elif isinstance(domains[0], openmc.Material):
        domain_filter = openmc.MaterialFilter(domains)
    elif isinstance(domains[0], openmc.Cell):
        domain_filter = openmc.CellFilter(domains)
    elif isinstance(domains[0], openmc.Universe):
        domain_filter = openmc.UniverseFilter(domains)
    else:
        raise ValueError(f"Unsupported domain type: {type(domains[0])}")

    flux_tally = openmc.Tally(name='GENDF flux')
    flux_tally.filters = [domain_filter, energy_filter]
    flux_tally.scores = ['flux']
    try:
        model.tallies = [flux_tally]

        if openmc.lib.is_initialized:
            openmc.lib.finalize()

            if comm.rank == 0:
                model.export_to_model_xml()
            comm.barrier()
            # Reinitialize with tallies
            openmc.lib.init(intracomm=comm)

        with TemporaryDirectory() as temp_dir:
            # Indicate to run in temporary directory unless being executed
            # through openmc.lib, in which case we don't need to specify the cwd
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
                flux_tally = sp.tallies[flux_tally.id]
                flux_tally._read_results()

        # Get flux values and make energy groups last dimension
        flux = flux_tally.get_reshaped_data()  # (domains, groups, 1, 1)
        flux = np.moveaxis(flux, 1, -1)  # (domains, 1, 1, groups)

        # Create list where each item corresponds to one domain
        fluxes = list(flux.squeeze((1, 2)))

        n_groups = len(energies) - 1
        if urr_material_dilution:
            # URR material dilution ON: one collapse per domain, because one
            # set of self-shielding factors serves one composition only. Each
            # domain is shielded by its own Material (a Cell by its fill,
            # resolved into ``dilution_materials`` above), through the scaler
            # built for that material before the transport solve, so a
            # material shared by several domains is folded once. The global
            # flux index i appears in any normaliser error, and a zero flux
            # stays all-zero through the normaliser and yields an all-zero
            # MicroXS. Analogue of FISPACT-II PROBTABLE multxs=1.
            micros = []
            for i, (mat, flux_i) in enumerate(zip(dilution_materials, fluxes)):
                phi = _normalize_flux_batch([flux_i], i, n_groups)
                collapsed = _collapse_gendf_streaming(
                    gendf_library, nuclides, reactions, mts, phi,
                    scaler=scalers[mat.id])[0]
                micros.append(MicroXS(collapsed[:, :, np.newaxis],
                                      nuclides, reactions))
        else:
            # Collapse every domain flux in one batched call, one streaming
            # pass per chunk of fluxes. A zero-sum flux yields an all-zero
            # MicroXS. The method applies the same GENDF filter to the
            # requested nuclides; an already filtered empty list would be
            # read as "not given" and refilled from the chain.
            micros = MicroXS.from_multigroup_flux_with_gendf(
                fluxes, gendf_library, chain_file=chain,
                nuclides=requested_nuclides, reactions=reactions)
            micros = [micros] if isinstance(micros, MicroXS) else list(micros)
    finally:
        # Reset tallies, also when the solve or the collapse raises
        model.tallies = original_tallies

    # Return Flux arrays carrying the energy grid for isomeric branching
    fluxes = [Flux(f, energy_bounds=energy_filter.values) for f in fluxes]
    return fluxes, micros


def _apply_gendf_mt4_fallback(micros, fluxes, gendf_library,
                              nuclides_with_data):
    """Substitute GENDF MT=4 data into the (n,n') column of each MicroXS.

    CE HDF5 libraries typically lack the lumped MT=4 reaction, so the (n,n')
    column is silently zero. Modifies micros in place, restricted to nuclides
    present in BOTH the CE library (nuclides_with_data) and the GENDF library
    so a GENDF-only nuclide is never activated through (n,n') alone.
    """
    if "(n,n')" not in micros[0].reactions:
        return

    available = gendf_library.available_nuclides_set()
    sigma4 = {}
    for nuc in micros[0].nuclides:
        if nuc in available and nuc in nuclides_with_data:
            xs = gendf_library.get_all_xs(nuc, mts=[4])
            if 4 in xs:
                sigma4[nuc] = xs[4]

    if comm.rank == 0:
        print(f" (n,n') cross sections computed from GENDF MT=4 "
              f"(not CE data) for {len(sigma4)} nuclides")

    n_groups = gendf_library.n_groups
    for micro, flux_i in zip(micros, fluxes):
        i_nn = micro.reactions.index("(n,n')")
        for nuc, sigma4_g in sigma4.items():
            i_nuc = micro._index_nuc.get(nuc)
            if i_nuc is None:
                continue
            if micro.data.shape[2] == n_groups:
                # direct mode: groups are GENDF-aligned (validated upstream)
                micro.data[i_nuc, i_nn, :] = sigma4_g
            elif micro.data.shape[2] == 1:
                flux_sum = flux_i.sum()
                if flux_sum > 0.0:
                    micro.data[i_nuc, i_nn, 0] = sigma4_g @ flux_i / flux_sum


def _from_multigroup_flux_with_gendf(
    cls,
    multigroup_flux: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    gendf_library: PathLike | 'openmc.deplete.gendf.GENDFLibrary',
    chain_file: PathLike | Chain | None = None,
    nuclides: Sequence[str] | None = None,
    reactions: Sequence[str] | None = None,
    *,
    urr_material_dilution: openmc.Material | Mapping[str, float] | bool | None = False,
    calendf_path: PathLike | None = None,
    mat_ssf_nuclides: Sequence[str] | None = None,
) -> MicroXS | list[MicroXS]:
    """Implementation of :meth:`MicroXS.from_multigroup_flux_with_gendf`.

    URR material dilution is applied row by row in the streaming collapse
    through a :class:`~openmc.deplete.gendf.calendf._CalendfRowScaler`.
    """
    # Fuse the URR dilution toggle with its composition: normalize
    # ``urr_material_dilution`` to a local ``densities`` mapping (or None
    # when off) here, before any expensive work, so the impossible "on but
    # no composition" state cannot be represented.
    if urr_material_dilution is False or urr_material_dilution is None:
        densities = None
    elif urr_material_dilution is True:
        raise ValueError(
            'urr_material_dilution=True is under-specified: the URR '
            'self-shielding sigma0_mat background needs a composition. Pass '
            'the openmc.Material being depleted, or a {nuclide: '
            'density-or-fraction} mapping, instead of True')
    elif isinstance(urr_material_dilution, openmc.Material):
        densities = urr_material_dilution.get_nuclide_atom_densities()
    elif isinstance(urr_material_dilution, Mapping):
        if not urr_material_dilution:
            raise ValueError(
                'urr_material_dilution mapping is empty: with no diluters '
                'the sigma0_mat background is zero and every flagged nuclide '
                'silently degrades to f=1. Pass the depleted composition, '
                'or omit the argument to disable the correction')
        densities = urr_material_dilution
    else:
        raise ValueError(
            'urr_material_dilution must be an openmc.Material, a {nuclide: '
            'density-or-fraction} mapping, or False; got '
            f'{type(urr_material_dilution).__name__}')

    # The correction needs the CALENDF probability tables; the GENDF path
    # takes an explicit CALENDF path.
    if densities is not None and calendf_path is None:
        raise ValueError(
            "urr_material_dilution requires `calendf_path` (the directory "
            "of CALENDF .tpe probability tables); the GENDF path uses an "
            "explicit CALENDF path")
    if densities is not None and not Path(calendf_path).is_dir():
        raise ValueError(
            f"calendf_path {calendf_path!s} is not a directory of CALENDF "
            f".tpe tables")

    # Handle GENDF library input
    if isinstance(gendf_library, (str, Path)):
        gendf_library = GENDFLibrary(gendf_library)
    elif not isinstance(gendf_library, _microxs._GENDF_TYPES):
        raise TypeError(
            f"gendf_library must be a path or GENDFLibrary instance, "
            f"not {type(gendf_library)}")

    # Use GENDF library's energy structure
    n_groups = len(gendf_library.energy_bounds) - 1

    # One flux or a batch, told apart without copying the batch: an ndarray by
    # its ndim, any other sequence by its first element. An input with no
    # length (a bare scalar) is rejected like any other bad shape.
    if isinstance(multigroup_flux, np.ndarray):
        ndim = multigroup_flux.ndim
    elif not hasattr(multigroup_flux, '__len__'):
        ndim = 0
    elif len(multigroup_flux) == 0:
        ndim = 2
    else:
        ndim = np.ndim(multigroup_flux[0]) + 1
    if ndim not in (1, 2):
        raise ValueError('multigroup_flux must be 1-D or 2-D')
    single = ndim == 1
    fluxes = [multigroup_flux] if single else multigroup_flux
    if len(fluxes) == 0:
        return []
    if isinstance(fluxes, np.ndarray) and fluxes.shape[1] != n_groups:
        raise ValueError(f'Multigroup flux 0 must have length {n_groups}')

    chain = _get_chain(chain_file)
    # get available GENDF nuclides
    available_nuclides = gendf_library.available_nuclides_set()

    # If no nuclides were specified, default to all nuclides from the chain
    # regardless, filter to those with GENDF data
    if not nuclides:
        nuclides = [nuc.name for nuc in chain.nuclides
                    if nuc.name in available_nuclides]
    else:
        # Filter user-provided nuclides to those with GENDF data
        nuclides = [nuc for nuc in nuclides if nuc in available_nuclides]


    # Get reaction MT values
    if reactions is None:
        reactions = list(chain.reactions)
    else:
        reactions = list(reactions)
    mts = [REACTION_MT[name] for name in reactions]

    # URR material dilution: one scaler per composition, applied to each row
    # as the streaming collapse stages it
    scaler = None if densities is None else _CalendfRowScaler(
        gendf_library, calendf_path, densities, mat_ssf_nuclides)

    # Normalize each chunk of fluxes (a zero flux stays all-zero, giving zero
    # cross sections) and collapse the GENDF rows in one streaming pass per
    # chunk. The chunk size is read from the module at call time.
    chunk = _microxs._COLLAPSE_CHUNK_SIZE
    micros = []
    for start in range(0, len(fluxes), chunk):
        rows = fluxes[start:start + chunk]
        if not isinstance(rows, np.ndarray):
            for i, f in enumerate(rows):
                if np.ndim(f) != 1 or len(f) != n_groups:
                    raise ValueError(
                        f'Multigroup flux {start + i} must have length '
                        f'{n_groups}')
        phi = _normalize_flux_batch(rows, start, n_groups)
        collapsed = _collapse_gendf_streaming(
            gendf_library, nuclides, reactions, mts, phi, scaler=scaler)
        micros.extend(cls(c[:, :, np.newaxis], nuclides, reactions)
                      for c in collapsed)

    return micros[0] if single else micros
