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
from collections.abc import Sequence
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
from ..microxs import DomainTypes, Flux, MicroXS, _build_sparse_xs_table
from .library import GENDFLibrary


def get_gendfxs_and_flux(
    model: openmc.Model,
    domains: DomainTypes,
    gendf_library: PathLike | 'openmc.deplete.gendf.GENDFLibrary',
    nuclides: Sequence[str] | None = None,
    reactions: Sequence[str] | None = None,
    chain_file: PathLike | Chain | None = None,
    path_statepoint: PathLike | None = None,
    path_input: PathLike | None = None,
    run_kwargs=None
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
        Domains in which to tally reaction rates, or a spatial tally filter.
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
    if not nuclides:
        nuclides = [nuc.name for nuc in chain.nuclides
                    if nuc.name in available_nuclides]
    else:
        nuclides = [nuc for nuc in nuclides if nuc in available_nuclides]

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
    model.tallies = [flux_tally]

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
            flux_tally = sp.tallies[flux_tally.id]
            flux_tally._read_results()

    # Get flux values and make energy groups last dimension
    flux = flux_tally.get_reshaped_data()  # (domains, groups, 1, 1)
    flux = np.moveaxis(flux, 1, -1)  # (domains, 1, 1, groups)

    # Create list where each item corresponds to one domain
    fluxes = list(flux.squeeze((1, 2)))

    # Build sparse XS table once (GENDF XS are domain-independent)
    mts = [REACTION_MT[name] for name in reactions]
    table = _build_sparse_xs_table(gendf_library, nuclides, reactions, mts)

    # Collapse per domain
    micros = []
    for flux_i in fluxes:
        flux_arr = np.asarray(flux_i, dtype=float)
        flux_sum = flux_arr.sum()
        if flux_sum == 0.0:
            micros.append(MicroXS(
                np.zeros((len(nuclides), len(reactions), 1)),
                nuclides, reactions))
            continue
        collapsed = table.collapse(flux_arr / flux_sum)
        micros.append(MicroXS(collapsed[:, :, np.newaxis],
                               nuclides, reactions))

    # Reset tallies
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
    multigroup_flux: Sequence[float],
    gendf_library: PathLike | 'openmc.deplete.gendf.GENDFLibrary',
    chain_file: PathLike | Chain | None = None,
    nuclides: Sequence[str] | None = None,
    reactions: Sequence[str] | None = None,
) -> MicroXS:
    """Implementation of :meth:`MicroXS.from_multigroup_flux_with_gendf`."""
    # Handle GENDF library input
    if isinstance(gendf_library, (str, Path)):
        gendf_library = GENDFLibrary(gendf_library)
    elif not isinstance(gendf_library, _microxs._GENDF_TYPES):
        raise TypeError(
            f"gendf_library must be a path or GENDFLibrary instance, "
            f"not {type(gendf_library)}")

    # Use GENDF library's energy structure
    energies = gendf_library.energy_bounds

    # Check dimension consistency
    if len(multigroup_flux) != len(energies) - 1:
        raise ValueError('Length of flux array should be len(energies)-1')


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
        reactions = chain.reactions
    mts = [REACTION_MT[name] for name in reactions]

    # If flux is zero, safely return zero cross sections
    multigroup_flux = np.asarray(multigroup_flux, dtype=float)
    if (flux_sum := multigroup_flux.sum()) == 0.0:
        return cls(np.zeros((len(nuclides), len(mts), 1)),
                   nuclides, reactions)

    # Build sparse table and collapse with normalized flux
    table = _build_sparse_xs_table(gendf_library, nuclides, reactions, mts)
    collapsed = table.collapse(multigroup_flux / flux_sum)

    return cls(collapsed[:, :, np.newaxis], nuclides, reactions)
