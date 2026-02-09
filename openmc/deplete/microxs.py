"""MicroXS module

A class for storing microscopic cross section data that can be used with the
IndependentOperator class for depletion.
"""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
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
from .gendf import GENDFLibrary, _PythonGENDFLibrary, _CppGENDFLibrary

# Build tuple of valid GENDF backend types for isinstance checks
_GENDF_TYPES = (_PythonGENDFLibrary,)
if _CppGENDFLibrary is not None:
    _GENDF_TYPES = (_PythonGENDFLibrary, _CppGENDFLibrary)

_valid_rxns = list(REACTIONS)
_valid_rxns.append('fission')
_valid_rxns.append('damage-energy')


# TODO: Replace with type statement when support is Python 3.12+
DomainTypes: TypeAlias = Union[
    Sequence[openmc.Material],
    Sequence[openmc.Cell],
    Sequence[openmc.Universe],
    openmc.MeshBase,
    openmc.Filter
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
    run_kwargs=None
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
    domains : list of openmc.Material or openmc.Cell or openmc.Universe, or openmc.MeshBase, or openmc.Filter
        Domains in which to tally reaction rates, or a spatial tally filter.
    nuclides : list of str
        Nuclides to get cross sections for. If not specified, all burnable
        nuclides from the depletion chain file are used.
    reactions : list of str
        Reactions to get cross sections for. If not specified, all neutron
        reactions listed in the depletion chain file are used.
    energies : iterable of float or str
        Energy group boundaries in [eV] or the name of the group structure.
        If left as None energies will default to [0.0, 100e6]
    reaction_rate_mode : {"direct", "flux"}, optional
        Indicate how reaction rates should be calculated. The "direct" method
        tallies reaction rates directly. The "flux" method tallies a multigroup
        flux spectrum and then collapses multigroup reaction rates after a
        transport solve (with an option to tally some reaction rates directly).
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

    # Set up the reaction rate and flux tallies
    if energies is None:
        energies = [0.0, 100.0e6]
    if isinstance(energies, str):
        energy_filter = openmc.EnergyFilter.from_group_structure(energies)
    else:
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

    flux_tally = openmc.Tally(name='MicroXS flux')
    flux_tally.filters = [domain_filter, energy_filter]
    flux_tally.scores = ['flux']
    model.tallies = [flux_tally]

    if reaction_rate_mode == 'direct':
        rr_tally = openmc.Tally(name='MicroXS RR')
        rr_tally.filters = [domain_filter, energy_filter]
        rr_tally.nuclides = nuclides
        rr_tally.multiply_density = False
        rr_tally.scores = reactions
        model.tallies.append(rr_tally)

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
            if reaction_rate_mode == 'direct':
                rr_tally = sp.tallies[rr_tally.id]
                rr_tally._read_results()
            flux_tally = sp.tallies[flux_tally.id]
            flux_tally._read_results()

    # Get flux values and make energy groups last dimension
    flux = flux_tally.get_reshaped_data()  # (domains, groups, 1, 1)
    flux = np.moveaxis(flux, 1, -1)  # (domains, 1, 1, groups)

    # Create list where each item corresponds to one domain
    fluxes = list(flux.squeeze((1, 2)))

    if reaction_rate_mode == 'direct':
        # Get reaction rates
        reaction_rates = rr_tally.get_reshaped_data()  # (domains, groups, nuclides, reactions)

        # Make energy groups last dimension
        reaction_rates = np.moveaxis(reaction_rates, 1, -1)  # (domains, nuclides, reactions, groups)

        # Divide RR by flux to get microscopic cross sections. The indexing
        # ensures that only non-zero flux values are used, and broadcasting is
        # applied to align the shapes of reaction_rates and flux for division.
        xs = np.empty_like(reaction_rates) # (domains, nuclides, reactions, groups)
        d, _, _, g = np.nonzero(flux)
        xs[d, ..., g] = reaction_rates[d, ..., g] / flux[d, :, :, g]

        # Create lists where each item corresponds to one domain
        micros = [MicroXS(xs_i, nuclides, reactions) for xs_i in xs]
    else:
        micros = [MicroXS.from_multigroup_flux(
            energies=energies,
            multigroup_flux=flux_i,
            chain_file=chain_file,
            nuclides=nuclides,
            reactions=reactions
        ) for flux_i in fluxes]

    # Reset tallies
    model.tallies = original_tallies

    # Package flux with energy information for isomeric branching
    fluxes_with_energy = [(f, energy_filter.values) for f in fluxes]

    return fluxes_with_energy, micros


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
) -> tuple[list[tuple[np.ndarray, Sequence[float]]], list[MicroXS]]:
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
    list of tuple of (numpy.ndarray, energies)
        For each domain, a tuple containing:
        - Flux in each group in [n-cm/src]
        - Energy boundaries used for the flux
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
    elif not isinstance(gendf_library, _GENDF_TYPES):
        raise TypeError(
            f"gendf_library must be a path or GENDFLibrary instance, "
            f"not {type(gendf_library)}")

    # Use GENDF library's energy structure
    energies = gendf_library.energy_bounds

    # Save any original tallies on the model
    original_tallies = model.tallies

    # Determine what reactions and nuclides are available
    chain = _get_chain(chain_file)
    if reactions is None:
        reactions = chain.reactions

    # Get nuclides from chain, filtered to those with GENDF data
    if not nuclides:
        available_nuclides = gendf_library.available_nuclides_set()
        nuclides = [nuc.name for nuc in chain.nuclides
                    if nuc.name in available_nuclides]

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

    # Generate microscopic cross sections using GENDF library
    micros = [MicroXS.from_multigroup_flux_with_gendf(
        multigroup_flux=flux_i,
        gendf_library=gendf_library,
        chain_file=chain_file,
        nuclides=nuclides,
        reactions=reactions
    ) for flux_i in fluxes]

    # Reset tallies
    model.tallies = original_tallies

    # Package flux with energy information for isomeric branching
    fluxes_with_energy = [(f, energy_filter.values) for f in fluxes]
    return fluxes_with_energy, micros


@dataclass
class _SparseXSTable:
    """Sparse storage of GENDF cross-sections for vectorized flux collapse.

    Stores only non-zero (nuclide, reaction) pairs. xs_matrix has shape
    (nnz, n_groups); nuc_indices and rxn_indices map rows to positions
    in the dense (n_nuclides, n_reactions) result.
    """
    nuclides: list[str]
    reactions: list[str]
    n_groups: int
    xs_matrix: np.ndarray
    nuc_indices: np.ndarray
    rxn_indices: np.ndarray

    def collapse(self, phi_norm: np.ndarray) -> np.ndarray:
        """Collapse to one-group XS. phi_norm must sum to 1."""
        collapsed_sparse = self.xs_matrix @ phi_norm
        result = np.zeros((len(self.nuclides), len(self.reactions)))
        result[self.nuc_indices, self.rxn_indices] = collapsed_sparse
        return result


def _build_sparse_xs_table(
    gendf_library,
    nuclides: list[str],
    reactions: list[str],
    mts: list[int]
) -> _SparseXSTable:
    """Build a sparse XS table from a GENDF library.

    Parameters
    ----------
    gendf_library : _PythonGENDFLibrary or _CppGENDFLibrary
        GENDF library instance
    nuclides : list of str
        Nuclide names to include
    reactions : list of str
        Reaction names (parallel to mts)
    mts : list of int
        MT numbers corresponding to reactions

    Returns
    -------
    _SparseXSTable
        Sparse table ready for vectorized collapse
    """
    if len(reactions) != len(mts):
        raise ValueError(
            f"reactions ({len(reactions)}) and mts ({len(mts)}) "
            f"must have same length")

    n_groups = gendf_library.n_groups
    rows = []
    nuc_idx_list = []
    rxn_idx_list = []

    mt_to_rxn_idx = {mt: i for i, mt in enumerate(mts)}
    is_cpp = (_CppGENDFLibrary is not None
              and isinstance(gendf_library, _CppGENDFLibrary))

    for nuc_idx, nuc in enumerate(nuclides):
        if is_cpp:
            all_xs = gendf_library.get_all_xs(nuc, mts=mts)
        else:
            all_xs = gendf_library.get_all_xs(nuc)
        for mt, xs_arr in all_xs.items():
            if mt not in mt_to_rxn_idx:
                continue
            rows.append(xs_arr)
            nuc_idx_list.append(nuc_idx)
            rxn_idx_list.append(mt_to_rxn_idx[mt])

    if rows:
        xs_matrix = np.vstack(rows)
    else:
        xs_matrix = np.empty((0, n_groups))

    return _SparseXSTable(
        nuclides=nuclides,
        reactions=reactions,
        n_groups=n_groups,
        xs_matrix=xs_matrix,
        nuc_indices=np.array(nuc_idx_list, dtype=np.int32),
        rxn_indices=np.array(rxn_idx_list, dtype=np.int32),
    )


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
        List of reactions. All reactions must match those in
        :data:`openmc.deplete.chain.REACTIONS`

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
        for reaction in reactions:
            check_value('reactions', reaction, _valid_rxns)

        self.data = data
        self.nuclides = nuclides
        self.reactions = reactions
        self._index_nuc = {nuc: i for i, nuc in enumerate(nuclides)}
        self._index_rx = {rx: i for i, rx in enumerate(reactions)}

    @classmethod
    def from_multigroup_flux(
        cls,
        energies: Sequence[float] | str,
        multigroup_flux: Sequence[float],
        chain_file: PathLike | None = None,
        temperature: float = 293.6,
        nuclides: Sequence[str] | None = None,
        reactions: Sequence[str] | None = None,
        **init_kwargs: dict,
    ) -> MicroXS:
        """Generated microscopic cross sections from a known flux.

        The size of the MicroXS matrix depends on the chain file and cross
        sections available. MicroXS entry will be 0 if the nuclide cross section
        is not found.

        It is recommended to make repeated calls to this method within a context
        manager using the :class:`openmc.lib.TemporarySession` class to avoid
        re-initializing OpenMC and loading cross sections each time.

        .. versionadded:: 0.15.0

        Parameters
        ----------
        energies : iterable of float or str
            Energy group boundaries in [eV] or the name of the group structure
        multigroup_flux : iterable of float
            Energy-dependent multigroup flux values
        chain_file : PathLike or Chain, optional
            Path to the depletion chain XML file or an instance of
            openmc.deplete.Chain. Defaults to ``openmc.config['chain_file']``.
        temperature : int, optional
            Temperature for cross section evaluation in [K].
        nuclides : list of str, optional
            Nuclides to get cross sections for. If not specified, all burnable
            nuclides from the depletion chain file are used.
        reactions : list of str, optional
            Reactions to get cross sections for. If not specified, all neutron
            reactions listed in the depletion chain file are used.
        **init_kwargs : dict
            Keyword arguments passed to :func:`openmc.lib.init`

        Returns
        -------
        MicroXS
        """

        check_type("temperature", temperature, (int, float))
        # if energy is string then use group structure of that name
        if isinstance(energies, str):
            energies = GROUP_STRUCTURES[energies]
        else:
            # if user inputs energies check they are ascending (low to high) as
            # some depletion codes use high energy to low energy.
            if not np.all(np.diff(energies) > 0):
                raise ValueError('Energy group boundaries must be in ascending order')

        # check dimension consistency
        if len(multigroup_flux) != len(energies) - 1:
            raise ValueError('Length of flux array should be len(energies)-1')

        chain = _get_chain(chain_file)
        cross_sections = _find_cross_sections(model=None)
        nuclides_with_data = _get_nuclides_with_data(cross_sections)

        # If no nuclides were specified, default to all nuclides from the chain
        if not nuclides:
            nuclides = chain.nuclides
            nuclides = [nuc.name for nuc in nuclides]

        # Get reaction MT values. If no reactions specified, default to the
        # reactions available in the chain file
        if reactions is None:
            reactions = chain.reactions
        mts = [REACTION_MT[name] for name in reactions]

        # Create 3D array for microscopic cross sections
        microxs_arr = np.zeros((len(nuclides), len(mts), 1))

        # If flux is zero, safely return zero cross sections
        multigroup_flux = np.array(multigroup_flux)
        if (flux_sum := multigroup_flux.sum()) == 0.0:
            return cls(microxs_arr, nuclides, reactions)

        # Normalize multigroup flux
        multigroup_flux /= flux_sum

        # Compute microscopic cross sections within a temporary session
        with openmc.lib.TemporarySession(**init_kwargs):
            # For each nuclide and reaction, compute the flux-averaged xs
            for nuc_index, nuc in enumerate(nuclides):
                if nuc not in nuclides_with_data:
                    continue
                lib_nuc = openmc.lib.load_nuclide(nuc)
                for mt_index, mt in enumerate(mts):
                    microxs_arr[nuc_index, mt_index, 0] = lib_nuc.collapse_rate(
                        mt, temperature, energies, multigroup_flux
                    )

        return cls(microxs_arr, nuclides, reactions)

    @classmethod
    def from_multigroup_flux_with_gendf(
        cls,
        multigroup_flux: Sequence[float],
        gendf_library: PathLike | 'openmc.deplete.gendf.GENDFLibrary',
        chain_file: PathLike | Chain | None = None,
        nuclides: Sequence[str] | None = None,
        reactions: Sequence[str] | None = None,
    ) -> MicroXS:
        """Generate one group microscopic cross sections by collapsing
        multigroup flux with multigroup cross-sections from a GENDF library.

        Uses pre-processed group-averaged cross-sections from a GENDF library 
        instead of calculating them from continuous-energy data, directly 
        retrieving cross-sections from a GENDF library.

        .. versionadded:: 0.15.4

        Parameters
        ----------
        multigroup_flux : iterable of float
            Energy-dependent multigroup flux values for each energy group
        gendf_library : path-like or GENDFLibrary
            Path to GENDF library directory or GENDFLibrary instance. If a path is
            provided, the energy structure is auto-detected from the GENDF library.
        chain_file : PathLike or Chain, optional
            Path to the depletion chain XML file or an instance of
            openmc.deplete.Chain. Defaults to ``openmc.config['chain_file']``.
        nuclides : list of str, optional
            Nuclides to get cross sections for. If not specified, all burnable
            nuclides from the depletion chain file that are available in the
            GENDF library are used.
        reactions : list of str, optional
            Reactions to get cross sections for. If not specified, all neutron
            reactions listed in the depletion chain file are used.

        Returns
        -------
        MicroXS
            Microscopic cross-section data object

        Notes
        -----
        The collapsed one-group cross-section for each nuclide and reaction is calculated as:

        .. math::
            \\langle\\sigma\\rangle = \\sum_g \\sigma_g \\phi_g / \\sum_g \\phi_g

        where σ_g is the group-averaged cross-section from GENDF and φ_g is the
        multigroup flux.

        """
        # Handle GENDF library input
        if isinstance(gendf_library, (str, Path)):
            gendf_library = GENDFLibrary(gendf_library)
        elif not isinstance(gendf_library, _GENDF_TYPES):
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

        # Create 3D array for microscopic cross sections
        microxs_arr = np.zeros((len(nuclides), len(mts), 1))

        # If flux is zero, safely return zero cross sections
        multigroup_flux = np.array(multigroup_flux)
        if (flux_sum := multigroup_flux.sum()) == 0.0:
            return cls(microxs_arr, nuclides, reactions)

        # Normalize multigroup flux
        multigroup_flux /= flux_sum

        # For each nuclide and reaction, get GENDF XS and compute flux-weighted value
        for nuc_index, nuc in enumerate(nuclides):
            for mt_index, mt in enumerate(mts):
                try:
                    # Get group-averaged cross-section from GENDF
                    xs_group = gendf_library.get_xs(nuc, mt, energies)

                    # Collapse cross-section
                    # Since flux is normalized: <σ> = Σ σ_g * φ_g
                    microxs_arr[nuc_index, mt_index, 0] = np.sum(xs_group * multigroup_flux)

                except (KeyError, openmc.exceptions.OpenMCError):
                    # Reaction not available in GENDF for this nuclide
                    # Leave as zero (e.g., H1 doesn't have MT=16 n,2n)
                    pass

        return cls(microxs_arr, nuclides, reactions)

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
