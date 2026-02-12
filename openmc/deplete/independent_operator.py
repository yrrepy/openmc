"""Transport-independent transport operator for depletion.

This module implements a transport operator that runs independently of any
transport solver by using user-provided multigroup fluxes and cross sections.

"""

from __future__ import annotations
from collections.abc import Iterable
import copy
import warnings

import numpy as np
from uncertainties import ufloat

import openmc
from openmc.checkvalue import check_type
from openmc.mpi import comm
from .abc import ReactionRateHelper, OperatorResult
from .openmc_operator import OpenMCOperator
from .pool import _distribute
from .microxs import MicroXS, read_local_microxs_hdf5
from .results import Results
from .helpers import (ChainFissionHelper, ConstantFissionYieldHelper, SourceRateHelper,
                      IsomericBranchingHelper)


class IndependentOperator(OpenMCOperator):
    """Transport-independent transport operator based on multigroup data.

    Instances of this class can be used to perform depletion using multigroup
    cross sections and multigroup fluxes. Normally, a user needn't call methods
    of this class directly. Instead, an instance of this class is passed to an
    integrator class, such as :class:`openmc.deplete.CECMIntegrator`.

    Note that passing an empty :class:`~openmc.deplete.MicroXS` instance to the
    ``micro_xs`` argument allows a decay-only calculation to be run.

    .. versionadded:: 0.13.1

    .. versionchanged:: 0.14.0
        Arguments updated to include list of fluxes and microscopic cross
        sections.

    Parameters
    ----------
    materials : iterable of openmc.Material
        Materials to deplete.
    fluxes : list of numpy.ndarray
        Flux in each group in [n-cm/src] for each domain
    micros : list of MicroXS
        Cross sections in [b] for each domain. If the
        :class:`~openmc.deplete.MicroXS` object is empty, a decay-only
        calculation will be run.
    chain_file : PathLike or Chain, optional
        Path to the depletion chain XML file or instance of openmc.deplete.Chain.
        Defaults to ``openmc.config['chain_file']``.
    keff : 2-tuple of float, optional
       keff eigenvalue and uncertainty from transport calculation.
    prev_results : Results, optional
        Results from a previous depletion calculation.
    normalization_mode : {"fission-q", "source-rate"}
        Indicate how reaction rates should be calculated. ``"fission-q"`` uses
        the fission Q values from the depletion chain to compute the flux based
        on the power. ``"source-rate"`` uses a the source rate (assumed to be
        neutron flux) to calculate the reaction rates.
    fission_q : dict, optional
        Dictionary of nuclides and their fission Q values [eV]. If not given,
        values will be pulled from the ``chain_file``. Only applicable if
        ``"normalization_mode" == "fission-q"``.
    reduce_chain_level : int, optional
        Depth of the search when reducing the depletion chain. The default
        value of ``None`` implies no limit on the depth.
    keep_isomeric_siblings : bool, optional
        Whether to keep all isomeric state siblings together during chain
        reduction, as isomers can be at different depths in the chain:
        - True (default): Always keep all isomeric siblings (ground + 
          metastables) when any state is reachable. Required for correct
          isomeric branching calculations. May increase chain size by 10-30%.
        - False: Original behavior. Isomeric states treated independently.
          May cause isomeric branching failures with partial exclusions.

        .. versionadded:: 0.15.4
    fission_yield_opts : dict of str to option, optional
        Optional arguments to pass to the
        :class:`openmc.deplete.helpers.FissionYieldHelper` object. Will be
        passed directly on to the helper. Passing a value of None will use the
        defaults for the associated helper.
    gendf_library : openmc.deplete.gendf.GENDFLibrary, optional
        GENDF library for σ×φ-weighted isomeric branching ratio calculation.
        -MicroXS has the flux-collapsed one-group cross-sections.
        -The chain has the multigroup-binned branching ratios (pre-processed from GENDF) per nuclide-reaction.
        -To combine the two;
        The contribution of each branching ratio bin to the collapsed one-group reaction rate must be determined by σ×φ weighting
        e.g.: the branching ratio itself must have a weighted collapse, this is done by accessing the GENDF XS of the nuclide-reaction

        Required for isomeric branching support. Default is None.

      .. versionadded:: 0.15.4

    Attributes
    ----------
    materials : openmc.Materials
        All materials present in the model
    cross_sections : list of MicroXS
        Object containing multigroup cross-sections in [b] for each material.        # *** wrong? I think this is multigroup XS collapsed with multigroup Flux to one group cross-sections ***
    output_dir : pathlib.Path
        Path to output directory to save results.
    round_number : bool
        Whether or not to round output to OpenMC to 8 digits. Useful in testing,
        as OpenMC is incredibly sensitive to exact values.
    number : openmc.deplete.AtomNumber
        Total number of atoms in simulation.
    nuclides_with_data : set of str
        A set listing all unique nuclides available from cross_sections.xml.
    chain : openmc.deplete.Chain
        The depletion chain information necessary to form matrices and tallies.
    reaction_rates : openmc.deplete.ReactionRates
        Reaction rates from the last operator step.
    burnable_mats : list of str
        All burnable material IDs
    heavy_metal : float
        Initial heavy metal inventory [g]
    local_mats : list of str
        All burnable material IDs being managed by a single process
    prev_res : Results or None
        Results from a previous depletion calculation. ``None`` if no results
        are to be used.

    """

    def __init__(self,
                 materials,
                 fluxes,
                 micros,
                 chain_file=None,
                 keff=None,
                 normalization_mode='fission-q',
                 fission_q=None,
                 prev_results=None,
                 reduce_chain_level=None,
                 keep_isomeric_siblings=True,
                 fission_yield_opts=None,
                 require_isomeric_branching=True,
                 gendf_library=None,
                 _prefiltered=False):
        # Validate micro-xs parameters
        check_type('materials', materials, Iterable, openmc.Material)
        check_type('micros', micros, Iterable, MicroXS)
        materials = openmc.Materials(materials)

        if not _prefiltered:
            if not (len(fluxes) == len(micros) == len(materials)):
                msg = (f'The length of fluxes ({len(fluxes)}) should be equal '
                       f'to the length of micros ({len(micros)}) and the '
                       f'length of materials ({len(materials)}).')
                raise ValueError(msg)

        if keff is not None:
            check_type('keff', keff, tuple, float)
            keff = ufloat(*keff)

        self._keff = keff

        if fission_yield_opts is None:
            fission_yield_opts = {}
        helper_kwargs = {'normalization_mode': normalization_mode,
                         'fission_yield_opts': fission_yield_opts}

        if not _prefiltered:
            # Sort fluxes and micros in same order that materials get sorted
            index_sort = np.argsort([mat.id for mat in materials])
            fluxes = [fluxes[i] for i in index_sort]
            micros = [micros[i] for i in index_sort]

        # Store energy bins if present in flux tuples
        self._energy_bins = None
        self._flux_with_energy = []
        for flux_item in fluxes:
            if isinstance(flux_item, tuple) and len(flux_item) == 2:
                self._flux_with_energy.append(flux_item)
                if self._energy_bins is None:
                    self._energy_bins = flux_item[1]
            else:
                self._flux_with_energy.append((flux_item, None))
        super().__init__(
            materials=materials,
            cross_sections=micros,
            chain_file=chain_file,
            prev_results=prev_results,
            fission_q=fission_q,
            helper_kwargs=helper_kwargs,
            reduce_chain_level=reduce_chain_level)

        # Filter to local materials only (MPI distribution)
        if comm.size > 1:
            if not _prefiltered:
                local_indices = [self._mat_index_map[m]
                                 for m in self.local_mats]
                self.cross_sections = [self.cross_sections[i] for i in local_indices]
                self._flux_with_energy = [self._flux_with_energy[i] for i in local_indices]

            # Remap to 0-based local indices; drop non-local materials
            self._mat_index_map = {
                lm: i for i, lm in enumerate(self.local_mats)}
            local_set = set(self.local_mats)
            self.materials = openmc.Materials(
                [m for m in self.materials if str(m.id) in local_set])

        # Store parameters for isomeric branching setup
        self._require_isomeric_branching = require_isomeric_branching
        self._gendf_library = gendf_library

        # Setup isomeric branching after initialization
        self._setup_isomeric_branching()

    @property
    def fluxes(self):
        """List of flux arrays, derived from _flux_with_energy."""
        return [f for f, _ in self._flux_with_energy]

    @classmethod
    def from_nuclides(cls, volume, nuclides,
                      flux,
                      micro_xs,
                      chain_file=None,
                      nuc_units='atom/b-cm',
                      keff=None,
                      normalization_mode='fission-q',
                      fission_q=None,
                      prev_results=None,
                      reduce_chain_level=None,
                      keep_isomeric_siblings=True,
                      fission_yield_opts=None,
                      require_isomeric_branching=True,
                      gendf_library=None):
        """
        Alternate constructor from a dictionary of nuclide concentrations

        volume : float
            Volume of the material being depleted in [cm^3]
        nuclides : dict of str to float
            Dictionary with nuclide names as keys and nuclide concentrations as
            values.
        flux : numpy.ndarray
            Flux in each group in [n-cm/src]
        micro_xs : MicroXS
            Cross sections in [b]. If the :class:`~openmc.deplete.MicroXS`
            object is empty, a decay-only calculation will be run.
        chain_file : PathLike or Chain, optional
            Path to the depletion chain XML file or instance of
            openmc.deplete.Chain. Defaults to ``openmc.config['chain_file']``.
        nuc_units : {'atom/cm3', 'atom/b-cm'}, optional
            Units for nuclide concentration.
        keff : 2-tuple of float, optional
           keff eigenvalue and uncertainty from transport calculation.
           Default is None.
        normalization_mode : {"fission-q", "source-rate"}
            Indicate how reaction rates should be calculated.
            ``"fission-q"`` uses the fission Q values from the depletion
            chain to compute the flux based on the power. ``"source-rate"`` uses
            the source rate (assumed to be neutron flux) to calculate the
            reaction rates.
        fission_q : dict, optional
            Dictionary of nuclides and their fission Q values [eV]. If not
            given, values will be pulled from the ``chain_file``. Only
            applicable if ``"normalization_mode" == "fission-q"``.
        prev_results : Results, optional
            Results from a previous depletion calculation.
        reduce_chain_level : int, optional
            Depth of the search when reducing the depletion chain. The default
            value of ``None`` implies no limit on the depth.
        keep_isomeric_siblings : bool, optional
            Whether to keep all isomeric state siblings together during chain
            reduction:

            - True (default): Always keep all isomeric siblings (ground +
              metastables) when any state is reachable. Required for correct
              isomeric branching calculations. May increase chain size by 10-30%.
            - False: Original behavior. Isomeric states treated independently.
              May cause isomeric branching failures with partial exclusions.

        fission_yield_opts : dict of str to option, optional
            Optional arguments to pass to the
            :class:`openmc.deplete.helpers.FissionYieldHelper` class. Will be
            passed directly on to the helper. Passing a value of None will use
            the defaults for the associated helper.
        require_isomeric_branching : bool, optional
            If True (default), raises RuntimeError when isomeric branching data
            exists in the chain but cannot be used (missing flux spectra or
            unsupported energy structure). If False, issues a warning and
            proceeds without isomeric branching.
            ** Maybe can be wholly removed **
        gendf_library : openmc.deplete.gendf.GENDFLibrary, optional
            GENDF library for on-the-fly multigroup cross-section lookup.
            Default is None.

        """
        check_type('nuclides', nuclides, dict, str)
        materials = cls._consolidate_nuclides_to_material(nuclides, nuc_units, volume)
        fluxes = [flux]
        micros = [micro_xs]
        return cls(materials,
                   fluxes,
                   micros,
                   chain_file,
                   keff=keff,
                   normalization_mode=normalization_mode,
                   fission_q=fission_q,
                   prev_results=prev_results,
                   reduce_chain_level=reduce_chain_level,
                   keep_isomeric_siblings=keep_isomeric_siblings,
                   fission_yield_opts=fission_yield_opts,
                   require_isomeric_branching=require_isomeric_branching,
                   gendf_library=gendf_library)

    @classmethod
    def from_microxs_file(
        cls,
        materials,
        microxs_file,
        chain_file=None,
        keff=None,
        normalization_mode='fission-q',
        fission_q=None,
        prev_results=None,
        reduce_chain_level=None,
        keep_isomeric_siblings=True,
        fission_yield_opts=None,
        require_isomeric_branching=True,
        gendf_library=None,
    ):
        """Construct operator from a pre-written MicroXS HDF5 file.

        Each MPI rank reads only its local material slices from the file,
        eliminating the memory peak from loading all cross sections at once.

        .. versionadded:: 0.15.4

        Parameters
        ----------
        materials : iterable of openmc.Material
            All materials in the model (not just local). Must include all
            depletable materials whose IDs appear in the HDF5 file.
        microxs_file : path-like
            Path to HDF5 file written by
            :func:`~openmc.deplete.write_global_microxs_hdf5`.
        chain_file : PathLike or Chain, optional
            Path to the depletion chain XML file or instance of
            openmc.deplete.Chain. Defaults to ``openmc.config['chain_file']``.
        keff : 2-tuple of float, optional
            keff eigenvalue and uncertainty from transport calculation.
        normalization_mode : {"fission-q", "source-rate"}
            How reaction rates should be calculated.
        fission_q : dict, optional
            Dictionary of nuclides and their fission Q values [eV].
        prev_results : Results, optional
            Results from a previous depletion calculation.
        reduce_chain_level : int, optional
            Depth of the search when reducing the depletion chain.
        keep_isomeric_siblings : bool, optional
            Whether to keep isomeric siblings. Defaults to True.
        fission_yield_opts : dict, optional
            Arguments for the FissionYieldHelper.
        require_isomeric_branching : bool, optional
            If True, require isomeric branching ratios. Defaults to True.
        gendf_library : GENDFLibrary, optional
            GENDF library for isomeric branching ratios.

        Returns
        -------
        IndependentOperator

        See Also
        --------
        write_global_microxs_hdf5 : Write the HDF5 file consumed here.
        read_local_microxs_hdf5 : Low-level reader used internally.

        """
        check_type('materials', materials, Iterable, openmc.Material)
        materials_obj = openmc.Materials(materials)

        # Determine burnable materials in sorted order (same logic as
        # OpenMCOperator._get_burnable_mats)
        burnable_mats = sorted(
            [str(mat.id) for mat in materials_obj if mat.depletable], key=int)
        local_mats = _distribute(burnable_mats)

        local_micros, local_flux_with_energy = read_local_microxs_hdf5(
            microxs_file, local_mats)

        # Build fluxes list from HDF5 data or default to unit flux.
        # Pass full (flux, energy_bounds) tuples so __init__ populates
        # _flux_with_energy for isomeric branching.
        if local_flux_with_energy is not None:
            local_fluxes = list(local_flux_with_energy)
        else:
            n_groups = local_micros[0].data.shape[2] if local_micros else 1
            local_fluxes = [np.ones(n_groups) for _ in local_mats]

        op = cls(
            materials_obj,
            local_fluxes,
            local_micros,
            chain_file=chain_file,
            keff=keff,
            normalization_mode=normalization_mode,
            fission_q=fission_q,
            prev_results=prev_results,
            reduce_chain_level=reduce_chain_level,
            keep_isomeric_siblings=keep_isomeric_siblings,
            fission_yield_opts=fission_yield_opts,
            require_isomeric_branching=require_isomeric_branching,
            gendf_library=gendf_library,
            _prefiltered=True,
        )

        # Verify _distribute agreement between from_microxs_file and __init__
        assert list(op.local_mats) == list(local_mats), (
            f"_distribute mismatch: from_microxs_file got {local_mats}, "
            f"but __init__ computed {op.local_mats}")

        return op

    @staticmethod
    def _consolidate_nuclides_to_material(nuclides, nuc_units, volume):
        """Puts nuclide list into an openmc.Materials object.

        """
        mat = openmc.Material()
        if nuc_units == 'atom/b-cm':
            for nuc, conc in nuclides.items():
                mat.add_nuclide(nuc, conc)
        elif nuc_units == 'atom/cm3':
            for nuc, conc in nuclides.items():
                mat.add_nuclide(nuc, conc * 1e-24)  # convert to at/b-cm
        else:
            raise ValueError(f"Unit '{nuc_units}' is invalid.")

        mat.volume = volume
        mat.depletable = True

        return openmc.Materials([mat])

    def _load_previous_results(self):
        """Load results from a previous depletion simulation"""
        # Reload volumes into geometry
        model = openmc.Model(materials=self.materials)
        self.prev_res[-1].transfer_volumes(model)
        self.materials = model.materials

        # Store previous results in operator
        # Distribute reaction rates according to those tracked
        # on this process
        if comm.size != 1:
            prev_results = self.prev_res
            self.prev_res = Results()
            mat_indexes = _distribute(range(len(self.burnable_mats)))
            for res_obj in prev_results:
                new_res = res_obj.distribute(self.local_mats, mat_indexes)
                self.prev_res.append(new_res)

    def _setup_isomeric_branching(self):
        """Set up isomeric branching helper if data exists.

        Raises
        ------
        RuntimeError
            If isomeric branching data exists but flux spectra or energy bins
            are missing, or if energy structure cannot be determined, and
            ``require_isomeric_branching=True`` (default).

        Notes
        -----
        If ``require_isomeric_branching=False`` was passed to ``__init__``,
        warnings are issued instead of errors, and isomeric branching is
        disabled for this operator.
        """
        self._isomeric_branching = None

        # Check if chain has isomeric branching data
        if not hasattr(self.chain, 'isomeric_branching'):
            return

        if self.chain.isomeric_branching is None:
            return

        # Helper for handling errors/warnings based on require_isomeric_branching
        def _handle_issue(message):
            if self._require_isomeric_branching:
                raise RuntimeError(message)
            else:
                warnings.warn(
                    f"{message} Isomeric branching will be disabled.",
                    UserWarning
                )
                return True  # Signal to return early

        if self._gendf_library is None:
            if _handle_issue(
                "Isomeric branching data is present in chain but no GENDF "
                "library was provided."
            ):
                return

        # Check flux spectra and energy bins
        if len(self._flux_with_energy) == 0 or self._energy_bins is None or len(self._energy_bins) == 0:
            if _handle_issue(
                "Isomeric branching data is present in chain but flux spectra "
                "or energy bins are missing. Energy-dependent isomeric branching "
                "requires flux spectra with energy information."
            ):
                return

        helper = IsomericBranchingHelper(
            self.chain,
            self._gendf_library,
        )
        self._isomeric_branching = []

        # Calculate σ×φ-weighted branching for each material
        for i, (flux_spectrum, energy) in enumerate(self._flux_with_energy):
            # All materials must have energy information
            if energy is None or not isinstance(flux_spectrum, np.ndarray):
                raise RuntimeError(
                    f"Material {i} is missing flux spectrum or energy information. "
                    f"All materials must have flux spectra with "
                    f"{helper.energy_structure} energy structure when using "
                    f"energy-dependent isomeric branching."
                )

            # Calculate σ×φ-weighted branching
            # The helper will perform strict validation and raise errors if mismatched
            weighted = helper.weighted_branching_ratios(flux_spectrum, energy)

            self._isomeric_branching.append(weighted)

        # Validate that branching was actually calculated
        has_branching = any(bool(d) for d in self._isomeric_branching)
        if not has_branching:
            warnings.warn(
                "Isomeric branching data exists in chain but σ×φ-weighted ratios "
                "could not be calculated. MicroXS stores flux-collapsed single-group "
                "cross-sections and cannot provide spectral information for weighting.\n"
                "To enable isomeric branching, provide gendf_library parameter with "
                "GENDF files containing multigroup cross-sections.\n"
                "Proceeding without isomeric branching.",
                UserWarning
            )
            self._isomeric_branching = None

    def _get_nuclides_with_data(self, cross_sections: list[MicroXS]) -> set[str]:
        """Finds nuclides with cross section data

        Parameters
        ----------
        cross_sections : iterable of :class`~openmc.deplete.MicroXS`
            List of multigroup cross-section data.

        Returns
        -------
        nuclides : set of str
            Set of nuclide names that have cross section data

        """
        if not cross_sections:
            return set()
        return set(cross_sections[0].nuclides)

    class _IndependentRateHelper(ReactionRateHelper):
        """Class for generating reaction rates with multigroup fluxes and
        multigroup cross sections.

        This class does not generate tallies and instead stores cross sections
        for each nuclide and transmutation reaction relevant for a depletion
        calculation. The reaction rate is calculated by multiplying the flux by
        the cross sections.

        Parameters
        ----------
        op : openmc.deplete.IndependentOperator
            Reference to the object encapsulate _IndependentRateHelper.
            We pass this so we don't have to duplicate the
            :attr:`IndependentOperator.number` object.

        Attributes
        ----------
        nuc_ind_map : dict of int to str
            Dictionary mapping the nuclide index to nuclide name
        rx_ind_map : dict of int to str
            Dictionary mapping reaction index to reaction name

        """

        def __init__(self, op: IndependentOperator):
            rates = op.reaction_rates
            super().__init__(rates.n_nuc, rates.n_react)

            self.nuc_ind_map = {ind: nuc for nuc, ind in rates.index_nuc.items()}
            self.rx_ind_map = {ind: rxn for rxn, ind in rates.index_rx.items()}
            self._op = op

        def generate_tallies(self, materials, scores):
            """Unused in this case"""
            pass

        def reset_tally_means(self):
            """Unused in this case"""
            pass

        def get_material_rates(self, mat_index, nuc_index, react_index):
            """Return 2D array of [nuclide, reaction] reaction rates

            Parameters
            ----------
            mat_index : int
                Index for the material
            nuc_index : list of str
                Ordering of desired nuclides
            react_index : list of str
                Ordering of reactions
            """
            self._results_cache.fill(0.0)

            # Get flux and microscopic cross sections from operator
            flux = self._op._flux_with_energy[mat_index][0]
            xs = self._op.cross_sections[mat_index]

            for i_nuc in nuc_index:
                nuc = self.nuc_ind_map[i_nuc]
                for i_rx in react_index:
                    rx = self.rx_ind_map[i_rx]

                    # Determine reaction rate by multiplying xs in [b] by flux
                    # in [n-cm/src] to give [(reactions/src)*b-cm/atom]
                    self._results_cache[i_nuc, i_rx] = (xs[nuc, rx] * flux).sum()

            return self._results_cache

    def _get_helper_classes(self, helper_kwargs):
        """Get helper classes for calculating reaction rates and fission yields

        Parameters
        ----------
        helper_kwargs : dict
            Keyword arguments for helper classes

        """

        normalization_mode = helper_kwargs['normalization_mode']
        fission_yield_opts = helper_kwargs['fission_yield_opts']

        self._rate_helper = self._IndependentRateHelper(self)
        if normalization_mode == "fission-q":
            self._normalization_helper = ChainFissionHelper()
        else:
            self._normalization_helper = SourceRateHelper()

        # Select and create fission yield helper
        fission_helper = ConstantFissionYieldHelper
        self._yield_helper = fission_helper.from_operator(
            self, **fission_yield_opts)

    def initial_condition(self):
        """Performs final setup and returns initial condition.

        Returns
        -------
        list of numpy.ndarray
            Total density for initial conditions.
        """

        # Return number density vector
        return super().initial_condition(self.materials)

    def __call__(self, vec, source_rate):
        """Obtain the reaction rates

        Parameters
        ----------
        vec : list of numpy.ndarray
            Total atoms to be used in function.
        source_rate : float
            Power in [W] or flux in [neutron/cm^2-s]

        Returns
        -------
        openmc.deplete.OperatorResult
            Eigenvalue and reaction rates resulting from transport operator

        """

        self._update_materials_and_nuclides(vec)

        # If the source rate is zero, return zero reaction rates
        if source_rate == 0.0:
            rates = self.reaction_rates.copy()
            rates.fill(0.0)
            return OperatorResult(ufloat(0.0, 0.0), rates)

        rates = self._calculate_reaction_rates(source_rate)
        keff = self._keff

        op_result = OperatorResult(keff, rates)
        return copy.deepcopy(op_result)

    def _update_materials(self):
        """Zero out negative nuclide densities on local materials."""
        for mat in self.number.materials:
            for nuc in self.number.nuclides:
                if nuc in self.nuclides_with_data:
                    val = 1.0e-24 * self.number.get_atom_density(mat, nuc)
                    if val < 0.0:
                        if val < -1.0e-21:
                            print(f'WARNING: nuclide {nuc} in material '
                                  f'{mat} is negative (density = {val}'
                                  ' atom/b-cm)')
                        self.number[mat, nuc] = 0.0
