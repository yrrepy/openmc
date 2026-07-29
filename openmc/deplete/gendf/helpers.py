"""GENDF reaction-rate and isomeric-branching helpers.

Holds the GENDF-specific helper classes that used to live in
:mod:`openmc.deplete.helpers`: :class:`DirectWithFluxHelper`,
:class:`GENDFFluxCollapseHelper` and :class:`IsomericBranchingHelper`.

This submodule is never imported at ``openmc.deplete.gendf`` package-init time,
so it may import from :mod:`openmc.deplete.helpers` at module level. For
backward compatibility the three classes remain reachable under their old
``openmc.deplete.helpers`` names through that module's ``__getattr__``.

.. versionadded:: 0.15.4
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING
import warnings

from numpy import asarray
import numpy as np

from openmc.data import REACTION_MT
from openmc.mgxs import GROUP_STRUCTURES
from openmc.exceptions import OpenMCError
from openmc.lib import Tally, MaterialFilter, EnergyFilter, load_nuclide
import openmc.lib
from ..abc import ReactionRateHelper
from ..helpers import (
    DirectReactionRateHelper, _GENDFMT4FallbackMixin,
    _warn_isomeric_normalize, _warn_weighted_xs)
from .library import REACTION_TO_MT

if TYPE_CHECKING:
    from ..chain import Chain


class DirectWithFluxHelper(_GENDFMT4FallbackMixin, ReactionRateHelper):
    """Direct reaction rates with flux tallying for isomeric branching.

    This helper provides the accuracy of direct reaction rate tallies
    while also tallying flux spectrum for isomeric branching calculations.
    It wraps a :class:`DirectReactionRateHelper` for reaction rate calculation
    and adds a separate flux tally with energy groups for isomeric branching.

    .. versionadded:: 0.15.4

    Parameters
    ----------
    n_nuc : int
        Number of nuclides tracked
    n_react : int
        Number of reactions tracked
    energies : numpy.ndarray
        Energy group boundaries (CCFE-709 or UKAEA-1102)
    gendf_library : openmc.deplete.gendf.GENDFLibrary, optional
        GENDF library used for the (n,n') MT=4 fallback
    gendf_mt4_fallback : bool, optional
        Whether to fill the (n,n') column by collapsing the tallied flux
        with GENDF MT=4 cross sections. Default is False.

    Attributes
    ----------
    flux_tally : openmc.lib.Tally or None
        Flux tally with energy filter for isomeric branching
    energies : numpy.ndarray
        Energy group boundaries in [eV]
    """

    def __init__(self, n_nuc: int, n_react: int, energies: np.ndarray,
                 gendf_library=None, gendf_mt4_fallback: bool = False) -> None:
        super().__init__(n_nuc, n_react)
        self._direct_helper = DirectReactionRateHelper(n_nuc, n_react)
        self._energies: np.ndarray = np.asarray(energies)
        self._flux_tally: Optional[Tally] = None
        self._materials: Optional[List] = None
        self._init_mt4_fallback(gendf_library, gendf_mt4_fallback)

    @ReactionRateHelper.nuclides.setter
    def nuclides(self, nuclides: List[str]) -> None:
        ReactionRateHelper.nuclides.fset(self, nuclides)
        self._direct_helper.nuclides = nuclides

    def generate_tallies(
        self,
        materials: Iterable,
        scores: Iterable[str]
    ) -> None:
        """Generate direct rate tally and flux tally.

        Parameters
        ----------
        materials : iterable of :class:`openmc.lib.Material`
            Burnable materials in the problem
        scores : iterable of str
            Reaction identifiers needed for the reaction rate tally
        """
        self._materials = list(materials)
        self._scores = list(scores)

        # Generate direct rate tallies (delegate to inner helper)
        self._direct_helper.generate_tallies(self._materials, scores)

        # Generate flux tally for isomeric branching
        self._flux_tally = Tally()
        self._flux_tally.writable = False
        self._flux_tally.filters = [
            MaterialFilter(self._materials),
            EnergyFilter(self._energies)
        ]
        self._flux_tally.scores = ['flux']

    def reset_tally_means(self) -> None:
        """Reset tally means for both direct and flux tallies."""
        self._direct_helper.reset_tally_means()
        # Flux tally means are handled by OpenMC directly

    def get_material_rates(
        self,
        mat_index: int,
        nuc_index: Iterable[int],
        rx_index: Iterable[int]
    ) -> np.ndarray:
        """Get reaction rate (delegate to direct helper).

        Parameters
        ----------
        mat_index : int
            Index for the material
        nuc_index : iterable of int
            Index for each nuclide
        rx_index : iterable of int
            Index for each reaction

        Returns
        -------
        rates : numpy.ndarray
            Array with shape ``(n_nuclides, n_rxns)`` with the
            reaction rates in this material
        """
        rates = self._direct_helper.get_material_rates(
            mat_index, nuc_index, rx_index)
        self._apply_mt4_fallback(rates, mat_index, nuc_index, rx_index)
        return rates

    @property
    def energies(self) -> np.ndarray:
        """Energy group boundaries in [eV]."""
        return self._energies

    @property
    def flux_tally_means(self) -> np.ndarray:
        """Return flux tally mean values."""
        return self._flux_tally.mean.ravel()

    def get_flux_spectrum(self, mat_index: int) -> np.ndarray:
        """Get flux spectrum for a specific material.

        Parameters
        ----------
        mat_index : int
            Index of the material

        Returns
        -------
        numpy.ndarray
            Flux spectrum with shape (n_groups,)
        """
        n_mats = len(self._materials)
        n_groups = len(self._energies) - 1
        shape = (n_mats, n_groups)
        return self._flux_tally.mean.reshape(shape)[mat_index]




class GENDFFluxCollapseHelper(ReactionRateHelper):
    """Class that generates one-group reaction rates from a GENDF flux collapse

    This class tallies a multigroup flux in the GENDF library's energy group
    structure (CCFE-709 or UKAEA-1102) and collapses it with GENDF group-wise
    cross sections using a sparse table (one matrix-vector product per
    material). GENDF MF=3 data includes lumped reactions such as (n,n') that
    are absent from most continuous-energy HDF5 libraries, so those rates are
    obtained natively. Select reactions can still be treated with a direct
    continuous-energy reaction rate tally; fission is direct-tallied by
    default so that normalization retains continuous-energy fidelity. The
    flux tally also provides the spectrum used for isomeric branching.

    .. versionadded:: 0.15.4

    Parameters
    ----------
    n_nucs : int
        Number of burnable nuclides tracked by
        :class:`openmc.deplete.CoupledOperator`
    n_reacts : int
        Number of reactions tracked by :class:`openmc.deplete.CoupledOperator`
    gendf_library : openmc.deplete.gendf.GENDFLibrary
        GENDF library providing group-wise cross sections
    reactions : iterable of str, optional
        Reactions for which rates should be directly tallied with
        continuous-energy data. Defaults to ``['fission']``; pass an empty
        list to collapse all reactions from GENDF data.
    nuclides : iterable of str, optional
        Nuclides for which some reaction rates should be directly tallied. If
        None, then ``reactions`` will be used for all nuclides.

    Attributes
    ----------
    nuclides : list of str
        All nuclides with desired reaction rates.
    energies : numpy.ndarray
        Energy group boundaries in [eV]
    """

    def __init__(self, n_nucs, n_reacts, gendf_library, reactions=None,
                 nuclides=None):
        super().__init__(n_nucs, n_reacts)
        if gendf_library.energy_structure is None:
            raise ValueError(
                "GENDF library has no detected energy group structure; "
                "'gendf-flux' mode requires a library using CCFE-709 or "
                "UKAEA-1102.")
        self._gendf_library = gendf_library
        self._energies = asarray(
            GROUP_STRUCTURES[gendf_library.energy_structure])
        self._reactions_direct = (
            ['fission'] if reactions is None else list(reactions))
        self._nuclides_direct = list(nuclides) if nuclides is not None else None
        self._xs_table = None
        self._table_index = {}
        self._table_nuclides = None
        self._missing = frozenset()

    @ReactionRateHelper.nuclides.setter
    def nuclides(self, nuclides):
        ReactionRateHelper.nuclides.fset(self, nuclides)
        if self._reactions_direct and self._nuclides_direct is None:
            # Direct tally spans all reaction nuclides; some may not be
            # loaded from the initial materials
            for nuclide in nuclides:
                if nuclide not in openmc.lib.nuclides:
                    load_nuclide(nuclide)
            self._rate_tally.nuclides = nuclides

    def generate_tallies(self, materials, scores):
        """Produce multigroup flux and direct reaction rate tallies

        Parameters
        ----------
        materials : iterable of :class:`openmc.lib.Material`
            Burnable materials in the problem. Used to construct a
            :class:`openmc.lib.MaterialFilter`
        scores : iterable of str
            Reaction identifiers, e.g. ``"(n, gamma)"``, needed for the
            reaction rate tally.
        """
        self._materials = materials
        self._mts = [REACTION_MT[x] for x in scores]
        self._scores = list(scores)

        # Direct tallies only make sense for tracked reactions
        self._reactions_direct = [
            r for r in self._reactions_direct if r in self._scores]

        # Create flux tally with material and energy filters
        self._flux_tally = Tally()
        self._flux_tally.writable = False
        self._flux_tally.filters = [
            MaterialFilter(materials),
            EnergyFilter(self._energies)
        ]
        self._flux_tally.scores = ['flux']
        self._flux_tally_means_cache = None

        # Create reaction rate tally
        if self._reactions_direct:
            self._rate_tally = Tally()
            self._rate_tally.writable = False
            self._rate_tally.scores = self._reactions_direct
            self._rate_tally.filters = [MaterialFilter(materials)]
            self._rate_tally.multiply_density = False
            self._rate_tally_means_cache = None
            if self._nuclides_direct is not None:
                # check if any direct tally nuclides are requested that are not
                # already loaded with the materials. Load separately if so.
                mat_nuclides = {n for mat in materials for n in mat.nuclides}
                extra_nuclides = set(self._nuclides_direct) - mat_nuclides
                for nuc in extra_nuclides:
                    load_nuclide(nuc)
                self._rate_tally.nuclides = self._nuclides_direct

    @property
    def rate_tally_means(self):
        """The mean results of the tally of every material's reaction rates for this cycle
        """
        if self._rate_tally_means_cache is None:
            self._rate_tally_means_cache = self._rate_tally.mean
        return self._rate_tally_means_cache

    @property
    def flux_tally_means(self):
        # If the mean cache is empty, fill it once for this transport cycle's results
        if self._flux_tally_means_cache is None:
            self._flux_tally_means_cache = self._flux_tally.mean
        return self._flux_tally_means_cache

    def reset_tally_means(self):
        """Reset the cached mean rate and flux tallies.
        .. note::

                This step must be performed after each transport cycle
        """
        self._flux_tally_means_cache = None
        if self._reactions_direct:
            self._rate_tally_means_cache = None

    @property
    def energies(self):
        """Energy group boundaries in [eV]."""
        return self._energies

    def get_flux_spectrum(self, mat_index):
        """Get flux spectrum for a specific material.

        Parameters
        ----------
        mat_index : int
            Index of the material

        Returns
        -------
        numpy.ndarray
            Flux spectrum with shape (n_groups,)
        """
        shape = (len(self._materials), len(self._energies) - 1)
        return self.flux_tally_means.reshape(shape)[mat_index]

    def _ensure_xs_table(self):
        """Build the sparse GENDF XS table; rebuild if the nuclide set grew."""
        if self._table_nuclides == self.nuclides:
            return
        from ..microxs import _build_sparse_xs_table

        available = self._gendf_library.available_nuclides_set()
        table_nucs = [n for n in self.nuclides if n in available]
        missing = frozenset(self.nuclides) - available
        if missing and missing != self._missing:
            names = ' '.join(sorted(missing)[:10])
            more = ' ...' if len(missing) > 10 else ''
            warnings.warn(
                f"{len(missing)} nuclides not in GENDF library will have "
                f"zero reaction rates unless directly tallied: {names}{more}")
        self._missing = missing
        self._xs_table = _build_sparse_xs_table(
            self._gendf_library, table_nucs, self._scores, self._mts)
        self._table_index = {n: i for i, n in enumerate(table_nucs)}
        self._table_nuclides = list(self.nuclides)

    def get_material_rates(self, mat_index, nuc_index, react_index):
        """Return an array of reaction rates for a material

        Parameters
        ----------
        mat_index : int
            Index for material
        nuc_index : iterable of int
            Index for each nuclide in :attr:`nuclides` in the
            desired reaction rate matrix
        react_index : iterable of int
            Index for each reaction scored in the tally

        Returns
        -------
        rates : numpy.ndarray
            Array with shape ``(n_nuclides, n_rxns)`` with the reaction rates
            in this material

        """
        self._results_cache.fill(0.0)
        self._ensure_xs_table()

        # Collapse GENDF group XS with this material's flux spectrum. Result
        # is sigma [b] times volume-integrated flux [particle-cm/src] -- the
        # same convention as direct tallies with multiply_density=False.
        flux = self.get_flux_spectrum(mat_index)
        collapsed = self._xs_table.collapse(flux)

        for name, i_nuc in zip(self.nuclides, nuc_index):
            i_table = self._table_index.get(name)
            if i_table is not None:
                self._results_cache[i_nuc, react_index] = collapsed[i_table]

        # Overlay direct tally results
        if self._reactions_direct:
            nuclides_direct = self._rate_tally.nuclides
            shape = (len(nuclides_direct), len(self._reactions_direct))
            rx_rates = self.rate_tally_means[mat_index].reshape(shape)
            direct_rx_index = {score: i for i, score in enumerate(self._reactions_direct)}
            direct_nuc_index = {nuc: i for i, nuc in enumerate(nuclides_direct)}
            for name, i_nuc in zip(self.nuclides, nuc_index):
                i_direct = direct_nuc_index.get(name)
                if i_direct is None:
                    continue
                for score, i_rx in zip(self._scores, react_index):
                    if score in direct_rx_index:
                        self._results_cache[i_nuc, i_rx] = \
                            rx_rates[i_direct, direct_rx_index[score]]

        return self._results_cache



class IsomericBranchingHelper:
    """Helper for automatic reaction-rate weighted isomeric branching calculations.

    Calculates reaction-rate weighted isomeric branching ratios for reactions that
    can produce both ground state and metastable products. The weighting
    is performed with group-wise GENDF ratios placed on the chain and group-wise
    cross-sections obtained from a user-provided GENDF library.

    Flux spectrum must use the same energy group structure(typically CCFE-709
    or UKAEA-1102) as the isomeric branching data on the chain files.
    Mismatches will raise ValueError.

    .. versionadded:: 0.15.4

    Parameters
    ----------
    chain : openmc.deplete.Chain
        Chain containing isomeric branching data
    gendf_library : GENDFLibrary
        GENDF library for on-the-fly multigroup cross-section lookup.
        Energy structure is derived from the library.

    Attributes
    ----------
    chain : openmc.deplete.Chain
        Reference to the depletion chain
    isomeric_data : dict or None
        Isomeric branching data from the chain
    energy_structure : str
        Name of the energy group structure (from gendf_library)
    expected_energies : numpy.ndarray
        Expected energy bin boundaries in eV
    n_groups : int
        Number of energy groups
    gendf_library : GENDFLibrary
        GENDF library for on-the-fly XS lookup
    """

    def __init__(
        self,
        chain: 'Chain',
        gendf_library,
    ) -> None:
        """Initialize with a depletion chain and GENDF library.

        Parameters
        ----------
        chain : openmc.deplete.Chain
            Chain containing isomeric branching data
        gendf_library : GENDFLibrary
            GENDF library for on-the-fly multigroup cross-section lookup.
        """
        if gendf_library is None:
            raise ValueError("gendf_library is required for IsomericBranchingHelper")

        self.chain: 'Chain' = chain
        self.isomeric_targets: Optional[Dict] = chain.isomeric_branching_targets
        self._branching_cache: Dict = {}
        self.gendf_library = gendf_library
        self.energy_structure: str = gendf_library.energy_structure
        self.expected_energies: np.ndarray = gendf_library.energy_bounds.copy()
        self.n_groups: int = len(self.expected_energies) - 1

    def _get_branching_data(self, nuclide, reaction):
        """Cached lookup: chain-embedded first, then GENDF."""
        key = (nuclide, reaction)
        if key not in self._branching_cache:
            if (self.chain.isomeric_branching_embedded
                    and key in self.chain.isomeric_branching_embedded):
                self._branching_cache[key] = self.chain.isomeric_branching_embedded[key]
            else:
                mt = REACTION_TO_MT.get(reaction)
                if mt is None:
                    self._branching_cache[key] = None
                else:
                    targets = (self.chain.isomeric_branching_targets or {}).get(
                        nuclide, {}).get(reaction)
                    lfs = (self.chain.isomeric_branching_lfs or {}).get(
                        nuclide, {}).get(reaction)
                    # Only use runtime mode when both targets and LFS are
                    # available; otherwise fall through to patcher mode
                    if targets is not None and lfs is None:
                        targets = None
                    try:
                        self._branching_cache[key] = \
                            self.gendf_library.get_branching_ratios(
                                nuclide, mt,
                                target_names=targets,
                                lfs_values=lfs)
                    except (KeyError, ValueError, NotImplementedError,
                            OpenMCError) as err:
                        warnings.warn(
                            f"Isomeric branching disabled for {nuclide} "
                            f"{reaction}: could not get branching ratios "
                            f"from GENDF ({err}). Metastable production "
                            f"falls back to the chain's static branching "
                            f"ratios.", UserWarning)
                        self._branching_cache[key] = None
        return self._branching_cache[key]

    def weighted_branching_ratios(
        self,
        flux_spectrum: np.ndarray
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate σ×φ-weighted branching ratios from GENDF at runtime.

        Fetches energy-dependent branching ratios from the GENDF library,
        filters to targets present in the chain, and computes σ×φ-weighted
        effective ratios. The energy group structure is sourced from the GENDF
        library (``self.expected_energies``); the flux supplies only the
        per-material spectrum values.

        Parameters
        ----------
        flux_spectrum : numpy.ndarray
            Neutron flux in each energy group [n-cm/src].

        Returns
        -------
        dict
            ``{nuclide: {reaction: {target: weighted_ratio}}}``
        """
        result = defaultdict(lambda: defaultdict(dict))

        if self.isomeric_targets is None:
            return {}

        # Group-count mismatch is the real dimension safety net; fail loudly.
        if len(flux_spectrum) != self.n_groups:
            raise ValueError(
                f"Flux has {len(flux_spectrum)} groups, "
                f"{self.energy_structure} defines {self.n_groups}"
            )

        for nuclide, reactions in self.isomeric_targets.items():
            for reaction, target_list in reactions.items():
                br = self._get_branching_data(nuclide, reaction)
                if br is None:
                    continue

                chain_target_set = set(target_list)

                if isinstance(br, dict):
                    # Chain-embedded ratios
                    data = {
                        'energies': br['energies'],
                        # Filter identically to branching_ratios so a target
                        # kept by the chain but missing its ratio row can't
                        # reach _calculate_weighted and KeyError.
                        'targets': [t for t in br['targets']
                                    if t in chain_target_set
                                    and t in br['branching_ratios']],
                        'branching_ratios': {
                            t: br['branching_ratios'][t]
                            for t in br['targets']
                            if t in chain_target_set
                            and t in br['branching_ratios']
                        }
                    }
                else:
                    # IsomericBranching from GENDF
                    data = {
                        'energies': br.energies,
                        'targets': [p for p in br.products
                                    if p in chain_target_set],
                        'branching_ratios': {
                            br.products[i]: br.branching_ratios[i]
                            for i, p in enumerate(br.products)
                            if p in chain_target_set
                        }
                    }

                if not data['targets']:
                    continue

                weighted = self._calculate_weighted(
                    data, flux_spectrum, self.expected_energies, nuclide, reaction
                )
                if weighted:
                    result[nuclide][reaction] = weighted

        return dict(result)

    def compute_for_materials(self, flux_spectra):
        """Compute σ×φ-weighted branching for a list of materials.

        Parameters
        ----------
        flux_spectra : list of numpy.ndarray
            Per-material flux spectrum on the GENDF library group structure.

        Returns
        -------
        list of dict or None
            Weighted branching dicts per material, or None if no branching.
        """
        results = []
        for flux_spectrum in flux_spectra:
            weighted = self.weighted_branching_ratios(flux_spectrum)
            results.append(weighted)

        if not any(bool(d) for d in results):
            warnings.warn(
                "Isomeric branching data exists in chain but σ×φ-weighted ratios "
                "could not be calculated. MicroXS stores flux-collapsed single-group "
                "cross-sections and cannot provide spectral information for weighting.\n"
                "To enable isomeric branching, provide gendf_library parameter with "
                "GENDF files containing multigroup cross-sections.\n"
                "Proceeding without isomeric branching.",
                UserWarning
            )
            return None

        return results

    def _compute_isomeric_indices(
        self,
        iso_energies: np.ndarray,
        energy_bins: np.ndarray,
        n_flux_groups: int
    ) -> tuple:
        """Compute index mapping between flux energy bins and isomeric data bins.

        Pre-computes the mapping from flux energy groups to isomeric data indices
        using vectorized np.searchsorted(). This mapping is then reused for all
        target nuclides, reducing computational cost from O(N_targets * N_groups)
        to O(N_groups).

        Groupwise GENDF, so uses flat-in-bin; value at bin boundary E_i applies
        to energy interval [E_i, E_{i+1}).

        Parameters
        ----------
        iso_energies : numpy.ndarray
            Energy boundaries from isomeric data (may be subset of flux structure)
        energy_bins : numpy.ndarray
            Full flux energy bin boundaries
        n_flux_groups : int
            Number of flux energy groups

        Returns
        -------
        tuple
            (iso_indices, flux_e_low, flux_e_high) where:
            - iso_indices: Index into isomeric data for each flux group, shape (n_flux_groups,)
            - flux_e_low: Lower energy boundary for each flux group
            - flux_e_high: Upper energy boundary for each flux group
        """
        # Extract energy boundaries for all flux groups
        flux_e_low = energy_bins[:n_flux_groups]
        flux_e_high = energy_bins[1:n_flux_groups + 1]

        # Pre-compute which isomeric bin each flux group corresponds to
        # Use 'right' side to get index where e_low would be inserted
        iso_indices = np.searchsorted(iso_energies, flux_e_low, side='right')

        # Adjust indices for the flat-in-bin convention (use value at highest E <= e_low)
        # Subtract 1 to get the index of the bin that contains or precedes e_low
        iso_indices = iso_indices - 1

        # Clip indices to valid range [0, n_ratios-1]
        n_ratios = len(iso_energies)
        iso_indices = np.clip(iso_indices, 0, n_ratios - 1)

        return iso_indices, flux_e_low, flux_e_high

    def _build_branching_array(
        self,
        ratios: np.ndarray,
        iso_indices: np.ndarray,
        below_range: np.ndarray,
        above_range: np.ndarray,
        in_range: np.ndarray,
        n_flux_groups: int
    ) -> np.ndarray:
        """Build the branching ratio array for a single target nuclide.

        Constructs a 1D array of branching ratios aligned with the flux energy
        groups. Uses vectorized NumPy operations to fill different energy regions
        (below, within, and above the isomeric data range).

        Parameters
        ----------
        ratios : numpy.ndarray
            Branching ratios for a single target from isomeric data
        iso_indices : numpy.ndarray
            Pre-computed index mapping from flux groups to isomeric bins
        below_range : numpy.ndarray
            Boolean mask for flux groups entirely below isomeric data range
        above_range : numpy.ndarray
            Boolean mask for flux groups entirely above isomeric data range
        in_range : numpy.ndarray
            Boolean mask for flux groups within isomeric data range
        n_flux_groups : int
            Number of flux energy groups

        Returns
        -------
        numpy.ndarray
            Branching ratios aligned with flux energy groups, shape (n_flux_groups,)

        Notes
        -----
        Out-of-range groups (below threshold or above data range) are set to zero.
        - Below threshold: reaction cannot occur (σ=0)
        - Above data range: typically GENDF branching ratio data stops at 30MeV, 
        itself already above OpenMC upper energy limit (φ=0)
        The σ×φ weighting will naturally zero these out,but explicit zeros
        are clearer and avoid numerical issues with extrapolated values.
        """
        # Initialize to zero - out-of-range groups contribute nothing
        br_array = np.zeros(n_flux_groups, dtype=float)

        # Only fill in-range groups with actual branching data
        # below_range and above_range remain zero (no extrapolation)
        if np.any(in_range):
            br_array[in_range] = ratios[iso_indices[in_range]]

        return br_array

    def _normalize_ratios(
        self,
        weighted_ratios: Dict[str, float],
        nuclide: str = "",
        reaction: str = ""
    ) -> Dict[str, float]:
        """Normalize weighted branching ratios to ensure probability conservation.

        Normalizes ratios to sum exactly to 1.0. Issues a warning if the
        original sum deviates significantly (>1%) from 1.0, which may
        indicate data quality issues. A zero or negative sum returns an
        empty dict (legitimate for threshold reactions with negligible
        cross-section) so the caller falls back to static branching.

        Parameters
        ----------
        weighted_ratios : dict
            Dictionary of {target: weighted_ratio} values
        nuclide : str, optional
            Parent nuclide name for warning messages
        reaction : str, optional
            Reaction type for warning messages

        Returns
        -------
        dict
            Normalized dictionary of {target: weighted_ratio} values.
            Empty dict if ratios sum to zero or negative.
        """
        if not weighted_ratios:
            return {}

        total = sum(weighted_ratios.values())

        if total <= 0:
            return {}

        # Normalize if not already summing to 1.0
        if not np.isclose(total, 1.0, rtol=1e-6):
            # Warn if deviation is significant (>1%)
            if abs(total - 1.0) > 0.01:
                products_str = ", ".join(
                    f"{target}: {ratio:.6f}" for target, ratio in weighted_ratios.items()
                )
                # Dedupe once per (nuclide, reaction): the message embeds
                # per-material varying values, so the default filter can't
                # collapse it across a 10k-material run.
                _warn_isomeric_normalize(
                    (nuclide, reaction),
                    f"Isomeric branching ratios sum to {total:.6f} for "
                    f"{nuclide} {reaction} (deviates >1% from 1.0). "
                    f"Products before normalization: {products_str}. "
                    f"Normalizing to preserve probability conservation."
                )
            # Normalize
            for target in weighted_ratios:
                weighted_ratios[target] /= total

        return weighted_ratios

    def _calculate_weighted(
        self,
        data: Dict,
        flux_spectrum: np.ndarray,
        energy_bins: np.ndarray,
        nuclide: str,
        reaction: str
    ) -> Dict[str, float]:
        """Calculate σ×φ-weighted average branching ratios.
    
        Uses reaction rate weighting (σ×φ) to compute effective isomeric
        branching ratios for a given nuclide and reaction. Flux flat-in-bin.
    
        Handles subset energy structures where isomeric data may only exist
        over a limited energy range (e.g., threshold reactions). Weighting is
        performed only over the energy range where isomeric data exists.
        
        Performance Optimizations:
        - Pre-computes index mapping for all flux groups using a single call
          to np.searchsorted(), then reuses this mapping for all target nuclides.
        - Vectorizes the weighting calculation using NumPy array operations
          and np.dot() for the weighted sum.
    
        Parameters
        ----------
        data : dict
            Isomeric data containing 'energies', 'targets', and
            'branching_ratios' keys. Energies may be a subset of flux_spectrum.
        flux_spectrum : numpy.ndarray
            Neutron flux in each energy group 
        energy_bins : numpy.ndarray
            Energy bin boundaries (full structure, typically CCFE-709 or UKAEA-1102)
        nuclide : str
            Nuclide name for cross-section lookup
        reaction : str
            Reaction type for cross-section lookup (e.g., "(n,gamma)")
    
        Returns
        -------
        dict
            {target: weighted_ratio} for each target nuclide.
            Returns empty dict if nuclide/reaction not found in GENDF library
            (reaction won't occur), or if all reaction rate is below threshold.
    
        Raises
        ------
        ValueError
            If GENDF group structure doesn't match flux groups.
    
        See Also
        --------
        _compute_isomeric_indices : Computes energy index mapping
        _build_branching_array : Constructs aligned branching ratio array
        _normalize_ratios : Normalizes ratios to sum to 1.0
        """
        # Validate flux spectrum is non-negative
        if np.any(flux_spectrum < 0):
            raise ValueError(
                f"Flux spectrum contains negative values. "
                f"Min value: {flux_spectrum.min():.6e}"
            )
    
        n_flux_groups = len(flux_spectrum)
    
        mt = REACTION_TO_MT.get(reaction)
        if mt is None:
            return {}

        try:
            sigma_g = self.gendf_library.get_xs(nuclide, mt, energy_bins)
        except KeyError:
            # Nuclide/reaction absent from GENDF: reaction won't occur.
            # Static-fallback by design, no warning.
            return {}
        except (ValueError, OpenMCError) as err:
            # Data-integrity failure (e.g. strict-alignment or oversized MF=10).
            # Runtime lane still falls back to static, but not silently.
            _warn_weighted_xs(
                (nuclide, reaction),
                f"Could not compute isomeric branching for {nuclide} "
                f"{reaction} from the GENDF library: {err}. "
                f"Falling back to static chain branching ratios."
            )
            return {}
    
        if len(sigma_g) != n_flux_groups:
            raise ValueError(
                f"GENDF groups ({len(sigma_g)}) != flux groups ({n_flux_groups}) "
                f"for {nuclide} {reaction}. Energy group structure mismatch."
            )
    
        iso_energies = data['energies']
        targets = data['targets']
        branching_ratios = data['branching_ratios']
    
        # Get energy range of isomeric data (subset of full structure)
        e_min_iso = iso_energies[0]
        e_max_iso = iso_energies[-1]
    
        # Step 1: Compute index mapping (vectorized, done once)
        iso_indices, flux_e_low, flux_e_high = self._compute_isomeric_indices(
            iso_energies, energy_bins, n_flux_groups
        )
    
        # Step 2: Pre-compute masks for vectorized operations
        below_range = flux_e_high <= e_min_iso  # Groups entirely below isomeric data
        above_range = flux_e_low > e_max_iso    # Groups entirely above isomeric data
        in_range = ~below_range & ~above_range  # Groups within or overlapping isomeric range
    
        # Step 3: Compute σ×φ weight array
        weight = sigma_g * flux_spectrum
        weight[~in_range] = 0.0  # Zero out-of-range (should already be ~0 below threshold)
        weight_sum = np.sum(weight)
    
        if weight_sum == 0:
            return {}
    
        weighted_ratios = {}
    
        # Step 4: Calculate σ×φ-weighted ratio for each target
        for target in targets:
            ratios = branching_ratios[target]
    
            # Build branching ratio array aligned with flux groups
            # (out-of-range groups are zero, not extrapolated)
            br_array = self._build_branching_array(
                ratios, iso_indices, below_range, above_range, in_range, n_flux_groups
            )
    
            # Compute σ×φ-weighted sum: Σ(BR × σ × φ) / Σ(σ × φ)
            weighted_sum = np.dot(br_array, weight)
            ratio = weighted_sum / weight_sum
    
            weighted_ratios[target] = ratio
    
        # Step 5: Normalize and return
        return self._normalize_ratios(weighted_ratios, nuclide, reaction)
