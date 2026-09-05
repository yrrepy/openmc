"""PENDF reaction-rate helper for the transport-coupled operator.

Holds :class:`PendfFluxCollapseHelper`, the reaction rate helper behind
``CoupledOperator(reaction_rate_mode="pendf-flux")``. It tallies a multigroup
flux per burnable material and collapses it against a PENDF library with the
block engine in :mod:`openmc.deplete.pendf.collapse`, so every reaction the
depletion chain carries -- product-qualified isomer columns (``(n,gamma)_m1``)
included -- gets a rate from PENDF data rather than from continuous-energy
libraries that do not resolve those channels.

Units. The flux tally mean is the volume-integrated flux [particle-cm/src] per
material and group, and the block contraction is linear, so contracting the RAW
tally rows yields ``sum_g sigma_g [b] * phi_g`` directly: cross section times
volume-integrated flux, the same convention a direct tally with
``multiply_density=False`` produces. The operator divides by ``1e24 * V``
afterwards and that is the whole normalization -- the flux is deliberately NOT
routed through :func:`~openmc.deplete.microxs._normalize_flux_batch`.

Chunk cache. Staging PENDF rows means HDF5 reads, so the collapse is done for a
whole chunk of materials at once (``openmc.deplete.microxs._COLLAPSE_CHUNK_SIZE``
materials, read at call time) and cached until the chunk changes or the tally
means are reset. With each rank's materials contiguous that is one staging pass
per chunk per transport cycle, not one per material.

``coupled_operator`` imports this module at import time, and ``microxs`` (which
``.collapse`` and ``.chain_check`` build on) imports ``coupled_operator``. To
keep that from cycling, ``.collapse``, ``.chain_check``, ``..microxs`` and
``openmc.data.pendf`` are imported inside functions only.

.. versionadded:: 0.15.4
"""

# TODO(pendf-flux, wanted): URR material-dilution self-shielding for the coupled path --
# build ``openmc.deplete.mat_ssf._MatSsfRowScaler(pendf_library, energies, densities,
# mat_ssf_nuclides)`` from the operator's LIVE atom densities each step and pass it as
# ``scaler=`` to ``_collapse_pendf_blocks`` -- and per-temperature PENDF libraries (one
# library per material temperature). Both deliberately left out of the first cut
# (user decision 2026-09-04); the option keys are rejected below until they are built.

import warnings

import numpy as np

from openmc.data import REACTION_MT
from openmc.lib import Tally, MaterialFilter, EnergyFilter, load_nuclide
import openmc.lib
from ..abc import ReactionRateHelper
from ..nuclide import _ISOMER_SUFFIX


# A material temperature this far (in K) from the library's preprocessed
# temperature is worth one warning; PENDF data is preprocessed at a single
# temperature and cannot be broadened at depletion time.
_TEMPERATURE_TOLERANCE = 1.0


class PendfFluxCollapseHelper(ReactionRateHelper):
    """Class that generates one-group reaction rates from a PENDF flux collapse

    This class tallies a multigroup flux for every burnable material and
    collapses it with pointwise (or pre-grouped) PENDF cross sections through
    the block engine, which stages the cross section rows a block at a time and
    contracts them against a whole chunk of material fluxes, so no cross section
    table is held between transport cycles. Because the collapse is chain-driven,
    the MF=10 isomeric partials become their own product-qualified columns
    (``(n,gamma)_m1`` and the like), which continuous-energy helpers cannot fill.
    Select reactions can still be treated with a direct continuous-energy
    reaction rate tally; by default nothing is, fission included.

    .. versionadded:: 0.15.4

    Parameters
    ----------
    n_nucs : int
        Number of burnable nuclides tracked by
        :class:`openmc.deplete.CoupledOperator`
    n_reacts : int
        Number of reactions tracked by :class:`openmc.deplete.CoupledOperator`
    pendf_library : openmc.data.PendfLibrary or openmc.data.GroupedPendfLibrary
        Open PENDF library providing the cross sections. Opening a library from
        a path is the operator's job.
    chain : openmc.deplete.Chain
        Depletion chain in use. It is the authority for the isomeric row names:
        each MF=10 ``LFS`` partial is bound to the chain reaction carrying that
        ``pendf_lfs``.
    energies : iterable of float
        Energy group boundaries for the flux tally in [eV], already resolved
        (a group structure name is resolved by the operator).
    reactions : iterable of str, optional
        Reactions for which rates should be directly tallied with
        continuous-energy data. Defaults to an empty list, i.e. every reaction
        is collapsed from PENDF data. A reaction the chain resolves into
        isomeric partials (``(n,gamma)`` alongside ``(n,gamma)_m1``) is
        rejected: the continuous-energy total would double count the isomer
        share, whose PENDF partial keeps its own column.
    nuclides : iterable of str, optional
        Nuclides for which some reaction rates should be directly tallied. If
        None, then ``reactions`` will be used for all nuclides.
    ce_nuclides : set of str, optional
        Nuclides that have continuous-energy data. Only these can be loaded and
        direct-tallied; the collapse itself needs no continuous-energy data, so
        a nuclide outside this set still gets its collapsed rates. Defaults to
        None, meaning every nuclide.
    partial_binding : bool or collection of (str, str), optional
        Opt-in binding of stock reactions to the MF=10 ground partial, passed
        through to the collapse engine. Defaults to False.

    Attributes
    ----------
    nuclides : list of str
        All nuclides with desired reaction rates.
    energies : numpy.ndarray
        Energy group boundaries in [eV]

    """

    def __init__(self, n_nucs, n_reacts, pendf_library, chain, energies,
                 reactions=None, nuclides=None, ce_nuclides=None,
                 partial_binding=False):
        super().__init__(n_nucs, n_reacts)
        self._pendf_library = pendf_library
        self._chain = chain
        self._energies = np.asarray(energies, dtype=float)
        self._reactions_direct = (
            list(reactions) if reactions is not None else [])
        self._nuclides_direct = list(nuclides) if nuclides is not None else None
        self._ce_nuclides = ce_nuclides
        self._partial_binding = partial_binding

        # A direct tally is a continuous-energy tally, so an override naming a
        # nuclide the transport data set does not carry can never be served.
        if ce_nuclides is not None and self._nuclides_direct is not None:
            for name in self._nuclides_direct:
                if name not in ce_nuclides:
                    raise ValueError(
                        f"Direct-tally nuclide {name!r} has no "
                        "continuous-energy data and cannot be direct-tallied; "
                        "drop it from reaction_rate_opts['nuclides'] and let "
                        'the PENDF collapse serve it.')

        # A direct tally is a continuous-energy tally, so it can only score
        # canonical reaction names; the product-qualified ones exist in PENDF
        # data alone.
        for reaction in self._reactions_direct:
            if _ISOMER_SUFFIX.search(reaction):
                raise ValueError(
                    f"Direct reaction {reaction!r} is product-qualified. "
                    "Continuous-energy tallies cannot score isomeric product "
                    "channels; drop the '_mN' suffix to direct-tally the whole "
                    "reaction, or leave it to the PENDF collapse.")
            if reaction not in REACTION_MT:
                raise ValueError(
                    f"Direct reaction {reaction!r} is not a known reaction; "
                    "reaction_rate_opts['reactions'] takes names from "
                    "openmc.data.REACTION_MT, e.g. 'fission' or '(n,gamma)'.")

        # A direct tally scores the continuous-energy TOTAL of a reaction, but
        # when the chain resolves that reaction into isomeric partials the
        # PENDF column carrying the base name is the GROUND partial. Overwriting
        # it with the total, while the sibling '_mN' column keeps its partial,
        # would count the isomer share twice.
        in_scope = (self._nuclides_direct if self._nuclides_direct is not None
                    else [nuclide.name for nuclide in chain.nuclides])
        for reaction in self._reactions_direct:
            for name in in_scope:
                if name not in chain:
                    continue
                sibling = next(
                    (rx.type for rx in chain[name].reactions
                     if rx.type != reaction
                     and _ISOMER_SUFFIX.sub('', rx.type) == reaction), None)
                if sibling is None:
                    continue
                raise ValueError(
                    f"Direct reaction {reaction!r} has isomer-resolved "
                    f"siblings in the chain (e.g. {name} {sibling!r}). A "
                    'direct tally would give the ground product the '
                    'continuous-energy total while the sibling keeps its '
                    'PENDF partial, counting the isomer share twice; drop it '
                    "from reaction_rate_opts['reactions'] or restrict "
                    "reaction_rate_opts['nuclides'] to nuclides without those "
                    'siblings.')

        self._materials = None
        self._scores = []
        # Base (unqualified) reaction names handed to the collapse engine
        self._base_reactions = None
        self._table_nucs = []
        self._table_index = {}
        self._table_nuclides = None
        self._missing = frozenset()
        # (chunk index, (n_chunk, n_table_nucs, n_expanded) values, name -> col)
        self._collapse_cache = None
        # Precomputed scatter index arrays and the objects they were built from
        self._scatter = None
        self._scatter_key = ()
        self._checked_pathways = False
        self._checked_fission = False
        self._warned_temperature = False
        self._flux_tally = None
        self._flux_tally_means_cache = None
        self._rate_tally = None
        self._rate_tally_means_cache = None

    @ReactionRateHelper.nuclides.setter
    def nuclides(self, nuclides):
        # A changed nuclide list invalidates the collapsed chunk: its rows are
        # indexed by the table nuclide order, which is about to be rebuilt.
        if nuclides != self._nuclides:
            self._collapse_cache = None
            self._scatter = None
        ReactionRateHelper.nuclides.fset(self, nuclides)
        # Only the direct tally needs continuous-energy data loaded; the PENDF
        # collapse never touches it.
        if self._reactions_direct and self._nuclides_direct is None:
            # A rate nuclide with no continuous-energy data cannot be loaded or
            # scored; it keeps the rates the collapse gives it.
            direct = [n for n in nuclides
                      if self._ce_nuclides is None or n in self._ce_nuclides]
            for nuclide in direct:
                if nuclide not in openmc.lib.nuclides:
                    load_nuclide(nuclide)
            self._rate_tally.nuclides = direct
        # The operator sets this before the transport solve of every step, and
        # ``generate_tallies`` (which fills ``_scores``) has run by then, so
        # building the index here makes the fission-row guard and the
        # missing-nuclide warning fire before a transport is paid for.
        if self._scores:
            self._ensure_nuclide_index()

    def generate_tallies(self, materials, scores):
        """Produce multigroup flux and direct reaction rate tallies

        Parameters
        ----------
        materials : iterable of :class:`openmc.lib.Material`
            Burnable materials in the problem. Used to construct a
            :class:`openmc.lib.MaterialFilter`
        scores : iterable of str
            Reaction identifiers, e.g. ``"(n,gamma)"`` or ``"(n,gamma)_m1"``,
            in the column order of the operator's reaction rate matrix.
        """
        self._materials = materials
        self._scores = list(scores)

        # The collapse takes base reaction names: strip the '_mN' qualifiers,
        # dedupe and drop chain reactions with no MT mapping (one warning).
        # Pathway expansion puts the qualified columns back.
        from .chain_check import _default_pendf_reactions
        self._base_reactions = _default_pendf_reactions(self._chain)

        # Direct tallies only make sense for tracked reactions
        self._reactions_direct = [
            r for r in self._reactions_direct if r in self._scores]

        self._check_temperature(materials)

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

    def _check_temperature(self, materials):
        """Warn once when material temperatures differ from the library's."""
        lib_temperature = getattr(self._pendf_library, 'temperature', None)
        if lib_temperature is None or self._warned_temperature:
            return
        mismatched = []
        for i, mat in enumerate(materials):
            temperature = getattr(mat, 'temperature', None)
            if temperature is None:
                continue
            if abs(temperature - lib_temperature) > _TEMPERATURE_TOLERANCE:
                mismatched.append(f'{getattr(mat, "id", i)} '
                                  f'({temperature:g} K)')
        if not mismatched:
            return
        self._warned_temperature = True
        names = ', '.join(mismatched[:10])
        more = ' ...' if len(mismatched) > 10 else ''
        warnings.warn(
            f'{len(mismatched)} material(s) differ from the PENDF library '
            f'temperature of {lib_temperature:g} K: {names}{more}. PENDF data '
            'is preprocessed at one temperature and is not broadened at '
            'depletion time, so those rates use the library temperature.')

    @property
    def rate_tally_means(self):
        """The mean results of the tally of every material's reaction rates for this cycle
        """
        # If the mean cache is empty, fill it once with this transport cycle's results
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
        self._collapse_cache = None
        self._scatter = None
        if self._reactions_direct:
            self._rate_tally_means_cache = None

    @property
    def energies(self):
        """Energy group boundaries in [eV]."""
        return self._energies

    def _ensure_nuclide_index(self):
        """Index the nuclides with PENDF data; rebuild if the nuclide set changed."""
        if self._table_nuclides == self.nuclides:
            return

        available = set(self._pendf_library.nuclides)
        table_nucs = [n for n in self.nuclides if n in available]
        missing = frozenset(self.nuclides) - available
        if missing and missing != self._missing:
            names = ' '.join(sorted(missing)[:10])
            more = ' ...' if len(missing) > 10 else ''
            warnings.warn(
                f"{len(missing)} nuclides not in PENDF library will have "
                f"zero reaction rates unless directly tallied: {names}{more}")
        self._missing = missing

        changed = table_nucs != self._table_nucs
        self._table_nucs = table_nucs
        self._table_index = {n: i for i, n in enumerate(table_nucs)}
        self._table_nuclides = list(self.nuclides)
        # The table nuclides moved, so any cached collapse has the wrong shape
        self._collapse_cache = None
        self._scatter = None

        if changed or not self._checked_fission:
            self._checked_fission = True
            self._check_fission_rows()

    def _check_fission_rows(self):
        """Fail early when the library carries no MT=18 for the fission nuclides."""
        if ('fission' not in self._scores
                or 'fission' in self._reactions_direct):
            return
        fissionable = [
            n for n in self._table_nucs
            if n in self._chain
            and any(r.type == 'fission' for r in self._chain[n].reactions)]
        if not fissionable:
            return
        without = [n for n in fissionable
                   if 18 not in self._pendf_library.reactions(n)]
        if not without:
            return
        if len(without) == len(fissionable):
            raise ValueError(
                f'PENDF library has no MT=18 rows for any of the '
                f'{len(fissionable)} fission nuclide(s) in the chain; grouped '
                'libraries built before the MT=18 fix carry none -- rebuild '
                'the grouped library, or direct-tally fission with '
                "reaction_rate_opts={'reactions': ['fission']}")
        names = ' '.join(sorted(without)[:10])
        more = ' ...' if len(without) > 10 else ''
        warnings.warn(
            f'{len(without)} of {len(fissionable)} fission nuclide(s) have no '
            f'MT=18 rows in the PENDF library and get a zero fission rate: '
            f'{names}{more}. Rebuild the library, or direct-tally fission '
            "with reaction_rate_opts={'reactions': ['fission']}.")

    def _fill_chunk(self, chunk, chunk_size):
        """Collapse the PENDF rows against every flux of one chunk of materials."""
        # Lazy imports: this module is loaded with ``coupled_operator``, which
        # both of these import back through ``microxs``.
        from . import collapse as _collapse
        from . import chain_check as _chain_check

        n_groups = self._energies.size - 1
        fluxes = self.flux_tally_means.reshape(len(self._materials), n_groups)
        start = chunk * chunk_size
        phi = np.array(fluxes[start:start + chunk_size], dtype=float)

        # A NaN or negative flux row would poison every rate of that material
        # silently, so it is reported by material index instead.
        bad = ~np.isfinite(phi).all(axis=1)
        if bad.any():
            raise ValueError(
                f'Flux tally for material index {start + int(bad.argmax())} '
                'has non-finite groups; the transport results are unusable.')
        bad = (phi < 0.0).any(axis=1)
        if bad.any():
            raise ValueError(
                f'Flux tally for material index {start + int(bad.argmax())} '
                'has negative groups; the transport results are unusable.')

        micros = _collapse._collapse_pendf_blocks(
            self._table_nucs, self._base_reactions, self._energies,
            self._pendf_library, self._chain, phi,
            partial_binding=self._partial_binding, scaler=None)
        # Same always-on backstop and heap return as the independent path
        _collapse._clamp_negative_pendf_microxs(micros)
        _collapse._trim_heap()

        values = np.stack([micro.data[:, :, 0] for micro in micros])

        # A chain/library isomeric-pathway mismatch is a property of the pair,
        # not of the flux, so it is checked once for the whole run. The check
        # reads which rows are non-zero, though, so it must not depend on which
        # material happens to carry flux: a cold or empty material 0 would let
        # a real mismatch through. Every material of the chunk is aggregated
        # into one flux-independent probe instead, and a chunk with no flux at
        # all leaves the flag unset so the next chunk (or cycle) with flux runs
        # the check.
        if not self._checked_pathways and phi.any():
            from ..microxs import MicroXS
            probe = MicroXS(np.abs(values).sum(axis=0)[:, :, np.newaxis],
                            list(micros[0].nuclides),
                            list(micros[0].reactions))
            _chain_check._check_pathway_consistency(self._chain, probe)
            self._checked_pathways = True

        col = {name: j for j, name in enumerate(micros[0].reactions)}
        self._collapse_cache = (chunk, values, col)

    def _scatter_index(self, nuc_index, react_index, col):
        """Index arrays mapping collapsed (nuclide, column) cells onto the results."""
        key = (nuc_index, react_index, col)
        if (self._scatter is not None
                and all(a is b for a, b in zip(self._scatter_key, key))):
            return self._scatter

        rows, table = [], []
        for name, i_nuc in zip(self.nuclides, nuc_index):
            i_table = self._table_index.get(name)
            if i_table is not None:
                rows.append(i_nuc)
                table.append(i_table)
        cols, columns = [], []
        for score, i_rx in zip(self._scores, react_index):
            j = col.get(score)
            if j is not None:
                cols.append(i_rx)
                columns.append(j)

        self._scatter = (np.array(rows, dtype=int), np.array(table, dtype=int),
                         np.array(cols, dtype=int),
                         np.array(columns, dtype=int))
        self._scatter_key = key
        return self._scatter

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
        self._ensure_nuclide_index()

        # Read the chunk size off the module at CALL time (never bound into a
        # local constant at import) so a test can monkeypatch it.
        from .. import microxs as _microxs
        chunk_size = _microxs._COLLAPSE_CHUNK_SIZE

        chunk, local = divmod(mat_index, chunk_size)
        if self._collapse_cache is None or self._collapse_cache[0] != chunk:
            self._fill_chunk(chunk, chunk_size)
        values, col = self._collapse_cache[1], self._collapse_cache[2]

        # One vectorized scatter over (nuclides x scores); scores with no
        # collapsed column -- an unmappable chain type, or an isomeric channel
        # the library never expanded -- keep their zero.
        rows, table, cols, columns = self._scatter_index(
            nuc_index, react_index, col)
        if rows.size and cols.size:
            self._results_cache[np.ix_(rows, cols)] = \
                values[local][np.ix_(table, columns)]

        # Overwrite the (nuclide, reaction) pairs tallied directly
        if self._reactions_direct:
            nuclides_direct = self._rate_tally.nuclides
            shape = (len(nuclides_direct), len(self._reactions_direct))
            rx_rates = self.rate_tally_means[mat_index].reshape(shape)
            direct_rx_index = {score: i
                               for i, score in enumerate(self._reactions_direct)}
            direct_nuc_index = {nuc: i
                                for i, nuc in enumerate(nuclides_direct)}
            for name, i_nuc in zip(self.nuclides, nuc_index):
                i_direct = direct_nuc_index.get(name)
                if i_direct is None:
                    continue
                for score, i_rx in zip(self._scores, react_index):
                    if score in direct_rx_index:
                        self._results_cache[i_nuc, i_rx] = rx_rates[
                            i_direct, direct_rx_index[score]]

        return self._results_cache
