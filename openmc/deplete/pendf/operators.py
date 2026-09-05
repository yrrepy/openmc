"""PENDF wiring for the transport-coupled depletion operator.

Holds the option resolution behind ``CoupledOperator(reaction_rate_mode=
"pendf-flux", pendf_library=...)``: it cross-checks the mode against the
library, opens a library given as a path, settles the energy group structure
and splits ``reaction_rate_opts`` into the keyword arguments of
:class:`~openmc.deplete.pendf.helpers.PendfFluxCollapseHelper`. The operator
keeps a thin hook that calls in here.

Imported at module level by ``coupled_operator``, which ``openmc.deplete``
initializes before ``microxs``: importing ``..microxs``, ``.collapse`` or
``openmc.data.pendf`` at module level here would cycle, so those imports are
function-local.

.. versionadded:: 0.15.4
"""

import os

import numpy as np

from openmc.mgxs import GROUP_STRUCTURES, _canonical_group_structure_name


# Keys ``reaction_rate_opts`` may carry in 'pendf-flux' mode. ``energies`` is
# consumed here; the rest become PendfFluxCollapseHelper keyword arguments.
_PENDF_FLUX_OPTS = ('energies', 'reactions', 'nuclides', 'partial_binding')

# Options for a correction this cut deliberately does not implement.
_PENDF_URR_OPTS = ('urr_material_dilution', 'mat_ssf_nuclides')


def _resolve_group_structure(energies):
    """Resolve a group structure name or edge sequence to an edge array.

    Descending or degenerate edges are rejected here rather than after the
    first transport solve: the collapse engine bins with ``add.reduceat`` and
    would die with an opaque ``IndexError``.
    """
    if isinstance(energies, str):
        name = energies
        edges = np.asarray(
            GROUP_STRUCTURES[_canonical_group_structure_name(energies)],
            dtype=float)
    else:
        name = None
        edges = np.asarray(energies, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        named = f' {name!r}' if name is not None else ''
        raise ValueError(
            f'energies{named} must be at least two strictly ascending energy '
            f'group boundaries in [eV] (low to high); got {edges.size} values.')
    return edges


def _resolve_pendf_flux_options(pendf_library, reaction_rate_mode,
                                reaction_rate_opts):
    """Validate the pendf-flux arguments; return (library, energies, helper_opts).

    Parameters
    ----------
    pendf_library : openmc.data.PendfLibrary or path-like or None
        Library given to the operator. A path is opened here and owned by the
        operator for the rest of the process.
    reaction_rate_mode : str
        Reaction rate mode requested on the operator.
    reaction_rate_opts : dict or None
        Reaction rate options given to the operator. Never modified.

    Returns
    -------
    library : openmc.data.PendfLibrary or None
        Open PENDF library, or None when the mode does not use one.
    energies : numpy.ndarray or None
        Energy group boundaries in [eV] for the flux tally.
    helper_opts : dict or None
        Keyword arguments for
        :class:`~openmc.deplete.pendf.helpers.PendfFluxCollapseHelper`.

    """
    if reaction_rate_mode != 'pendf-flux':
        if pendf_library is not None:
            raise ValueError(
                "pendf_library requires reaction_rate_mode='pendf-flux'; "
                f"got {reaction_rate_mode!r}.")
        return None, None, None

    if pendf_library is None:
        raise ValueError(
            "reaction_rate_mode='pendf-flux' requires the pendf_library "
            'argument: an open PENDF library object or a path to a PENDF '
            'HDF5 library.')

    library = pendf_library
    if isinstance(library, (str, os.PathLike)):
        from openmc.data.pendf import open_pendf_library
        library = open_pendf_library(library)

    opts = dict(reaction_rate_opts) if reaction_rate_opts else {}

    # TODO(pendf-flux, wanted): URR material-dilution self-shielding for the coupled path --
    # build ``openmc.deplete.mat_ssf._MatSsfRowScaler(pendf_library, energies, densities,
    # mat_ssf_nuclides)`` from the operator's LIVE atom densities each step and pass it as
    # ``scaler=`` to ``_collapse_pendf_blocks`` -- and per-temperature PENDF libraries (one
    # library per material temperature). Both deliberately left out of the first cut
    # (user decision 2026-09-04); the option keys are rejected below until they are built.
    rejected = [key for key in _PENDF_URR_OPTS if key in opts]
    if rejected:
        raise NotImplementedError(
            f"reaction_rate_opts key(s) {', '.join(rejected)} are not "
            "supported in 'pendf-flux' mode. URR material-dilution "
            'self-shielding for the coupled path -- building a '
            '_MatSsfRowScaler from the live atom densities each step and '
            'passing it as scaler= to the collapse -- and per-temperature '
            'PENDF libraries are deliberately left out of the first cut; use '
            'the transport-independent path '
            '(MicroXS.from_multigroup_flux) for a URR-corrected collapse.')

    unknown = sorted(set(opts) - set(_PENDF_FLUX_OPTS))
    if unknown:
        raise ValueError(
            f"Unknown reaction_rate_opts key(s) for 'pendf-flux' mode: "
            f"{', '.join(unknown)}. Accepted keys: "
            f"{', '.join(_PENDF_FLUX_OPTS)}.")

    energies = opts.pop('energies', None)
    # A grouped library carries no pointwise data to rebin, so its own edges
    # are the only structure it can be collapsed on.
    group_edges = getattr(library, 'group_edges', None)
    if group_edges is not None:
        edges = np.asarray(group_edges, dtype=float)
        if energies is not None:
            given = _resolve_group_structure(energies)
            if not np.array_equal(given, edges):
                raise ValueError(
                    'energies must match the grouped PENDF library group '
                    f'structure ({edges.size - 1} groups) or be omitted; the '
                    f'given structure has {given.size - 1} groups. A grouped '
                    'library cannot be rebinned.')
    elif energies is None:
        raise ValueError(
            "reaction_rate_mode='pendf-flux' with a pointwise PENDF library "
            "requires reaction_rate_opts['energies']: a group structure name "
            "(e.g. 'CASMO-40') or an array of energy group boundaries in "
            '[eV]. Only a grouped library defines its own structure.')
    else:
        edges = _resolve_group_structure(energies)

    helper_opts = {
        'reactions': opts.pop('reactions', None),
        'nuclides': opts.pop('nuclides', None),
        'partial_binding': opts.pop('partial_binding', False),
    }
    return library, edges, helper_opts
