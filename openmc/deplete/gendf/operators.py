"""GENDF wiring for the depletion operators.

Holds the GENDF-specific setup blocks that used to sit inline in
:class:`openmc.deplete.CoupledOperator`: validation of the opt-in (n,n')
fallback and construction of the isomeric-branching helper. The operator keeps
thin hooks that call in here.

Imported at module level by ``coupled_operator``, which ``openmc.deplete``
initializes before ``microxs``: this module may import ``..abc``/``..helpers``
at module level, but importing ``..microxs`` there would cycle (use a
function-local import if it is ever needed).

.. versionadded:: 0.15.4
"""

from warnings import warn

import numpy as np

from openmc.mgxs import GROUP_STRUCTURES
from .helpers import IsomericBranchingHelper


def _validate_mt4_fallback(gendf_library, reaction_rate_mode, gendf_mt4_fallback):
    """Check the opt-in (n,n') fallback against the reaction rate mode."""
    # Opt-in (n,n') fallback for CE modes lacking lumped MT=4 data
    if gendf_mt4_fallback:
        if gendf_library is None:
            raise ValueError(
                "gendf_mt4_fallback requires the gendf_library argument.")
        if reaction_rate_mode == "direct":
            raise ValueError(
                "gendf_mt4_fallback requires a flux-tallying reaction "
                "rate mode ('direct_with_flux' or 'flux').")
        if reaction_rate_mode == "gendf-flux":
            warn("gendf_mt4_fallback has no effect in 'gendf-flux' mode; "
                 "(n,n') rates are computed natively from GENDF MT=4.")
            gendf_mt4_fallback = False
    return gendf_mt4_fallback


def _setup_coupled_isomeric_branching(operator):
    """Set up isomeric branching for CoupledOperator.

    Requires a GENDF library, a flux-tallying reaction rate mode
    ("direct_with_flux" or "gendf-flux") on the GENDF group structure, and
    a backend exposing get_branching_ratios(). Chains carrying isomeric
    branching targets raise when any of these is missing; chains without
    such targets skip branching silently.
    """
    operator._isomeric_helper = None
    operator._isomeric_branching = None

    if not operator.chain.isomeric_branching_targets:
        return

    if operator._gendf_library is None:
        raise ValueError(
            "Chain has isomeric branching targets but no GENDF library "
            "was provided. Pass gendf_library= to the operator, or use a "
            "chain without isomeric branching data."
        )

    if not hasattr(operator._rate_helper, 'get_flux_spectrum'):
        raise ValueError(
            "Chain has isomeric branching targets but the reaction rate "
            "helper does not tally a flux spectrum. Use 'direct_with_flux' "
            "or 'gendf-flux' mode, or use a chain without isomeric "
            "branching data."
        )

    # The flux tally must use the GENDF group structure for the
    # branching-ratio weights to apply (only 'flux' mode can differ)
    expected = GROUP_STRUCTURES[operator._gendf_library.energy_structure]
    if not np.array_equal(operator._rate_helper.energies, expected):
        raise ValueError(
            "Isomeric branching requires the flux tally energy group "
            "boundaries to match the GENDF library's "
            f"'{operator._gendf_library.energy_structure}' group "
            f"structure ({len(expected) - 1} groups); got "
            f"{len(operator._rate_helper.energies) - 1} groups. Omit "
            "reaction_rate_opts['energies'] to use the GENDF "
            "structure automatically.")

    if not hasattr(operator._gendf_library, 'get_branching_ratios'):
        raise ValueError(
            "GENDF library backend does not support get_branching_ratios(). "
            "Re-patch chain with the updated patcher tool to add the "
            "gendf_lfs attribute, use the Python backend with a decay_file, "
            "or use a chain without isomeric branching data."
        )

    operator._isomeric_helper = IsomericBranchingHelper(
        operator.chain,
        operator._gendf_library,
    )


def _setup_independent_isomeric_branching(operator):
    """Set up isomeric branching helper if chain has targets and GENDF library is available."""
    operator._isomeric_branching = None

    if not operator.chain.isomeric_branching_targets:
        return

    if operator._gendf_library is None:
        raise ValueError(
            "Chain has isomeric branching targets but no GENDF library "
            "was provided. Pass gendf_library= to the operator, or use a "
            "chain without isomeric branching data."
        )

    if not hasattr(operator._gendf_library, 'get_branching_ratios'):
        raise ValueError(
            "GENDF library backend does not support get_branching_ratios(). "
            "Re-patch chain with the updated patcher tool to add the "
            "gendf_lfs attribute, use the Python backend with a decay_file, "
            "or use a chain without isomeric branching data."
        )

    helper = IsomericBranchingHelper(
        operator.chain,
        operator._gendf_library,
    )

    # The group structure is authoritative on the GENDF library; flux-carried
    # energy bounds are inert metadata here. The helper takes spectra only and
    # always collapses on the library grid.
    operator._isomeric_branching = helper.compute_for_materials(
        [flux_spectrum for flux_spectrum, _ in operator._flux_with_energy]
    )
