"""
openmc.deplete
==============

A depletion front-end tool.
"""

# Energy group structure constants for isomeric branching
# These must be defined before submodule imports to avoid circular import
CCFE_709_N_GROUPS = 709
UKAEA_1102_N_GROUPS = 1102

ISOMERIC_ENERGY_STRUCTURES = {
    'CCFE-709': CCFE_709_N_GROUPS,
    'UKAEA-1102': UKAEA_1102_N_GROUPS
}


def get_energy_structure_name(n_groups):
    """Get energy structure name from number of groups.

    Parameters
    ----------
    n_groups : int
        Number of energy groups

    Returns
    -------
    str or None
        'CCFE-709', 'UKAEA-1102', or None if unknown
    """
    for name, count in ISOMERIC_ENERGY_STRUCTURES.items():
        if n_groups == count:
            return name
    return None


from .nuclide import *
from .chain import *
from .openmc_operator import *
from .coupled_operator import *
from .independent_operator import *
from .microxs import *
from .gendf import *
from .reaction_rates import *
from .atom_number import *
from .stepresult import *
from .results import *
from .integrators import *
from .transfer_rates import *
from .r2s import *
from . import abc
from . import cram
from . import helpers
