"""Dedicated module containing depletion function

Provided to avoid some circular imports
"""
import inspect
import warnings
from functools import lru_cache
from itertools import repeat, starmap
from multiprocessing import Pool
from typing import Any, Callable, List, Optional, TYPE_CHECKING

import numpy as np
from scipy.sparse import hstack

from openmc.mpi import comm
from .._sparse_compat import block_array

if TYPE_CHECKING:
    from .chain import Chain
    from .abc import TransportOperator
    from .reaction_rates import ReactionRates
    from .transfer_rates import TransferRates

# Configurable switch that enables / disables the use of
# multiprocessing routines during depletion
USE_MULTIPROCESSING = True

# Allow user to override the number of worker processes to use for depletion
# calculations
NUM_PROCESSES = None

def _distribute(items: List) -> List:
    """Distribute items across MPI communicator

    Parameters
    ----------
    items : list
        List of items of distribute

    Returns
    -------
    list
        Items assigned to process that called

    """
    min_size, extra = divmod(len(items), comm.size)
    j = 0
    for i in range(comm.size):
        chunk_size = min_size + int(i < extra)
        if comm.rank == i:
            return items[j:j + chunk_size]
        j += chunk_size


@lru_cache(maxsize=None)
def _accepts_isomeric_branching(matrix_func: Callable) -> bool:
    """Cached check for whether matrix_func accepts an isomeric_branching arg."""
    return 'isomeric_branching' in inspect.signature(matrix_func).parameters


def deplete(
    func: Callable,
    chain: 'Chain',
    n: List[np.ndarray],
    rates: 'ReactionRates',
    dt: float,
    current_timestep: Optional[int] = None,
    matrix_func: Optional[Callable] = None,
    transfer_rates: Optional['TransferRates'] = None,
    external_source_rates: Optional[Any] = None,
    *matrix_args: Any,
    operator: Optional['TransportOperator'] = None,
) -> List[np.ndarray]:
    """Deplete materials using given reaction rates for a specified time

    Parameters
    ----------
    func : callable
        Function to use to get new compositions. Expected to have the signature
        ``func(A, n0, t) -> n1``
    chain : openmc.deplete.Chain
        Depletion chain
    n : list of numpy.ndarray
        List of atom number arrays for each material. Each array in the list
        contains the number of [atom] of each nuclide.
    rates : openmc.deplete.ReactionRates
        Reaction rates (from transport operator)
    dt : float
        Time in [s] to deplete for
    current_timestep : int
        Current timestep index
    matrix_func : callable, optional
        Function to form the depletion matrix after calling ``matrix_func(chain,
        rates, fission_yields, isomeric_branching)``, where ``fission_yields = {parent: {product:
        yield_frac}}`` and ``isomeric_branching = {parent: {reaction: {target: ratio}}}``
        Expected to return the depletion matrix required by ``func``
    transfer_rates : openmc.deplete.TransferRates, Optional
        Transfer rates for continuous removal/feed.

        .. versionadded:: 0.14.0
    external_source_rates : openmc.deplete.ExternalSourceRates, Optional
        External source rates for continuous removal/feed.

        .. versionadded:: 0.15.3
    matrix_args: Any, optional
        Additional arguments passed to matrix_func
    operator : OperatorBase, keyword-only, optional
        Transport operator instance. If provided, isomeric branching data
        will be extracted from operator._isomeric_branching

        .. versionadded:: 0.15.4

    Returns
    -------
    n_result : list of numpy.ndarray
        Updated list of atom number arrays for each material. Each array in the
        list contains the number of [atom] of each nuclide.

    """
    # Operator now passed explicitly as parameter (clearer than hasattr detection)

    fission_yields = chain.fission_yields
    if len(fission_yields) == 1:
        fission_yields = repeat(fission_yields[0])
    elif len(fission_yields) != len(n):
        raise ValueError(
            "Number of material fission yield distributions {} is not "
            "equal to the number of compositions {}".format(
                len(fission_yields), len(n)))

    # Get isomeric branching from operator if available
    isomeric_branching = None
    if operator is not None:
        isomeric_branching = getattr(operator, '_isomeric_branching', None)

        if isomeric_branching is not None:
            if len(isomeric_branching) == 1:
                isomeric_branching = repeat(isomeric_branching[0])
            elif len(isomeric_branching) != len(n):
                warnings.warn(
                    f"Isomeric branching list length ({len(isomeric_branching)}) "
                    f"!= materials ({len(n)}). Disabling isomeric branching.",
                    UserWarning
                )
                isomeric_branching = None

    if matrix_func is None:
        if isomeric_branching is not None:
            matrices = []
            for rate, fy, iso in zip(rates, fission_yields, isomeric_branching):
                matrix = chain.form_matrix(rate, fission_yields=fy,
                                          isomeric_branching=iso)
                matrices.append(matrix)
        else:
            matrices = []
            for rate, fy in zip(rates, fission_yields):
                matrix = chain.form_matrix(rate, fission_yields=fy)
                matrices.append(matrix)
    else:
        if isomeric_branching is not None:
            accepts_iso = _accepts_isomeric_branching(matrix_func)
            matrices = []
            for c, r, fy, iso in zip(repeat(chain), rates,
                                    fission_yields, isomeric_branching):
                if accepts_iso:
                    m = matrix_func(c, r, fy, iso, *matrix_args)
                else:
                    m = matrix_func(c, r, fy, *matrix_args)
                matrices.append(m)
        else:
            matrices = []
            for c, r, fy in zip(repeat(chain), rates, fission_yields):
                m = matrix_func(c, r, fy, *matrix_args)
                matrices.append(m)

    if (transfer_rates is not None and
        current_timestep in transfer_rates.external_timesteps):
        # Calculate transfer rate terms as diagonal matrices
        transfers = map(chain.form_rr_term, repeat(transfer_rates),
                        repeat(current_timestep), transfer_rates.local_mats)

        # Subtract transfer rate terms from Bateman matrices
        matrices = [matrix - transfer for (matrix, transfer) in zip(matrices,
                                                                    transfers)]

        if transfer_rates.redox:
            for mat_idx, mat_id in enumerate(transfer_rates.local_mats):
                if mat_id in transfer_rates.redox:
                    matrices[mat_idx] = chain.add_redox_term(matrices[mat_idx],
                                                transfer_rates.redox[mat_id][0],
                                                transfer_rates.redox[mat_id][1])

        if current_timestep in transfer_rates.index_transfer:
            # Gather all on comm.rank 0
            matrices = comm.gather(matrices)
            n = comm.gather(n)

            if comm.rank == 0:
                # Expand lists
                matrices = [elm for matrix in matrices for elm in matrix]
                n = [n_elm for n_mat in n for  n_elm in n_mat]

                # Calculate transfer rate terms as diagonal matrices
                transfer_pair = {}
                for mat_pair in transfer_rates.index_transfer[current_timestep]:
                    transfer_matrix = chain.form_rr_term(transfer_rates,
                                                         current_timestep,
                                                         mat_pair)

                    # check if destination material has a redox control
                    if mat_pair[0] in transfer_rates.redox:
                        transfer_matrix = chain.add_redox_term(transfer_matrix,
                                          transfer_rates.redox[mat_pair[0]][0],
                                          transfer_rates.redox[mat_pair[0]][1])
                    transfer_pair[mat_pair] = transfer_matrix

                # Combine all matrices together in a single matrix of matrices
                # to be solved in one go
                n_rows = n_cols = len(transfer_rates.burnable_mats)
                rows = []
                for row in range(n_rows):
                    cols = []
                    for col in range(n_cols):
                        mat_pair = (transfer_rates.burnable_mats[row],
                                    transfer_rates.burnable_mats[col])
                        if row == col:
                            # Fill the diagonals with the Bateman matrices
                            cols.append(matrices[row])
                        elif mat_pair in transfer_rates.index_transfer[current_timestep]:
                            # Fill the off-diagonals with the transfer pair matrices
                            cols.append(transfer_pair[mat_pair])
                        else:
                            cols.append(None)

                    rows.append(cols)
                matrix = block_array(rows)

                # Concatenate vectors of nuclides in one
                n_multi = np.concatenate(n)
                n_result = func(matrix, n_multi, dt)

                # Split back the nuclide vector result into the original form
                n_result = np.split(n_result, np.cumsum([len(i) for i in n])[:-1])

            else:
                n_result = None

            # Broadcast result to other ranks
            n_result = comm.bcast(n_result)
            # Distribute results across MPI
            n_result = _distribute(n_result)

            return n_result

    if (external_source_rates is not None and
        current_timestep in external_source_rates.external_timesteps):
        # Calculate external source term vectors
        sources = map(chain.form_ext_source_term, repeat(external_source_rates),
                      repeat(current_timestep), external_source_rates.local_mats)

        # stack vector column at the end of the matrix
        matrices = [
            hstack([matrix, source])
            for matrix, source in zip(matrices, sources)
        ]

        # Add a last row of zeroes to the matrices and append 1 to the last row
        # of the nuclide vectors
        for i, matrix in enumerate(matrices):
            if not np.equal(*matrix.shape):
                matrix.resize(matrix.shape[1], matrix.shape[1])
                n[i] = np.append(n[i], 1.0)

    inputs = zip(matrices, n, repeat(dt))

    if USE_MULTIPROCESSING:
        with Pool(NUM_PROCESSES) as pool:
            n_result = list(pool.starmap(func, inputs))
    else:
        n_result = list(starmap(func, inputs))

    # Remove extra value at the end of the nuclide vectors
    if (external_source_rates is not None and
        current_timestep in external_source_rates.external_timesteps):
        external_source_rates.reformat_nuclide_vectors(n)
        external_source_rates.reformat_nuclide_vectors(n_result)

    return n_result
