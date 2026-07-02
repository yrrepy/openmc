#include "openmc/message_passing.h"

namespace openmc {
namespace mpi {

int rank {0};
int n_procs {1};
bool master {true};

#ifdef OPENMC_MPI
// Shared/rma tally storage relies on MPI-3 shared-memory windows and
// MPI_Comm_split_type.
static_assert(MPI_VERSION >= 3,
  "OpenMC's shared/rma tally storage requires an MPI-3 implementation.");

MPI_Comm intracomm {MPI_COMM_NULL};
MPI_Comm node_comm {MPI_COMM_NULL};
MPI_Comm internode_comm {MPI_COMM_NULL};
int node_rank {0};
bool node_leader {true};
int n_nodes {1};
MPI_Datatype source_site {MPI_DATATYPE_NULL};
MPI_Datatype collision_track_site {MPI_DATATYPE_NULL};
#endif

extern "C" bool openmc_master()
{
  return mpi::master;
}

vector<int64_t> calculate_parallel_index_vector(int64_t size)
{
  vector<int64_t> result;
  result.resize(n_procs + 1);
  result[0] = 0;

#ifdef OPENMC_MPI

  // Populate the result with cumulative sum of the number of
  // surface source banks per process
  int64_t scan_total;
  MPI_Scan(&size, &scan_total, 1, MPI_INT64_T, MPI_SUM, intracomm);
  MPI_Allgather(
    &scan_total, 1, MPI_INT64_T, result.data() + 1, 1, MPI_INT64_T, intracomm);
#else
  result[1] = size;
#endif

  return result;
}

#ifdef OPENMC_MPI
// Specializations of the MPITypeMap template struct
template<>
const MPI_Datatype MPITypeMap<int>::mpi_type = MPI_INT;
template<>
const MPI_Datatype MPITypeMap<double>::mpi_type = MPI_DOUBLE;
#endif

} // namespace mpi

} // namespace openmc
