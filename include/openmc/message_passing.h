#ifndef OPENMC_MESSAGE_PASSING_H
#define OPENMC_MESSAGE_PASSING_H

#include <cstdint>

#ifdef OPENMC_MPI
#include <mpi.h>
#endif

#include "openmc/vector.h"

namespace openmc {
namespace mpi {

extern int rank;
extern int n_procs;
extern bool master;

#ifdef OPENMC_MPI
extern MPI_Datatype source_site;
extern MPI_Datatype collision_track_site;
extern MPI_Comm intracomm;

// Communicators for shared-storage tallies. node_comm groups ranks that share
// memory (one shared accumulator plane per node); internode_comm groups the
// node leaders (MPI_COMM_NULL on non-leaders). n_nodes is the number of nodes,
// known on every rank.
extern MPI_Comm node_comm;
extern MPI_Comm internode_comm;
extern int node_rank;    //!< rank within node_comm
extern bool node_leader; //!< node_rank == 0
extern int n_nodes;      //!< number of shared-memory nodes
#endif

//==============================================================================
// Template struct used to map types to MPI datatypes
// By having a single static data member, the template can
// be specialized for each type we know of. The specializations appear in the
// .cpp file since they are definitions.
//==============================================================================
#ifdef OPENMC_MPI
template<typename T>
struct MPITypeMap {
  static const MPI_Datatype mpi_type;
};
#endif

// Calculates global indices of the bank particles
// across all ranks using a parallel scan. This is used to write
// the surface source file in parallel runs. It will probably
// be used in the future for other types of bank like particles
// in flight used to kick off transient simulations.
//
// More abstractly, this just takes a number from each MPI rank,
// and returns a vector which is the exclusive parallel scan across
// all of those numbers, having a length of the number of MPI ranks
// plus one.
vector<int64_t> calculate_parallel_index_vector(int64_t size);

} // namespace mpi
} // namespace openmc

#endif // OPENMC_MESSAGE_PASSING_H
