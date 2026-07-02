// Cross-process atomic hammer test for the shared tally storage mode.
//
// Proves that openmc::atomic_score_add performs an address-free atomic RMW on
// an MPI-3 shared-memory window: every rank and thread hammers the same window
// elements and the totals must be exact. Lost updates (non-atomic cross-process
// adds) would leave the totals short. Run under two ranks on one node with
// several OpenMP threads to exercise both contention axes:
//
//   mpiexec -n 2 ./test_shared_window   (with OMP_NUM_THREADS >= 2)
//
// All MPI collectives complete before any assertion so a failing check on one
// rank cannot deadlock the other.

#include <mpi.h>

#include <cstdint>

#include "openmc/openmp_interface.h" // for openmc::atomic_score_add

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("shared window cross-process atomic hammer", "[hammer]")
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Intra-node shared communicator (one plane per node).
  MPI_Comm node_comm;
  MPI_Comm_split_type(
    MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
  int node_rank;
  int node_size;
  MPI_Comm_rank(node_comm, &node_rank);
  MPI_Comm_size(node_comm, &node_size);
  bool leader = (node_rank == 0);

  constexpr int64_t N = 16;     // window elements
  constexpr int64_t K = 200000; // add operations issued per rank

  // Leader allocates the whole plane; everyone queries its contiguous segment.
  MPI_Aint bytes = leader ? N * static_cast<MPI_Aint>(sizeof(double)) : 0;
  void* base = nullptr;
  MPI_Win win;
  MPI_Win_allocate_shared(
    bytes, sizeof(double), MPI_INFO_NULL, node_comm, &base, &win);
  MPI_Aint seg_bytes;
  int seg_disp;
  MPI_Win_shared_query(win, 0, &seg_bytes, &seg_disp, &base);
  double* plane = static_cast<double*>(base);

  int* model = nullptr;
  int model_flag = 0;
  MPI_Win_get_attr(win, MPI_WIN_MODEL, &model, &model_flag);
  bool unified = model_flag && (*model == MPI_WIN_UNIFIED);

  MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

  if (leader)
    for (int64_t i = 0; i < N; ++i)
      plane[i] = 0.0;
  MPI_Win_sync(win);
  MPI_Barrier(node_comm);
  MPI_Win_sync(win);

  // Every rank issues K adds of 1.0, spread over the N elements; the parallel
  // for partitions the K adds across threads, so element e receives K/N adds
  // per rank and node_size * (K/N) across the node. Threads on a rank and ranks
  // on the node collide on the same elements -- the atomicity being tested.
#pragma omp parallel for
  for (int64_t i = 0; i < K; ++i)
    openmc::atomic_score_add(&plane[i % N], 1.0);

  MPI_Win_sync(win);
  MPI_Barrier(node_comm);
  MPI_Win_sync(win);

  bool values_ok = true;
  double expected = static_cast<double>(node_size) * static_cast<double>(K / N);
  if (leader) {
    for (int64_t e = 0; e < N; ++e)
      if (plane[e] != expected)
        values_ok = false;
  }

  MPI_Win_unlock_all(win);
  MPI_Win_free(&win);
  MPI_Comm_free(&node_comm);

  // Assert only after every collective has completed.
  REQUIRE(unified);
  if (leader)
    REQUIRE(values_ok);
}

// Reproduces the shared end-of-batch protocol on a single machine by splitting
// MPI_COMM_WORLD into artificial "nodes" (all ranks still share memory, so an
// MPI_Win_allocate_shared works on each sub-communicator). This reaches the
// paths a real single node cannot: the leader-only internode reduce (step 2)
// and the non-master-node-leader plane zero (step 3). Needs a rank count that
// is a multiple of NODE_SIZE with at least two nodes, e.g. mpiexec -n 4.
TEST_CASE("shared multi-node end-of-batch protocol", "[multinode]")
{
  constexpr int NODE_SIZE = 2; // ranks per artificial node
  int rank;
  int nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (nprocs % NODE_SIZE != 0 || nprocs < 2 * NODE_SIZE) {
    SUCCEED("needs a multiple of NODE_SIZE and >= 2 nodes; skipping");
    return;
  }

  // Artificial node split: contiguous ranks form a node. All ranks are on the
  // real machine, so each sub-communicator can back a shared-memory window.
  MPI_Comm node_comm;
  MPI_Comm_split(MPI_COMM_WORLD, rank / NODE_SIZE, rank, &node_comm);
  int node_rank;
  int node_size;
  MPI_Comm_rank(node_comm, &node_rank);
  MPI_Comm_size(node_comm, &node_size);
  bool leader = (node_rank == 0);

  MPI_Comm internode_comm;
  MPI_Comm_split(
    MPI_COMM_WORLD, leader ? 0 : MPI_UNDEFINED, rank, &internode_comm);

  constexpr int64_t N = 16;
  constexpr int64_t K = 100000;

  MPI_Aint bytes = leader ? N * static_cast<MPI_Aint>(sizeof(double)) : 0;
  void* base = nullptr;
  MPI_Win win;
  MPI_Win_allocate_shared(
    bytes, sizeof(double), MPI_INFO_NULL, node_comm, &base, &win);
  MPI_Aint seg_bytes;
  int seg_disp;
  MPI_Win_shared_query(win, 0, &seg_bytes, &seg_disp, &base);
  double* plane = static_cast<double*>(base);

  MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
  if (leader)
    for (int64_t i = 0; i < N; ++i)
      plane[i] = 0.0;
  MPI_Win_sync(win);
  MPI_Barrier(node_comm);
  MPI_Win_sync(win);

  // Score: each rank issues K adds of 1.0 spread over the N elements.
#pragma omp parallel for
  for (int64_t i = 0; i < K; ++i)
    openmc::atomic_score_add(&plane[i % N], 1.0);

  // Step 1: publish node-local scores.
  MPI_Win_sync(win);
  MPI_Barrier(node_comm);
  MPI_Win_sync(win);

  // Step 2: node leaders reduce their planes onto the world master (rank 0 of
  // internode_comm), mirroring reduce_in_place_chunked's IN_PLACE-on-root form.
  bool world_master = (rank == 0);
  if (leader) {
    int inode_rank;
    MPI_Comm_rank(internode_comm, &inode_rank);
    if (inode_rank == 0)
      MPI_Reduce(
        MPI_IN_PLACE, plane, N, MPI_DOUBLE, MPI_SUM, 0, internode_comm);
    else
      MPI_Reduce(plane, nullptr, N, MPI_DOUBLE, MPI_SUM, 0, internode_comm);
  }

  // The master now holds the global per-element sum; each of the nprocs ranks
  // added K/N to each element. A non-root leader's plane is untouched by the
  // reduce (it is a send buffer) and still holds its node-local sum.
  double expected_master = static_cast<double>(nprocs) * (K / N);
  double expected_node = static_cast<double>(node_size) * (K / N);
  bool master_ok = true;
  bool nonroot_leader_ok = true;
  if (world_master) {
    for (int64_t e = 0; e < N; ++e)
      if (plane[e] != expected_master)
        master_ok = false;
  } else if (leader) {
    for (int64_t e = 0; e < N; ++e)
      if (plane[e] != expected_node)
        nonroot_leader_ok = false;
  }

  // Step 3: master zeros its plane; every non-master leader zeros its own.
  if (leader)
    for (int64_t e = 0; e < N; ++e)
      plane[e] = 0.0;

  // Step 4: resume.
  MPI_Win_sync(win);
  MPI_Barrier(node_comm);
  MPI_Win_sync(win);

  bool zeroed_ok = true;
  for (int64_t e = 0; e < N; ++e)
    if (plane[e] != 0.0)
      zeroed_ok = false;

  MPI_Win_unlock_all(win);
  MPI_Win_free(&win);
  if (internode_comm != MPI_COMM_NULL)
    MPI_Comm_free(&internode_comm);
  MPI_Comm_free(&node_comm);

  // Assert only after every collective has completed.
  REQUIRE(master_ok);
  REQUIRE(nonroot_leader_ok);
  REQUIRE(zeroed_ok);
}

int main(int argc, char* argv[])
{
  // Threads never call MPI (only window loads/stores), so FUNNELED is the
  // honest level requested here.
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
