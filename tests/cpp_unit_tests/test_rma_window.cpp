// Remote-scoring hammer test for the rma tally storage mode.
//
// Drives a real Tally in rma mode through the actual score_add seam: every rank
// (and, where the MPI provides thread serialization, every OpenMP thread)
// scores integer-valued contributions into ALL filter-bin rows. Remote bins go
// through the row coalescer -> per-(thread, target) double-buffered staging ->
// batched hindexed MPI_Accumulate path; local bins go to the private plane. The
// end-of-batch drain + publish complete every accumulate, and the fold on each
// owner must recover the exact cross-rank sum for the rows it owns. A lost or
// double-counted staged row (a wrap/flush or drain bug) would leave a row off.
//
// Run under several ranks on one node with several OpenMP threads:
//   mpiexec -n 2 ./test_rma_window "[rma]"   (with OMP_NUM_THREADS >= 2)
//
// The three cases share one driver but vary the staging depth K via the
// OPENMC_RMA_STAGING_ROWS test seam: default (mostly drain), K=2 (repeated
// buffer wrap + both-in-flight MPI_Win_flush_local(target)), and K larger than
// any target's row count (pure end-of-batch drain of partial buffers). All MPI
// collectives (window teardown included) complete before any assertion so a
// failing check on one rank cannot deadlock another.

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib> // for setenv, unsetenv
#include <string>
#include <vector>

#include "openmc/constants.h"
#include "openmc/message_passing.h"
#include "openmc/openmp_interface.h" // for num_threads
#include "openmc/settings.h"
#include "openmc/tallies/filter.h"
#include "openmc/tallies/filter_energy.h"
#include "openmc/tallies/tally.h"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace openmc;

// Build an rma tally with n_bins energy bins and n_scores scores, score
// contrib=(rank+1) into every (bin, score) from every rank/thread through the
// real seam, publish + fold, and verify each owned moment row holds the exact
// cross-rank sum 1 + 2 + ... + n_procs. Returns whether every owned row
// matched; the tally (and its collective window) is torn down before returning.
static bool run_rma_cycle(int64_t n_bins, int n_scores)
{
  // The Tally rma paths read these globals; initialize_mpi would set them in a
  // real run, but this test drives Tally directly.
  mpi::intracomm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi::rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi::n_procs);
  mpi::master = (mpi::rank == 0);

  // Make the fold normalization exactly 1: EIGENVALUE => total_source = 1,
  // contributing particles = n_particles = 1, gen_per_batch = 1.
  settings::run_mode = RunMode::EIGENVALUE;
  settings::solver_type = SolverType::MONTE_CARLO;
  settings::reduce_tallies = true;
  settings::n_particles = 1;
  settings::gen_per_batch = 1;

  // One energy filter with unit-width bins 0..n_bins, plus n_scores scores.
  Filter* f = Filter::create("energy");
  auto* ef = dynamic_cast<EnergyFilter*>(f);
  std::vector<double> boundaries(n_bins + 1);
  for (int64_t i = 0; i <= n_bins; ++i)
    boundaries[i] = static_cast<double>(i);
  ef->set_bins({boundaries.data(), boundaries.size()});

  std::vector<std::string> scores;
  scores.push_back("flux");
  if (n_scores >= 2)
    scores.push_back("total");

  Tally* tally = Tally::create();
  tally->set_filters({&f, 1});
  tally->set_scores(scores);
  tally->set_storage(TallyStorage::RMA);
  tally->set_strides();
  tally->init_results();

  // Score every (bin, score) once per rank through the real seam. Round-robin
  // scheduling spreads each thread's bins across all owner ranks, so several
  // threads stage to the same target concurrently (contending on the named
  // critical) and each thread talks to multiple targets. Fall back to a single
  // thread if the MPI cannot serialize threaded MPI calls.
  int provided;
  MPI_Query_thread(&provided);
  const int nt = (provided >= MPI_THREAD_SERIALIZED) ? num_threads() : 1;
  const double contrib = static_cast<double>(mpi::rank + 1);
#pragma omp parallel for num_threads(nt) schedule(static, 1)
  for (int64_t bin = 0; bin < n_bins; ++bin)
    for (int s = 0; s < n_scores; ++s)
      tally->score_add(bin, s, contrib);

  // End of batch: drain staging + complete accumulates, then fold owned rows.
  tally->rma_publish();
  tally->accumulate();

  // Each rank added (rank + 1) to every bin; the owning rank's fold sums them.
  const int64_t P = mpi::n_procs;
  const double expected = static_cast<double>(P * (P + 1) / 2);
  const int64_t bpr = std::max<int64_t>(1, (n_bins + P - 1) / P);
  const int64_t first =
    std::min<int64_t>(static_cast<int64_t>(mpi::rank) * bpr, n_bins);
  const int64_t last =
    std::min<int64_t>(static_cast<int64_t>(mpi::rank + 1) * bpr, n_bins);
  const int64_t n_rows = last - first;

  bool ok = true;
  const auto& m = tally->moments();
  for (int64_t i = 0; i < n_rows; ++i)
    for (int s = 0; s < n_scores; ++s)
      if (m(i, s, TallyMoment::SUM) != expected)
        ok = false;

  // Each bin is scored once per rank, so no staging row should have been merged
  // into another as a duplicate displacement.
  ok &= (tally->rma_merged_rows() == 0);

  // Collective window teardown on every rank before the caller asserts.
  free_memory_tally();
  return ok;
}

// Like run_rma_cycle, but each rank scores every (bin, score) `revisits` times
// in a round-robin order [0,1,..,n_bins-1, 0,1,..] on a single thread, so a
// remote bin is revisited NON-consecutively (other bins fall in between). The
// coalescer only merges consecutive scores, so each revisit appends a separate
// staging row carrying the same target displacement; the staging must merge
// those before issuing or it hands MPI_Accumulate a target datatype with
// overlapping entries (forbidden). Verifies the exact sum survives.
static bool run_rma_revisit(int64_t n_bins, int n_scores, int revisits)
{
  mpi::intracomm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi::rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi::n_procs);
  mpi::master = (mpi::rank == 0);

  settings::run_mode = RunMode::EIGENVALUE;
  settings::solver_type = SolverType::MONTE_CARLO;
  settings::reduce_tallies = true;
  settings::n_particles = 1;
  settings::gen_per_batch = 1;

  Filter* f = Filter::create("energy");
  auto* ef = dynamic_cast<EnergyFilter*>(f);
  std::vector<double> boundaries(n_bins + 1);
  for (int64_t i = 0; i <= n_bins; ++i)
    boundaries[i] = static_cast<double>(i);
  ef->set_bins({boundaries.data(), boundaries.size()});

  std::vector<std::string> scores;
  scores.push_back("flux");
  if (n_scores >= 2)
    scores.push_back("total");

  Tally* tally = Tally::create();
  tally->set_filters({&f, 1});
  tally->set_scores(scores);
  tally->set_storage(TallyStorage::RMA);
  tally->set_strides();
  tally->init_results();

  // Single-threaded so the round-robin revisit order is deterministic; the
  // staging still batches and (must) dedup. Default K keeps the revisits of a
  // bin in the same buffer, which is where the overlap would occur.
  const double contrib = static_cast<double>(mpi::rank + 1);
  for (int64_t idx = 0; idx < n_bins * revisits; ++idx) {
    const int64_t bin = idx % n_bins;
    for (int s = 0; s < n_scores; ++s)
      tally->score_add(bin, s, contrib);
  }

  tally->rma_publish();
  tally->accumulate();

  const int64_t P = mpi::n_procs;
  const double expected = static_cast<double>(revisits) * (P * (P + 1) / 2);
  const int64_t bpr = std::max<int64_t>(1, (n_bins + P - 1) / P);
  const int64_t first =
    std::min<int64_t>(static_cast<int64_t>(mpi::rank) * bpr, n_bins);
  const int64_t last =
    std::min<int64_t>(static_cast<int64_t>(mpi::rank + 1) * bpr, n_bins);
  const int64_t n_rows = last - first;

  bool ok = true;
  const auto& m = tally->moments();
  for (int64_t i = 0; i < n_rows; ++i)
    for (int s = 0; s < n_scores; ++s)
      if (m(i, s, TallyMoment::SUM) != expected)
        ok = false;

  // The non-consecutive revisits must have produced duplicate-displacement rows
  // that the staging merged before issuing. If none were merged, the scenario
  // did not exercise the path -- and on a strict MPI the un-merged overlapping
  // target datatype would have been illegal.
  ok &= (tally->rma_merged_rows() > 0);

  free_memory_tally();
  return ok;
}

TEST_CASE("rma remote-scoring hammer", "[rma][hammer]")
{
  // Default staging depth: end-to-end correctness through the real seam.
  unsetenv("OPENMC_RMA_STAGING_ROWS");
  REQUIRE(run_rma_cycle(/*n_bins=*/200, /*n_scores=*/2));
}

TEST_CASE("rma staging buffer wrap and flush", "[rma][liveness]")
{
  // K forced to 2: each remote target receives many rows, so the double buffers
  // wrap repeatedly and the both-in-flight -> MPI_Win_flush_local(target) path
  // must retire an origin payload before it is reused. A premature reuse would
  // corrupt an in-flight accumulate and drop the sum.
  setenv("OPENMC_RMA_STAGING_ROWS", "2", 1);
  bool ok = run_rma_cycle(/*n_bins=*/64, /*n_scores=*/1);
  unsetenv("OPENMC_RMA_STAGING_ROWS");
  REQUIRE(ok);
}

TEST_CASE("rma end-of-batch drain of partial buffers", "[rma][drain]")
{
  // K larger than the rows any target receives: no buffer fills during scoring,
  // so correctness rests entirely on the end-of-batch drain issuing the final
  // partial buffers.
  setenv("OPENMC_RMA_STAGING_ROWS", "1024", 1);
  bool ok = run_rma_cycle(/*n_bins=*/12, /*n_scores=*/1);
  unsetenv("OPENMC_RMA_STAGING_ROWS");
  REQUIRE(ok);
}

TEST_CASE("rma non-consecutive remote-bin revisits", "[rma][revisit]")
{
  // Each remote bin is scored several times, non-consecutively, so the staging
  // buffer accrues multiple rows at the same target displacement. They must be
  // merged before the MPI_Accumulate; otherwise the target datatype has
  // overlapping entries (undefined). Default K keeps the revisits co-resident.
  unsetenv("OPENMC_RMA_STAGING_ROWS");
  REQUIRE(run_rma_revisit(/*n_bins=*/24, /*n_scores=*/2, /*revisits=*/5));
}

int main(int argc, char* argv[])
{
  // The rma remote arm issues MPI from inside the OpenMP scoring region under a
  // named critical, so request thread serialization exactly as a real run does.
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
