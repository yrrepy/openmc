#ifndef OPENMC_TALLIES_EXATALLY_PROFILE_H
#define OPENMC_TALLIES_EXATALLY_PROFILE_H

//! \file exatally_profile.h
//! \brief Per-rank profiling of the ExaTallies accumulator modes (profiling
//! branch only; observation only, results are unchanged).
//!
//! Environment variables, read once at initialization:
//!   OPENMC_PROFILE=<dir>         write <dir>/profile_rank<R>.json per rank at
//!                                the end of each simulation; <dir> must exist
//!   OPENMC_RMA_PROGRESS_POKE=<N> every N source particles per thread, enter
//!                                critical(openmc_rma) and MPI_Iprobe, so the
//!                                MPI progress engine runs during transport.
//!                                Only in history-based transport without the
//!                                shared secondary bank, only with > 1 rank
//!                                and an rma tally. 0 (default) = off.
//!   OPENMC_RMA_STAGING_ROWS=<K>  (existing test seam) force the staging K
//!
//! JSON schema (one object per file, one line per batch record):
//!   "static": host, ranks, node layout, threads, MPI thread level and library,
//!             commit, run sizes; "tallies": id, modes, bins, and for rma the
//!             owned rows, K, staging and window MiB
//!   "mem_after_init|mem_after_batch1|mem_final": {"VmHWM", "VmRSS"} in kB
//!             (after simulation init, after the first batch, at finalize)
//!   "batches": batch, active, "seconds" per Phase, particles, scores,
//!             scores_remote/local, rows_issued, accumulates, merged,
//!             flush_local, pokes, and per rma tally: rows_to[t], mass_to[t]
//!             (issued per target), recv_sum (window block before the fold,
//!             null when no fold ran), priv_sum (private plane)
//!   "tail": "seconds" per Phase spent after the last batch (finalize)

#include <cstdint>

#include "openmc/openmp_interface.h"
#include "openmc/timer.h"
#include "openmc/vector.h"

namespace openmc {
namespace profile {

//==============================================================================
// Global variables
//==============================================================================

extern bool enabled;      //!< OPENMC_PROFILE is set
extern int poke_interval; //!< OPENMC_RMA_PROGRESS_POKE, 0 = off

//! Timed phases of a batch. Timers accumulate across tallies within a batch.
enum class Phase {
  TRANSPORT,
  RMA_DRAIN,
  RMA_FLUSH_ALL,
  RMA_BARRIER,
  RMA_FOLD,
  SHARED_NODE_SYNC,
  SHARED_INTERNODE_REDUCE,
  SHARED_ZERO,
  REPLICATED_REDUCE,
  FOLD,
  TRIGGER_WALK,
  TRIGGER_REDUCE,
  STATEPOINT,
  BATCH,
  N
};

//! Per-thread event counters, folded at batch_end(). One cache line each.
struct alignas(64) ThreadCounters {
  int64_t scores {0}; //!< (filter bin, score bin) pairs scored
  int64_t pokes {0};  //!< progress pokes made
};

extern vector<ThreadCounters> thread_counters;
extern Timer phase_timers[static_cast<int>(Phase::N)];

//==============================================================================
// Functions
//==============================================================================

//! Read the environment (once) and check the output directory.
void init();

//! Open a batch record (before initialize_batch()). The first batch of a
//! simulation also records the static header and the memory after init.
void batch_begin();

//! Close the batch record (after finalize_batch()).
void batch_end();

//! Fold the counters and write this rank's JSON file (from
//! openmc_simulation_finalize()).
void finalize();

inline void phase_begin(Phase p)
{
  if (enabled)
    phase_timers[static_cast<int>(p)].start();
}

inline void phase_end(Phase p)
{
  if (enabled)
    phase_timers[static_cast<int>(p)].stop();
}

//! Times one phase for the life of the object.
class ScopedPhase {
public:
  explicit ScopedPhase(Phase p) : p_(p) { phase_begin(p_); }
  ~ScopedPhase() { phase_end(p_); }
  ScopedPhase(const ScopedPhase&) = delete;
  ScopedPhase& operator=(const ScopedPhase&) = delete;

private:
  Phase p_;
};

//! Count n scored (filter bin, score bin) pairs on the calling thread.
inline void add_scores(int64_t n)
{
  if (enabled)
    thread_counters[thread_num()].scores += n;
}

//! Count n progress pokes on the calling thread.
void add_pokes(int64_t n);

//! Particles between progress pokes for this simulation: poke_interval when
//! the poke applies (> 1 rank and an rma tally), else 0. Set by batch_begin().
int poke_every();

//! One progress poke: MPI_Iprobe under critical(openmc_rma), counted.
void rma_progress_poke();

} // namespace profile
} // namespace openmc

#endif // OPENMC_TALLIES_EXATALLY_PROFILE_H
