#ifndef OPENMC_TALLIES_EXATALLY_H
#define OPENMC_TALLIES_EXATALLY_H

//! \file exatally.h
//! \brief ExaTallies accumulator/moments mode switch

#include <cstdint>
#include <string>

#ifdef OPENMC_MPI
#include <mpi.h>
#endif

#include "pugixml.hpp"

namespace openmc {

//==============================================================================
// ExaTallies mode switch
//==============================================================================

// Where a Tally's per-batch accumulator (and, for rma, its moments) is homed
// (the accumulator mode):
//   REPLICATED - every rank owns a full private copy (default)
//   SHARED     - one accumulator plane per shared-memory node (MPI-3 window)
//   RMA        - accumulator and moments block-distributed across all ranks
enum class AccumulatorMode { REPLICATED, SHARED, RMA };

// Which cross-batch moments a Tally keeps per bin (the moments mode):
//   NONE - no moments during the run: the accumulator is never zeroed and
//          holds the run total, collected and scaled into one SUM plane only
//          for a statepoint and at the end of the run (no standard deviation)
//   MEAN - the sum only (one moment plane; no standard deviation)
//   FULL - sum and sum of squares, plus the higher moments when requested
enum class MomentsMode { NONE, MEAN, FULL };

//! The modes selected by one <exatally> block: per tally in tallies.xml, or
//! the global default at the top level of settings.xml.
struct ExaTally {
  AccumulatorMode accumulator {AccumulatorMode::REPLICATED};
  MomentsMode moments {MomentsMode::FULL};
};

//! \brief Parse the <exatally> block under `node` on top of `defaults`.
//!
//! A missing block or a missing child keeps the corresponding default.
//! \param node      Element that may hold an <exatally> child
//! \param defaults  Modes used for any setting the block does not give
//! \param context   Where the block is, for error text ("settings.xml" or
//!                  "tally N")
//! \return The resolved modes
//! \throw std::runtime_error on an invalid value
ExaTally parse_exatally(
  pugi::xml_node node, const ExaTally& defaults, const std::string& context);

//! The input word of a moments mode ("none", "mean" or "full"), as written to
//! the statepoint and used in messages.
const char* moments_mode_name(MomentsMode mode);

#ifdef OPENMC_MPI
//! \brief Sum a contiguous accumulator onto the master (rank 0 of `comm`) in
//! place, in chunks of at most MAX_MPI_CHUNK_DOUBLES.
//!
//! The chunking also bounds the MPI library's own reduction temporaries, which
//! otherwise scale with the whole plane. Non-master ranks pass their buffer as
//! the send side; only the master's buffer holds the sum afterwards.
//! \param accum  Accumulator of n doubles (in place on the master)
//! \param n      Number of doubles
//! \param comm   Communicator whose rank 0 is the world master
void reduce_accum_to_master(double* accum, int64_t n, MPI_Comm comm);

//! \brief Print the run statistics of every distributed rma tally at
//! verbosity 8: remote rows issued, MPI_Accumulate calls, merged rows, and
//! remote rows per source particle, summed over all ranks.
//!
//! Collective on all ranks (one MPI_Reduce per distributed rma tally, in tally
//! order); a no-op below verbosity 8. Called at the end of the run.
void report_rma_statistics();
#endif

} // namespace openmc

#endif // OPENMC_TALLIES_EXATALLY_H
