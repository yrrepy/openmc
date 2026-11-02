#ifndef OPENMC_TALLIES_EXATALLY_H
#define OPENMC_TALLIES_EXATALLY_H

//! \file exatally.h
//! \brief ExaTallies accumulator/moments mode switch

#include <cstdint>
#include <string>
#include <utility> // for pair

#ifdef OPENMC_MPI
#include <mpi.h>
#endif

#include "openmc/tensor.h"
#include "openmc/vector.h"
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

//==============================================================================
// Projected moments
//==============================================================================

//! \brief One projection of a tally: a coarser view of the tally's filter
//! bins whose batch realizations get their own sum and sum of squares.
//!
//! The projected filter list follows the tally's filter order. Each tally
//! filter is kept (listed itself), collapsed (an energy filter nested in it is
//! listed instead), or summed over (not listed). Each batch, the fold adds
//! every bin's batch value into the projected bin of its row (the realization
//! y = P x), and then SUM += y and SUM_SQ += y * y, so the variance of the
//! projected quantity includes the covariance of the fine bins it sums.
struct TallyProjection {
  std::string name;        //!< Statepoint group name, unique within the tally
  vector<int32_t> filters; //!< Projected filters (model::tally_filters indices)

  // Derived by Tally::init_projections():

  //! Per tally filter: fine bin -> bin on its projected axis, -1 dropping the
  //! bin (outside the coarse energy range). Empty for a kept filter (the
  //! identity) and for a summed-over filter.
  vector<vector<int32_t>> maps;

  //! Per tally filter: stride of its projected axis in the projected
  //! filter-bin index (last projected filter fastest); 0 when summed over.
  vector<int64_t> strides;

  int64_t n_bins {0};     //!< Projected filter bins F_p (1 when all summed)
  bool fast_path {false}; //!< Every filter but the tally's last kept (case A)

  //! [F_p, n_score_bins, 2] SUM and SUM_SQ on the rank that folds them (the
  //! master, or every rank without tally reduction); empty elsewhere. Two
  //! moments in every moments mode that allows projections.
  tensor::Tensor<double> moments;

  //! This batch's projected realization y (under rma: this rank's partial,
  //! summed onto the master before the moments fold), holding projected
  //! filter-bin rows [first_row, last_row] x n_score_bins. Zero between
  //! batches.
  vector<double> realization;
  int64_t first_row {0};
  int64_t last_row {-1};

  //! Case B: one private realization per OpenMP thread, summed into
  //! realization after the fold (so no two threads write one value).
  vector<vector<double>> thread_partials;
};

//! \brief Parse the <projection> children of the <exatally> block under
//! `node`.
//!
//! \param node     Tally element that may hold an <exatally> child
//! \param context  Where the block is, for error text ("tally N")
//! \return One {name, filter ids} pair per projection, in input order
//! \throw std::runtime_error on a projection without a name or a filters
//!        attribute
vector<std::pair<std::string, vector<int32_t>>> parse_projections(
  pugi::xml_node node, const std::string& context);

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
