#ifndef OPENMC_TALLIES_EXATALLY_H
#define OPENMC_TALLIES_EXATALLY_H

//! \file exatally.h
//! \brief ExaTallies accumulator/moments mode switch

#include <string>

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
//   NONE - no moments (reserved, not implemented yet)
//   MEAN - the running sum only (reserved, not implemented yet)
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
//! A missing block or a missing child keeps the corresponding default. Only
//! the 'full' moments mode is implemented; 'mean' and 'none' are rejected.
//! \param node      Element that may hold an <exatally> child
//! \param defaults  Modes used for any setting the block does not give
//! \param context   Where the block is, for error text ("settings.xml" or
//!                  "tally N")
//! \return The resolved modes
//! \throw std::runtime_error on an invalid or unimplemented value
ExaTally parse_exatally(
  pugi::xml_node node, const ExaTally& defaults, const std::string& context);

} // namespace openmc

#endif // OPENMC_TALLIES_EXATALLY_H
