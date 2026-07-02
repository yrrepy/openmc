#ifndef OPENMC_TALLIES_TRIGGER_H
#define OPENMC_TALLIES_TRIGGER_H

#include <string>

#include "pugixml.hpp"

namespace openmc {

//==============================================================================
// Type definitions
//==============================================================================

enum class TriggerMetric {
  variance,
  relative_error,
  standard_deviation,
  not_active
};

//! Stops the simulation early if a desired tally uncertainty is reached.

struct Trigger {
  TriggerMetric metric; //!< The type of uncertainty (e.g. std dev) measured
  double threshold;     //!< Uncertainty value below which trigger is satisfied
  bool ignore_zeros;    //!< Whether to allow zero tally bins to be ignored
  int score_index;      //!< Index of the relevant score in the tally's arrays
};

//! Stops the simulation early if a desired k-effective uncertainty is reached.

struct KTrigger {
  TriggerMetric metric {TriggerMetric::not_active};
  double threshold {0.};
};

//==============================================================================
// Global variable declarations
//==============================================================================

// TODO: consider a different namespace
namespace settings {
extern KTrigger keff_trigger;
}

//==============================================================================
// Non-memeber functions
//==============================================================================

void check_triggers();

//! \brief Whether any rma-storage tally carries an active trigger.
//!
//! When true, the tally-uncertainty walk must run collectively across ranks
//! (each rank owns only part of the moments), so the trigger check is invoked
//! on every rank rather than the master alone. The scan is over global tally
//! metadata, so it returns the same value on every rank.
bool has_rma_triggers();

} // namespace openmc
#endif // OPENMC_TALLIES_TRIGGER_H
