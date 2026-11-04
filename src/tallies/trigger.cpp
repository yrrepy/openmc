#include "openmc/tallies/trigger.h"

#include <cmath>
#include <utility> // for std::pair

#include <fmt/core.h>

#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/message_passing.h"
#include "openmc/reaction.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/tallies/exatally_profile.h"
#include "openmc/tallies/tally.h"

namespace openmc {

//==============================================================================
// Global variable definitions
//==============================================================================

namespace settings {
KTrigger keff_trigger;
}

//==============================================================================
// Non-member functions
//==============================================================================

std::pair<double, double> get_tally_uncertainty(
  double sum, double sum_sq, int n)
{
  auto mean = sum / n;

  // if the result has no contributions, return an invalid pair
  if (mean == 0)
    return {-1, -1};

  double std_dev = std::sqrt((sum_sq / n - mean * mean) / (n - 1));
  double rel_err = (mean != 0.) ? std_dev / std::abs(mean) : 0.;

  return {std_dev, rel_err};
}

//! Find the limiting limiting tally trigger.
//
//! param[out] ratio The uncertainty/threshold ratio for the most limiting
//!   tally trigger
//! param[out] tally_id The ID number of the most limiting tally
//! param[out] score The most limiting tally score bin

void check_tally_triggers(double& ratio, int& tally_id, int& score)
{
  ratio = 0.;
  tally_id = 0;
  score = 0;
  bool found_no_contribution = false;

  profile::phase_begin(profile::Phase::TRIGGER_WALK);
  for (auto i_tally = 0;
       i_tally < model::tallies.size() && !found_no_contribution; ++i_tally) {
    const Tally& t {*model::tallies[i_tally]};

#ifdef OPENMC_MPI
    // Under rma each rank walks only the moment rows it owns; every other mode
    // homes the moments on the master, so only the master walks those. A rank
    // that owns nothing (or a non-master rank on a replicated tally) simply
    // contributes ratio 0 to the reduction below.
    const bool is_rma = (t.accumulator() == AccumulatorMode::RMA);
    if (!is_rma && !mpi::master)
      continue;
    const int64_t n_rows = is_rma ? t.rma_n_rows() : t.n_filter_bins();
#else
    const int64_t n_rows = t.n_filter_bins();
#endif

    // Ignore tallies with less than two realizations.
    if (t.n_realizations_ < 2)
      continue;

    for (const auto& trigger : t.triggers_) {
      // Skip trigger if it is not active
      if (trigger.metric == TriggerMetric::not_active)
        continue;

      // Walk the rows on all threads. Each thread keeps the largest ratio of
      // its rows; the maximum is order-independent, so the result is the
      // serial walk's. A non-ignored zero bin anywhere gives the same result
      // as the first one the serial walk stops at, so once any thread finds
      // one, every thread skips its remaining rows. The raw moments pointer
      // and strides avoid indexing through the tensor per bin; the doubles
      // read are the same.
      const double* m = t.moments().data();
      const int64_t n_mom = static_cast<int64_t>(t.moments().shape(2));
      const int64_t row_stride = t.n_score_bins() * n_mom;
      const int64_t col = trigger.score_index * n_mom;
      const int n = t.n_realizations_;
      double best = ratio;
      bool zero_bin = false;

#pragma omp parallel for schedule(static) reduction(max : best)
      for (int64_t filter_index = 0; filter_index < n_rows; ++filter_index) {
        bool stop;
#pragma omp atomic read
        stop = zero_bin;
        if (stop)
          continue;

        // Compute the tally uncertainty metrics.
        const double* cell = m + filter_index * row_stride + col;
        auto uncert_pair =
          get_tally_uncertainty(cell[static_cast<int>(TallyMoment::SUM)],
            cell[static_cast<int>(TallyMoment::SUM_SQ)], n);

        // A score without contributions is skipped when zero scores are
        // ignored for this trigger; otherwise it forces the maximum ratio
        // (handled after the loop).
        if (uncert_pair.first == -1) {
          if (!trigger.ignore_zeros) {
#pragma omp atomic write
            zero_bin = true;
          }
          continue;
        }

        double std_dev = uncert_pair.first;
        double rel_err = uncert_pair.second;

        // Pick out the relevant uncertainty metric for this trigger.
        double uncertainty;
        switch (trigger.metric) {
        case TriggerMetric::variance:
          uncertainty = std_dev * std_dev;
          break;
        case TriggerMetric::standard_deviation:
          uncertainty = std_dev;
          break;
        case TriggerMetric::relative_error:
          uncertainty = rel_err;
          break;
        case TriggerMetric::not_active:
          UNREACHABLE();
        }

        // Compute the uncertainty / threshold ratio.
        double this_ratio = uncertainty / trigger.threshold;
        if (trigger.metric == TriggerMetric::variance) {
          // Batch prediction in check_triggers() squares the limiting ratio.
          // Since variance scales as 1/N, use sqrt(variance / threshold) so
          // the predicted batch multiplier is variance / threshold. This
          // preserves the convergence condition that the ratio is <= 1.
          this_ratio = std::sqrt(this_ratio);
        }

        if (this_ratio > best)
          best = this_ratio;
      }

      // A score without contributions forces the maximum ratio. Record it and
      // stop scanning -- but fall through to the cross-rank reduction rather
      // than returning, so a collective rma check is still reached on every
      // rank.
      if (zero_bin) {
        ratio = INFINITY;
        score = t.scores_[trigger.score_index];
        tally_id = t.id_;
        found_no_contribution = true;
        break;
      }

      // If this is the most uncertain value, set the output variables.
      if (best > ratio) {
        ratio = best;
        score = t.scores_[trigger.score_index];
        tally_id = t.id_;
      }
    }
  }

  profile::phase_end(profile::Phase::TRIGGER_WALK);

#ifdef OPENMC_MPI
  if (has_rma_triggers()) {
    profile::ScopedPhase timer(profile::Phase::TRIGGER_REDUCE);
    rma_reduce_trigger(ratio, tally_id, score);
  }
#endif
}

//! Compute the uncertainty/threshold ratio for the eigenvalue trigger.

double check_keff_trigger()
{
  if (settings::run_mode != RunMode::EIGENVALUE)
    return 0.0;

  double k_combined[2];
  openmc_get_keff(k_combined);

  double uncertainty = 0.;
  switch (settings::keff_trigger.metric) {
  case TriggerMetric::variance:
    uncertainty = k_combined[1] * k_combined[1];
    break;
  case TriggerMetric::standard_deviation:
    uncertainty = k_combined[1];
    break;
  case TriggerMetric::relative_error:
    uncertainty = k_combined[1] / k_combined[0];
    break;
  default:
    // If it's an unrecognized TriggerMetric or no keff trigger is on,
    // return 0 to stop division by zero where "ratio" is calculated.
    return 0.0;
  }

  double ratio = uncertainty / settings::keff_trigger.threshold;
  if (settings::keff_trigger.metric == TriggerMetric::variance)
    ratio = std::sqrt(ratio);
  return ratio;
}

//! See if tally and eigenvalue uncertainties are under trigger thresholds.

void check_triggers()
{
  // Make some aliases.
  const auto current_batch {simulation::current_batch};
  const auto n_batches {settings::n_batches};
  const auto interval {settings::trigger_batch_interval};

  // See if the current batch is one for which the triggers must be checked.
  if (!settings::trigger_on)
    return;
  if (current_batch < n_batches)
    return;
  if (((current_batch - n_batches) % interval) != 0)
    return;

  // Check the eigenvalue and tally triggers. The eigenvalue metric and the
  // satisfy/continue decision are master-only, but check_tally_triggers is
  // collective when an rma tally carries a trigger (its owned-row walks are
  // partial), so every rank must reach it -- finalize_batch invokes
  // check_triggers on all ranks in exactly that case.
  double keff_ratio = mpi::master ? check_keff_trigger() : 0.0;
  double tally_ratio;
  int tally_id, score;
  check_tally_triggers(tally_ratio, tally_id, score);

  // The remaining work (messaging, batch prediction, and setting
  // satisfy_triggers) belongs to the master; the flag is then broadcast from
  // finalize_batch as the sole decision channel.
  if (!mpi::master)
    return;

  // If all the triggers are satisfied, alert the user and return.
  if (std::max(keff_ratio, tally_ratio) <= 1.) {
    simulation::satisfy_triggers = true;
    write_message(7, "Triggers satisfied for batch {}", current_batch);
    return;
  }

  // At least one trigger is unsatisfied.  Let the user know which one.
  simulation::satisfy_triggers = false;
  std::string msg;
  if (keff_ratio >= tally_ratio) {
    msg = fmt::format("Triggers unsatisfied, max unc./thresh. is {} for "
                      "eigenvalue",
      keff_ratio);
  } else {
    if (tally_ratio == INFINITY) {
      msg = fmt::format(
        "Triggers unsatisfied, no result tallied for score {} in tally {}",
        reaction_name(score), tally_id);
    } else {
      msg = fmt::format(
        "Triggers unsatisfied, max unc./thresh. is {} for {} in tally {}",
        tally_ratio, reaction_name(score), tally_id);
    }
  }
  write_message(msg, 7);

  // Estimate batches til triggers are satisfied.
  if (settings::trigger_predict) {
    // This calculation assumes tally variance is proportional to 1/N where N is
    // the number of batches.
    auto max_ratio = std::max(keff_ratio, tally_ratio);
    auto n_active = current_batch - settings::n_inactive;
    auto n_pred_batches = static_cast<int>(n_active * max_ratio * max_ratio) +
                          settings::n_inactive + 1;

    if (max_ratio == INFINITY) {
      std::string msg =
        fmt::format("One or more tallies with triggers have no scores. Unable "
                    "to estimate the number of remaining batches.");
      write_message(msg, 7);
    } else {
      std::string msg =
        fmt::format("The estimated number of batches is {}", n_pred_batches);
      if (n_pred_batches > settings::n_max_batches) {
        msg.append(" --- greater than max batches");
        warning(msg);
      } else {
        write_message(msg, 7);
      }
    }
  }
}

} // namespace openmc
