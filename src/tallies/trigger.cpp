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
  int i_tally, int score_index, int64_t filter_index)
{
  const auto& tally {model::tallies[i_tally]};

  auto sum = tally->moments()(filter_index, score_index, TallyMoment::SUM);
  auto sum_sq =
    tally->moments()(filter_index, score_index, TallyMoment::SUM_SQ);

  int n = tally->n_realizations_;
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

bool has_rma_triggers()
{
#ifdef OPENMC_MPI
  for (const auto& t : model::tallies) {
    if (t->storage_ != TallyStorage::RMA)
      continue;
    for (const auto& trigger : t->triggers_) {
      if (trigger.metric != TriggerMetric::not_active)
        return true;
    }
  }
#endif
  return false;
}

void check_tally_triggers(double& ratio, int& tally_id, int& score)
{
  ratio = 0.;
  tally_id = 0;
  score = 0;
  bool found_no_contribution = false;

  for (auto i_tally = 0;
       i_tally < model::tallies.size() && !found_no_contribution; ++i_tally) {
    const Tally& t {*model::tallies[i_tally]};

#ifdef OPENMC_MPI
    // Under rma each rank walks only the moment rows it owns; every other mode
    // homes the moments on the master, so only the master walks those. A rank
    // that owns nothing (or a non-master rank on a replicated tally) simply
    // contributes ratio 0 to the reduction below.
    const bool is_rma = (t.storage_ == TallyStorage::RMA);
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

      for (int64_t filter_index = 0; filter_index < n_rows; ++filter_index) {
        // Compute the tally uncertainty metrics.
        auto uncert_pair =
          get_tally_uncertainty(i_tally, trigger.score_index, filter_index);

        // If there is a score without contributions, force the maximum ratio,
        // unless zero scores are ignored for this trigger. Record it and stop
        // scanning -- but fall through to the cross-rank reduction rather than
        // returning, so a collective rma check is still reached on every rank.
        if (uncert_pair.first == -1 && !trigger.ignore_zeros) {
          ratio = INFINITY;
          score = t.scores_[trigger.score_index];
          tally_id = t.id_;
          found_no_contribution = true;
          break;
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
          this_ratio = std::sqrt(ratio);
        }

        // If this is the most uncertain value, set the output variables.
        if (this_ratio > ratio) {
          ratio = this_ratio;
          score = t.scores_[trigger.score_index];
          tally_id = t.id_;
        }
      }
      if (found_no_contribution)
        break;
    }
  }

#ifdef OPENMC_MPI
  // For rma tallies each rank has only walked its owned rows, so the most
  // limiting trigger lives on whichever rank owns it. Reduce {ratio, rank} with
  // MAXLOC to find the global maximum and the (lowest) rank holding it; that
  // rank ships its tally id / score to the master, which alone forms the
  // diagnostic message and the satisfy/continue decision from the global ratio.
  // The reduction runs iff has_rma_triggers() -- exactly when every rank calls
  // this function (see finalize_batch) -- so it is never a half-collective.
  if (has_rma_triggers()) {
    struct {
      double val;
      int rank;
    } in {ratio, mpi::rank}, out {};
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, mpi::intracomm);
    ratio = out.val;
    if (out.rank != 0) {
      constexpr int TRIGGER_TAG = 42;
      int buf[2] {tally_id, score};
      if (mpi::rank == out.rank) {
        MPI_Send(buf, 2, MPI_INT, 0, TRIGGER_TAG, mpi::intracomm);
      } else if (mpi::master) {
        MPI_Recv(buf, 2, MPI_INT, out.rank, TRIGGER_TAG, mpi::intracomm,
          MPI_STATUS_IGNORE);
        tally_id = buf[0];
        score = buf[1];
      }
    }
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
