#include "openmc/tallies/tally.h"

#include <cstdint>
#include <string>
#include <vector>

#include "openmc/constants.h"
#include "openmc/settings.h"
#include "openmc/tallies/filter.h"
#include "openmc/tallies/filter_energy.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>

using namespace openmc;

TEST_CASE("Invalid tally nuclide names remain catchable")
{
  Tally* tally = Tally::create();
  REQUIRE_THROWS_WITH(tally->set_nuclides({"not-a-nuclide"}),
    "Could not add nuclide 'not-a-nuclide' to a tally: "
    "Nuclide 'not-a-nuclide' is not present in library.");
}

TEST_CASE("Test add/set_filter")
{
  // create a new tally object
  Tally* tally = Tally::create();

  // create a new particle filter
  Filter* particle_filter = Filter::create("particle");

  // add the particle filter to the tally
  tally->add_filter(particle_filter);

  // the filter should be added to the tally
  REQUIRE(tally->filters().size() == 1);
  REQUIRE(model::filter_map[particle_filter->id()] == tally->filters(0));

  // add the particle filter to the tally again
  tally->add_filter(particle_filter);
  // the tally should have the same number of filters
  REQUIRE(tally->filters().size() == 1);

  // create a cell filter
  Filter* cell_filter = Filter::create("cell");
  tally->add_filter(cell_filter);

  // now the size of the filters should have increased
  REQUIRE(tally->filters().size() == 2);
  REQUIRE(model::filter_map[cell_filter->id()] == tally->filters(1));

  // if we set the filters explicitly there shouldn't be extra filters hanging
  // around
  tally->set_filters({&cell_filter, 1});

  REQUIRE(tally->filters().size() == 1);
  REQUIRE(model::filter_map[cell_filter->id()] == tally->filters(0));

  // set filters again using both filters
  std::vector<Filter*> filters = {cell_filter, particle_filter};
  tally->set_filters(filters);

  REQUIRE(tally->filters().size() == 2);
  REQUIRE(model::filter_map[cell_filter->id()] == tally->filters(0));
  REQUIRE(model::filter_map[particle_filter->id()] == tally->filters(1));

  // set filters with a duplicate filter, should only add the filter to the
  // tally once
  filters = {cell_filter, cell_filter};
  tally->set_filters(filters);
  REQUIRE(tally->filters().size() == 1);
  REQUIRE(model::filter_map[cell_filter->id()] == tally->filters(0));
}

// Regression test for 64-bit tally filter-bin counts (mesh x groups > 2^31).
TEST_CASE("Tally filter-bin count does not overflow 32 bits")
{
  // Two energy filters whose bin counts multiply to 2.5e9, above INT32_MAX.
  constexpr int64_t bins_per_filter = 50000;

  // Only the bin count matters here, so the edge values are an arbitrary ramp.
  std::vector<double> edges(bins_per_filter + 1);
  for (int64_t i = 0; i < bins_per_filter + 1; ++i)
    edges[i] = static_cast<double>(i);

  Tally* tally = Tally::create();
  for (int i = 0; i < 2; ++i) {
    Filter* filter = Filter::create("energy");
    dynamic_cast<EnergyFilter*>(filter)->set_bins(edges);
    tally->add_filter(filter);
  }
  tally->set_strides();

  // set_strides() previously accumulated this product in a 32-bit int.
  REQUIRE(tally->n_filter_bins() == bins_per_filter * bins_per_filter);
  REQUIRE(tally->n_filter_bins() > 2147483647);

  // Free the globally-registered filters/tally so later cases start clean.
  free_memory_tally();
}

TEST_CASE("Test tally score_add, accumulate fold, and reset")
{
  // Run once per moments mode that folds every batch: 'full' keeps SUM and
  // SUM_SQ (M = 2), 'mean' keeps SUM only (M = 1).
  const auto moments_mode = GENERATE(MomentsMode::FULL, MomentsMode::MEAN);
  const bool full = moments_mode == MomentsMode::FULL;

  // Save globals this case mutates; restore them at the end so the shared test
  // binary has no order dependence.
  const auto saved_run_mode = settings::run_mode;
  const auto saved_solver_type = settings::solver_type;
  const auto saved_reduce_tallies = settings::reduce_tallies;
  const auto saved_n_particles = settings::n_particles;
  const auto saved_gen_per_batch = settings::gen_per_batch;

  // Two energy bins x two scores => n_filter_bins = 2, n_score_bins = 2.
  Filter* f = Filter::create("energy");
  auto* ef = dynamic_cast<EnergyFilter*>(f);
  REQUIRE(ef != nullptr);
  std::vector<double> boundaries = {0.0, 1.0, 2.0};
  ef->set_bins({boundaries.data(), boundaries.size()});

  Tally* tally = Tally::create();
  tally->set_filters({&f, 1});
  tally->set_scores(std::vector<std::string> {"flux", "total"});
  tally->set_exatally({AccumulatorMode::REPLICATED, moments_mode});
  tally->set_strides();
  tally->init_results();

  REQUIRE(tally->n_filter_bins() == 2);
  REQUIRE(tally->n_score_bins() == 2);
  REQUIRE(tally->n_moments() == (full ? 2 : 1));
  REQUIRE(tally->moments().shape(2) == tally->n_moments());
  REQUIRE(tally->has_moments());

  // Configure normalization so accumulate()'s norm factor is exactly 1
  // (EIGENVALUE => total_source = 1, contributing = n_particles = 1).
  settings::run_mode = RunMode::EIGENVALUE;
  settings::solver_type = SolverType::MONTE_CARLO;
  settings::reduce_tallies = true;
  settings::n_particles = 1;
  settings::gen_per_batch = 1;

  // score_add writes into the accumulator; accum() reads it back.
  tally->score_add(0, 0, 2.0); // filter bin 0, score 0
  tally->score_add(1, 1, 3.0); // filter bin 1, score 1
  REQUIRE(tally->accum(0, 0) == 2.0);
  REQUIRE(tally->accum(1, 1) == 3.0);
  REQUIRE(tally->accum(0, 1) == 0.0);

  // First realization: fold the accumulator into the moments and zero it.
  tally->accumulate();
  const auto& m = tally->moments();
  REQUIRE(m(0, 0, TallyMoment::SUM) == 2.0);
  REQUIRE(m(1, 1, TallyMoment::SUM) == 3.0);
  if (full) {
    REQUIRE(m(0, 0, TallyMoment::SUM_SQ) == 4.0);
    REQUIRE(m(1, 1, TallyMoment::SUM_SQ) == 9.0);
  } else {
    // With one plane the SUM entries are contiguous: bin (0, 1) follows (0, 0)
    // directly, so a stray SUM_SQ write would land on a neighbour's SUM.
    REQUIRE(m(0, 1, TallyMoment::SUM) == 0.0);
    REQUIRE(m(1, 0, TallyMoment::SUM) == 0.0);
  }
  REQUIRE(tally->accum(0, 0) == 0.0);
  REQUIRE(tally->accum(1, 1) == 0.0);

  // Second realization accumulates on top of the first.
  tally->score_add(0, 0, 5.0);
  tally->accumulate();
  REQUIRE(m(0, 0, TallyMoment::SUM) == 7.0); // 2 + 5
  if (full) {
    REQUIRE(m(0, 0, TallyMoment::SUM_SQ) == 29.0); // 4 + 25
  } else {
    REQUIRE(m(0, 1, TallyMoment::SUM) == 0.0);
  }
  REQUIRE(tally->n_realizations_ == 2);

  // reset() clears both the accumulator and the moments.
  tally->score_add(0, 0, 1.0);
  tally->reset();
  REQUIRE(tally->accum(0, 0) == 0.0);
  REQUIRE(tally->moments()(0, 0, TallyMoment::SUM) == 0.0);
  REQUIRE(tally->n_realizations_ == 0);

  // Restore mutated globals and free the registered filter/tally.
  settings::run_mode = saved_run_mode;
  settings::solver_type = saved_solver_type;
  settings::reduce_tallies = saved_reduce_tallies;
  settings::n_particles = saved_n_particles;
  settings::gen_per_batch = saved_gen_per_batch;
  free_memory_tally();
}