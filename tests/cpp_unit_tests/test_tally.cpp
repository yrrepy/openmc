#include "openmc/tallies/tally.h"

#include <cstdint>
#include <string>
#include <vector>

#include "openmc/constants.h"
#include "openmc/settings.h"
#include "openmc/tallies/filter.h"
#include "openmc/tallies/filter_energy.h"
#include <catch2/catch_test_macros.hpp>

using namespace openmc;

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

  // set filters with a duplicate filter, should only add the filter to the tally once
  filters = {cell_filter, cell_filter};
  tally->set_filters(filters);
  REQUIRE(tally->filters().size() == 1);
  REQUIRE(model::filter_map[cell_filter->id()] == tally->filters(0));
}

TEST_CASE("Test int64 filter-bin indexing for large tally shapes")
{
  // Two filters whose *product* of bin counts exceeds 2^31. Each individual
  // filter fits comfortably in int32 (mirroring the real target workload: a
  // fine mesh crossed with a fine energy filter), but their product overflows
  // a 32-bit stride accumulator. This exercises the int64 widening of the
  // filter-bin stride/index math without allocating the (multi-GB) results.
  const int n_bins_0 = 100000;
  const int n_bins_1 = 25000;

  auto make_energy_filter = [](int n_bins) {
    Filter* f = Filter::create("energy");
    auto* ef = dynamic_cast<EnergyFilter*>(f);
    REQUIRE(ef != nullptr);
    // Monotonically increasing boundaries: 0, 1, ..., n_bins.
    std::vector<double> boundaries(n_bins + 1);
    for (int i = 0; i <= n_bins; ++i)
      boundaries[i] = static_cast<double>(i);
    ef->set_bins({boundaries.data(), boundaries.size()});
    return f;
  };

  Filter* filter0 = make_energy_filter(n_bins_0);
  Filter* filter1 = make_energy_filter(n_bins_1);

  Tally* tally = Tally::create();
  tally->add_filter(filter0);
  tally->add_filter(filter1);
  tally->set_strides();

  // Strides: filters are traversed in reverse so the last filter has unit
  // stride and the first filter's stride is the last filter's bin count.
  REQUIRE(tally->strides(1) == 1);
  REQUIRE(tally->strides(0) == static_cast<int64_t>(n_bins_1));

  // Total number of filter bins is the product; it exceeds 2^31 and must not
  // overflow (a 32-bit accumulator would wrap to a wrong, likely negative,
  // value).
  const int64_t expected_bins =
    static_cast<int64_t>(n_bins_0) * static_cast<int64_t>(n_bins_1);
  REQUIRE(expected_bins > static_cast<int64_t>(INT32_MAX));
  REQUIRE(tally->n_filter_bins() == expected_bins);

  // The flat index of the highest bin combination must also be computed in
  // 64 bits (this mirrors FilterBinIter::compute_index_weight).
  const int64_t bin0 = n_bins_0 - 1;
  const int64_t bin1 = n_bins_1 - 1;
  const int64_t flat_index =
    bin0 * tally->strides(0) + bin1 * tally->strides(1);
  REQUIRE(flat_index > static_cast<int64_t>(INT32_MAX));
  REQUIRE(flat_index == expected_bins - 1);

  // Free the globally-registered filters/tally so later cases start clean.
  free_memory_tally();
}

TEST_CASE("Test tally score_add, accumulate fold, and reset")
{
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
  tally->set_strides();
  tally->init_results();

  REQUIRE(tally->n_filter_bins() == 2);
  REQUIRE(tally->n_score_bins() == 2);
  REQUIRE(tally->n_moments() == 2);
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
  REQUIRE(m(0, 0, TallyMoment::SUM_SQ) == 4.0);
  REQUIRE(m(1, 1, TallyMoment::SUM) == 3.0);
  REQUIRE(m(1, 1, TallyMoment::SUM_SQ) == 9.0);
  REQUIRE(tally->accum(0, 0) == 0.0);
  REQUIRE(tally->accum(1, 1) == 0.0);

  // Second realization accumulates on top of the first.
  tally->score_add(0, 0, 5.0);
  tally->accumulate();
  REQUIRE(m(0, 0, TallyMoment::SUM) == 7.0);       // 2 + 5
  REQUIRE(m(0, 0, TallyMoment::SUM_SQ) == 29.0);   // 4 + 25
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