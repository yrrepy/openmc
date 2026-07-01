#include "openmc/tallies/tally.h"

#include <cstdint>
#include <vector>

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
}