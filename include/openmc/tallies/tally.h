#ifndef OPENMC_TALLIES_TALLY_H
#define OPENMC_TALLIES_TALLY_H

#include "openmc/constants.h"
#include "openmc/memory.h" // for unique_ptr
#include "openmc/span.h"
#include "openmc/tallies/filter.h"
#include "openmc/tallies/trigger.h"
#include "openmc/vector.h"

#include "openmc/tensor.h"
#include "pugixml.hpp"

#include <string>
#include <unordered_map>

#ifdef OPENMC_MPI
#include <mpi.h>
#endif

namespace openmc {

//==============================================================================
//! A user-specified flux-weighted (or current) measurement.
//==============================================================================

class Tally {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors, factory functions
  explicit Tally(int32_t id);
  explicit Tally(pugi::xml_node node);
  ~Tally();

  // accum_ points into accum_buffer_, so a copy/move would leave the pointer
  // dangling into the source's buffer. Tallies live only in
  // vector<unique_ptr<Tally>> and are never copied or moved, so forbid it
  // rather than write a pointer-reseating copy.
  Tally(const Tally&) = delete;
  Tally& operator=(const Tally&) = delete;

  static Tally* create(int32_t id = -1);

  //----------------------------------------------------------------------------
  // Accessors

  void set_id(int32_t id);

  int id() const { return id_; }

  void set_active(bool active) { active_ = active; }

  void set_multiply_density(bool value) { multiply_density_ = value; }

  void set_writable(bool writable) { writable_ = writable; }

  void set_scores(pugi::xml_node node);

  void set_scores(const vector<std::string>& scores);

  std::vector<std::string> scores() const;

  int32_t n_scores() const { return scores_.size(); }

  void set_nuclides(pugi::xml_node node);

  void set_nuclides(const vector<std::string>& nuclides);

  //! Cross-batch moments array, shape [n_filter_bins, n_score_bins, n_moments].
  tensor::Tensor<double>& moments() { return moments_; }
  const tensor::Tensor<double>& moments() const { return moments_; }

  //! Number of moment columns stored per (filter, score) bin (2 or 4).
  int n_moments() const { return higher_moments_ ? 4 : 2; }

  //! Whether this tally currently holds an allocated moments array.
  bool has_moments() const { return moments_.size() != 0; }

  //! Length of the innermost (score x nuclide) dimension of the results.
  int n_score_bins() const { return n_score_bins_; }

  //! \brief Add a contribution to the per-batch accumulator. This is the ONLY
  //! way transport writes a tally, and it is on the hot path.
  //!
  //! For the local storage modes (replicated/shared) accum_ points at an
  //! owned or shared plane and the add is a single atomic RMW. The rma branch
  //! (predicted-not-taken) routes remote bins through an out-of-line handler.
  void score_add(int64_t filter_index, int score_index, double val)
  {
#ifdef OPENMC_MPI
    if (storage_ == TallyStorage::RMA) {
      rma_score_add(filter_index, score_index, val);
      return;
    }
#endif
    double& c = accum_[filter_index * n_score_bins_ + score_index];
#pragma omp atomic
    c += val;
  }

  //! \brief Direct (non-atomic) accumulator element access. Used by random ray
  //! for the serial per-bin volume normalization outside the scoring loop.
  double& accum(int64_t filter_index, int score_index)
  {
    return accum_[filter_index * n_score_bins_ + score_index];
  }

  //! returns vector of indices corresponding to the tally this is called on
  const vector<int32_t>& filters() const { return filters_; }

  //! returns a vector of filter types for the tally
  std::vector<FilterType> filter_types() const;

  //! returns a mapping of filter types to index into the tally's filters
  std::unordered_map<FilterType, int32_t> filter_indices() const;

  //! \brief Returns the tally filter at index i
  int32_t filters(int i) const { return filters_[i]; }

  //! \brief Return a const pointer to a filter instance based on type. Always
  //! returns the first matching filter type
  template<class T>
  const T* get_filter() const
  {
    const T* out;
    for (auto filter_idx : filters_) {
      if ((out = dynamic_cast<T*>(model::tally_filters[filter_idx].get())))
        return out;
    }
    return nullptr;
  }

  template<class T>
  const T* get_filter(int idx) const
  {
    if (const T* out = dynamic_cast<T*>(model::tally_filters[filters_.at(idx)]))
      return out;
    return nullptr;
  }

  //! \brief Check if this tally has a specified type of filter
  bool has_filter(FilterType filter_type) const;

  void set_filters(span<Filter*> filters);

  //! Given already-set filters, set the stride lengths
  void set_strides();

  int64_t strides(int i) const { return strides_[i]; }

  int64_t n_filter_bins() const { return n_filter_bins_; }

  bool multiply_density() const { return multiply_density_; }

  bool writable() const { return writable_; }

  bool higher_moments() const { return higher_moments_; }

  //----------------------------------------------------------------------------
  // Other methods.

  void add_filter(Filter* filter);

  void init_triggers(pugi::xml_node node);

  void init_results();

  void reset();

  void accumulate();

  //! return the index of a score specified by name
  int score_index(const std::string& score) const;

  //! Tally results reshaped according to filter sizes
  tensor::Tensor<double> get_reshaped_data() const;

  //! A string representing the i-th score on this tally
  std::string score_name(int score_idx) const;

  //! A string representing the i-th nuclide on this tally
  std::string nuclide_name(int nuclide_idx) const;

  //----------------------------------------------------------------------------
  // Major public data members.

  int id_ {C_NONE}; //!< User-defined identifier

  std::string name_; //!< User-defined name

  TallyType type_ {TallyType::VOLUME}; //!< e.g. volume, surface current

  //! Event type that contributes to this tally
  TallyEstimator estimator_ {TallyEstimator::TRACKLENGTH};

  //! Whether this tally is currently being updated
  bool active_ {false};

  //! Number of realizations
  int n_realizations_ {0};

  vector<int> scores_; //!< Filter integrands (e.g. flux, fission)

  //! Index of each nuclide to be tallied.  -1 indicates total material.
  vector<int> nuclides_ {-1};

  //! Per-batch accumulator plane (the s_k plane), logically shaped
  //! [n_filter_bins, n_score_bins] and stored contiguously. This is the only
  //! object transport writes to. accum_ is a raw pointer so that shared/rma
  //! modes can re-home the storage (an MPI window) without changing the hot
  //! path; in replicated mode it points at accum_buffer_ below.
  double* accum_ {nullptr};
  int64_t accum_size_ {0};       //!< n_filter_bins * n_score_bins
  vector<double> accum_buffer_;  //!< backing storage for accum_ (replicated)

  //! Cross-batch moments, shape [n_filter_bins, n_score_bins, n_moments] with
  //! n_moments = higher_moments_ ? 4 : 2. Written only by the once-per-batch
  //! fold in accumulate(). This layout matches the on-disk statepoint dataset.
  tensor::Tensor<double> moments_;

  //! Where accum_ (and, for rma, moments_) is homed.
  TallyStorage storage_ {TallyStorage::REPLICATED};

  //! Keep moments_ allocated on every rank, not just the master. In reduced
  //! mode moments are otherwise homed on the master alone and distributed to
  //! the other ranks only at the end-of-run broadcast. Tallies whose moments
  //! are read off the master during the run set this so those reads see a live
  //! array -- weight-window generation reads them on all ranks with no reduce.
  bool moments_all_ranks_ {false};

  //! True if this tally should be written to statepoint files
  bool writable_ {true};

  //----------------------------------------------------------------------------
  // Miscellaneous public members.

  // We need to have quick access to some filters.  The following gives indices
  // for various filters that could be in the tally or C_NONE if they are not
  // present.
  int energyout_filter_ {C_NONE};
  int delayedgroup_filter_ {C_NONE};

  vector<Trigger> triggers_;

  int deriv_ {C_NONE}; //!< Index of a TallyDerivative object for diff tallies.

private:
  //----------------------------------------------------------------------------
  // Private data.

  vector<int32_t> filters_; //!< Filter indices in global filters array

  //! Index strides assigned to each filter to support 1D indexing.
  //! int64 so the product of per-filter bin counts (e.g. mesh x fine energy)
  //! cannot overflow 2^31 for large tallies.
  vector<int64_t> strides_;

  int64_t n_filter_bins_ {0};

  //! Innermost dimension of the results: number of score x nuclide combinations
  int n_score_bins_ {0};

#ifdef OPENMC_MPI
  //! Out-of-line handler for the rma storage mode; only reached when
  //! storage_ == RMA.
  void rma_score_add(int64_t filter_index, int score_index, double val);
#endif

  //! Whether to multiply by atom density for reaction rates
  bool multiply_density_ {true};

  //! Whether to accumulate higher moments (third and fourth)
  bool higher_moments_ {false};

  int64_t index_;
};

//==============================================================================
// Global variable declarations
//==============================================================================

namespace model {
extern std::unordered_map<int, int> tally_map;
extern vector<unique_ptr<Tally>> tallies;
extern vector<int> active_tallies;
extern vector<int> active_analog_tallies;
extern vector<int> active_tracklength_tallies;
extern vector<int> active_timed_tracklength_tallies;
extern vector<int> active_collision_tallies;
extern vector<int> active_meshsurf_tallies;
extern vector<int> active_surface_tallies;
extern vector<int> active_pulse_height_tallies;
extern vector<int32_t> pulse_height_cells;
extern vector<double> time_grid;

} // namespace model

namespace simulation {
//! Global tallies (such as k-effective estimators)
extern tensor::StaticTensor2D<double, N_GLOBAL_TALLIES, 3> global_tallies;

//! Number of realizations for global tallies
extern "C" int32_t n_realizations;
} // namespace simulation

extern double global_tally_absorption;
extern double global_tally_collision;
extern double global_tally_tracklength;
extern double global_tally_leakage;

//==============================================================================
// Non-member functions
//==============================================================================

//! Read tally specification from tallies.xml
void read_tallies_xml();

//! Read tally specification from an XML node
//! \param[in] root node of tallies XML element
void read_tallies_xml(pugi::xml_node root);

//! \brief Accumulate the sum of the contributions from each history within the
//! batch to a new random variable
void accumulate_tallies();

//! Determine distance to next time boundary
//
//! \param time Current time of particle
//! \param speed Speed of particle
//! \return Distance to next time boundary (or INFTY if none)
double distance_to_time_boundary(double time, double speed);

//! Determine which tallies should be active
void setup_active_tallies();

#ifdef OPENMC_MPI
//! Collect all tally results onto master process
void reduce_tally_results();

//! \brief In-place MPI_SUM reduction onto the master, split into chunks of at
//! most 2^27 elements so tally-sized planes never exceed the int MPI count
//! limit. Master reduces with MPI_IN_PLACE; other ranks send with a null recv.
void reduce_in_place_chunked(double* data, int64_t n, MPI_Comm comm);
#endif

void free_memory_tally();

} // namespace openmc

#endif // OPENMC_TALLIES_TALLY_H
