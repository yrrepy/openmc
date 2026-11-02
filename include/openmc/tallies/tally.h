#ifndef OPENMC_TALLIES_TALLY_H
#define OPENMC_TALLIES_TALLY_H

#include "openmc/constants.h"
#include "openmc/memory.h" // for unique_ptr
#include "openmc/openmp_interface.h"
#include "openmc/span.h"
#include "openmc/tallies/exatally.h"
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

  //! The ExaTallies modes of this tally (accumulator and moments).
  const ExaTally& exatally() const { return exatally_; }

  //! Accumulator mode: where accum_ (and, for rma, the moments) is homed.
  AccumulatorMode accumulator() const { return exatally_.accumulator; }

  //! Moments mode: which cross-batch moments are kept (full, mean, or none).
  MomentsMode moments_mode() const { return exatally_.moments; }

  //! \brief Set the ExaTallies modes, validating the distributed accumulator
  //! modes' prerequisites (MPI + OpenMP build, tally reduction on, Monte Carlo
  //! solver, MPI thread level, not event-based) and the reduced moments modes'
  //! restrictions (no triggers or higher moments; for 'none' also tally
  //! reduction on and not the random ray solver). Must run after the triggers
  //! and higher_moments are read and before init_results(), which homes the
  //! storage. Tallies created through the C API keep the replicated/full
  //! default; the global settings::exatally default applies only to tallies
  //! read from XML.
  void set_exatally(const ExaTally& exatally);

  //! True when the moments are genuinely block-distributed across ranks (rma
  //! accumulator mode on a multi-rank run). A single-rank rma tally holds the
  //! complete, globally-indexed moments and serves every replicated readout
  //! path.
  bool rma_distributed() const;

  void set_scores(pugi::xml_node node);

  void set_scores(const vector<std::string>& scores);

  std::vector<std::string> scores() const;

  int32_t n_scores() const { return scores_.size(); }

  void set_nuclides(pugi::xml_node node);

  void set_nuclides(const vector<std::string>& nuclides);

  //! Cross-batch moments array, shape [n_filter_bins, n_score_bins, n_moments].
  tensor::Tensor<double>& moments() { return moments_; }
  const tensor::Tensor<double>& moments() const { return moments_; }

  //! Number of moment planes in the results array -- the moments_ array, the
  //! statepoint "results" dataset, and the in-memory readout: 2 or 4 for the
  //! full moments mode, 1 (SUM only) for mean and none. A 'none' tally holds
  //! no planes during the run; its one SUM plane exists only once
  //! collect_none() has materialised it.
  int n_moments() const
  {
    if (exatally_.moments != MomentsMode::FULL)
      return 1;
    return higher_moments_ ? 4 : 2;
  }

  //! Whether this tally currently holds an allocated moments array.
  bool has_moments() const { return moments_.size() != 0; }

  //! Length of the innermost (score x nuclide) dimension of the results.
  int n_score_bins() const { return n_score_bins_; }

#ifdef OPENMC_MPI
  //! Number of filter-bin rows this rank owns under rma block distribution
  //! (0 in every other accumulator mode). Under rma the moments array is homed
  //! on the owning rank and locally indexed, so this is its outer dimension.
  int64_t rma_n_rows() const { return rma_n_rows_; }

  //! First global filter-bin row owned by `rank` under rma block distribution.
  //! Uses the same block map as init_rma_accum(), so the master can compute any
  //! rank's ownership (for statepoint gather/scatter) without communication.
  int64_t rma_first_row(int rank) const;

  //! Number of filter-bin rows owned by `rank` under rma block distribution.
  int64_t rma_rows_owned(int rank) const;

  //! Duplicate rma staging rows merged so far (remote-coalescing diagnostic).
  int64_t rma_merged_rows() const { return rma_merged_rows_; }

  //! Remote rows issued so far (unique rows after merging, all threads).
  int64_t rma_rows_issued() const { return rma_rows_issued_; }

  //! MPI_Accumulate calls issued so far (all threads).
  int64_t rma_accumulates() const { return rma_accumulates_; }
#endif

  //! \brief Add a contribution to the per-batch accumulator. This is the ONLY
  //! way transport writes a tally, and it is on the hot path.
  //!
  //! For the local accumulator modes (replicated/shared) accum_ points at an
  //! owned or shared plane and the add is a single atomic RMW. The rma branch
  //! (predicted-not-taken) routes remote bins through an out-of-line handler.
  void score_add(int64_t filter_index, int score_index, double val)
  {
#ifdef OPENMC_MPI
    if (exatally_.accumulator == AccumulatorMode::RMA) {
      rma_score_add(filter_index, score_index, val);
      return;
    }
#endif
    atomic_score_add(&accum_[filter_index * n_score_bins_ + score_index], val);
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

  //! \brief Combine the run total of a 'none' tally and materialise its one
  //! SUM plane (SUM = accumulator total x norm) into moments_: [F, S, 1] on the
  //! master for the replicated and shared accumulator modes, the owned rows
  //! under rma. Collective on all ranks; a no-op when the plane is already
  //! current. The accumulator keeps the run total, so the run can continue.
  //! Called before each statepoint write and at the end of the run.
  void collect_none();

  //! Free the materialised SUM plane of a 'none' tally (after a mid-run
  //! statepoint write, or when the next batch makes it stale).
  void release_none_moments();

  //! True when a 'none' tally's SUM plane is materialised and current.
  bool none_materialised() const { return none_materialised_; }

#ifdef OPENMC_MPI
  //! \brief End-of-batch publish + internode combine for a shared tally.
  //! Publishes this node's atomic scores (win_sync + node barrier), then node
  //! leaders reduce the node planes onto the world master (skipped for a
  //! single-node run). After this the master's plane holds the global sum; the
  //! plane belongs to the node leader until the node barrier in
  //! shared_batch_end() -- non-leaders must not touch accum_ before it. Called
  //! from reduce_tally_results().
  void shared_publish();

  //! \brief End-of-batch publish for an rma tally. Drains any thread-local
  //! staging, completes this rank's outstanding accumulates at their targets,
  //! barriers so no accumulate is in flight anywhere, then issues a local
  //! memory barrier before accumulate() folds this rank's owned window rows.
  //! Called from reduce_tally_results().
  void rma_publish();
#endif

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
  int64_t accum_size_ {0};      //!< n_filter_bins * n_score_bins
  vector<double> accum_buffer_; //!< backing storage for accum_ (replicated)

#ifdef OPENMC_MPI
  //! Shared-memory window backing accum_ in the shared accumulator mode (one
  //! plane per node). MPI_WIN_NULL in every other mode.
  MPI_Win accum_win_ {MPI_WIN_NULL};
#endif

  //! Cross-batch moments, shape [n_filter_bins, n_score_bins, n_moments()].
  //! Written only by the once-per-batch fold in accumulate() (or, for a 'none'
  //! tally, by collect_none()). This layout matches the on-disk statepoint
  //! dataset.
  tensor::Tensor<double> moments_;

  //! Keep moments_ allocated on every rank, not just the master. In reduced
  //! mode non-master ranks never fold, so their moments_ holds nothing until
  //! the end-of-run broadcast; this flag keeps the array but does not fill it.
  //! Set for tallies read on every rank during the run (weight-window
  //! generation), which stay 'replicated': see the warning in
  //! WeightWindowsGenerator::create_tally() and the fatal in init_results().
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
  vector<int64_t> strides_;

  int64_t n_filter_bins_ {0};

  //! Innermost dimension of the results: number of score x nuclide combinations
  int n_score_bins_ {0};

  //! ExaTallies modes: where accum_ (and, for rma, moments_) is homed, and
  //! which moments are kept. Set through set_exatally(), which validates the
  //! distributed modes' prerequisites.
  ExaTally exatally_;

#ifdef OPENMC_MPI
  //! Distributed-window state for the rma accumulator mode. accum_win_ (reused
  //! from the shared mode) holds this rank's contiguous block of owned
  //! filter-bin rows and receives only MPI_Accumulate; accum_/accum_buffer_ is
  //! a private same-size plane for this rank's own scores to bins it owns. Both
  //! planes are locally indexed (row 0 == rma_first_row_). The fold sums them.
  double* rma_win_base_ {nullptr}; //!< base of this rank's owned window block
  int64_t rma_bins_per_rank_ {0};  //!< block size of the ownership map
  int64_t rma_first_row_ {0};      //!< first global filter-bin row owned
  int64_t rma_n_rows_ {0};         //!< number of owned filter-bin rows
  int64_t rma_plane_size_ {0};     //!< rma_n_rows_ * n_score_bins_

  //! Rows per staging buffer (the coalescer depth K) for the remote-scoring
  //! arm, sized from a fixed per-rank memory budget (or the test override).
  //! Zero when this tally never scores remotely (single-rank runs).
  int64_t rma_k_ {0};

  //! Number of 9-bit radix passes that sort a staging buffer by displacement:
  //! enough digits for the largest key (disp / 8) in any rank's block.
  int rma_radix_passes_ {0};

  //! Per-thread state for the rma remote-scoring arm (C2): a row coalescer plus
  //! per-target double-buffered staging feeding batched MPI_Accumulate calls.
  //! Sized num_threads(); each element is written by one thread alone, so the
  //! only cross-thread contention is the named critical around the MPI calls.
  //! Cache-line aligned so neighboring threads' hot coalescer fields never
  //! share a line.
  struct alignas(64) RmaThreadStaging {
    //! Row coalescer: consecutive score_add calls to the same remote filter bin
    //! accumulate into row before it is flushed to a target's staging buffer.
    int64_t coalesce_bin {-1}; //!< global filter bin held open, -1 == none
    int coalesce_owner {-1};   //!< owner rank of coalesce_bin
    vector<double> row;        //!< n_score_bins_ coalesced score values

    //! Owner-block cache for a coalescer miss: the owner of the last remote
    //! row opened and its global rows [owner_lo, owner_hi). Empty until the
    //! first miss, so the first lookup always computes the owner.
    int owner_rank {-1};
    int64_t owner_lo {0};
    int64_t owner_hi {0};

    // clang-format off
    //! Per-target double buffers. For target rank t and buffer b in {0, 1}, with
    //! slot = t * 2 + b:
    //!   data[(slot * rma_k_ + r) * n_score_bins_ + s] : staged score values
    //!   disp[slot * rma_k_ + r] : byte displacement of row r in target t's block
    //!   fill[slot]      : rows staged in that buffer (0..rma_k_)
    //!   in_flight[slot] : accumulate issued, origin not yet locally retired
    //!   cur[t]          : active buffer index for target t
    // clang-format on
    vector<double> data;
    vector<MPI_Aint> disp;
    vector<int> fill;
    vector<char> in_flight;
    vector<int> cur;

    // clang-format off
    //! Reusable scratch used when a buffer is issued, so the MPI_Accumulate
    //! target datatype never has overlapping entries:
    //!   perm, perm2 : index permutation sorted by displacement (rma_k_ ints
    //!                 each; the radix sort scatters between the two)
    //!   hist        : 512-entry digit histogram of the radix sort
    //!   comp_disp, comp_data : compacted, duplicate-free rows, written back
    //!                 into the buffer
    // clang-format on
    vector<int> perm;
    vector<int> perm2;
    vector<int> hist;
    vector<MPI_Aint> comp_disp;
    vector<double> comp_data;

    //! Count of staging rows merged into an earlier same-displacement row (a
    //! coalescing diagnostic; > 0 exactly when a bin was staged
    //! non-consecutively and would otherwise have produced an overlapping
    //! accumulate).
    int64_t merged {0};

    //! Run statistics, folded into the tally totals at each publish: unique
    //! rows issued and MPI_Accumulate calls made.
    int64_t rows_issued {0};
    int64_t accumulates {0};
  };
  vector<RmaThreadStaging> rma_staging_;

  //! Cumulative count of duplicate staging rows merged across the run (folded
  //! from the per-thread counters at each publish). A remote-coalescing quality
  //! metric, also asserted by the unit tests.
  int64_t rma_merged_rows_ {0};

  //! Cumulative remote rows issued and MPI_Accumulate calls (folded like
  //! rma_merged_rows_) for the end-of-run statistics line.
  int64_t rma_rows_issued_ {0};
  int64_t rma_accumulates_ {0};

  //! Out-of-line handler for the rma accumulator mode; only reached when
  //! exatally_.accumulator == RMA. Local bins score into the private plane;
  //! remote bins are coalesced and staged for a batched MPI_Accumulate.
  void rma_score_add(int64_t filter_index, int score_index, double val);

  //! Owner rank of a global filter-bin row under rma block distribution.
  int rma_owner(int64_t filter_index) const;

  //! Allocate the per-thread remote-scoring staging from the memory budget (or
  //! the OPENMC_RMA_STAGING_ROWS test override). No-op on single-rank runs.
  void init_rma_staging();

  //! Append a thread's open coalescer row to its target's active staging
  //! buffer, issuing an MPI_Accumulate (and swapping buffers) when it fills.
  void rma_coalescer_flush(RmaThreadStaging& ts);

  //! Issue one hindexed-block MPI_Accumulate for a (target, buffer) staging
  //! buffer under critical(openmc_rma). No-op if the buffer is empty.
  void rma_stage_issue(RmaThreadStaging& ts, int target, int buf);

  //! \brief Sort the index of a staging buffer's f rows by displacement:
  //! std::sort for a few rows, else a stable 9-bit LSD radix sort on disp / 8.
  //! \return The sorted index (ts.perm or ts.perm2)
  const int* rma_sort_by_disp(
    RmaThreadStaging& ts, const MPI_Aint* disp, int f) const;

  //! End-of-batch: flush every thread's open coalescer row and partial staging
  //! buffer so all of this rank's contributions become outstanding accumulates.
  void rma_drain();

  //! Allocate this rank's block of the distributed rma window plus the private
  //! owned-rows plane, and open the window's passive-target epoch.
  void init_rma_accum();

  //! \brief End-of-batch resume for an rma tally: publish the zeroed window
  //! block (win_sync + intracomm barrier) so the next batch accumulates into
  //! all-zeros.
  void rma_resume();

  //! \brief End-of-batch fold for an rma tally: fold this rank's owned rows
  //! (window block plus private plane) into the moments, zero both planes,
  //! then rma_resume(). Called by accumulate().
  void rma_fold();

  //! \brief End-of-batch resume for a shared tally: publish the zeroed plane
  //! (win_sync + node barrier) so the next batch scores into all-zeros.
  void shared_resume();

  //! \brief End-of-batch tail for a shared tally: on a non-master node every
  //! node rank zeroes its shared_slice() of the node plane (after a node
  //! barrier), then every node rank calls shared_resume(). Called by
  //! accumulate() after the master's fold, and by collect_none().
  void shared_batch_end();

  //! \brief This node rank's slice [begin, end) of the shared node plane:
  //! equal counts of whole 4 KiB pages per node rank, clamped to accum_size_
  //! (trailing ranks may get an empty slice). Each rank zeroes its own slice,
  //! so first touch places those pages on its NUMA domain.
  void shared_slice(int64_t& begin, int64_t& end) const;

  //! Close the passive-target epoch and free accum_win_ (idempotent).
  void free_accum_window();

  //! Allocate the per-node shared accumulator plane and open its epoch.
  void init_shared_accum();
#endif

  //! Per-batch normalization applied in the fold (source strength divided by
  //! contributing particles per generation; 1 for the random ray solver).
  double tally_normalization() const;

  //! \brief Fold this batch's accumulator plane (the private plane, or the
  //! shared node plane) into the moments and zero the folded bins. Sparse
  //! stretches of the plane skip their zero bins, which is exact. Called by
  //! accumulate() on the ranks that hold the batch sum.
  void fold_moments();

  //! Master side of collect_none() for the replicated and shared accumulator
  //! modes: allocate moments_ as [F, S, 1] and fill SUM = accum_ x norm.
  void materialise_none_sum(double norm);

  //! For a 'none' tally: moments_ holds the SUM plane materialised by
  //! collect_none() and no batch has been accumulated since. Rank-identical.
  bool none_materialised_ {false};

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
extern int32_t n_realizations;
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

#endif

void free_memory_tally();

} // namespace openmc

#endif // OPENMC_TALLIES_TALLY_H
