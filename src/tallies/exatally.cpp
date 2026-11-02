//! \file exatally.cpp
//! \brief ExaTallies machinery: the <exatally> mode parser, node-shared
//! accumulator windows and one-sided RMA distribution. The hot-path seam stays
//! in tally.h/tally.cpp.

#include "openmc/tallies/exatally.h"

#include "openmc/error.h"
#include "openmc/message_passing.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/tallies/filter_energy.h"
#include "openmc/tallies/filter_mesh.h"
#include "openmc/tallies/tally.h"
#include "openmc/tallies/trigger.h"
#include "openmc/xml_interface.h"

#include <fmt/core.h>

#include <algorithm> // for min, max, clamp, copy, count, fill, find, sort
#include <cassert>
#include <cmath>     // for abs
#include <cstdint>   // for uintptr_t
#include <cstdlib>   // for getenv, atoll
#include <stdexcept> // for runtime_error

#if defined(OPENMC_MPI) && defined(__linux__)
#include <array>
#include <bitset>
#include <climits> // for CHAR_BIT
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace openmc {

//==============================================================================
// ExaTallies mode switch
//==============================================================================

ExaTally parse_exatally(
  pugi::xml_node node, const ExaTally& defaults, const std::string& context)
{
  ExaTally result = defaults;
  pugi::xml_node block = node.child("exatally");
  if (!block)
    return result;

  if (check_for_node(block, "accumulator")) {
    std::string mode = get_node_value(block, "accumulator", true, true);
    if (mode == "replicated") {
      result.accumulator = AccumulatorMode::REPLICATED;
    } else if (mode == "shared") {
      result.accumulator = AccumulatorMode::SHARED;
    } else if (mode == "rma") {
      result.accumulator = AccumulatorMode::RMA;
    } else {
      throw std::runtime_error {fmt::format(
        "Invalid <exatally> accumulator mode '{}' in {}. Valid options are "
        "'replicated', 'shared', and 'rma'.",
        mode, context)};
    }
  }

  if (check_for_node(block, "moments")) {
    std::string mode = get_node_value(block, "moments", true, true);
    if (mode == "full") {
      result.moments = MomentsMode::FULL;
    } else if (mode == "mean") {
      result.moments = MomentsMode::MEAN;
    } else if (mode == "none") {
      result.moments = MomentsMode::NONE;
    } else {
      throw std::runtime_error {
        fmt::format("Invalid <exatally> moments mode '{}' in {}. Valid "
                    "options are 'full', 'mean', and 'none'.",
          mode, context)};
    }
  }

  return result;
}

const char* moments_mode_name(MomentsMode mode)
{
  switch (mode) {
  case MomentsMode::NONE:
    return "none";
  case MomentsMode::MEAN:
    return "mean";
  default:
    return "full";
  }
}

#ifdef OPENMC_MPI
void reduce_accum_to_master(double* accum, int64_t n, MPI_Comm comm)
{
  // World rank 0 is rank 0 of both intracomm and internode_comm, so the master
  // is the root on either communicator.
  for (int64_t offset = 0; offset < n; offset += MAX_MPI_CHUNK_DOUBLES) {
    const int64_t count = std::min(n - offset, MAX_MPI_CHUNK_DOUBLES);
    if (mpi::master) {
      mpi::reduce(MPI_IN_PLACE, accum + offset, count, MPI_SUM, 0, comm);
    } else {
      mpi::reduce<double>(accum + offset, nullptr, count, MPI_SUM, 0, comm);
    }
  }
}
#endif

//==============================================================================
// Batch fold kernel
//==============================================================================

namespace {

//! Which moments a batch fold updates.
enum class FoldKind {
  MEAN,  //!< SUM only (the mean moments mode)
  FULL,  //!< SUM and SUM_SQ
  HIGHER //!< SUM through SUM_FOURTH (higher moments)
};

//! Number of moments per bin a fold kind keeps (the moments_ inner extent).
template<FoldKind KIND>
constexpr int fold_n_moments()
{
  return KIND == FoldKind::MEAN ? 1 : (KIND == FoldKind::FULL ? 2 : 4);
}

//! Bins per chunk of the batch fold, the unit of its dense/sparse choice, and
//! the divisor of the sparse threshold: a chunk with at most 1/16 of its bins
//! nonzero is sparse. Measured on a 4 GiB plane, this split is never slower
//! than folding every bin, at one thread or four.
constexpr int FOLD_CHUNK_BINS = 2048;
constexpr int FOLD_SPARSE_DIVISOR = 16;

//! Whether bin q of a chunk received any score this batch.
template<bool TWO_PLANES>
inline bool fold_bin_nonzero(const double* acc, const double* win, int q)
{
  if constexpr (TWO_PLANES) {
    return (win[q] != 0.0) | (acc[q] != 0.0);
  } else {
    return acc[q] != 0.0;
  }
}

//! Fold bin q of a chunk into its moments (mom, fold_n_moments() per bin) and
//! zero its accumulator. One plane: the batch value is acc * norm. Two planes
//! (rma): it is (win + acc) * norm, the window block plus the private plane,
//! and both are zeroed.
//! \return The batch value folded
template<bool TWO_PLANES, FoldKind KIND>
inline double fold_bin(
  double* acc, double* win, double* mom, int q, double norm)
{
  double val;
  if constexpr (TWO_PLANES) {
    val = (win[q] + acc[q]) * norm;
    win[q] = 0.0;
  } else {
    val = acc[q] * norm;
  }
  acc[q] = 0.0;
  double* m = mom + static_cast<int64_t>(q) * fold_n_moments<KIND>();
  if constexpr (KIND == FoldKind::MEAN) {
    m[static_cast<int>(TallyMoment::SUM)] += val;
  } else if constexpr (KIND == FoldKind::HIGHER) {
    double val2 = val * val;
    m[static_cast<int>(TallyMoment::SUM)] += val;
    m[static_cast<int>(TallyMoment::SUM_SQ)] += val2;
    m[static_cast<int>(TallyMoment::SUM_THIRD)] += val2 * val;
    m[static_cast<int>(TallyMoment::SUM_FOURTH)] += val2 * val2;
  } else {
    m[static_cast<int>(TallyMoment::SUM)] += val;
    m[static_cast<int>(TallyMoment::SUM_SQ)] += val * val;
  }
  return val;
}

//! \brief Fold one chunk of len <= FOLD_CHUNK_BINS bins, dense or sparse.
//!
//! A dense chunk folds every bin, exactly as a plain loop, and counts its
//! nonzero bins on the way; a sparse chunk first lists its nonzero bins, then
//! folds only those, so the moments of an empty bin are neither read nor
//! written. `sparse` holds this thread's choice for the chunk on entry and its
//! choice for the next chunk on return: sparse when at most 1/16 of this
//! chunk's bins were nonzero. on_dense(val) receives the value of every bin of
//! a dense chunk in order; on_sparse(q, val) receives each folded bin of a
//! sparse chunk with its offset q (the projection arm; empty otherwise).
template<bool TWO_PLANES, FoldKind KIND, class DENSE, class SPARSE>
inline void fold_chunk(double* a, double* w, double* m, int len, double norm,
  int* nonzero, bool& sparse, DENSE&& on_dense, SPARSE&& on_sparse)
{
  int n_nonzero = 0;
  if (sparse) {
    for (int q = 0; q < len; ++q) {
      nonzero[n_nonzero] = q;
      n_nonzero += fold_bin_nonzero<TWO_PLANES>(a, w, q);
    }
    for (int i = 0; i < n_nonzero; ++i) {
      const int q = nonzero[i];
      on_sparse(q, fold_bin<TWO_PLANES, KIND>(a, w, m, q, norm));
    }
  } else {
    for (int q = 0; q < len; ++q) {
      n_nonzero += fold_bin_nonzero<TWO_PLANES>(a, w, q);
      on_dense(fold_bin<TWO_PLANES, KIND>(a, w, m, q, norm));
    }
  }
  sparse = n_nonzero * FOLD_SPARSE_DIVISOR <= len;
}

//! The projection arm of a tally without projections: nothing to add.
struct NoProjection {
  static constexpr bool ENABLED = false;
};

//! \brief The projection arm of the batch fold: adds the batch value of every
//! folded bin into the realization y of each projection of the tally.
//!
//! The plane being folded holds bins [first_bin, first_bin + n) of the tally
//! (all bins, or an rma rank's owned rows), bin g = row * S + score bin, and a
//! row's projected row is sum_i map_i(b_i) * stride_i over the tally filters
//! (b_i its fine bin of filter i, map_i the identity for a kept or summed
//! filter, the coarse map for a collapsed one). A dense chunk walks the bins
//! with an odometer over (b_0, ..., b_{nf-1}, score bin); a row step looks up
//! the last filter only, and the other filters' part is recomputed on a carry
//! out of the last filter. A sparse chunk's bins are decomposed by division,
//! once per bin for all projections. Zero-skipped bins add nothing to y,
//! which is exact.
//!
//! Threads: a case-A projection (every filter but the last kept) maps whole
//! outer rows (row / n_last) to disjoint projected rows, so its threads add
//! straight into y if each thread folds whole outer rows. The parallel unit is
//! therefore a block of whole outer rows (block_bins()), aligned to the
//! GLOBAL outer-row boundaries: under rma a rank's owned rows may start or end
//! inside an outer row, which then forms a lead block of its own, and the
//! other owner projects the rest of that row (the master sums both partials
//! before squaring). A case-B projection adds into one private y per thread
//! (thread_partials), summed into y by reduce_thread_partials().
class ProjectionFold {
public:
  static constexpr bool ENABLED = true;

  ProjectionFold(vector<TallyProjection>& projections,
    const vector<int32_t>& filters, int n_score_bins, int64_t first_bin)
    : nf_(static_cast<int>(filters.size())), np_(projections.size()),
      s_(n_score_bins), first_bin_(first_bin), n_bins_(nf_)
  {
    assert(nf_ > 0); // a projection of a filterless tally is the tally itself
    for (int i = 0; i < nf_; ++i)
      n_bins_[i] = model::tally_filters[filters[i]]->n_bins();

    const int nt = num_threads();
    bool any_case_a = false;
    proj_.resize(np_);
    maps_.resize(np_ * nf_);
    strides_.resize(np_ * nf_);
    for (size_t p = 0; p < np_; ++p) {
      TallyProjection& tp = projections[p];
      proj_[p].first_row = tp.first_row;
      proj_[p].case_a = tp.fast_path;
      proj_[p].realization = tp.realization.data();
      for (int i = 0; i < nf_; ++i) {
        maps_[p * nf_ + i] = tp.maps[i].empty() ? nullptr : tp.maps[i].data();
        strides_[p * nf_ + i] = tp.strides[i];
      }
      // Realization offset of each fine bin of the last filter (its
      // projected-row part times the score bins), or DROPPED.
      proj_[p].last_off.resize(n_bins_[nf_ - 1]);
      for (int64_t b = 0; b < n_bins_[nf_ - 1]; ++b) {
        const int64_t c = part(p, nf_ - 1, b);
        proj_[p].last_off[b] = c < 0 ? DROPPED : c * s_;
      }
      if (tp.fast_path) {
        any_case_a = true;
      } else {
        // One private realization per thread; kept between batches (they
        // are zero after each reduce), resized if the thread count grew.
        const size_t len = tp.realization.size();
        if (tp.thread_partials.size() < static_cast<size_t>(nt))
          tp.thread_partials.resize(nt);
        for (auto& part : tp.thread_partials)
          part.resize(len, 0.0);
        proj_[p].partials = &tp.thread_partials;
      }
    }

    if (any_case_a) {
      // Whole outer rows per block, and at least FOLD_CHUNK_BINS bins when
      // an outer row is shorter; the lead block runs to the first outer-row
      // boundary at or after first_bin.
      const int64_t outer = n_bins_[nf_ - 1] * s_;
      block_bins_ = std::max<int64_t>(1, FOLD_CHUNK_BINS / outer) * outer;
      lead_bins_ = (outer - first_bin_ % outer) % outer;
    } else {
      block_bins_ = FOLD_CHUNK_BINS;
      lead_bins_ = 0;
    }
  }

  int64_t block_bins() const { return block_bins_; }
  int64_t lead_bins() const { return lead_bins_; }

  //! One thread's cursor: the fine-bin odometer and, per projection, the
  //! realization offset of the current tally row (negative when the row is
  //! dropped). Only a carry out of the last filter recomputes the projected
  //! row part of the other filters.
  class Thread {
  public:
    explicit Thread(const ProjectionFold& fold)
      : f_(fold), b_(fold.nf_), base_(fold.np_), drop_(fold.np_),
        off_(fold.np_), last_off_(fold.np_), dst_(fold.np_)
    {
      const int t = thread_num();
      for (size_t p = 0; p < f_.np_; ++p) {
        const Proj& pr = f_.proj_[p];
        dst_[p] = pr.case_a ? pr.realization : (*pr.partials)[t].data();
        last_off_[p] = pr.last_off.data();
      }
    }

    //! Position the cursor at plane bin k (the first bin of a dense chunk).
    void begin(int64_t k)
    {
      j_ = f_.decompose(k, b_.data());
      set_bases();
      set_rows();
    }

    //! Add the cursor bin's value to every projection, then advance one bin.
    void dense(double val)
    {
      for (size_t p = 0; p < f_.np_; ++p) {
        if (off_[p] >= 0)
          dst_[p][off_[p] + j_] += val;
      }
      if (++j_ == f_.s_) {
        j_ = 0;
        next_row();
      }
    }

    //! Add the value of plane bin k (a bin of a sparse chunk).
    void sparse(int64_t k, double val)
    {
      int64_t* b = b_.data();
      const int64_t j = f_.decompose(k, b);
      for (size_t p = 0; p < f_.np_; ++p) {
        const int64_t outer = f_.outer_part(p, b);
        const int64_t last = f_.last_part(p, b[f_.nf_ - 1]);
        if (outer >= 0 && last >= 0)
          dst_[p][(outer + last - f_.proj_[p].first_row) * f_.s_ + j] += val;
      }
    }

  private:
    //! Recompute each projection's part of filters 0..nf-2 (as an offset).
    void set_bases()
    {
      for (size_t p = 0; p < f_.np_; ++p) {
        const int64_t outer = f_.outer_part(p, b_.data());
        drop_[p] = outer < 0;
        base_[p] = (outer - f_.proj_[p].first_row) * f_.s_;
      }
    }

    //! Set every projection's realization offset for the current row.
    void set_rows()
    {
      const int64_t b_last = b_[f_.nf_ - 1];
      for (size_t p = 0; p < f_.np_; ++p) {
        const int64_t last = last_off_[p][b_last];
        off_[p] = (drop_[p] || last < 0) ? DROPPED : base_[p] + last;
      }
    }

    //! Advance the odometer one filter-bin row (last filter fastest).
    void next_row()
    {
      const int last = f_.nf_ - 1;
      if (++b_[last] == f_.n_bins_[last]) {
        b_[last] = 0;
        for (int i = last - 1; i >= 0; --i) {
          if (++b_[i] < f_.n_bins_[i])
            break;
          b_[i] = 0;
        }
        set_bases();
      }
      set_rows();
    }

    const ProjectionFold& f_;
    vector<int64_t> b_;    //!< fine bin of each tally filter
    int64_t j_ {0};        //!< score bin
    vector<int64_t> base_; //!< [p]: offset part of filters 0..nf-2
    vector<char> drop_;    //!< [p]: filters 0..nf-2 drop the row
    vector<int64_t> off_;  //!< [p]: realization offset of the current row
    vector<const int64_t*> last_off_; //!< [p]: Proj::last_off
    vector<double*> dst_; //!< [p]: realization or this thread's partial
  };

  //! Sum the case-B thread partials into the realizations and zero them, in
  //! thread order (deterministic for a given thread count).
  void reduce_thread_partials()
  {
    for (size_t p = 0; p < np_; ++p) {
      if (proj_[p].case_a)
        continue;
      auto& parts = *proj_[p].partials;
      double* y = proj_[p].realization;
      const int64_t len = parts.empty() ? 0 : parts[0].size();
#pragma omp parallel for schedule(static)
      for (int64_t x = 0; x < len; ++x) {
        double sum = 0.0;
        for (auto& part : parts) {
          sum += part[x];
          part[x] = 0.0;
        }
        y[x] += sum;
      }
    }
  }

private:
  //! Decompose plane bin k into the fine bin of each tally filter (into b)
  //! and return its score bin.
  int64_t decompose(int64_t k, int64_t* b) const
  {
    const int64_t g = first_bin_ + k;
    int64_t row = g / s_;
    const int64_t j = g - row * s_;
    for (int i = nf_ - 1; i >= 0; --i) {
      const int64_t q = row / n_bins_[i];
      b[i] = row - q * n_bins_[i];
      row = q;
    }
    return j;
  }

  //! Projected-row contribution of filter i's fine bin `bin` for projection
  //! p, or a negative value when the bin is dropped.
  int64_t part(size_t p, int i, int64_t bin) const
  {
    const size_t pi = p * nf_ + i;
    const int32_t* map = maps_[pi];
    const int64_t v = map ? map[bin] : bin;
    return v < 0 ? DROPPED : v * strides_[pi];
  }

  //! Projected-row part of filters 0..nf-2 (negative: dropped).
  int64_t outer_part(size_t p, const int64_t* b) const
  {
    int64_t sum = 0;
    for (int i = 0; i < nf_ - 1; ++i) {
      const int64_t c = part(p, i, b[i]);
      if (c < 0)
        return DROPPED;
      sum += c;
    }
    return sum;
  }

  //! Projected-row part of the last filter (negative: dropped).
  int64_t last_part(size_t p, int64_t bin) const
  {
    return part(p, nf_ - 1, bin);
  }

  static constexpr int64_t DROPPED {-1};

  struct Proj {
    int64_t first_row {0};
    bool case_a {false};
    double* realization {nullptr};
    vector<vector<double>>* partials {nullptr};
    vector<int64_t> last_off; //!< last filter bin -> offset, or DROPPED
  };

  int nf_;                 //!< tally filters
  size_t np_;              //!< projections
  int s_;                  //!< score bins
  int64_t first_bin_;      //!< tally bin of plane bin 0
  vector<int64_t> n_bins_; //!< bins per tally filter
  vector<Proj> proj_;
  vector<const int32_t*> maps_; //!< [p, i]: bin map, nullptr = identity
  vector<int64_t> strides_;     //!< [p, i]: projected stride
  int64_t block_bins_ {FOLD_CHUNK_BINS};
  int64_t lead_bins_ {0};
};

//! \brief Fold one batch's accumulator into the moments and zero the folded
//! bins.
//!
//! The n bins are folded in chunks of FOLD_CHUNK_BINS (fold_chunk()); each
//! thread folds a chunk sparse when its previous chunk was sparse. Skipping a
//! zero bin is exact: folding it would add +0.0 to each moment, and no moment
//! is ever -0.0 (they start at +0.0 and +0.0 + -0.0 == +0.0); a NaN or Inf bin
//! is not == 0.0 and is folded. A skipped bin holds +0.0 or -0.0, which gives
//! the next batch the same moments as a zeroed bin. Per-bin listing alone
//! would make a moderately occupied plane slower to fold than the plain loop
//! (scattered moment accesses defeat prefetching), hence the dense mode.
//!
//! With projections (PROJ::ENABLED) the parallel unit is a block of
//! proj.block_bins() bins after a lead block of proj.lead_bins(), each folded
//! as consecutive chunks by one thread (see ProjectionFold); without them the
//! unit is one chunk and this compiles to the plain chunked fold.
//! \param acc   Accumulator plane (private, shared, or rma private), n bins
//! \param win   rma window block of the same shape (two planes only)
//! \param mom   Moments, fold_n_moments() contiguous values per bin
//! \param n     Number of bins (filter-bin rows x score bins)
//! \param norm  Per-batch normalization
//! \param proj  Projection arm (NoProjection or ProjectionFold)
template<bool TWO_PLANES, FoldKind KIND, class PROJ>
void fold_bins(
  double* acc, double* win, double* mom, int64_t n, double norm, PROJ& proj)
{
  constexpr int NM = fold_n_moments<KIND>();
  if constexpr (!PROJ::ENABLED) {
    const int64_t n_chunks = (n + FOLD_CHUNK_BINS - 1) / FOLD_CHUNK_BINS;
#pragma omp parallel
    {
      int nonzero[FOLD_CHUNK_BINS]; // offsets of a sparse chunk's nonzero bins
      bool sparse = false;
#pragma omp for schedule(static)
      for (int64_t c = 0; c < n_chunks; ++c) {
        const int64_t first = c * FOLD_CHUNK_BINS;
        const int len =
          static_cast<int>(std::min<int64_t>(FOLD_CHUNK_BINS, n - first));
        double* w = nullptr;
        if constexpr (TWO_PLANES)
          w = win + first;
        fold_chunk<TWO_PLANES, KIND>(
          acc + first, w, mom + first * NM, len, norm, nonzero, sparse,
          [](double) {}, [](int, double) {});
      }
    }
  } else {
    const int64_t block = proj.block_bins();
    const int64_t lead = std::min(proj.lead_bins(), n);
    const int64_t n_blocks =
      (lead > 0 ? 1 : 0) + (n - lead + block - 1) / block;
#pragma omp parallel
    {
      int nonzero[FOLD_CHUNK_BINS];
      bool sparse = false;
      typename PROJ::Thread cursor(proj);
#pragma omp for schedule(static)
      for (int64_t blk = 0; blk < n_blocks; ++blk) {
        int64_t start, end;
        if (lead > 0 && blk == 0) {
          start = 0;
          end = lead;
        } else {
          start = lead + (blk - (lead > 0 ? 1 : 0)) * block;
          end = std::min(n, start + block);
        }
        for (int64_t first = start; first < end; first += FOLD_CHUNK_BINS) {
          const int len =
            static_cast<int>(std::min<int64_t>(FOLD_CHUNK_BINS, end - first));
          double* w = nullptr;
          if constexpr (TWO_PLANES)
            w = win + first;
          if (!sparse)
            cursor.begin(first);
          fold_chunk<TWO_PLANES, KIND>(
            acc + first, w, mom + first * NM, len, norm, nonzero, sparse,
            [&cursor](double val) { cursor.dense(val); },
            [&cursor, first](
              int q, double val) { cursor.sparse(first + q, val); });
        }
      }
    }
    proj.reduce_thread_partials();
  }
}

//! \brief Fold a tally's accumulator plane(s) into its moments: dispatch
//! fold_bins() on the moments kind once per fold.
//!
//! The mean moments mode keeps the sum only; otherwise higher moments add the
//! third and fourth. moments must hold n bins of the kind's moment count.
template<bool TWO_PLANES, class PROJ>
void fold_planes(MomentsMode mode, bool higher_moments, double* acc,
  double* win, tensor::Tensor<double>& moments, int64_t n, double norm,
  PROJ& proj)
{
  if (mode == MomentsMode::MEAN) {
    assert(moments.size() == static_cast<size_t>(n) * 1);
    fold_bins<TWO_PLANES, FoldKind::MEAN>(
      acc, win, moments.data(), n, norm, proj);
  } else if (higher_moments) {
    assert(moments.size() == static_cast<size_t>(n) * 4);
    fold_bins<TWO_PLANES, FoldKind::HIGHER>(
      acc, win, moments.data(), n, norm, proj);
  } else {
    assert(moments.size() == static_cast<size_t>(n) * 2);
    fold_bins<TWO_PLANES, FoldKind::FULL>(
      acc, win, moments.data(), n, norm, proj);
  }
}

} // namespace

void Tally::fold_moments()
{
  // accum_ is stored contiguously as [n_filter_bins, n_score_bins] and
  // moments_ as [n_filter_bins, n_score_bins, n_moments], so bin k of the
  // plane owns moments [k * n_moments, (k + 1) * n_moments).
  if (projections_.empty()) {
    NoProjection none;
    fold_planes<false>(exatally_.moments, higher_moments_, accum_, nullptr,
      moments_, accum_size_, tally_normalization(), none);
    return;
  }
  ProjectionFold proj(projections_, filters_, n_score_bins_, 0);
  fold_planes<false>(exatally_.moments, higher_moments_, accum_, nullptr,
    moments_, accum_size_, tally_normalization(), proj);
  fold_projection_moments();
}

#ifdef OPENMC_MPI
void report_rma_statistics()
{
  // The verbosity is rank-identical, so every rank skips the collective alike.
  if (settings::verbosity < 8)
    return;

  // Batches whose scores the counters cover: the active batches simulated in
  // this run. The counters start at zero at simulation init, so a restart
  // counts from the restart batch, not from the first active batch.
  int first = settings::n_inactive;
  if (settings::restart_run)
    first = std::max(first, simulation::restart_batch);
  const int n_batches = simulation::current_batch - first;
  const double n_source = static_cast<double>(settings::n_particles) *
                          n_batches * settings::gen_per_batch;

  for (const auto& t : model::tallies) {
    if (!t->rma_distributed())
      continue;
    int64_t local[3] {
      t->rma_rows_issued(), t->rma_accumulates(), t->rma_merged_rows()};
    int64_t total[3] {0, 0, 0};
    MPI_Reduce(local, total, 3, MPI_INT64_T, MPI_SUM, 0, mpi::intracomm);
    write_message(8,
      "rma tally {}: {} remote rows issued in {} accumulates ({} merged) "
      "over {} batches; {:.2f} remote rows per source particle",
      t->id_, total[0], total[1], total[2], n_batches,
      n_source > 0.0 ? total[0] / n_source : 0.0);
  }
}
#endif

//==============================================================================
// Tally member functions: accumulator modes
//==============================================================================

bool Tally::rma_distributed() const
{
#ifdef OPENMC_MPI
  return exatally_.accumulator == AccumulatorMode::RMA && mpi::n_procs > 1;
#else
  return false;
#endif
}

#ifdef OPENMC_MPI
namespace {

// Issue-time sort of a staging buffer by displacement (rma_sort_by_disp()):
// LSD radix digits of 9 bits, and the buffer size below which std::sort is
// faster than several 512-bucket passes (measured break-even at 32 rows for a
// three-pass key).
constexpr int RADIX_DIGIT_BITS = 9;
constexpr int RADIX_BUCKETS = 1 << RADIX_DIGIT_BITS;
constexpr int RADIX_MIN_ROWS = 32;

} // namespace

void Tally::rma_score_add(int64_t filter_index, int score_index, double val)
{
  // rma_k_ > 0 exactly when a remote bin is possible (a multi-rank run), and
  // then init_rma_staging() has allocated a slot for every thread. The state
  // is per-thread, so nothing here races another thread; only the accumulate
  // itself is serialized (a critical).
  if (rma_k_ > 0) {
    // Coalescer hit first: the dominant pattern is the scores loop -- several
    // consecutive calls with the same filter bin -- and an open coalescer row
    // only ever holds a REMOTE bin, so this one compare routes the repeat case
    // with no ownership arithmetic at all.
    RmaThreadStaging& ts = rma_staging_[thread_num()];
    if (filter_index == ts.coalesce_bin) {
      ts.row[score_index] += val;
      return;
    }

    // Local arm: a bin this rank owns is scored into the private owned-rows
    // plane (accum_), locally indexed. The window block itself receives only
    // MPI_Accumulate, so a NIC-side remote atomic never races a CPU store on
    // the same cell (the two classes target disjoint address ranges by
    // construction). The unsigned compare is the range test
    // rma_first_row_ <= filter_index < rma_first_row_ + rma_n_rows_.
    const int64_t local_row = filter_index - rma_first_row_;
    if (static_cast<uint64_t>(local_row) < static_cast<uint64_t>(rma_n_rows_)) {
      atomic_score_add(&accum_[local_row * n_score_bins_ + score_index], val);
      return;
    }

    // Remote arm (C2), coalescer miss: open a row for this bin, flushing any
    // previous one to its target's staging buffer for a batched
    // MPI_Accumulate to the owning rank. Design follows Dun et al. (2015):
    // origin-side buffering of one-sided accumulates cuts the per-score RMA
    // cost. The owner lookup runs only here -- once per coalesced row, not
    // once per score -- and the division only when the row leaves the block
    // of the last owner (the same unsigned range test as the local arm).
    if (ts.coalesce_bin >= 0)
      rma_coalescer_flush(ts);
    if (static_cast<uint64_t>(filter_index - ts.owner_lo) >=
        static_cast<uint64_t>(ts.owner_hi - ts.owner_lo)) {
      ts.owner_rank = rma_owner(filter_index);
      ts.owner_lo = static_cast<int64_t>(ts.owner_rank) * rma_bins_per_rank_;
      ts.owner_hi = ts.owner_lo + rma_bins_per_rank_;
    }
    ts.coalesce_bin = filter_index;
    ts.coalesce_owner = ts.owner_rank;
    std::fill(ts.row.begin(), ts.row.end(), 0.0);
    ts.row[score_index] += val;
    return;
  }

  // Single-rank run: every bin is local (rma_first_row_ == 0).
  atomic_score_add(&accum_[filter_index * n_score_bins_ + score_index], val);
}

int Tally::rma_owner(int64_t filter_index) const
{
  // Whole filter-bin rows are block-distributed: rows [r*bpr, (r+1)*bpr) belong
  // to rank r. bpr is the ceiling of bins/procs, so filter_index / bpr is
  // always a valid rank in [0, n_procs).
  return static_cast<int>(filter_index / rma_bins_per_rank_);
}

int64_t Tally::rma_first_row(int rank) const
{
  // Same block map as init_rma_accum(): rank r owns rows starting at r*bpr,
  // clamped to n_filter_bins_ (trailing ranks may own nothing).
  return std::min<int64_t>(
    static_cast<int64_t>(rank) * rma_bins_per_rank_, n_filter_bins_);
}

int64_t Tally::rma_rows_owned(int rank) const
{
  const int64_t last = std::min<int64_t>(
    static_cast<int64_t>(rank + 1) * rma_bins_per_rank_, n_filter_bins_);
  return last - rma_first_row(rank);
}

void Tally::rma_coalescer_flush(RmaThreadStaging& ts)
{
  // Append the open coalescer row to the active buffer for its target rank.
  const int t = ts.coalesce_owner;
  const int64_t bin = ts.coalesce_bin;
  const int b = ts.cur[t];
  const int slot = t * 2 + b;
  const int64_t f = ts.fill[slot];

  double* dst =
    &ts.data[(static_cast<int64_t>(slot) * rma_k_ + f) * n_score_bins_];
  std::copy(ts.row.begin(), ts.row.end(), dst);
  // Byte displacement of this row within the owner's window block. The owner's
  // first global row is t * bins_per_rank (exact for a bin it owns), so the
  // local row is bin - that, and each row spans n_score_bins_ doubles.
  const int64_t first_owned_row = static_cast<int64_t>(t) * rma_bins_per_rank_;
  ts.disp[static_cast<int64_t>(slot) * rma_k_ + f] =
    static_cast<MPI_Aint>((bin - first_owned_row) * n_score_bins_) *
    static_cast<MPI_Aint>(sizeof(double));
  ts.fill[slot] = static_cast<int>(f) + 1;
  ts.coalesce_bin = -1;

  if (ts.fill[slot] < rma_k_)
    return;

  // Buffer full: issue its accumulate, then move to the other buffer.
  rma_stage_issue(ts, t, b);
  const int nb = 1 - b;
  if (ts.in_flight[t * 2 + nb]) {
    // Both buffers for this target are outstanding, so the one we are about to
    // reuse still owns an in-flight payload. flush_local(t) locally retires
    // every accumulate this rank has issued to that one target -- unlike a
    // flush_local_all it does not stall on the other targets' pipelines -- so
    // both of this thread's buffers for t are reusable. Clear only those two
    // flags (other threads' flags for t stay conservatively marked: a
    // redundant flush at worst, never a reused-in-flight payload).
#pragma omp critical(openmc_rma)
    {
      MPI_Win_flush_local(t, accum_win_);
    }
    ts.in_flight[t * 2] = 0;
    ts.in_flight[t * 2 + 1] = 0;
  }
  ts.cur[t] = nb;
  ts.fill[t * 2 + nb] = 0;
}

void Tally::rma_stage_issue(RmaThreadStaging& ts, int target, int buf)
{
  const int slot = target * 2 + buf;
  const int f = ts.fill[slot];
  if (f == 0)
    return;
  double* data = &ts.data[static_cast<int64_t>(slot) * rma_k_ * n_score_bins_];
  MPI_Aint* disp = &ts.disp[static_cast<int64_t>(slot) * rma_k_];

  // The coalescer only merges *consecutive* scores to a bin, so a bin scored
  // non-consecutively (interleaved with other bins across particles) appears as
  // several rows at the same target displacement. MPI_Accumulate forbids a
  // target datatype with overlapping entries, so merge equal-displacement rows
  // first: sort an index by displacement, sum duplicates into the thread's
  // compaction scratch, then write the unique rows back to the front of this
  // buffer -- keeping the accumulate origin inside the double-buffered storage
  // whose in-flight liveness we track.
  const int* order = rma_sort_by_disp(ts, disp, f);

  int u = 0;
  for (int i = 0; i < f; ++i) {
    const int src = order[i];
    const double* srow = &data[static_cast<int64_t>(src) * n_score_bins_];
    if (u > 0 && disp[src] == ts.comp_disp[u - 1]) {
      double* drow = &ts.comp_data[static_cast<int64_t>(u - 1) * n_score_bins_];
      for (int s = 0; s < n_score_bins_; ++s)
        drow[s] += srow[s];
      ++ts.merged;
    } else {
      ts.comp_disp[u] = disp[src];
      std::copy(srow, srow + n_score_bins_,
        &ts.comp_data[static_cast<int64_t>(u) * n_score_bins_]);
      ++u;
    }
  }
  std::copy(ts.comp_data.begin(),
    ts.comp_data.begin() + static_cast<int64_t>(u) * n_score_bins_, data);
  std::copy(ts.comp_disp.begin(), ts.comp_disp.begin() + u, disp);

  // One MPI_Accumulate moves u rows of n_score_bins_ contiguous doubles from
  // the origin buffer into u scattered rows of the target's block. Accumulate
  // atomicity is per basic element (MPI_DOUBLE), so the indexed type is fine.
  // The type is created, committed, used, and freed inside the critical --
  // freeing after posting is legal (the posted operation completes normally)
  // and keeps every MPI call serialized under MPI_THREAD_SERIALIZED.
#pragma omp critical(openmc_rma)
  {
    MPI_Datatype dt;
    MPI_Type_create_hindexed_block(u, n_score_bins_, disp, MPI_DOUBLE, &dt);
    MPI_Type_commit(&dt);
    MPI_Accumulate(data, u * n_score_bins_, MPI_DOUBLE, target, 0, 1, dt,
      MPI_SUM, accum_win_);
    MPI_Type_free(&dt);
  }
  ts.in_flight[slot] = 1;
  ts.rows_issued += u;
  ++ts.accumulates;
}

const int* Tally::rma_sort_by_disp(
  RmaThreadStaging& ts, const MPI_Aint* disp, int f) const
{
  for (int i = 0; i < f; ++i)
    ts.perm[i] = i;

  if (f < RADIX_MIN_ROWS) {
    std::sort(ts.perm.begin(), ts.perm.begin() + f,
      [disp](int a, int b) { return disp[a] < disp[b]; });
    return ts.perm.data();
  }

  // Stable LSD radix sort on the key disp / 8 (every displacement is a whole
  // number of doubles), 9 bits per pass, scattering the index between perm and
  // perm2. Being stable, it keeps equal displacements in staging order, so the
  // merge sums duplicate rows in the order they were staged.
  int* src = ts.perm.data();
  int* dst = ts.perm2.data();
  int* hist = ts.hist.data();
  for (int pass = 0; pass < rma_radix_passes_; ++pass) {
    const int shift = 3 + pass * RADIX_DIGIT_BITS;
    auto digit = [disp, shift](int row) {
      return (static_cast<uint64_t>(disp[row]) >> shift) & (RADIX_BUCKETS - 1);
    };
    std::fill(hist, hist + RADIX_BUCKETS, 0);
    for (int i = 0; i < f; ++i)
      ++hist[digit(i)];
    int offset = 0;
    for (int b = 0; b < RADIX_BUCKETS; ++b) {
      const int count = hist[b];
      hist[b] = offset;
      offset += count;
    }
    for (int i = 0; i < f; ++i)
      dst[hist[digit(src[i])]++] = src[i];
    std::swap(src, dst);
  }
  return src;
}

void Tally::rma_drain()
{
  // Runs single-threaded at end of batch (the transport region has ended), so
  // there is no contention; the main thread walks every thread's slot. Flush
  // each open coalescer row, then issue whatever remains in each target's
  // active buffer (the inactive buffer, if full, was already issued at swap
  // time).
  for (auto& ts : rma_staging_) {
    if (ts.coalesce_bin >= 0)
      rma_coalescer_flush(ts);
    for (int t = 0; t < mpi::n_procs; ++t) {
      if (t == mpi::rank)
        continue;
      rma_stage_issue(ts, t, ts.cur[t]);
    }
  }
}

void Tally::free_accum_window()
{
  if (accum_win_ == MPI_WIN_NULL)
    return;
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  if (!mpi_finalized) {
    // Close the persistent passive-target epoch, then free. Both are collective
    // over the window's group (node_comm for shared, intracomm for rma); every
    // rank destroys its tallies in the same order, so the calls line up.
    MPI_Win_unlock_all(accum_win_);
    MPI_Win_free(&accum_win_);
  }
  accum_win_ = MPI_WIN_NULL;
  rma_win_base_ = nullptr;
}

namespace {

//! Interleave the calling task's memory policy for the duration of an
//! allocation, so pages an MPI library pre-faults inside
//! MPI_Win_allocate_shared (registration pinning on some fabrics) spread over
//! all NUMA domains the task may use instead of landing on the allocating
//! rank's domain. Restores the previous policy on exit. Engages only when the
//! task runs the default policy, so an explicit user policy (numactl
//! --membind/--interleave) is honoured as is. Best effort: any failed call
//! leaves the policy alone. A no-op outside Linux.
class InterleaveDuringAlloc {
public:
  explicit InterleaveDuringAlloc(bool engage)
  {
#ifdef __linux__
    if (!engage)
      return;
    int mode;
    if (syscall(SYS_get_mempolicy, &mode, nullptr, 0UL, nullptr, 0UL) != 0 ||
        mode != MPOL_DEFAULT)
      return;
    std::array<unsigned long, 64> mask {};
    const unsigned long maxnode =
      mask.size() * sizeof(unsigned long) * CHAR_BIT;
    // Variadic syscall(): pass unsigned long where the kernel takes one.
    if (syscall(SYS_get_mempolicy, nullptr, mask.data(), maxnode, nullptr,
          static_cast<unsigned long>(MPOL_F_MEMS_ALLOWED)) != 0)
      return;
    int n = 0;
    for (unsigned long word : mask)
      n += std::bitset<sizeof(unsigned long) * CHAR_BIT>(word).count();
    if (n <= 1)
      return;
    if (syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, mask.data(), maxnode) != 0)
      return;
    n_domains_ = n;
#endif
  }

  ~InterleaveDuringAlloc()
  {
#ifdef __linux__
    if (n_domains_ > 0)
      syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0UL);
#endif
  }

  InterleaveDuringAlloc(const InterleaveDuringAlloc&) = delete;
  InterleaveDuringAlloc& operator=(const InterleaveDuringAlloc&) = delete;

  //! \return Number of NUMA domains interleaved over, 0 if not engaged
  int n_domains() const { return n_domains_; }

private:
  int n_domains_ {0}; //!< NUMA domains interleaved over; 0 if not engaged
};

} // namespace

void Tally::init_shared_accum()
{
  // Re-init (e.g. openmc.lib re-runs): drop any previous window first. This is
  // collective on node_comm and lines up because every rank re-inits its
  // tallies in the same order.
  free_accum_window();

  // The shared plane replaces the private buffer entirely -- that is the RAM
  // win, so make sure no per-rank copy lingers.
  accum_buffer_.clear();
  accum_buffer_.shrink_to_fit();

  // The node leader allocates the whole plane; everyone else contributes 0
  // bytes. The default (contiguous) layout means querying the leader's segment
  // returns one contiguous plane mapped into every rank -- do not request
  // alloc_shared_noncontig, which would break that assumption.
  MPI_Aint bytes =
    mpi::node_leader ? static_cast<MPI_Aint>(accum_size_) * sizeof(double) : 0;
  void* base = nullptr;
  int n_interleaved = 0;
  {
    InterleaveDuringAlloc policy(mpi::node_leader);
    MPI_Win_allocate_shared(
      bytes, sizeof(double), MPI_INFO_NULL, mpi::node_comm, &base, &accum_win_);
    n_interleaved = policy.n_domains();
  }
  if (n_interleaved > 0) {
    write_message(8,
      "shared tally {}: node plane allocated under an interleave memory "
      "policy over {} NUMA domains (restored after allocation)",
      id_, n_interleaved);
  }

  MPI_Aint seg_bytes;
  int seg_disp;
  MPI_Win_shared_query(accum_win_, 0, &seg_bytes, &seg_disp, &base);
  accum_ = static_cast<double*>(base);

  // The atomic scoring path needs natural alignment and the unified memory
  // model (public == private window copy) so load/store atomics stay coherent.
  assert(reinterpret_cast<uintptr_t>(accum_) % alignof(double) == 0);
  int* model;
  int flag;
  MPI_Win_get_attr(accum_win_, MPI_WIN_MODEL, &model, &flag);
  if (!flag || *model != MPI_WIN_UNIFIED) {
    fatal_error(
      "The shared accumulator mode requires an MPI_WIN_UNIFIED "
      "shared-memory window (non-cache-coherent hardware is not supported).");
  }

  // Open a persistent passive-target epoch for the window's whole life so the
  // win_sync memory barriers used at every batch boundary are legal MPI.
  MPI_Win_lock_all(MPI_MODE_NOCHECK, accum_win_);

  // Every node rank zeroes its own page-aligned slice, then the plane is
  // published so every rank sees all-zeros before the first score. If the MPI
  // library does not touch the pages at allocation, this fill is their first
  // touch: Linux places each page on the NUMA domain of the rank that zeroes
  // it, which spreads the plane over the node instead of putting it all on the
  // leader's domain. If the library pre-faults them during allocation
  // (registration pinning on some fabrics), the interleave policy around the
  // allocation has already spread them over the domains.
  int64_t begin, end;
  shared_slice(begin, end);
  std::fill(accum_ + begin, accum_ + end, 0.0);
  MPI_Win_sync(accum_win_);
  MPI_Barrier(mpi::node_comm);
  MPI_Win_sync(accum_win_);
}

void Tally::shared_slice(int64_t& begin, int64_t& end) const
{
  // Whole 4 KiB pages (512 doubles) per node rank, so no page is shared by two
  // slices and page placement follows the slicing. The ceiling division makes
  // the slices cover the plane; trailing ranks may get an empty slice.
  constexpr int64_t PAGE_DOUBLES = 4096 / sizeof(double);
  const int64_t n_pages = (accum_size_ + PAGE_DOUBLES - 1) / PAGE_DOUBLES;
  const int64_t pages_per_rank =
    (n_pages + mpi::node_size - 1) / mpi::node_size;
  const int64_t slice = pages_per_rank * PAGE_DOUBLES;
  begin = std::min<int64_t>(mpi::node_rank * slice, accum_size_);
  end = std::min<int64_t>((mpi::node_rank + 1) * slice, accum_size_);
}

void Tally::shared_publish()
{
  // Step 1 -- publish every node-local atomic score. The symmetric
  // win_sync/barrier/win_sync is the MPI-3 shared-memory idiom: the first sync
  // flushes writers, the barrier orders, the second refreshes the reader.
  MPI_Win_sync(accum_win_);
  MPI_Barrier(mpi::node_comm);
  MPI_Win_sync(accum_win_);

  // Step 2 -- combine the per-node planes onto the world master. A single-node
  // run skips this: the one plane already holds the global sum and the master
  // folds it directly. World rank 0 is rank 0 of internode_comm, so the
  // reduction roots on the master; it runs in 1 GiB chunks
  // (reduce_accum_to_master).
  if (mpi::n_nodes > 1 && mpi::node_leader) {
    reduce_accum_to_master(accum_, accum_size_, mpi::internode_comm);
  }
}

void Tally::shared_resume()
{
  // Step 4 -- publish the zeroed plane so the next batch scores into all-zeros.
  MPI_Win_sync(accum_win_);
  MPI_Barrier(mpi::node_comm);
  MPI_Win_sync(accum_win_);
}

void Tally::init_rma_accum()
{
  // One-sided tally accumulation follows Romano et al. (2011) and Dun et al.
  // (2015), implemented independently on MPI-3.

  // Re-init (e.g. openmc.lib re-runs): drop any previous window first.
  // Idempotent and collective over the window's group; every rank re-inits in
  // the same order so the calls line up.
  free_accum_window();

  // Block-distribute whole filter-bin rows across all ranks. bins_per_rank is
  // the ceiling of bins/procs so the rows partition exactly; trailing ranks may
  // own zero rows (a zero-size window block is legal). max(1, ...) guards the
  // ownership division against a degenerate zero-bin tally.
  const int64_t n_procs = mpi::n_procs;
  rma_bins_per_rank_ =
    std::max<int64_t>(1, (n_filter_bins_ + n_procs - 1) / n_procs);
  rma_first_row_ = std::min<int64_t>(
    static_cast<int64_t>(mpi::rank) * rma_bins_per_rank_, n_filter_bins_);
  const int64_t last_row = std::min<int64_t>(
    static_cast<int64_t>(mpi::rank + 1) * rma_bins_per_rank_, n_filter_bins_);
  rma_n_rows_ = last_row - rma_first_row_;
  rma_plane_size_ = rma_n_rows_ * n_score_bins_;

  // Private owned-rows plane for this rank's own scores to bins it owns. Reuses
  // accum_buffer_ so the hot path writes it exactly like replicated mode, just
  // sized to the owned block and locally indexed.
  accum_buffer_.assign(rma_plane_size_, 0.0);
  accum_ = accum_buffer_.data();

  // Distributed window: each rank contributes its owned block. same_op (every
  // accumulate is MPI_SUM; MPI_NO_OP is never used) + no ordering lets the
  // implementation use hardware atomics for the SUM-only accumulates and drops
  // ordering overhead (we never mix ops or rely on order).
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "accumulate_ops", "same_op");
  MPI_Info_set(info, "accumulate_ordering", "none");
  MPI_Aint bytes = static_cast<MPI_Aint>(rma_plane_size_) * sizeof(double);
  void* base = nullptr;
  MPI_Win_allocate(
    bytes, sizeof(double), info, mpi::intracomm, &base, &accum_win_);
  MPI_Info_free(&info);
  rma_win_base_ = static_cast<double*>(base);

  // Require the unified memory model, exactly as the shared window does, so
  // MPI_Win_sync is a plain memory barrier and the SEPARATE-model self-lock
  // problem stays out of scope.
  int* model;
  int flag;
  MPI_Win_get_attr(accum_win_, MPI_WIN_MODEL, &model, &flag);
  if (!flag || *model != MPI_WIN_UNIFIED) {
    fatal_error("The rma accumulator mode requires an MPI_WIN_UNIFIED window "
                "(non-cache-coherent hardware is not supported).");
  }

  // Persistent passive-target epoch for the window's whole life so the
  // per-batch flush/sync operations are legal MPI.
  MPI_Win_lock_all(MPI_MODE_NOCHECK, accum_win_);

  // Zero this rank's window block (MPI_Win_allocate memory is uninitialized)
  // and publish it so remote accumulates land on all-zeros.
  if (rma_plane_size_ > 0)
    std::fill(rma_win_base_, rma_win_base_ + rma_plane_size_, 0.0);
  MPI_Win_sync(accum_win_);
  MPI_Barrier(mpi::intracomm);
  MPI_Win_sync(accum_win_);

  // First-batch diagnostics (high verbosity only): the master reports the
  // layout.
  write_message(8,
    "rma tally {}: {} filter bins block-distributed across {} ranks "
    "(~{} rows/rank, {:.1f} MiB/rank window)",
    id_, n_filter_bins_, mpi::n_procs, rma_bins_per_rank_,
    static_cast<double>(rma_plane_size_) * sizeof(double) / (1024.0 * 1024.0));

  // Allocate the per-thread remote-scoring staging (skipped on single-rank
  // runs, which never score remotely).
  init_rma_staging();
}

void Tally::init_rma_staging()
{
  rma_staging_.clear();
  rma_k_ = 0;

  // A single-rank run owns every bin, so score_add never reaches the remote arm
  // -- no staging is needed and the 3.A local path is preserved untouched.
  if (mpi::n_procs <= 1)
    return;

  // Choose the staging depth K (rows per buffer). The test seam
  // OPENMC_RMA_STAGING_ROWS forces a tiny K to exercise buffer wrap + flush;
  // otherwise K comes from a fixed per-rank budget split across tallies,
  // threads, targets, and the two buffers, so total staging stays near the
  // budget.
  const int nt = num_threads();
  const int64_t np = mpi::n_procs;

  // Bytes one thread holds per unit of K: per target and buffer, K staged rows
  // plus their displacements; plus one buffer's worth of issue scratch
  // (comp_data row, the perm and perm2 index entries, comp_disp entry). The
  // fixed RADIX_BUCKETS-int histogram per thread (2 KiB) is not counted.
  const int64_t row_bytes =
    static_cast<int64_t>(n_score_bins_) * sizeof(double);
  const int64_t disp_bytes =
    sizeof(decltype(RmaThreadStaging::disp)::value_type);
  const int64_t perm_bytes =
    sizeof(decltype(RmaThreadStaging::perm)::value_type);
  const int64_t bytes_per_k = np * 2 * (row_bytes + disp_bytes) +
                              (row_bytes + 2 * perm_bytes + disp_bytes);

  // Fixed per-rank budget covering ALL rma tallies (each carries its own
  // staging), shared in proportion to plane size (filter bins x score bins),
  // so a small companion tally does not halve a large tally's K. Later tallies
  // have not run set_strides() and init_results() yet, so their plane size
  // comes from their filters, scores and nuclides. The share of a lone rma
  // tally is the whole budget, exactly.
  constexpr int64_t BUDGET = int64_t {64} << 20; // 64 MiB / rank, total
  double total_bins = 0.0;
  for (const auto& t : model::tallies) {
    if (t->accumulator() != AccumulatorMode::RMA)
      continue;
    double bins = static_cast<double>(t->scores_.size() * t->nuclides_.size());
    for (int32_t i_filt : t->filters())
      bins *= model::tally_filters[i_filt]->n_bins();
    total_bins += bins;
  }
  const double own_bins = static_cast<double>(n_filter_bins_) * n_score_bins_;
  const double share = total_bins > 0.0
                         ? static_cast<double>(BUDGET) * (own_bins / total_bins)
                         : static_cast<double>(BUDGET);

  if (const char* env = std::getenv("OPENMC_RMA_STAGING_ROWS")) {
    rma_k_ = std::max<int64_t>(1, std::atoll(env));
  } else {
    const double per_k = static_cast<double>(nt) * bytes_per_k;
    rma_k_ = std::clamp<int64_t>(static_cast<int64_t>(share / per_k), 8, 4096);
  }

  // Radix passes for the issue-time sort: the key disp / 8 is below the
  // largest block a rank owns (rma_bins_per_rank_ rows x n_score_bins_
  // doubles), so ceil(log2) of that bounds the key width.
  const uint64_t max_key = static_cast<uint64_t>(rma_bins_per_rank_) *
                           static_cast<uint64_t>(n_score_bins_);
  int key_bits = 1;
  while (key_bits < 64 && (uint64_t {1} << key_bits) < max_key)
    ++key_bits;
  rma_radix_passes_ = (key_bits + RADIX_DIGIT_BITS - 1) / RADIX_DIGIT_BITS;

  // One slot per thread; self-target slots are allocated but never used (the
  // local arm handles owned bins). Every per-thread buffer starts empty.
  rma_merged_rows_ = 0;
  rma_rows_issued_ = 0;
  rma_accumulates_ = 0;
  rma_staging_.resize(nt);
  for (auto& ts : rma_staging_) {
    ts.row.assign(n_score_bins_, 0.0);
    ts.data.assign(np * 2 * rma_k_ * n_score_bins_, 0.0);
    ts.disp.assign(np * 2 * rma_k_, 0);
    ts.fill.assign(np * 2, 0);
    ts.in_flight.assign(np * 2, 0);
    ts.cur.assign(np, 0);
    // Per-thread issue scratch (one buffer's worth), reused at each issue.
    ts.perm.assign(rma_k_, 0);
    ts.perm2.assign(rma_k_, 0);
    ts.hist.assign(RADIX_BUCKETS, 0);
    ts.comp_disp.assign(rma_k_, 0);
    ts.comp_data.assign(rma_k_ * n_score_bins_, 0.0);
    ts.merged = 0;
    ts.rows_issued = 0;
    ts.accumulates = 0;
  }

  write_message(8,
    "rma tally {}: staging K={} rows/buffer, {:.1f} MiB/rank incl. "
    "displacements and scratch ({} threads x {} targets x 2 buffers; "
    "budget share {:.1f} MiB)",
    id_, rma_k_,
    static_cast<double>(nt) * rma_k_ * bytes_per_k / (1024.0 * 1024.0), nt, np,
    share / (1024.0 * 1024.0));
}

void Tally::rma_publish()
{
  // Step 1 -- drain thread-local staging: issue every open coalescer row and
  // partial buffer so all of this rank's contributions become accumulates.
  rma_drain();
  // Step 2 -- complete this rank's outstanding accumulates at every target
  // (both locally and remotely). All origin buffers are now retired.
  MPI_Win_flush_all(accum_win_);
  // The next batch starts from empty staging; reset the per-thread bookkeeping
  // and fold the per-thread merge counts into the run-level diagnostic.
  for (auto& ts : rma_staging_) {
    std::fill(ts.in_flight.begin(), ts.in_flight.end(), 0);
    std::fill(ts.fill.begin(), ts.fill.end(), 0);
    std::fill(ts.cur.begin(), ts.cur.end(), 0);
    ts.coalesce_bin = -1;
    rma_merged_rows_ += ts.merged;
    rma_rows_issued_ += ts.rows_issued;
    rma_accumulates_ += ts.accumulates;
    ts.merged = 0;
    ts.rows_issued = 0;
    ts.accumulates = 0;
  }
  // Step 3 -- after this barrier no accumulate is in flight anywhere.
  MPI_Barrier(mpi::intracomm);
  // Step 4 -- local memory barrier before the fold reads this rank's own block.
  MPI_Win_sync(accum_win_);
}

void Tally::rma_resume()
{
  // Publish the zeroed window block so the next batch accumulates into zeros.
  MPI_Win_sync(accum_win_);
  MPI_Barrier(mpi::intracomm);
  MPI_Win_sync(accum_win_);
}

void Tally::rma_fold()
{
  // Every rank folds its owned rows. The batch sum for a bin is its window
  // block (remote contributions, zero while scoring is local) plus the
  // private plane (this rank's own scores); the folded bins of both planes are
  // then zeroed (bins zero in both may be skipped, see fold_bins()). norm
  // is rank-identical: it depends only on global model data and rma requires
  // reduce_tallies. n_realizations_ was incremented on all ranks in
  // accumulate().
  if (projections_.empty()) {
    NoProjection none;
    fold_planes<true>(exatally_.moments, higher_moments_, accum_, rma_win_base_,
      moments_, rma_plane_size_, tally_normalization(), none);
  } else {
    // Each rank projects its owned rows (plane bin 0 is tally bin
    // rma_first_row_ * n_score_bins_); the master sums the partials and
    // folds the projected moments.
    ProjectionFold proj(
      projections_, filters_, n_score_bins_, rma_first_row_ * n_score_bins_);
    fold_planes<true>(exatally_.moments, higher_moments_, accum_, rma_win_base_,
      moments_, rma_plane_size_, tally_normalization(), proj);
    combine_rma_projections();
  }
  // Publish the zeroed window block before the next batch accumulates.
  rma_resume();
}

void Tally::shared_batch_end()
{
  if (!mpi::master_node) {
    // A non-master node does not fold, but its node plane was the source of the
    // internode reduce and still holds this node's batch sum. Zero it so it
    // does not carry into the next batch. (The master zeroes its own plane in
    // the fold in accumulate(); collect_none() reuses this routine, so there
    // the master node keeps its run total.) After this barrier the leader has
    // returned from the internode reduce and the plane is dead data;
    // non-leaders must not write it before that. Each rank then zeroes its own
    // slice, so it writes the pages that first touch placed on its NUMA domain.
    MPI_Barrier(mpi::node_comm);
    int64_t begin, end;
    shared_slice(begin, end);
#pragma omp parallel for
    for (int64_t i = begin; i < end; ++i)
      accum_[i] = 0.0;
  }

  // Publish the zeroed plane before the next batch scores into it. Every node
  // rank participates in the barrier inside shared_resume().
  shared_resume();
}
#endif

//==============================================================================
// Tally member functions: projected moments
//==============================================================================

namespace {

//! \brief Map the bins of a fine energy filter onto a coarser energy filter
//! nested in it.
//!
//! Every boundary of the coarse filter must be a boundary of the fine filter
//! (to a relative 1e-12, since both come from input text), and the coarse
//! range must lie within the fine range. Fine bins outside the coarse range
//! map to -1 (dropped).
//! \param fine    The tally's energy filter
//! \param coarse  The projection's energy filter
//! \param where   "Tally N projection 'name'", for error text
//! \return Coarse bin of each fine bin, or -1
vector<int32_t> nested_energy_map(const EnergyFilter& fine,
  const EnergyFilter& coarse, const std::string& where)
{
  const auto& e = fine.bins();
  const auto& c = coarse.bins();
  auto close = [](double a, double b) {
    return std::abs(a - b) <= 1e-12 * std::max(std::abs(a), std::abs(b));
  };
  if ((c.front() < e.front() && !close(c.front(), e.front())) ||
      (c.back() > e.back() && !close(c.back(), e.back()))) {
    fatal_error(fmt::format(
      "{}: energy filter {} ([{}, {}] eV) extends outside the range of the "
      "tally's energy filter {} ([{}, {}] eV).",
      where, coarse.id(), c.front(), c.back(), fine.id(), e.front(), e.back()));
  }

  // Fine boundary index of each coarse boundary.
  vector<int64_t> index(c.size());
  for (size_t m = 0; m < c.size(); ++m) {
    const auto it = std::lower_bound(e.begin(), e.end(), c[m]);
    int64_t k = it - e.begin();
    if (k < static_cast<int64_t>(e.size()) && close(e[k], c[m])) {
      index[m] = k;
    } else if (k > 0 && close(e[k - 1], c[m])) {
      index[m] = k - 1;
    } else {
      fatal_error(fmt::format(
        "{}: boundary {} eV of energy filter {} is not a boundary of the "
        "tally's energy filter {}, so the coarse groups are not unions of "
        "fine groups.",
        where, c[m], coarse.id(), fine.id()));
    }
  }

  vector<int32_t> map(e.size() - 1, -1);
  for (size_t m = 0; m + 1 < c.size(); ++m) {
    for (int64_t g = index[m]; g < index[m + 1]; ++g)
      map[g] = static_cast<int32_t>(m);
  }
  return map;
}

} // namespace

vector<std::pair<std::string, vector<int32_t>>> parse_projections(
  pugi::xml_node node, const std::string& context)
{
  vector<std::pair<std::string, vector<int32_t>>> result;
  pugi::xml_node block = node.child("exatally");
  if (!block)
    return result;
  for (pugi::xml_node p : block.children("projection")) {
    if (!p.attribute("name")) {
      throw std::runtime_error {
        fmt::format("A <projection> of {} has no 'name' attribute.", context)};
    }
    std::string name = get_node_value(p, "name", false, true);
    if (!p.attribute("filters")) {
      throw std::runtime_error {
        fmt::format("Projection '{}' of {} has no 'filters' attribute (give "
                    "an empty list to sum over every filter).",
          name, context)};
    }
    result.emplace_back(name, get_node_array<int32_t>(p, "filters"));
  }
  return result;
}

void Tally::init_projections()
{
  if (projections_.empty())
    return;

  const int nf = filters_.size();
  const int64_t n_score = n_score_bins_;
  for (auto& proj : projections_) {
    const std::string where =
      fmt::format("Tally {} projection '{}'", id_, proj.name);

    // The name is an HDF5 group name in the statepoint.
    if (proj.name.empty() || proj.name == "." ||
        proj.name.find('/') != std::string::npos) {
      fatal_error(fmt::format("{}: a projection name must be non-empty, not "
                              "'.', and contain no '/'.",
        where));
    }
    for (const auto& other : projections_) {
      if (&other != &proj && other.name == proj.name) {
        fatal_error(fmt::format("{} is declared more than once.", where));
      }
    }

    // Match each projected filter to a tally filter, in the tally's order: the
    // filter itself (kept) or an energy filter nested in it (collapsed).
    proj.maps.assign(nf, {});
    proj.strides.assign(nf, 0);
    vector<char> kept(nf, 0);
    vector<int> axis_filter; // tally filter of each projected axis
    vector<int64_t> axis_size;
    int next = 0;
    for (int32_t i_filt : proj.filters) {
      const Filter* pf = model::tally_filters[i_filt].get();
      const auto it = std::find(filters_.begin(), filters_.end(), i_filt);
      int j;
      if (it != filters_.end()) {
        j = static_cast<int>(it - filters_.begin());
        kept[j] = 1;
      } else if (dynamic_cast<const MeshFilter*>(pf)) {
        fatal_error(fmt::format(
          "{}: filter {} is a mesh filter that is not one of the tally's "
          "filters; projections onto a coarser mesh are not supported yet "
          "(only energy filters can be collapsed).",
          where, pf->id()));
      } else {
        const auto* coarse = dynamic_cast<const EnergyFilter*>(pf);
        vector<int> candidates;
        if (coarse) {
          for (int k = 0; k < nf; ++k) {
            if (model::tally_filters[filters_[k]]->type() == pf->type())
              candidates.push_back(k);
          }
        }
        if (candidates.empty()) {
          fatal_error(fmt::format(
            "{}: filter {} is neither one of the tally's filters nor an "
            "energy filter nested in the tally's energy filter.",
            where, pf->id()));
        }
        if (candidates.size() > 1) {
          fatal_error(fmt::format(
            "{}: energy filter {} could collapse any of the tally's {} "
            "filters of its type; a tally with more than one such energy "
            "filter cannot collapse one.",
            where, pf->id(), candidates.size()));
        }
        j = candidates[0];
        const auto* fine = dynamic_cast<const EnergyFilter*>(
          model::tally_filters[filters_[j]].get());
        proj.maps[j] = nested_energy_map(*fine, *coarse, where);
      }
      if (j < next) {
        fatal_error(fmt::format(
          "{}: its filters must follow the tally's filter order, each tally "
          "filter at most once.",
          where));
      }
      next = j + 1;
      axis_filter.push_back(j);
      axis_size.push_back(pf->n_bins());
    }
    if (std::count(kept.begin(), kept.end(), 1) == nf) {
      fatal_error(fmt::format(
        "{} is identical to the tally (it keeps every filter).", where));
    }

    // Projected strides, last projected filter fastest (as set_strides()).
    int64_t stride = 1;
    for (int k = static_cast<int>(axis_filter.size()) - 1; k >= 0; --k) {
      proj.strides[axis_filter[k]] = stride;
      stride *= axis_size[k];
    }
    proj.n_bins = stride;

    // Case A: every filter but the last kept, the last collapsed or summed.
    proj.fast_path = nf >= 2 && !kept[nf - 1] &&
                     std::count(kept.begin(), kept.end() - 1, 1) == nf - 1;

    // Storage: the moments live where the fold runs (the master, or every rank
    // without tally reduction); the realization on every rank that folds
    // bins, under rma as this rank's partial (case A: the projected rows its
    // owned rows reach; case B: dense).
    const bool holds_moments = mpi::master || !settings::reduce_tallies;
    proj.moments = tensor::Tensor<double>();
    proj.thread_partials.clear();
    proj.first_row = 0;
    proj.last_row = -1;
    if (holds_moments) {
      proj.moments =
        tensor::Tensor<double>({static_cast<size_t>(proj.n_bins),
                                 static_cast<size_t>(n_score), size_t {2}},
          0.0);
      proj.last_row = proj.n_bins - 1;
    }
#ifdef OPENMC_MPI
    if (rma_distributed() && !mpi::master && rma_n_rows_ > 0) {
      if (proj.fast_path) {
        const int64_t n_last = model::tally_filters[filters_[nf - 1]]->n_bins();
        const int64_t g = proj.maps[nf - 1].empty() ? 1 : axis_size.back();
        proj.first_row = (rma_first_row_ / n_last) * g;
        proj.last_row =
          ((rma_first_row_ + rma_n_rows_ - 1) / n_last) * g + g - 1;
      } else {
        proj.last_row = proj.n_bins - 1;
      }
    }
#endif
    proj.realization.assign(
      (proj.last_row - proj.first_row + 1) * n_score, 0.0);

    const double mib = 1.0 / (1024.0 * 1024.0);
    const double plane = static_cast<double>(proj.n_bins) * n_score * 8 * mib;
    std::string extra;
    if (!proj.fast_path) {
      extra = fmt::format(
        "; case B, {:.3g} MiB of per-thread realizations on each folding rank",
        plane * num_threads());
    } else {
      extra = "; case A, threads add in place";
    }
#ifdef OPENMC_MPI
    if (rma_distributed()) {
      double received = 0.0;
      for (int r = 1; r < mpi::n_procs; ++r) {
        const int64_t rows = rma_rows_owned(r);
        if (rows == 0)
          continue;
        if (proj.fast_path) {
          const int64_t n_last =
            model::tally_filters[filters_[nf - 1]]->n_bins();
          const int64_t g = proj.maps[nf - 1].empty() ? 1 : axis_size.back();
          const int64_t lo = (rma_first_row(r) / n_last) * g;
          const int64_t hi = ((rma_first_row(r) + rows - 1) / n_last) * g + g;
          received += static_cast<double>(hi - lo) * n_score * 8 * mib;
        } else {
          received += plane;
        }
      }
      extra +=
        fmt::format("; {:.3g} MiB received by the master per batch", received);
    }
#endif
    write_message(8,
      "tally {} projection '{}': {} projected filter bins x {} score bins, "
      "{:.3g} MiB on the master (sum, sum of squares, realization){}",
      id_, proj.name, proj.n_bins, n_score, 3.0 * plane, extra);
  }
}

void Tally::reset_projections()
{
  for (auto& proj : projections_) {
    if (proj.moments.size() != 0)
      proj.moments.fill(0.0);
    std::fill(proj.realization.begin(), proj.realization.end(), 0.0);
    for (auto& part : proj.thread_partials)
      std::fill(part.begin(), part.end(), 0.0);
  }
}

void Tally::fold_projection_moments()
{
  constexpr int SUM = static_cast<int>(TallyMoment::SUM);
  constexpr int SUM_SQ = static_cast<int>(TallyMoment::SUM_SQ);
  for (auto& proj : projections_) {
    if (proj.moments.size() == 0)
      continue;
    const int64_t n = proj.n_bins * n_score_bins_;
    assert(static_cast<int64_t>(proj.realization.size()) == n);
    double* mom = proj.moments.data();
    double* y = proj.realization.data();
#pragma omp parallel for schedule(static)
    for (int64_t x = 0; x < n; ++x) {
      const double v = y[x];
      mom[2 * x + SUM] += v;
      mom[2 * x + SUM_SQ] += v * v;
      y[x] = 0.0;
    }
  }
}

#ifdef OPENMC_MPI
void Tally::combine_rma_projections()
{
  // Single rank: the master's realization already holds the whole batch.
  if (!rma_distributed()) {
    fold_projection_moments();
    return;
  }

  // Every non-master sends, per projection in declaration order, one header
  // {first_row, last_row} (first_row > last_row for an empty slice) and then
  // its slice in chunks of at most MAX_MPI_CHUNK_DOUBLES; the master receives
  // from ranks 1..n-1 in rank order and adds each slice into its
  // realization, so a projected row straddling two owners sums both partials
  // before it is squared. Called for the same tallies in the same order on
  // every rank (accumulate() over the active tallies), and messages between
  // two ranks with one tag do not overtake, so the pairing is exact.
  const int64_t n_score = n_score_bins_;
  for (auto& proj : projections_) {
    if (mpi::master) {
      vector<double> buf;
      for (int r = 1; r < mpi::n_procs; ++r) {
        int64_t header[2];
        MPI_Recv(header, 2, MPI_INT64_T, r, TAG_RMA_PROJECTION, mpi::intracomm,
          MPI_STATUS_IGNORE);
        if (header[0] > header[1])
          continue;
        assert(header[0] >= 0 && header[1] < proj.n_bins);
        double* dst = proj.realization.data() + header[0] * n_score;
        const int64_t n = (header[1] - header[0] + 1) * n_score;
        for (int64_t off = 0; off < n; off += MAX_MPI_CHUNK_DOUBLES) {
          const int64_t count = std::min(n - off, MAX_MPI_CHUNK_DOUBLES);
          buf.resize(count);
          MPI_Recv(buf.data(), static_cast<int>(count), MPI_DOUBLE, r,
            TAG_RMA_PROJECTION, mpi::intracomm, MPI_STATUS_IGNORE);
          double* d = dst + off;
#pragma omp parallel for schedule(static)
          for (int64_t x = 0; x < count; ++x)
            d[x] += buf[x];
        }
      }
    } else {
      int64_t header[2] {proj.first_row, proj.last_row};
      MPI_Send(header, 2, MPI_INT64_T, 0, TAG_RMA_PROJECTION, mpi::intracomm);
      const int64_t n = proj.realization.size();
      for (int64_t off = 0; off < n; off += MAX_MPI_CHUNK_DOUBLES) {
        const int64_t count = std::min(n - off, MAX_MPI_CHUNK_DOUBLES);
        MPI_Send(proj.realization.data() + off, static_cast<int>(count),
          MPI_DOUBLE, 0, TAG_RMA_PROJECTION, mpi::intracomm);
      }
      std::fill(proj.realization.begin(), proj.realization.end(), 0.0);
    }
  }
  if (mpi::master)
    fold_projection_moments();
}
#endif

//==============================================================================
// Tally member functions: the 'none' moments mode
//==============================================================================

void Tally::collect_none()
{
  // A 'none' tally keeps no moments during the run: its accumulator is never
  // zeroed and holds the run total. This routine combines that total and
  // materialises SUM = total x norm into a one-plane moments_ ([F, S, 1] on
  // the master, or the owned rows under rma), so the generic statepoint writer
  // (including the rma gather) and the end-of-run readout serve it exactly as
  // they serve a 'mean' tally. One norm for the whole run is valid because
  // tally_normalization() has no writer during a Monte Carlo run (random ray,
  // whose scores are scaled per batch, rejects 'none'). Materialising first and
  // writing generically costs one transient plane on the master but reuses all
  // writer code, rather than scaling chunk by chunk inside the writer.
  //
  // Collective on every rank (each accumulator mode below communicates), and
  // called for the same tallies in the same order on every rank: the moments
  // mode is a rank-identical per-tally property and none_materialised_ is set
  // and cleared identically on every rank.
  if (none_materialised_)
    return;
  const double norm = tally_normalization();

#ifdef OPENMC_MPI
  if (exatally_.accumulator == AccumulatorMode::RMA) {
    // Complete every rank's accumulates (drain, flush_all, barrier, win_sync).
    // The window is only read here, never zeroed or written by the CPU, so it
    // keeps accumulating the run total (the invariant that the CPU writes the
    // window only inside the collective per-batch fold).
    rma_publish();
    moments_ = tensor::Tensor<double>({static_cast<size_t>(rma_n_rows_),
      static_cast<size_t>(n_score_bins_), size_t {1}});
#pragma omp parallel for
    for (int64_t i = 0; i < rma_n_rows_; ++i) {
      for (int j = 0; j < n_score_bins_; ++j) {
        const int64_t k = i * n_score_bins_ + j;
        moments_(i, j, TallyMoment::SUM) =
          (rma_win_base_[k] + accum_[k]) * norm;
      }
    }
    // Every rank must finish this snapshot before any rank issues the next
    // batch's accumulates, or a slow rank would read next-batch mass into this
    // statepoint. rma_resume() is exactly that barrier (win_sync + intracomm
    // barrier + win_sync).
    rma_resume();
    none_materialised_ = true;
    return;
  }

  if (exatally_.accumulator == AccumulatorMode::SHARED) {
    // Node sync, then the node leaders reduce their planes onto the master's
    // node plane, which then holds the run total so far.
    shared_publish();
    if (mpi::master) {
      materialise_none_sum(norm);
    }
    // Non-master nodes zero their (now combined) node plane, so a later
    // collect adds only new scores (a running total, never double-counted);
    // the master node keeps its plane. Then every node rank publishes the plane
    // and barriers before scoring resumes, so no rank scores into the master's
    // plane while the master reads it above.
    shared_batch_end();
    none_materialised_ = true;
    return;
  }

  if (mpi::n_procs > 1) {
    // Replicated: reduce the run totals in place onto the master. Non-master
    // ranks then restart from zero, so a later collect adds only their new
    // scores (the same running-total pattern as above).
    reduce_accum_to_master(accum_, accum_size_, mpi::intracomm);
    if (!mpi::master) {
      std::fill(accum_, accum_ + accum_size_, 0.0);
    }
  }
#endif

  if (mpi::master) {
    materialise_none_sum(norm);
  }
  none_materialised_ = true;
}

void Tally::materialise_none_sum(double norm)
{
  // Master side of collect_none() for the full-plane accumulator modes
  // (replicated, shared): accum_ holds the global run total.
  moments_ = tensor::Tensor<double>({static_cast<size_t>(n_filter_bins_),
    static_cast<size_t>(n_score_bins_), size_t {1}});
#pragma omp parallel for
  for (int64_t i = 0; i < n_filter_bins_; ++i) {
    for (int j = 0; j < n_score_bins_; ++j) {
      moments_(i, j, TallyMoment::SUM) = accum_[i * n_score_bins_ + j] * norm;
    }
  }
}

void Tally::release_none_moments()
{
  moments_ = tensor::Tensor<double>();
  none_materialised_ = false;
}

//==============================================================================
// Trigger support
//==============================================================================

bool has_rma_triggers()
{
#ifdef OPENMC_MPI
  for (const auto& t : model::tallies) {
    if (t->accumulator() != AccumulatorMode::RMA)
      continue;
    for (const auto& trigger : t->triggers_) {
      if (trigger.metric != TriggerMetric::not_active)
        return true;
    }
  }
#endif
  return false;
}

#ifdef OPENMC_MPI
// For rma tallies each rank has only walked its owned rows, so the most
// limiting trigger lives on whichever rank owns it. Reduce {ratio, rank} with
// MAXLOC to find the global maximum and the (lowest) rank holding it; that
// rank ships its tally id / score to the master, which alone forms the
// diagnostic message and the satisfy/continue decision from the global ratio.
// The reduction runs iff has_rma_triggers() -- exactly when every rank calls
// check_tally_triggers() (see finalize_batch) -- so it is never a
// half-collective.
void rma_reduce_trigger(double& ratio, int& tally_id, int& score)
{
  struct {
    double val;
    int rank;
  } in {ratio, mpi::rank}, out {};
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, mpi::intracomm);
  ratio = out.val;
  if (out.rank != 0) {
    int buf[2] {tally_id, score};
    if (mpi::rank == out.rank) {
      MPI_Send(buf, 2, MPI_INT, 0, TAG_RMA_TRIGGER, mpi::intracomm);
    } else if (mpi::master) {
      MPI_Recv(buf, 2, MPI_INT, out.rank, TAG_RMA_TRIGGER, mpi::intracomm,
        MPI_STATUS_IGNORE);
      tally_id = buf[0];
      score = buf[1];
    }
  }
}
#endif

} // namespace openmc
