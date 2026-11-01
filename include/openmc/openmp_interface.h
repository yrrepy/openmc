#ifndef OPENMC_OPENMP_INTERFACE_H
#define OPENMC_OPENMP_INTERFACE_H

#include <algorithm> // for fill
#include <atomic>
#include <cstdint> // for int64_t

#ifdef _OPENMP
#include <omp.h>
#endif

namespace openmc {

#ifdef OPENMC_MPI
//! Lock-free 8-byte atomics are a prerequisite for the shared accumulator
//! mode, where several processes atomically update one shared-memory plane.
static_assert(std::atomic<double>::is_always_lock_free,
  "the shared accumulator mode requires lock-free 8-byte atomics");
#endif

//! \brief Atomically add a contribution to a tally accumulator element.
//!
//! On the tally hot path this is one OpenMP atomic RMW. For the shared
//! accumulator mode accum_ points into an MPI-3 shared-memory window mapped by
//! several processes: #pragma omp atomic on an 8-byte double lowers to a
//! hardware atomic RMW (lock-prefixed on x86, LL/SC on ARM) whose atomicity is
//! enforced by the cache-coherence protocol at the physical address. It is
//! therefore address-free -- valid across processes sharing the memory, the
//! same guarantee lock-free std::atomic relies on. The static_assert above is a
//! necessary (not sufficient) platform check; the cross-rank hammer test is the
//! empirical proof. Without _OPENMP the pragma vanishes and this is not atomic,
//! so the shared accumulator mode is rejected at input validation.
inline void atomic_score_add(double* ptr, double val)
{
#pragma omp atomic
  *ptr += val;
}

//! \brief Fill n doubles starting at p with v, splitting the range evenly over
//! the OpenMP threads.
//!
//! schedule(static) gives thread t the t-th contiguous slice, the same
//! partition the tally fold uses. A plane zeroed here for the first time is
//! therefore first-touched slice by slice on the NUMA domains of the threads
//! that later fold it, and every later fill writes the pages its thread owns.
//! Ranges below 1 MiB are filled serially, so small tallies do not pay for a
//! parallel region. Must be called outside a parallel region to use the
//! threads.
inline void parallel_fill(double* p, int64_t n, double v)
{
  if (n < (int64_t {1} << 17)) {
    std::fill(p, p + n, v);
    return;
  }
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i)
    p[i] = v;
}

//==============================================================================
//! Accessor functions related to number of threads and thread number
//==============================================================================
inline int num_threads()
{
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

inline int thread_num()
{
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

//==============================================================================
//! An object used to prevent concurrent access to a piece of data.
//
//! This type meets the C++ "Lockable" requirements.
//==============================================================================

class OpenMPMutex {
public:
  void init()
  {
#ifdef _OPENMP
    omp_init_lock(&mutex_);
#endif
  }

  OpenMPMutex() { init(); }

  ~OpenMPMutex()
  {
#ifdef _OPENMP
    omp_destroy_lock(&mutex_);
#endif
  }

  // omp_lock_t objects cannot be deep copied, they can only be shallow
  // copied. Thus, while shallow copying of an omp_lock_t object is
  // completely valid (provided no race conditions exist), true copying
  // of an OpenMPMutex object is not valid due to the action of the
  // destructor. However, since locks are fungible, we can simply replace
  // copying operations with default construction. This allows storage of
  // OpenMPMutex objects within containers that may need to move/copy them
  // (e.g., std::vector). It is left to the caller to understand that
  // copying of OpenMPMutex does not produce two handles to the same mutex,
  // rather, it produces two different mutexes.

  // Copy constructor
  OpenMPMutex(const OpenMPMutex& other) { init(); }

  // Copy assignment operator
  OpenMPMutex& operator=(const OpenMPMutex& other) { return *this; }

  //! Lock the mutex.
  //
  //! This function blocks execution until the lock succeeds.
  void lock()
  {
#ifdef _OPENMP
    omp_set_lock(&mutex_);
#endif
  }

  //! Try to lock the mutex and indicate success.
  //
  //! This function does not block.  It returns immediately and gives false if
  //! the lock is unavailable.
  bool try_lock() noexcept
  {
#ifdef _OPENMP
    return omp_test_lock(&mutex_);
#else
    return true;
#endif
  }

  //! Unlock the mutex.
  void unlock() noexcept
  {
#ifdef _OPENMP
    omp_unset_lock(&mutex_);
#endif
  }

private:
#ifdef _OPENMP
  omp_lock_t mutex_;
#endif
};

} // namespace openmc
#endif // OPENMC_OPENMP_INTERFACE_H
