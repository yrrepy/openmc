//! \file exatally_profile.cpp
//! \brief Per-rank profiling of the ExaTallies accumulator modes (see
//! exatally_profile.h). Observation only: counters, timers and memory reads.

#include "openmc/tallies/exatally_profile.h"

#include "openmc/constants.h" // for VERSION_COMMIT_HASH
#include "openmc/error.h"
#include "openmc/file_utils.h"
#include "openmc/message_passing.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/tallies/exatally.h"
#include "openmc/tallies/tally.h"

#include <fmt/core.h>

#include <algorithm> // for max
#include <cmath>     // for isfinite
#include <cstdlib>   // for getenv, atoi, atol
#include <fstream>
#include <string>

#ifdef __linux__
#include <unistd.h> // for gethostname
#endif

#ifdef OPENMC_MPI
#include <mpi.h>
#endif

namespace openmc {
namespace profile {

//==============================================================================
// Global variables
//==============================================================================

bool enabled {false};
int poke_interval {0};
vector<ThreadCounters> thread_counters;
Timer phase_timers[static_cast<int>(Phase::N)];

namespace {

constexpr int N_PHASES = static_cast<int>(Phase::N);

constexpr const char* PHASE_NAMES[N_PHASES] = {"TRANSPORT", "RMA_DRAIN",
  "RMA_FLUSH_ALL", "RMA_BARRIER", "RMA_FOLD", "SHARED_NODE_SYNC",
  "SHARED_INTERNODE_REDUCE", "SHARED_ZERO", "REPLICATED_REDUCE", "FOLD",
  "TRIGGER_WALK", "TRIGGER_REDUCE", "STATEPOINT", "BATCH"};

std::string out_dir;             //!< OPENMC_PROFILE directory
bool sim_open {false};           //!< a simulation's first batch has begun
bool batch1_done {false};        //!< mem_after_batch1 recorded
int poke_active {0};             //!< poke interval in force, 0 = none
std::string static_json;         //!< "static" object
std::string tallies_json;        //!< "tallies" array
std::string mem_init_json;       //!< memory after simulation init
std::string mem_batch1_json;     //!< memory after the first batch
vector<std::string> batch_lines; //!< one JSON object per batch

//! A double as a JSON number ({:.17g}), or null when not finite.
std::string jnum(double x)
{
  return std::isfinite(x) ? fmt::format("{:.17g}", x) : std::string("null");
}

//! A string as a JSON string literal.
std::string jstr(const std::string& s)
{
  std::string out = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += static_cast<char>(c);
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\t') {
      out += "\\t";
    } else if (c < 0x20) {
      out += fmt::format("\\u{:04x}", c);
    } else {
      out += static_cast<char>(c);
    }
  }
  return out + "\"";
}

//! VmHWM and VmRSS of this process in kB (-1 when unavailable).
std::string mem_json()
{
  long hwm = -1;
  long rss = -1;
#ifdef __linux__
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmHWM:", 0) == 0)
      hwm = std::atol(line.c_str() + 6);
    else if (line.rfind("VmRSS:", 0) == 0)
      rss = std::atol(line.c_str() + 6);
  }
#endif
  return fmt::format("{{\"VmHWM\": {}, \"VmRSS\": {}}}", hwm, rss);
}

const char* accumulator_name(AccumulatorMode mode)
{
  switch (mode) {
  case AccumulatorMode::SHARED:
    return "shared";
  case AccumulatorMode::RMA:
    return "rma";
  default:
    return "replicated";
  }
}

//! The "seconds" object of every phase timer; the timers are reset.
std::string take_seconds()
{
  std::string out = "{";
  for (int i = 0; i < N_PHASES; ++i) {
    out += fmt::format("{}\"{}\": {}", i ? ", " : "", PHASE_NAMES[i],
      jnum(phase_timers[i].elapsed()));
    phase_timers[i].reset();
  }
  return out + "}";
}

//! The rma counters (summed over the rma tallies) and the per-tally "rma"
//! array; every tally's per-batch record is cleared.
std::string take_rma_fields()
{
  int64_t scores_remote = 0;
  int64_t scores_local = 0;
  int64_t rows_issued = 0;
  int64_t accumulates = 0;
  int64_t merged = 0;
  int64_t flush_local = 0;
  std::string rma = "[";
#ifdef OPENMC_MPI
  bool first = true;
  for (auto& t : model::tallies) {
    if (!t->rma_distributed())
      continue;
    auto& r = t->rma_profile();
    scores_remote += r.scores_remote;
    scores_local += r.scores_local;
    rows_issued += r.rows_issued;
    accumulates += r.accumulates;
    merged += r.merged;
    flush_local += r.flush_local;
    std::string rows_to;
    std::string mass_to;
    for (size_t i = 0; i < r.rows_to.size(); ++i) {
      rows_to += fmt::format("{}{}", i ? ", " : "", r.rows_to[i]);
      mass_to += fmt::format("{}{}", i ? ", " : "", jnum(r.mass_to[i]));
    }
    rma += fmt::format("{}{{\"tally\": {}, \"rows_to\": [{}], \"mass_to\": "
                       "[{}], \"recv_sum\": {}, \"priv_sum\": {}}}",
      first ? "" : ", ", t->id(), rows_to, mass_to,
      r.sums_valid ? jnum(r.recv_sum) : "null",
      r.sums_valid ? jnum(r.priv_sum) : "null");
    first = false;
    r.scores_remote = 0;
    r.scores_local = 0;
    r.rows_issued = 0;
    r.accumulates = 0;
    r.merged = 0;
    r.flush_local = 0;
    std::fill(r.rows_to.begin(), r.rows_to.end(), 0);
    std::fill(r.mass_to.begin(), r.mass_to.end(), 0.0);
    r.sums_valid = false;
  }
#endif
  rma += "]";
  return fmt::format(
    "\"scores_remote\": {}, \"scores_local\": {}, \"rows_issued\": {}, "
    "\"accumulates\": {}, \"merged\": {}, \"flush_local\": {}, \"rma\": {}",
    scores_remote, scores_local, rows_issued, accumulates, merged, flush_local,
    rma);
}

//! Record the static header and the tally table at a simulation's start.
void record_static()
{
  std::string host = "unknown";
#ifdef __linux__
  char buf[256];
  if (gethostname(buf, sizeof(buf)) == 0) {
    buf[sizeof(buf) - 1] = '\0';
    host = buf;
  }
#endif
  int node_rank = 0;
  int node_size = 1;
  int n_nodes = 1;
  std::string level = "none";
  std::string library = "none";
#ifdef OPENMC_MPI
  node_rank = mpi::node_rank;
  node_size = mpi::node_size;
  n_nodes = mpi::n_nodes;
  int provided;
  MPI_Query_thread(&provided);
  level = provided == MPI_THREAD_MULTIPLE     ? "multiple"
          : provided == MPI_THREAD_SERIALIZED ? "serialized"
          : provided == MPI_THREAD_FUNNELED   ? "funneled"
                                              : "single";
  char lib[MPI_MAX_LIBRARY_VERSION_STRING];
  int len = 0;
  MPI_Get_library_version(lib, &len);
  library.assign(lib, len);
#endif
  const char* k_env = std::getenv("OPENMC_RMA_STAGING_ROWS");
  static_json = fmt::format(
    "{{\"host\": {}, \"rank\": {}, \"n_procs\": {}, \"node_rank\": {}, "
    "\"node_size\": {}, \"n_nodes\": {}, \"threads\": {}, "
    "\"mpi_thread_level\": {}, \"mpi_library\": {}, \"commit\": {}, "
    "\"n_particles\": {}, \"n_batches\": {}, \"n_inactive\": {}, "
    "\"gen_per_batch\": {}, \"work_per_rank\": {}, \"poke_interval\": {}, "
    "\"poke_active\": {}, \"staging_rows_env\": {}}}",
    jstr(host), mpi::rank, mpi::n_procs, node_rank, node_size, n_nodes,
    num_threads(), jstr(level), jstr(library), jstr(VERSION_COMMIT_HASH),
    settings::n_particles, settings::n_batches, settings::n_inactive,
    settings::gen_per_batch, simulation::work_per_rank, poke_interval,
    poke_active, k_env ? jstr(k_env) : "null");

  tallies_json = "[";
  bool first = true;
  for (const auto& t : model::tallies) {
    std::string rma;
#ifdef OPENMC_MPI
    if (t->rma_distributed()) {
      const double mib = 1024.0 * 1024.0;
      rma = fmt::format(", \"owned_rows\": {}, \"K\": {}, \"staging_mib\": {}, "
                        "\"window_mib\": {}",
        t->rma_n_rows(), t->rma_k(), jnum(t->rma_staging_bytes() / mib),
        jnum(static_cast<double>(t->rma_n_rows()) * t->n_score_bins() *
             sizeof(double) / mib));
    }
#endif
    tallies_json += fmt::format(
      "{}{{\"id\": {}, \"accumulator\": {}, \"moments\": {}, "
      "\"n_filter_bins\": {}, \"n_score_bins\": {}, \"rma_distributed\": "
      "{}{}}}",
      first ? "" : ", ", t->id(), jstr(accumulator_name(t->accumulator())),
      jstr(moments_mode_name(t->moments_mode())), t->n_filter_bins(),
      t->n_score_bins(), t->rma_distributed() ? "true" : "false", rma);
    first = false;
  }
  tallies_json += "]";
}

} // namespace

//==============================================================================
// Functions
//==============================================================================

void init()
{
  const char* dir = std::getenv("OPENMC_PROFILE");
  enabled = (dir != nullptr && dir[0] != '\0');
  if (enabled) {
    out_dir = dir;
    if (mpi::master && !dir_exists(out_dir)) {
      fatal_error(
        fmt::format("OPENMC_PROFILE directory '{}' does not exist.", out_dir));
    }
  }
  poke_interval = 0;
  if (const char* env = std::getenv("OPENMC_RMA_PROGRESS_POKE"))
    poke_interval = std::max(0, std::atoi(env));
  sim_open = false;
}

void batch_begin()
{
  if (!enabled && poke_interval == 0)
    return;

  if (!sim_open) {
    // First batch of this simulation: tallies are allocated and the run sizes
    // are known.
    sim_open = true;
    thread_counters.assign(num_threads(), ThreadCounters {});
    poke_active = 0;
#ifdef OPENMC_MPI
    if (poke_interval > 0 && mpi::n_procs > 1) {
      for (const auto& t : model::tallies) {
        if (t->rma_distributed())
          poke_active = poke_interval;
      }
    }
#endif
    if (enabled) {
      for (auto& timer : phase_timers)
        timer.reset();
      record_static();
      mem_init_json = mem_json();
      mem_batch1_json = "null";
      batch1_done = false;
      batch_lines.clear();
      write_message(8, "profiling: writing {}/profile_rank<R>.json{}", out_dir,
        poke_active > 0
          ? fmt::format(", rma progress poke every {} particles", poke_active)
          : std::string());
    }
  }

  if (enabled)
    phase_timers[static_cast<int>(Phase::BATCH)].start();
}

void batch_end()
{
  if (!enabled)
    return;
  phase_timers[static_cast<int>(Phase::BATCH)].stop();

  int64_t scores = 0;
  int64_t pokes = 0;
  for (auto& c : thread_counters) {
    scores += c.scores;
    pokes += c.pokes;
    c = ThreadCounters {};
  }
  const std::string seconds = take_seconds();
  const std::string rma = take_rma_fields();
  batch_lines.push_back(fmt::format(
    "{{\"batch\": {}, \"active\": {}, \"seconds\": {}, \"particles\": {}, "
    "\"scores\": {}, {}, \"pokes\": {}}}",
    simulation::current_batch,
    simulation::current_batch > settings::n_inactive ? "true" : "false",
    seconds, simulation::work_per_rank * settings::gen_per_batch, scores, rma,
    pokes));

  if (!batch1_done) {
    mem_batch1_json = mem_json();
    batch1_done = true;
  }
}

void finalize()
{
  if (!enabled) {
    sim_open = false;
    return;
  }
  if (!sim_open)
    return;
  sim_open = false;

  // Phases after the last batch (the end-of-run collect of a 'none' tally).
  int64_t pokes = 0;
  for (auto& c : thread_counters) {
    pokes += c.pokes;
    c = ThreadCounters {};
  }
  const std::string seconds = take_seconds();
  const std::string rma = take_rma_fields();
  const std::string tail =
    fmt::format("{{\"seconds\": {}, {}, \"pokes\": {}}}", seconds, rma, pokes);

  const std::string path =
    fmt::format("{}/profile_rank{}.json", out_dir, mpi::rank);
  std::ofstream f(path);
  if (!f) {
    warning(fmt::format("Cannot write the profile file '{}'.", path));
    return;
  }
  f << "{\"schema\": \"exatallies-profile-1\",\n";
  f << "\"static\": " << static_json << ",\n";
  f << "\"tallies\": " << tallies_json << ",\n";
  f << "\"mem_after_init\": " << mem_init_json << ",\n";
  f << "\"mem_after_batch1\": " << mem_batch1_json << ",\n";
  f << "\"mem_final\": " << mem_json() << ",\n";
  f << "\"batches\": [\n";
  for (size_t i = 0; i < batch_lines.size(); ++i)
    f << batch_lines[i] << (i + 1 < batch_lines.size() ? ",\n" : "\n");
  f << "],\n";
  f << "\"tail\": " << tail << "\n}\n";
  batch_lines.clear();
}

void add_pokes(int64_t n)
{
  thread_counters[thread_num()].pokes += n;
}

int poke_every()
{
  return poke_active;
}

void rma_progress_poke()
{
#ifdef OPENMC_MPI
  int flag;
#pragma omp critical(openmc_rma)
  {
    MPI_Iprobe(
      MPI_ANY_SOURCE, MPI_ANY_TAG, mpi::intracomm, &flag, MPI_STATUS_IGNORE);
  }
#endif
  add_pokes(1);
}

} // namespace profile
} // namespace openmc
