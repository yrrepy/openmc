#include "openmc/state_point.h"

#include <algorithm>
#include <cassert>
#include <cstdint> // for int64_t
#include <string>

#include "openmc/tensor.h"
#include <fmt/core.h>

#include "openmc/bank.h"
#include "openmc/bank_io.h"
#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/eigenvalue.h"
#include "openmc/error.h"
#include "openmc/file_utils.h"
#include "openmc/hdf5_interface.h"
#include "openmc/mcpl_interface.h"
#include "openmc/mesh.h"
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/nuclide.h"
#include "openmc/output.h"
#include "openmc/particle_type.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/tallies/derivative.h"
#include "openmc/tallies/filter.h"
#include "openmc/tallies/filter_mesh.h"
#include "openmc/tallies/tally.h"
#include "openmc/timer.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
// rma tally statepoint gather / restart scatter
//
// Under the rma storage mode a tally's moments are block-distributed by
// filter-bin row across all ranks, so no rank holds the whole array. The
// statepoint therefore gathers rows to the master on write and scatters them
// back to their owners on restart, both in bounded chunks so an individual MPI
// call never exceeds the int element-count limit. The reduce-path master-serial
// file format is unchanged (a single [n_filter, n_score, n_moments] "results"
// dataset), so Python readers are untouched and there is no version bump.
//==============================================================================

namespace {

#ifdef OPENMC_MPI
// Doubles per filter-bin row of a tally results dataset.
inline int64_t results_row_len(const Tally& tally)
{
  return static_cast<int64_t>(tally.n_score_bins()) * tally.n_moments();
}

// Filter-bin rows moved per MPI/HDF5 chunk: the shared 1 GiB cap applied at
// row granularity (at least one row).
inline int64_t results_rows_per_chunk(int64_t row_len)
{
  return std::max<int64_t>(
    1, MAX_MPI_CHUNK_DOUBLES / std::max<int64_t>(1, row_len));
}

// Write a contiguous block of n_rows filter-bin rows, starting at global row
// first_row, into a tally "results" dataset. The reduce-path statepoint is
// written by the master alone on a serial file handle, so plain writes suffice.
void write_results_hyperslab(hid_t dset, int64_t first_row, int64_t n_rows,
  hsize_t n_score, hsize_t n_moments, const double* buffer)
{
  hsize_t start[3] {static_cast<hsize_t>(first_row), 0, 0};
  hsize_t count[3] {static_cast<hsize_t>(n_rows), n_score, n_moments};
  hid_t filespace = H5Dget_space(dset);
  H5Sselect_hyperslab(
    filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);
  hid_t memspace = H5Screate_simple(3, count, nullptr);
  H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, buffer);
  H5Sclose(memspace);
  H5Sclose(filespace);
}

// Read this rank's owned block of n_rows filter-bin rows (starting at global
// row first_row) from a tally "results" dataset straight into buffer. Every
// rank reads a disjoint hyperslab, so under Parallel HDF5 the reads run
// collectively -- the read_source_bank idiom (state_point.cpp) -- rather than
// master-scatter: a lone rank's independent read on a collectively-opened MPIO
// file does not complete on all fabrics. With a serial build each rank reads
// independently from its own file handle. A rank owning zero rows issues an
// empty collective read.
void read_owned_rows_hyperslab(hid_t dset, int64_t first_row, int64_t n_rows,
  hsize_t n_score, hsize_t n_moments, double* buffer)
{
  hsize_t start[3] {static_cast<hsize_t>(first_row), 0, 0};
  hsize_t count[3] {static_cast<hsize_t>(n_rows), n_score, n_moments};
  hid_t filespace = H5Dget_space(dset);
  if (n_rows > 0) {
    H5Sselect_hyperslab(
      filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);
  } else {
    H5Sselect_none(filespace);
  }
  hid_t memspace = H5Screate_simple(3, count, nullptr);
  // A rank that owns no rows still participates (empty selection) but has an
  // empty moments tensor, whose data() is null; hand HDF5 a dummy buffer since
  // some versions reject a null pointer even when nothing will be transferred.
  double dummy;
  if (n_rows == 0) {
    H5Sselect_none(memspace);
    buffer = &dummy;
  }
#ifdef PHDF5
  if (using_mpio_device(dset)) {
    hid_t plist = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
    H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, filespace, plist, buffer);
    H5Pclose(plist);
  } else
#endif
    H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, buffer);
  H5Sclose(memspace);
  H5Sclose(filespace);
}

// Master side of the statepoint write: create the full results dataset, write
// the master's own owned rows, then receive each other rank's owned rows in
// chunks and write them into their hyperslab.
void gather_rma_tally_results(hid_t tally_group, const Tally& tally)
{
  const hsize_t n_filter = static_cast<hsize_t>(tally.n_filter_bins());
  const hsize_t n_score = static_cast<hsize_t>(tally.n_score_bins());
  const hsize_t n_moments = static_cast<hsize_t>(tally.n_moments());
  hsize_t dims[3] {n_filter, n_score, n_moments};
  hid_t dspace = H5Screate_simple(3, dims, nullptr);
  hid_t dset = H5Dcreate(tally_group, "results", H5T_NATIVE_DOUBLE, dspace,
    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  const int64_t row_len = results_row_len(tally);
  const int64_t rows_per_chunk = results_rows_per_chunk(row_len);

  // One receive buffer reused for every chunk (chunks share a common size
  // except tails), instead of a fresh up-to-1-GiB allocation per chunk.
  vector<double> buf;
  for (int r = 0; r < mpi::n_procs; ++r) {
    const int64_t first = tally.rma_first_row(r);
    const int64_t nrows = tally.rma_rows_owned(r);
    for (int64_t off = 0; off < nrows; off += rows_per_chunk) {
      const int64_t crows = std::min(rows_per_chunk, nrows - off);
      const int64_t count = crows * row_len;
      if (r == 0) {
        // Master owns rows [0, nrows); its moments are locally indexed from 0.
        write_results_hyperslab(dset, first + off, crows, n_score, n_moments,
          tally.moments().data() + off * row_len);
      } else {
        buf.resize(count);
        MPI_Recv(buf.data(), static_cast<int>(count), MPI_DOUBLE, r,
          TAG_RMA_GATHER, mpi::intracomm, MPI_STATUS_IGNORE);
        write_results_hyperslab(
          dset, first + off, crows, n_score, n_moments, buf.data());
      }
    }
  }
  H5Dclose(dset);
  H5Sclose(dspace);
}

// Non-master side of the statepoint write: ship this rank's owned moment rows
// to the master in the same bounded chunks the master receives.
void send_rma_tally_results(const Tally& tally)
{
  const int64_t row_len = results_row_len(tally);
  const int64_t rows_per_chunk = results_rows_per_chunk(row_len);
  const int64_t nrows = tally.rma_n_rows();
  const double* data = tally.moments().data();
  for (int64_t off = 0; off < nrows; off += rows_per_chunk) {
    const int64_t crows = std::min(rows_per_chunk, nrows - off);
    const int64_t count = crows * row_len;
    MPI_Send(data + off * row_len, static_cast<int>(count), MPI_DOUBLE, 0,
      TAG_RMA_GATHER, mpi::intracomm);
  }
}

// Restart read for the distributed rma moments (all ranks). Every rank has the
// statepoint open, so the group/dataset opens are collective under Parallel
// HDF5; each rank then reads only the moment rows it owns straight into its
// (owned-rows) moments. No rank allocates the full moments, so the PR-2.B
// temp-alloc idiom is avoided; and because every rank reads a disjoint owned
// hyperslab the reads compose into a single collective transfer rather than a
// master-scatter (a lone rank's independent read on a collectively-opened MPIO
// file does not complete on all fabrics).
void read_distributed_rma_tally_results(hid_t file_id)
{
  // Nothing to do without a genuinely distributed rma tally.
  bool any = false;
  for (const auto& t : model::tallies) {
    if (t->rma_distributed()) {
      any = true;
      break;
    }
  }
  if (!any)
    return;

  // Results are only present if the writing run had active, reduced tallies.
  // Read on every rank (each has its own file handle in a serial build).
  bool present = false;
  if (attribute_exists(file_id, "tallies_present"))
    read_attribute(file_id, "tallies_present", present);
  if (!present)
    return;

  hid_t tallies_group = open_group(file_id, "tallies");
  for (auto& tally : model::tallies) {
    if (!tally->rma_distributed())
      continue;

    std::string name = "tally " + std::to_string(tally->id_);
    hid_t tally_group = open_group(tallies_group, name.c_str());

    int internal = 0;
    if (attribute_exists(tally_group, "internal"))
      read_attribute(tally_group, "internal", internal);
    if (internal) {
      tally->writable_ = false;
      close_group(tally_group);
      continue;
    }

    // Every rank needs n_realizations for its owned-row fold. Read it on every
    // rank: this is a collective dataset read under Parallel HDF5 (each rank
    // otherwise has its own serial file handle), so it must not be master-only.
    read_dataset(tally_group, "n_realizations", tally->n_realizations_);

    hid_t dset = open_dataset(tally_group, "results");
    const hsize_t n_filter = static_cast<hsize_t>(tally->n_filter_bins());
    const hsize_t n_score = static_cast<hsize_t>(tally->n_score_bins());
    const hsize_t n_moments = static_cast<hsize_t>(tally->n_moments());
    // Every rank reads the same on-disk extent and holds the same model shape,
    // so this check is reached identically on all ranks -- a mismatch fatals on
    // every rank with no collective divergence. Validate the full extent before
    // selecting each rank's owned-row hyperslab; the read otherwise assumes the
    // statepoint layout matches the current model.
    auto shape = object_shape(dset);
    if (shape.size() != 3 || shape[0] != n_filter || shape[1] != n_score ||
        shape[2] != n_moments) {
      std::string found;
      for (auto d : shape)
        found += (found.empty() ? "" : ", ") + std::to_string(d);
      fatal_error(fmt::format(
        "Tally {} results shape mismatch on rma restart: the current model "
        "expects [{}, {}, {}] but the statepoint \"results\" dataset is [{}]. "
        "The restart model's tally layout must match the statepoint.",
        tally->id_, n_filter, n_score, n_moments, found));
    }
    read_owned_rows_hyperslab(dset, tally->rma_first_row(mpi::rank),
      tally->rma_n_rows(), n_score, n_moments, tally->moments().data());

    close_dataset(dset);
    close_group(tally_group);
  }
  close_group(tallies_group);
}
#endif // OPENMC_MPI

} // namespace

extern "C" int openmc_statepoint_write(const char* filename, bool* write_source)
{
  simulation::time_statepoint.start();

  // If a nullptr is passed in, we assume that the user
  // wants a default name for this, of the form like output/statepoint.20.h5
  std::string filename_;
  if (filename) {
    filename_ = filename;
  } else {
    // Determine width for zero padding
    int w = std::to_string(settings::n_max_batches).size();

    // Set filename for state point
    filename_ = fmt::format("{0}statepoint.{1:0{2}}.h5", settings::path_output,
      simulation::current_batch, w);
  }

  // If a file name was specified, ensure it has .h5 file extension
  const auto extension = get_file_extension(filename_);
  if (extension != "h5") {
    warning("openmc_statepoint_write was passed a file extension differing "
            "from .h5, but an hdf5 file will be written.");
  }

  // Determine whether or not to write the source bank
  bool write_source_ = write_source ? *write_source : true;

  // Write message
  write_message("Creating state point " + filename_ + "...", 5);

  hid_t file_id;
  if (mpi::master) {
    // Create statepoint file
    file_id = file_open(filename_, 'w');

    // Write file type
    write_attribute(file_id, "filetype", "statepoint");

    // Write revision number for state point file
    write_attribute(file_id, "version", VERSION_STATEPOINT);

    // Write OpenMC version
    write_attribute(file_id, "openmc_version", VERSION);
#ifdef GIT_SHA1
    write_attribute(file_id, "git_sha1", GIT_SHA1);
#endif

    // Write current date and time
    write_attribute(file_id, "date_and_time", time_stamp());

    // Write path to input
    write_attribute(file_id, "path", settings::path_input);

    // Write out random number seed
    write_dataset(file_id, "seed", openmc_get_seed());

    // Write out random number stride
    write_dataset(file_id, "stride", openmc_get_stride());

    // Write run information
    write_dataset(file_id, "energy_mode",
      settings::run_CE ? "continuous-energy" : "multi-group");
    switch (settings::run_mode) {
    case RunMode::FIXED_SOURCE:
      write_dataset(file_id, "run_mode", "fixed source");
      break;
    case RunMode::EIGENVALUE:
      write_dataset(file_id, "run_mode", "eigenvalue");
      break;
    default:
      break;
    }
    write_attribute(file_id, "photon_transport", settings::photon_transport);
    write_dataset(file_id, "n_particles", settings::n_particles);
    write_dataset(file_id, "n_batches", settings::n_batches);

    // Write out current batch number
    write_dataset(file_id, "current_batch", simulation::current_batch);

    // Indicate whether source bank is stored in statepoint
    write_attribute(file_id, "source_present", write_source_);

    // Write out information for eigenvalue run
    if (settings::run_mode == RunMode::EIGENVALUE)
      write_eigenvalue_hdf5(file_id);

    hid_t tallies_group = create_group(file_id, "tallies");

    // Write meshes
    meshes_to_hdf5(tallies_group);

    // Write information for derivatives
    if (!model::tally_derivs.empty()) {
      hid_t derivs_group = create_group(tallies_group, "derivatives");
      for (const auto& deriv : model::tally_derivs) {
        hid_t deriv_group =
          create_group(derivs_group, "derivative " + std::to_string(deriv.id));
        write_dataset(deriv_group, "material", deriv.diff_material);
        if (deriv.variable == DerivativeVariable::DENSITY) {
          write_dataset(deriv_group, "independent variable", "density");
        } else if (deriv.variable == DerivativeVariable::NUCLIDE_DENSITY) {
          write_dataset(deriv_group, "independent variable", "nuclide_density");
          write_dataset(
            deriv_group, "nuclide", data::nuclides[deriv.diff_nuclide]->name_);
        } else if (deriv.variable == DerivativeVariable::TEMPERATURE) {
          write_dataset(deriv_group, "independent variable", "temperature");
        } else {
          fatal_error("Independent variable for derivative " +
                      std::to_string(deriv.id) +
                      " not defined in state_point.cpp");
        }
        close_group(deriv_group);
      }
      close_group(derivs_group);
    }

    // Write information for filters
    hid_t filters_group = create_group(tallies_group, "filters");
    write_attribute(filters_group, "n_filters", model::tally_filters.size());
    if (!model::tally_filters.empty()) {
      // Write filter IDs
      vector<int32_t> filter_ids;
      filter_ids.reserve(model::tally_filters.size());
      for (const auto& filt : model::tally_filters)
        filter_ids.push_back(filt->id());
      write_attribute(filters_group, "ids", filter_ids);

      // Write info for each filter
      for (const auto& filt : model::tally_filters) {
        hid_t filter_group =
          create_group(filters_group, "filter " + std::to_string(filt->id()));
        filt->to_statepoint(filter_group);
        close_group(filter_group);
      }
    }
    close_group(filters_group);

    // Write information for tallies
    write_attribute(tallies_group, "n_tallies", model::tallies.size());
    if (!model::tallies.empty()) {
      // Write tally IDs
      vector<int32_t> tally_ids;
      tally_ids.reserve(model::tallies.size());
      for (const auto& tally : model::tallies)
        tally_ids.push_back(tally->id_);
      write_attribute(tallies_group, "ids", tally_ids);

      // Write all tally information except results
      for (const auto& tally : model::tallies) {
        hid_t tally_group =
          create_group(tallies_group, "tally " + std::to_string(tally->id_));

        write_dataset(tally_group, "name", tally->name_);

        if (tally->writable_) {
          write_attribute(tally_group, "internal", 0);
        } else {
          write_attribute(tally_group, "internal", 1);
          close_group(tally_group);
          continue;
        }

        if (tally->multiply_density()) {
          write_attribute(tally_group, "multiply_density", 1);
        } else {
          write_attribute(tally_group, "multiply_density", 0);
        }

        if (tally->higher_moments()) {
          write_attribute(tally_group, "higher_moments", 1);
        } else {
          write_attribute(tally_group, "higher_moments", 0);
        }

        if (tally->estimator_ == TallyEstimator::ANALOG) {
          write_dataset(tally_group, "estimator", "analog");
        } else if (tally->estimator_ == TallyEstimator::TRACKLENGTH) {
          write_dataset(tally_group, "estimator", "tracklength");
        } else if (tally->estimator_ == TallyEstimator::COLLISION) {
          write_dataset(tally_group, "estimator", "collision");
        }

        write_dataset(tally_group, "n_realizations", tally->n_realizations_);

        // Write the ID of each filter attached to this tally
        write_dataset(tally_group, "n_filters", tally->filters().size());
        if (!tally->filters().empty()) {
          vector<int32_t> filter_ids;
          filter_ids.reserve(tally->filters().size());
          for (auto i_filt : tally->filters())
            filter_ids.push_back(model::tally_filters[i_filt]->id());
          write_dataset(tally_group, "filters", filter_ids);
        }

        // Write the nuclides this tally scores
        vector<std::string> nuclides;
        for (auto i_nuclide : tally->nuclides_) {
          if (i_nuclide == -1) {
            nuclides.push_back("total");
          } else {
            if (settings::run_CE) {
              nuclides.push_back(data::nuclides[i_nuclide]->name_);
            } else {
              nuclides.push_back(data::mg.nuclides_[i_nuclide].name);
            }
          }
        }
        write_dataset(tally_group, "nuclides", nuclides);

        if (tally->deriv_ != C_NONE)
          write_dataset(
            tally_group, "derivative", model::tally_derivs[tally->deriv_].id);

        // Write the tally score bins
        vector<std::string> scores;
        for (auto sc : tally->scores_)
          scores.push_back(reaction_name(sc));
        write_dataset(tally_group, "n_score_bins", scores.size());
        write_dataset(tally_group, "score_bins", scores);

        close_group(tally_group);
      }
    }

    if (settings::reduce_tallies) {
      // Write global tallies
      write_dataset(file_id, "global_tallies", simulation::global_tallies);

      // Write tallies
      if (model::active_tallies.size() > 0) {
        // Indicate that tallies are on
        write_attribute(file_id, "tallies_present", 1);

        // Write all tally results
        for (const auto& tally : model::tallies) {
          if (!tally->writable_)
            continue;

          // Write results for each bin
          std::string name = "tally " + std::to_string(tally->id_);
          hid_t tally_group = open_group(tallies_group, name.c_str());
#ifdef OPENMC_MPI
          // Under multi-rank rma the master holds only its own owned rows, so
          // gather every rank's rows into the full dataset. The matching sends
          // are issued by the non-master ranks just after this master-only
          // block.
          if (tally->rma_distributed()) {
            gather_rma_tally_results(tally_group, *tally);
            close_group(tally_group);
            continue;
          }
#endif
          auto& moments = tally->moments();
          write_tally_results(tally_group, moments.shape(0), moments.shape(1),
            moments.shape(2), moments.data());
          close_group(tally_group);
        }
      } else {
        // Indicate tallies are off
        write_attribute(file_id, "tallies_present", 0);
      }
    }

    close_group(tallies_group);
  }

#ifdef OPENMC_MPI
  // Non-master ranks ship their owned rma moment rows to the master, which
  // receives and writes them inside its writer loop above. Guard exactly as the
  // master's writer does (reduced tallies, at least one active tally) so the
  // sends and receives pair up over model::tallies in the same order.
  if (!mpi::master && settings::reduce_tallies &&
      model::active_tallies.size() > 0) {
    for (const auto& tally : model::tallies) {
      if (!tally->writable_)
        continue;
      if (tally->rma_distributed())
        send_rma_tally_results(*tally);
    }
  }
#endif

  // Check for the no-tally-reduction method
  if (!settings::reduce_tallies) {
    // If using the no-tally-reduction method, we need to collect tally
    // results before writing them to the state point file.
    write_tally_results_nr(file_id);

  } else if (mpi::master) {
    // Write number of global realizations
    write_dataset(file_id, "n_realizations", simulation::n_realizations);
  }

  if (mpi::master) {
    // Write out the runtime metrics.
    using namespace simulation;
    hid_t runtime_group = create_group(file_id, "runtime");
    write_dataset(
      runtime_group, "total initialization", time_initialize.elapsed());
    write_dataset(
      runtime_group, "reading cross sections", time_read_xs.elapsed());
    write_dataset(runtime_group, "simulation",
      time_inactive.elapsed() + time_active.elapsed());
    write_dataset(runtime_group, "transport", time_transport.elapsed());
    if (settings::run_mode == RunMode::EIGENVALUE) {
      write_dataset(runtime_group, "inactive batches", time_inactive.elapsed());
    }
    write_dataset(runtime_group, "active batches", time_active.elapsed());
    if (settings::run_mode == RunMode::EIGENVALUE) {
      write_dataset(
        runtime_group, "synchronizing fission bank", time_bank.elapsed());
      write_dataset(
        runtime_group, "sampling source sites", time_bank_sample.elapsed());
      write_dataset(
        runtime_group, "SEND-RECV source sites", time_bank_sendrecv.elapsed());
    }
    write_dataset(
      runtime_group, "accumulating tallies", time_tallies.elapsed());
    write_dataset(runtime_group, "total", time_total.elapsed());
    write_dataset(
      runtime_group, "writing statepoints", time_statepoint.elapsed());
    close_group(runtime_group);

    file_close(file_id);
  }

#ifdef PHDF5
  bool parallel = true;
#else
  bool parallel = false;
#endif

  // Write the source bank if desired
  if (write_source_) {
    if (mpi::master || parallel)
      file_id = file_open(filename_, 'a', true);
    write_source_bank(file_id, simulation::source_bank, simulation::work_index);
    if (mpi::master || parallel)
      file_close(file_id);
  }

#if defined(OPENMC_LIBMESH_ENABLED) || defined(OPENMC_DAGMC_ENABLED)
  // write unstructured mesh tally files
  write_unstructured_mesh_results();
#endif

  simulation::time_statepoint.stop();

  return 0;
}

void restart_set_keff()
{
  if (simulation::restart_batch > settings::n_inactive) {
    for (int i = settings::n_inactive; i < simulation::restart_batch; ++i) {
      simulation::k_sum[0] += simulation::k_generation[i];
      simulation::k_sum[1] += std::pow(simulation::k_generation[i], 2);
    }
    int n = settings::gen_per_batch * simulation::n_realizations;
    simulation::keff = simulation::k_sum[0] / n;
  } else {
    simulation::keff = simulation::k_generation.back();
  }
}

void load_state_point()
{
  write_message(
    fmt::format("Loading state point {}...", settings::path_statepoint_c), 5);
  openmc_statepoint_load(settings::path_statepoint.c_str());
}

void statepoint_version_check(hid_t file_id)
{
  // Read revision number for state point file and make sure it matches with
  // current version
  array<int, 2> version_array;
  read_attribute(file_id, "version", version_array);
  if (version_array != VERSION_STATEPOINT) {
    fatal_error(
      "State point version does not match current version in OpenMC.");
  }
}

extern "C" int openmc_statepoint_load(const char* filename)
{
  // Open file for reading
  hid_t file_id = file_open(filename, 'r', true);

  // Read filetype
  std::string word;
  read_attribute(file_id, "filetype", word);
  if (word != "statepoint") {
    fatal_error("OpenMC tried to restart from a non-statepoint file.");
  }

  statepoint_version_check(file_id);

  // Read and overwrite random number seed
  int64_t seed;
  read_dataset(file_id, "seed", seed);
  openmc_set_seed(seed);

  // Read and overwrite random number stride
  uint64_t stride;
  read_dataset(file_id, "stride", stride);
  openmc_set_stride(stride);

  // It is not impossible for a state point to be generated from a CE run but
  // to be loaded in to an MG run (or vice versa), check to prevent that.
  read_dataset(file_id, "energy_mode", word);
  if (word == "multi-group" && settings::run_CE) {
    fatal_error("State point file is from multigroup run but current run is "
                "continous energy.");
  } else if (word == "continuous-energy" && !settings::run_CE) {
    fatal_error("State point file is from continuous-energy run but current "
                "run is multigroup!");
  }

  // Read and overwrite run information except number of batches
  read_dataset(file_id, "run_mode", word);
  if (word == "fixed source") {
    settings::run_mode = RunMode::FIXED_SOURCE;
  } else if (word == "eigenvalue") {
    settings::run_mode = RunMode::EIGENVALUE;
  }
  read_attribute(file_id, "photon_transport", settings::photon_transport);
  read_dataset(file_id, "n_particles", settings::n_particles);
  int temp;
  read_dataset(file_id, "n_batches", temp);

  // Take maximum of statepoint n_batches and input n_batches
  settings::n_batches = std::max(settings::n_batches, temp);

  // Read batch number to restart at
  read_dataset(file_id, "current_batch", simulation::restart_batch);

  if (settings::restart_run &&
      simulation::restart_batch >= settings::n_max_batches) {
    warning(fmt::format(
      "The number of batches specified for simulation ({}) is smaller "
      "than or equal to the number of batches in the restart statepoint file "
      "({})",
      settings::n_max_batches, simulation::restart_batch));
  }

  // Logical flag for source present in statepoint file
  bool source_present;
  read_attribute(file_id, "source_present", source_present);

  // Read information specific to eigenvalue run
  if (settings::run_mode == RunMode::EIGENVALUE) {
    read_dataset(file_id, "n_inactive", temp);
    read_eigenvalue_hdf5(file_id);

    // Take maximum of statepoint n_inactive and input n_inactive
    settings::n_inactive = std::max(settings::n_inactive, temp);

    // Check to make sure source bank is present
    if (settings::path_sourcepoint == settings::path_statepoint &&
        !source_present) {
      fatal_error("Source bank must be contained in statepoint restart file");
    }
  }

  // Read number of realizations for global tallies
  read_dataset(file_id, "n_realizations", simulation::n_realizations);

  // Set k_sum, keff, and current_batch based on whether restart file is part
  // of active cycle or inactive cycle
  if (settings::run_mode == RunMode::EIGENVALUE) {
    restart_set_keff();
  }

  // Set current batch number
  simulation::current_batch = simulation::restart_batch;

  // Read tallies to master. If we are using Parallel HDF5, all processes
  // need to be included in the HDF5 calls.
#ifdef PHDF5
  if (true) {
#else
  if (mpi::master) {
#endif
    // Read global tally data
    read_dataset_lowlevel(file_id, "global_tallies", H5T_NATIVE_DOUBLE, H5S_ALL,
      false, simulation::global_tallies.data());

    // Check if tally results are present
    bool present;
    read_attribute(file_id, "tallies_present", present);

    // Read in sum and sum squared
    if (present) {
      hid_t tallies_group = open_group(file_id, "tallies");

      for (auto& tally : model::tallies) {
        // rma tally moments are block-distributed; each rank reads its owned
        // rows in a separate all-ranks pass
        // (read_distributed_rma_tally_results, called below) so no rank
        // temp-allocates the full moments. Skip them here -- consistently on
        // every rank, keeping the Parallel HDF5 collective group/dataset calls
        // below in lock-step.
        if (tally->rma_distributed()) {
          continue;
        }

        // Read sum, sum_sq, and N for each bin
        std::string name = "tally " + std::to_string(tally->id_);
        hid_t tally_group = open_group(tallies_group, name.c_str());

        int internal = 0;
        if (attribute_exists(tally_group, "internal")) {
          read_attribute(tally_group, "internal", internal);
        }
        if (internal) {
          tally->writable_ = false;
        } else {
          // Under Parallel HDF5 every rank collectively opens, reads, and
          // closes the results dataset, so a rank that dropped its moments
          // (owner-only reduced mode) allocates a receiving buffer for the read
          // and releases it again afterward. Only the master's copy needs to
          // survive the restart; the others stay empty for the resumed run and
          // are refilled by the end-of-run broadcast.
          bool owns_moments = tally->has_moments();
          if (!owns_moments) {
            tally->moments() = tensor::Tensor<double>(
              {static_cast<size_t>(tally->n_filter_bins()),
                static_cast<size_t>(tally->n_score_bins()),
                static_cast<size_t>(tally->n_moments())});
          }
          auto& moments = tally->moments();
          read_tally_results(tally_group, moments.shape(0), moments.shape(1),
            moments.shape(2), moments.data());

          read_dataset(tally_group, "n_realizations", tally->n_realizations_);
          close_group(tally_group);

          if (!owns_moments) {
            tally->moments() = tensor::Tensor<double>();
          }
        }
      }
      close_group(tallies_group);
    }
  }

#ifdef OPENMC_MPI
  // Read the block-distributed moments of any multi-rank rma tally: each rank
  // reads its owned rows (all ranks participate; the statepoint file is still
  // open on every rank here).
  read_distributed_rma_tally_results(file_id);
#endif

  // Read source if in eigenvalue mode
  if (settings::run_mode == RunMode::EIGENVALUE) {

    // Check if source was written out separately
    if (!source_present) {

      // Close statepoint file
      file_close(file_id);

      // Write message
      write_message(
        "Loading source file " + settings::path_sourcepoint + "...", 5);

      // Open source file
      file_id = file_open(settings::path_sourcepoint.c_str(), 'r', true);
    }

    // Read source
    read_source_bank(file_id, simulation::source_bank, true);
  }

  // Close file
  file_close(file_id);

  return 0;
}

hid_t h5banktype(bool memory)
{
  // Create compound type for position
  hid_t postype = H5Tcreate(H5T_COMPOUND, sizeof(struct Position));
  H5Tinsert(postype, "x", HOFFSET(Position, x), H5T_NATIVE_DOUBLE);
  H5Tinsert(postype, "y", HOFFSET(Position, y), H5T_NATIVE_DOUBLE);
  H5Tinsert(postype, "z", HOFFSET(Position, z), H5T_NATIVE_DOUBLE);

  // Create bank datatype
  //
  // If you make changes to the compound datatype here, make sure you update:
  // - openmc/source.py
  // - openmc/statepoint.py
  // - docs/source/io_formats/statepoint.rst
  // - docs/source/io_formats/source.rst
  auto n = sizeof(SourceSite);
  if (!memory)
    n = 2 * sizeof(struct Position) + 3 * sizeof(double) + 3 * sizeof(int);
  hid_t banktype = H5Tcreate(H5T_COMPOUND, n);
  H5Tinsert(banktype, "r", HOFFSET(SourceSite, r), postype);
  H5Tinsert(banktype, "u", HOFFSET(SourceSite, u), postype);
  H5Tinsert(banktype, "E", HOFFSET(SourceSite, E), H5T_NATIVE_DOUBLE);
  H5Tinsert(banktype, "time", HOFFSET(SourceSite, time), H5T_NATIVE_DOUBLE);
  H5Tinsert(banktype, "wgt", HOFFSET(SourceSite, wgt), H5T_NATIVE_DOUBLE);
  H5Tinsert(banktype, "delayed_group", HOFFSET(SourceSite, delayed_group),
    H5T_NATIVE_INT);
  H5Tinsert(banktype, "surf_id", HOFFSET(SourceSite, surf_id), H5T_NATIVE_INT);
  H5Tinsert(
    banktype, "particle", HOFFSET(SourceSite, particle), H5T_NATIVE_INT);

  H5Tclose(postype);
  return banktype;
}

void write_source_point(std::string filename, span<SourceSite> source_bank,
  const vector<int64_t>& bank_index, bool use_mcpl)
{
  std::string ext = use_mcpl ? "mcpl" : "h5";

  int total_surf_particles = source_bank.size();
#ifdef OPENMC_MPI
  int num_particles = source_bank.size();
  MPI_Allreduce(
    &num_particles, &total_surf_particles, 1, MPI_INT, MPI_SUM, mpi::intracomm);
#endif

  write_message("Creating source file {}.{} with {} particles ...", filename,
    ext, total_surf_particles, 5);

  // Dispatch to appropriate function based on file type
  if (use_mcpl) {
    filename.append(".mcpl");
    write_mcpl_source_point(filename.c_str(), source_bank, bank_index);
  } else {
    filename.append(".h5");
    write_h5_source_point(filename.c_str(), source_bank, bank_index);
  }
}

void write_h5_source_point(const char* filename, span<SourceSite> source_bank,
  const vector<int64_t>& bank_index)
{
  // When using parallel HDF5, the file is written to collectively by all
  // processes. With MPI-only, the file is opened and written by the master
  // (note that the call to write_source_bank is by all processes since slave
  // processes need to send source bank data to the master.
#ifdef PHDF5
  bool parallel = true;
#else
  bool parallel = false;
#endif

  if (!filename)
    fatal_error("write_source_point filename needs a nonempty name.");

  std::string filename_(filename);
  const auto extension = get_file_extension(filename_);
  if (extension != "h5") {
    warning("write_source_point was passed a file extension differing "
            "from .h5, but an hdf5 file will be written.");
  }

  hid_t file_id;
  if (mpi::master || parallel) {
    file_id = file_open(filename_.c_str(), 'w', true);
    write_attribute(file_id, "filetype", "source");
    write_attribute(file_id, "version", VERSION_STATEPOINT);
  }

  // Get pointer to source bank and write to file
  write_source_bank(file_id, source_bank, bank_index);

  if (mpi::master || parallel)
    file_close(file_id);
}

void write_source_bank(hid_t group_id, span<SourceSite> source_bank,
  const vector<int64_t>& bank_index)
{
  hid_t membanktype = h5banktype(true);
  hid_t filebanktype = h5banktype(false);

#ifdef OPENMC_MPI
  write_bank_dataset("source_bank", group_id, source_bank, bank_index,
    membanktype, filebanktype, mpi::source_site);
#else
  write_bank_dataset("source_bank", group_id, source_bank, bank_index,
    membanktype, filebanktype);
#endif

  H5Tclose(membanktype);
  H5Tclose(filebanktype);
}

// Determine member names of a compound HDF5 datatype
std::string dtype_member_names(hid_t dtype_id)
{
  int nmembers = H5Tget_nmembers(dtype_id);
  std::string names;
  for (int i = 0; i < nmembers; i++) {
    char* name = H5Tget_member_name(dtype_id, i);
    names = names.append(name);
    H5free_memory(name);
    if (i < nmembers - 1)
      names += ", ";
  }
  return names;
}

void read_source_bank(
  hid_t group_id, vector<SourceSite>& sites, bool distribute)
{
  bool legacy_particle_codes = true;
  if (attribute_exists(group_id, "version")) {
    array<int, 2> version;
    read_attribute(group_id, "version", version);
    if (version[0] > VERSION_STATEPOINT[0] ||
        (version[0] == VERSION_STATEPOINT[0] && version[1] >= 2)) {
      legacy_particle_codes = false;
    }
  }

  hid_t banktype = h5banktype(true);

  // Open the dataset
  hid_t dset = H5Dopen(group_id, "source_bank", H5P_DEFAULT);

  // Make sure number of members matches
  hid_t dtype = H5Dget_type(dset);
  auto file_member_names = dtype_member_names(dtype);
  auto bank_member_names = dtype_member_names(banktype);
  if (file_member_names != bank_member_names) {
    fatal_error(fmt::format(
      "Source site attributes in file do not match what is "
      "expected for this version of OpenMC. File attributes = ({}). Expected "
      "attributes = ({})",
      file_member_names, bank_member_names));
  }

  hid_t dspace = H5Dget_space(dset);
  hsize_t n_sites;
  H5Sget_simple_extent_dims(dspace, &n_sites, nullptr);

  // Make sure vector is big enough in case where we're reading entire source on
  // each process
  if (!distribute)
    sites.resize(n_sites);

  hid_t memspace;
  if (distribute) {
    if (simulation::work_index[mpi::n_procs] > n_sites) {
      fatal_error("Number of source sites in source file is less "
                  "than number of source particles per generation.");
    }

    // Create another data space but for each proc individually
    hsize_t n_sites_local = simulation::work_per_rank;
    memspace = H5Screate_simple(1, &n_sites_local, nullptr);

    // Select hyperslab for each process
    hsize_t offset = simulation::work_index[mpi::rank];
    H5Sselect_hyperslab(
      dspace, H5S_SELECT_SET, &offset, nullptr, &n_sites_local, nullptr);
  } else {
    memspace = H5S_ALL;
  }

#ifdef PHDF5
  // Read data in parallel
  hid_t plist = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
  H5Dread(dset, banktype, memspace, dspace, plist, sites.data());
  H5Pclose(plist);
#else
  H5Dread(dset, banktype, memspace, dspace, H5P_DEFAULT, sites.data());
#endif

  // Close all ids
  H5Sclose(dspace);
  if (distribute)
    H5Sclose(memspace);
  H5Dclose(dset);
  H5Tclose(banktype);

  if (legacy_particle_codes) {
    for (auto& site : sites) {
      site.particle = legacy_particle_index_to_type(site.particle.pdg_number());
    }
  }
}

void write_unstructured_mesh_results()
{

  for (auto& tally : model::tallies) {

    vector<std::string> tally_scores;
    for (auto filter_idx : tally->filters()) {
      auto& filter = model::tally_filters[filter_idx];
      if (filter->type() != FilterType::MESH)
        continue;

      // check if the filter uses an unstructured mesh
      auto mesh_filter = dynamic_cast<MeshFilter*>(filter.get());
      auto mesh_idx = mesh_filter->mesh();
      auto umesh =
        dynamic_cast<UnstructuredMesh*>(model::meshes[mesh_idx].get());

      if (!umesh)
        continue;

      if (!umesh->output_)
        continue;

      // A multi-rank rma tally holds only owned moment rows on each rank; the
      // gather this per-rank mesh output would need is not yet implemented, so
      // point the user at the statepoint. A single-rank rma tally holds the
      // full array and writes normally.
      if (tally->rma_distributed()) {
        if (mpi::master)
          warning(fmt::format(
            "Unstructured mesh output for tally {} is skipped because it uses "
            "'rma' storage. Read the results from the statepoint file.",
            tally->id_));
        continue;
      }

      if (umesh->library() == "moab") {
        if (mpi::master)
          warning(fmt::format(
            "Output for a MOAB mesh (mesh {}) was "
            "requested but will not be written. Please use the Python "
            "API to generated the desired VTK tetrahedral mesh.",
            umesh->id_));
        continue;
      }

      // if this tally has more than one filter, print
      // warning and skip writing the mesh
      if (tally->filters().size() > 1) {
        warning(fmt::format("Skipping unstructured mesh writing for tally "
                            "{}. More than one filter is present on the tally.",
          tally->id_));
        break;
      }

      int n_realizations = tally->n_realizations_;

      for (int score_idx = 0; score_idx < tally->scores_.size(); score_idx++) {
        for (int nuc_idx = 0; nuc_idx < tally->nuclides_.size(); nuc_idx++) {
          // combine the score and nuclide into a name for the value
          auto score_str = fmt::format("{}_{}", tally->score_name(score_idx),
            tally->nuclide_name(nuc_idx));
          // add this score to the mesh
          // (this is in a separate loop because all variables need to be added
          //  to libMesh's equation system before any are initialized, which
          //  happens in set_score_data)
          umesh->add_score(score_str);
        }
      }

      for (int score_idx = 0; score_idx < tally->scores_.size(); score_idx++) {
        for (int nuc_idx = 0; nuc_idx < tally->nuclides_.size(); nuc_idx++) {
          // combine the score and nuclide into a name for the value
          auto score_str = fmt::format("{}_{}", tally->score_name(score_idx),
            tally->nuclide_name(nuc_idx));

          // index for this nuclide and score
          int nuc_score_idx = score_idx + nuc_idx * tally->scores_.size();

          // construct result vectors
          vector<double> mean_vec(umesh->n_bins()),
            std_dev_vec(umesh->n_bins());
          // Only ranks that own moments fill these from the tally; the values
          // are then broadcast from the master below, so non-owner ranks leave
          // them zero here and receive the master's copy.
          if (tally->has_moments()) {
            const auto& moments = tally->moments();
            for (int j = 0; j < moments.shape(0); j++) {
              // get the volume for this bin
              double volume = umesh->volume(j);
              // compute the mean
              double mean =
                moments(j, nuc_score_idx, TallyMoment::SUM) / n_realizations;
              mean_vec.at(j) = mean / volume;

              // compute the standard deviation
              double sum_sq = moments(j, nuc_score_idx, TallyMoment::SUM_SQ);
              double std_dev {0.0};
              if (n_realizations > 1) {
                std_dev = sum_sq / n_realizations - mean * mean;
                std_dev = std::sqrt(std_dev / (n_realizations - 1));
              }
              std_dev_vec[j] = std_dev / volume;
            }
          }
#ifdef OPENMC_MPI
          MPI_Bcast(
            mean_vec.data(), mean_vec.size(), MPI_DOUBLE, 0, mpi::intracomm);
          MPI_Bcast(std_dev_vec.data(), std_dev_vec.size(), MPI_DOUBLE, 0,
            mpi::intracomm);
#endif
          // set the data for this score
          umesh->set_score_data(score_str, mean_vec, std_dev_vec);
        }
      }

      // Generate a file name based on the tally id
      // and the current batch number
      size_t batch_width {std::to_string(settings::n_max_batches).size()};
      std::string filename = fmt::format("tally_{0}.{1:0{2}}", tally->id_,
        simulation::current_batch, batch_width);

      // Write the unstructured mesh and data to file
      umesh->write(filename);

      // remove score data added for this mesh write
      umesh->remove_scores();
    }
  }
}

void write_tally_results_nr(hid_t file_id)
{
  // ==========================================================================
  // COLLECT AND WRITE GLOBAL TALLIES

  hid_t tallies_group;
  if (mpi::master) {
    // Write number of realizations
    write_dataset(file_id, "n_realizations", simulation::n_realizations);

    tallies_group = open_group(file_id, "tallies");
  }

  // Get global tallies
  auto& gt = simulation::global_tallies;

#ifdef OPENMC_MPI
  // Reduce global tallies
  tensor::Tensor<double> gt_reduced({N_GLOBAL_TALLIES, 3});
  MPI_Reduce(gt.data(), gt_reduced.data(), gt.size(), MPI_DOUBLE, MPI_SUM, 0,
    mpi::intracomm);

  // Transfer values to value on master
  if (mpi::master) {
    if (simulation::current_batch == settings::n_max_batches ||
        simulation::satisfy_triggers) {
      std::copy(gt_reduced.begin(), gt_reduced.end(), gt.begin());
    }
  }
#endif

  // Write out global tallies sum and sum_sq
  if (mpi::master) {
    write_dataset(file_id, "global_tallies", gt);
  }

  for (const auto& t : model::tallies) {
    // Skip any tallies that are not active
    if (!t->active_)
      continue;
    if (!t->writable_)
      continue;

    // rma requires reduce_tallies at input validation, so an rma tally can
    // never reach the no-reduction writer.
    assert(t->storage() != TallyStorage::RMA);

    if (mpi::master && !attribute_exists(file_id, "tallies_present")) {
      write_attribute(file_id, "tallies_present", 1);
    }

    // Reduce a copy of the moments onto the master. moments_ is already the
    // on-disk layout, so the copy reduces contiguously (chunked to respect the
    // int MPI count limit). The copy keeps each rank's own moments intact for
    // the next batch's accumulation. All n_moments columns are reduced and
    // written, including SUM_THIRD/SUM_FOURTH for higher-moments tallies.
    tensor::Tensor<double> values = t->moments();

    if (mpi::master) {
      // Open group for tally
      std::string groupname {"tally " + std::to_string(t->id_)};
      hid_t tally_group = open_group(tallies_group, groupname.c_str());

#ifdef OPENMC_MPI
      reduce_in_place_chunked(values.data(), values.size(), mpi::intracomm);
#endif

      // At the end of the simulation, store the reduced moments back into the
      // tally so post-run consumers (openmc.lib, output) see combined results.
      if (simulation::current_batch == settings::n_max_batches ||
          simulation::satisfy_triggers) {
        t->moments() = values;
      }

      // Write reduced tally moments to file
      auto shape = values.shape();
      write_tally_results(
        tally_group, shape[0], shape[1], shape[2], values.data());

      close_group(tally_group);
    } else {
      // Receive buffer not significant at other processors
#ifdef OPENMC_MPI
      reduce_in_place_chunked(values.data(), values.size(), mpi::intracomm);
#endif
    }
  }

  if (mpi::master) {
    if (!object_exists(file_id, "tallies_present")) {
      // Indicate that tallies are off
      write_dataset(file_id, "tallies_present", 0);
    }

    close_group(tallies_group);
  }
}

} // namespace openmc
