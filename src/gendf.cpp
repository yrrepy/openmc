//! \file gendf.cpp
//! \brief Implementation of GENDF library classes

#include "openmc/gendf.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "openmc/error.h"
#include "openmc/file_utils.h"
#include "openmc/mgxs.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace data {
std::unordered_map<int, unique_ptr<GENDFLibrary>> gendf_libraries;
int n_gendf_libraries {0};
} // namespace data

//==============================================================================
// GENDFMaterial implementation
//==============================================================================

GENDFMaterial::GENDFMaterial(const std::string& filename, bool fast_parser)
{
  load_from_file(filename, fast_parser);
}

void GENDFMaterial::load_from_file(const std::string& filename, bool fast_parser)
{
  if (fast_parser) {
    parse_mf3_only(filename);
  } else {
    parse_full(filename);
  }
}

vector<double> GENDFMaterial::get_xs(int mt, int n_groups,
                                      const vector<double>& library_bounds) const
{
  auto xs_it = xs_data_.find(mt);
  if (xs_it == xs_data_.end()) {
    throw std::runtime_error("MT=" + std::to_string(mt) +
                             " not found in GENDF material " + nuclide_name_);
  }

  const auto& xs = xs_it->second;

  // Check if we have energy data for this reaction
  auto energy_it = energy_data_.find(mt);
  bool has_energy = (energy_it != energy_data_.end() && !energy_it->second.empty());

  // Full energy range case - direct return
  if (xs.size() == static_cast<size_t>(n_groups)) {
    return xs;
  } else if (xs.size() == static_cast<size_t>(n_groups + 1)) {
    // GENDF file has n_groups+1 values (per energy bound instead of per group)
    // Return first n_groups values (drop the last one)
    return vector<double>(xs.begin(), xs.begin() + n_groups);
  } else if (xs.size() < static_cast<size_t>(n_groups)) {
    // Threshold reaction - need energy-aware placement
    vector<double> padded(n_groups, 0.0);

    // Use energy data for correct alignment if available
    if (has_energy && library_bounds.size() > 0) {
      const auto& energies = energy_it->second;
      double start_energy = energies.front();

      // Find start index in library bounds using relative tolerance
      // H1 fix: Use unified tolerance constants from gendf.h
      int start_idx = -1;

      for (size_t i = 0; i < library_bounds.size(); ++i) {
        double diff = std::abs(library_bounds[i] - start_energy);
        double threshold = GENDF_RTOL_MATCH * std::abs(start_energy) + GENDF_ATOL;
        if (diff <= threshold) {
          start_idx = static_cast<int>(i);
          break;
        }
      }

      // Fall back to nearest match if exact match not found
      if (start_idx < 0) {
        double min_diff = std::numeric_limits<double>::max();
        for (size_t i = 0; i < library_bounds.size(); ++i) {
          double diff = std::abs(library_bounds[i] - start_energy);
          if (diff < min_diff) {
            min_diff = diff;
            start_idx = static_cast<int>(i);
          }
        }
        double rel_diff = min_diff / std::abs(start_energy);
        if (rel_diff > GENDF_RTOL_WARN) {  // H1 fix: Use unified warning threshold
          warning("Energy alignment uncertainty for " + nuclide_name_ + " MT=" +
                  std::to_string(mt) + ": GENDF starts at " +
                  std::to_string(start_energy) + " eV, using nearest boundary at " +
                  std::to_string(library_bounds[start_idx]) + " eV (rel diff: " +
                  std::to_string(rel_diff) + ")");
        }
      }

      // Copy XS data to correct position
      size_t n_to_copy = std::min(xs.size(),
                                   static_cast<size_t>(n_groups - start_idx));
      for (size_t i = 0; i < n_to_copy; ++i) {
        padded[start_idx + i] = xs[i];
      }
    } else {
      // Fallback: no energy data available, use old behavior (high-energy end)
      // This maintains backward compatibility but may be inaccurate
      size_t offset = n_groups - xs.size();
      std::copy(xs.begin(), xs.end(), padded.begin() + offset);
    }

    return padded;
  } else {
    throw std::runtime_error("Cross-section size mismatch for MT=" + std::to_string(mt) +
                             ": expected " + std::to_string(n_groups) +
                             " or " + std::to_string(n_groups + 1) +
                             ", got " + std::to_string(xs.size()));
  }
}

bool GENDFMaterial::has_mt(int mt) const
{
  return xs_data_.find(mt) != xs_data_.end();
}

const vector<double>& GENDFMaterial::get_energies(int mt) const
{
  auto it = energy_data_.find(mt);
  if (it == energy_data_.end()) {
    throw std::runtime_error("MT=" + std::to_string(mt) +
                             " energy data not found in GENDF material " + nuclide_name_);
  }
  return it->second;
}

bool GENDFMaterial::has_energies(int mt) const
{
  return energy_data_.find(mt) != energy_data_.end();
}

void GENDFMaterial::parse_mf3_only(const std::string& filename)
{
  // Extract nuclide name from filename
  std::filesystem::path p(filename);
  nuclide_name_ = p.stem().string();

  // Remove trailing 'g' or 'm' (ground/metastable state indicators)
  if (!nuclide_name_.empty()) {
    char last = nuclide_name_.back();
    if (last == 'g' || last == 'm') {
      nuclide_name_.pop_back();
    }
  }

  // Call parser (now extracts both xs_data and energy_data)
  parse_gendf_mf3_only(filename, xs_data_, energy_data_, za_, zam_);
}

void GENDFMaterial::parse_full(const std::string& filename)
{
  // Extract nuclide name from filename
  std::filesystem::path p(filename);
  nuclide_name_ = p.stem().string();

  // Remove trailing 'g' or 'm'
  if (!nuclide_name_.empty()) {
    char last = nuclide_name_.back();
    if (last == 'g' || last == 'm') {
      nuclide_name_.pop_back();
    }
  }

  // Call parser (now extracts both xs_data and energy_data)
  parse_gendf_full(filename, xs_data_, energy_data_, za_, zam_);
}

//==============================================================================
// GENDFLibrary implementation
//==============================================================================

GENDFLibrary::GENDFLibrary(
  const std::string& library_path,
  const vector<double>& energy_bounds,
  const std::string& energy_structure_name)
  : library_path_(library_path),
    energy_bounds_(energy_bounds),
    energy_structure_(energy_structure_name),
    n_groups_(static_cast<int>(energy_bounds.size()) - 1)
{
  // Validate energy bounds
  if (energy_bounds_.size() < 2) {
    throw std::runtime_error("Energy bounds must have at least 2 elements");
  }

  // Check library path exists
  if (!dir_exists(library_path_)) {
    throw std::runtime_error("GENDF library path does not exist: " + library_path_);
  }

  // Build file index
  build_file_index();
}

//==============================================================================
// Helper function: Normalize nuclide name by removing leading zeros from mass
//==============================================================================

std::string normalize_nuclide_name(const std::string& name)
{
  // Parse nuclide name: Element + Mass + optional metastable state
  // Examples: "Al027" -> "Al27", "Ag109" -> "Ag109", "Am242m1" -> "Am242m1"

  std::string result;
  size_t i = 0;

  // Extract element symbol (letters only)
  while (i < name.length() && std::isalpha(name[i])) {
    result += name[i];
    ++i;
  }

  // Extract mass number (digits), stripping leading zeros
  std::string mass_str;
  while (i < name.length() && std::isdigit(name[i])) {
    mass_str += name[i];
    ++i;
  }

  // Remove leading zeros by converting to int and back to string
  if (!mass_str.empty()) {
    int mass = std::stoi(mass_str);
    result += std::to_string(mass);
  }

  // Append any remaining characters (metastable state: m, m1, m2, etc.)
  while (i < name.length()) {
    result += name[i];
    ++i;
  }

  return result;
}

void GENDFLibrary::build_file_index()
{
  file_index_.clear();

  // FISPACT metastable naming convention:
  // - 'g'  = ground state -> Element{A}
  // - 'mg' = m1 (1st metastable) -> Element{A}_m1
  // - 'ng' = m2 (2nd metastable) -> Element{A}_m2
  // - 'og' = m3 (3rd metastable) -> Element{A}_m3
  // - 'pg' = m4 (4th metastable) -> Element{A}_m4
  // - 'qg' = m5 (5th metastable) -> Element{A}_m5

  // Scan directory for .asc files
  for (const auto& entry : std::filesystem::directory_iterator(library_path_)) {
    if (entry.is_regular_file()) {
      std::string filename = entry.path().filename().string();

      // Skip macOS metadata files
      if (filename.length() >= 2 && filename.substr(0, 2) == "._") {
        continue;
      }

      // Check for .asc extension
      if (filename.length() > 4 &&
          filename.substr(filename.length() - 4) == ".asc") {

        // Extract nuclide name (remove .asc extension)
        std::string nuclide = filename.substr(0, filename.length() - 4);
        std::string openmc_suffix;

        // Check for metastable suffixes (2-char suffixes before single 'g')
        // Must check these before checking for single 'g'
        if (nuclide.length() >= 2) {
          std::string last_two = nuclide.substr(nuclide.length() - 2);

          if (last_two == "mg") {
            nuclide = nuclide.substr(0, nuclide.length() - 2);
            openmc_suffix = "_m1";
          } else if (last_two == "ng") {
            nuclide = nuclide.substr(0, nuclide.length() - 2);
            openmc_suffix = "_m2";
          } else if (last_two == "og") {
            nuclide = nuclide.substr(0, nuclide.length() - 2);
            openmc_suffix = "_m3";
          } else if (last_two == "pg") {
            nuclide = nuclide.substr(0, nuclide.length() - 2);
            openmc_suffix = "_m4";
          } else if (last_two == "qg") {
            nuclide = nuclide.substr(0, nuclide.length() - 2);
            openmc_suffix = "_m5";
          } else if (!nuclide.empty() && nuclide.back() == 'g') {
            // Ground state: single 'g' suffix
            nuclide.pop_back();
          }
        } else if (!nuclide.empty() && nuclide.back() == 'g') {
          // Ground state: single 'g' suffix (short names)
          nuclide.pop_back();
        }

        // Normalize nuclide name (strip leading zeros from mass number)
        // This ensures "Al027" -> "Al27" to match OpenMC naming convention
        nuclide = normalize_nuclide_name(nuclide);

        // Append OpenMC metastable suffix if present
        nuclide += openmc_suffix;

        file_index_[nuclide] = entry.path().string();
      }
    }
  }

  if (file_index_.empty()) {
    throw std::runtime_error("No GENDF files found in: " + library_path_);
  }
}

std::string GENDFLibrary::get_file_path(const std::string& nuclide) const
{
  auto it = file_index_.find(nuclide);
  if (it == file_index_.end()) {
    throw std::runtime_error("Nuclide not found in GENDF library: " + nuclide);
  }
  return it->second;
}

bool GENDFLibrary::has_nuclide(const std::string& nuclide) const
{
  return file_index_.find(nuclide) != file_index_.end();
}

vector<std::string> GENDFLibrary::available_nuclides() const
{
  vector<std::string> nuclides;
  nuclides.reserve(file_index_.size());

  for (const auto& pair : file_index_) {
    nuclides.push_back(pair.first);
  }

  std::sort(nuclides.begin(), nuclides.end());
  return nuclides;
}

GENDFMaterial& GENDFLibrary::load_material(const std::string& nuclide)
{
  // Thread safety: Uses std::shared_mutex for fine-grained read/write locking.
  // Multiple threads can read the cache simultaneously (shared lock), but only
  // one thread can write (unique lock). This is more scalable than omp critical.

  // First, try read-only cache lookup with shared lock (allows concurrent readers)
  {
    std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
    auto it = material_cache_.find(nuclide);
    if (it != material_cache_.end()) {
      return *it->second;
    }
  }

  // Check if nuclide exists BEFORE acquiring write lock
  // (avoid throwing exceptions while holding locks)
  std::string filepath = get_file_path(nuclide);  // May throw if not found

  // Acquire exclusive write lock for cache insertion
  std::unique_lock<std::shared_mutex> write_lock(cache_mutex_);

  // Double-check pattern: another thread may have loaded while we waited for lock
  auto it = material_cache_.find(nuclide);
  if (it == material_cache_.end()) {
    auto material = make_unique<GENDFMaterial>(filepath, true);
    material_cache_[nuclide] = std::move(material);
    it = material_cache_.find(nuclide);
  }

  return *it->second;
}

vector<double> GENDFLibrary::get_xs(
  const std::string& nuclide,
  int mt,
  const vector<double>& energy_bounds)
{
  // Validate energy bounds using unified tolerance constants (H1 fix)
  // Aligned with OpenMC Python standards (detect_energy_structure(), branching validation)
  // NOTE: May need to relax GENDF_RTOL_MATCH if too strict for some FISPACT GENDF files

  if (energy_bounds.size() != energy_bounds_.size()) {
    throw std::runtime_error(
      "Energy bounds size mismatch: expected " +
      std::to_string(energy_bounds_.size()) + ", got " +
      std::to_string(energy_bounds.size()));
  }

  for (size_t i = 0; i < energy_bounds.size(); ++i) {
    double diff = std::abs(energy_bounds[i] - energy_bounds_[i]);
    double threshold = GENDF_ATOL + GENDF_RTOL_MATCH * std::abs(energy_bounds_[i]);

    if (diff > threshold) {
      throw std::runtime_error(
        "Energy bounds do not match library energy structure '" +
        energy_structure_ + "'");
    }
  }

  // Load material and get cross-section (pass library bounds for threshold alignment)
  auto& material = load_material(nuclide);
  return material.get_xs(mt, n_groups_, energy_bounds_);
}

//==============================================================================
// C API IMPLEMENTATION
//==============================================================================

extern "C" {

// Global registry for GENDF library instances (managed via integer IDs)
namespace {
std::unordered_map<int, unique_ptr<GENDFLibrary>> g_gendf_libs;
int g_next_lib_id = 1;
constexpr int MAX_GENDF_LIB_ID = 2000000000;  // ~2 billion, well below INT_MAX

// Helper to get library by ID
GENDFLibrary* get_library(int32_t lib_id)
{
  auto it = g_gendf_libs.find(lib_id);
  if (it == g_gendf_libs.end()) {
    set_errmsg("Invalid GENDF library ID");
    return nullptr;
  }
  return it->second.get();
}
} // anonymous namespace

int openmc_gendf_library_create(const char* library_path, int n_energy_bounds,
  const double* energy_bounds, const char* energy_structure_name,
  int32_t* lib_id)
{
  try {
    // Validate inputs
    if (!library_path || !energy_bounds || !lib_id) {
      set_errmsg("Null pointer argument to openmc_gendf_library_create");
      return OPENMC_E_INVALID_ARGUMENT;
    }

    if (n_energy_bounds < 2) {
      set_errmsg("Energy bounds must have at least 2 elements");
      return OPENMC_E_INVALID_SIZE;
    }

    // Create energy bounds vector
    vector<double> bounds(energy_bounds, energy_bounds + n_energy_bounds);

    // Create library
    std::string energy_name =
      energy_structure_name ? std::string(energy_structure_name) : std::string();
    auto lib = make_unique<GENDFLibrary>(library_path, bounds, energy_name);

    // Store in global registry with overflow protection
    if (g_next_lib_id >= MAX_GENDF_LIB_ID) {
      set_errmsg("Maximum number of GENDF libraries exceeded");
      return OPENMC_E_ALLOCATE;
    }
    int id = g_next_lib_id++;
    g_gendf_libs[id] = std::move(lib);
    *lib_id = id;

    return 0;

  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
}

int openmc_gendf_library_free(int32_t lib_id)
{
  auto it = g_gendf_libs.find(lib_id);
  if (it == g_gendf_libs.end()) {
    set_errmsg("Invalid GENDF library ID");
    return OPENMC_E_INVALID_ID;
  }

  g_gendf_libs.erase(it);
  return 0;
}

int openmc_gendf_library_get_n_groups(int32_t lib_id, int* n_groups)
{
  if (!n_groups) {
    set_errmsg("Null pointer for n_groups");
    return OPENMC_E_INVALID_ARGUMENT;
  }

  GENDFLibrary* lib = get_library(lib_id);
  if (!lib)
    return OPENMC_E_INVALID_ID;

  *n_groups = lib->n_groups();
  return 0;
}

int openmc_gendf_library_get_energy_bounds(
  int32_t lib_id, const double** bounds, int* n)
{
  if (!bounds || !n) {
    set_errmsg("Null pointer for bounds or n");
    return OPENMC_E_INVALID_ARGUMENT;
  }

  GENDFLibrary* lib = get_library(lib_id);
  if (!lib)
    return OPENMC_E_INVALID_ID;

  *bounds = lib->energy_bounds().data();
  *n = lib->energy_bounds().size();
  return 0;
}

int openmc_gendf_library_has_nuclide(
  int32_t lib_id, const char* nuclide, bool* has)
{
  if (!nuclide || !has) {
    set_errmsg("Null pointer for nuclide or has");
    return OPENMC_E_INVALID_ARGUMENT;
  }

  GENDFLibrary* lib = get_library(lib_id);
  if (!lib)
    return OPENMC_E_INVALID_ID;

  *has = lib->has_nuclide(nuclide);
  return 0;
}

int openmc_gendf_library_available_nuclides(
  int32_t lib_id, char*** nuclides, int* n)
{
  if (!nuclides || !n) {
    set_errmsg("Null pointer for nuclides or n");
    return OPENMC_E_INVALID_ARGUMENT;
  }

  GENDFLibrary* lib = get_library(lib_id);
  if (!lib)
    return OPENMC_E_INVALID_ID;

  try {
    auto nuc_list = lib->available_nuclides();
    *n = nuc_list.size();

    // Allocate array of C strings
    *nuclides = (char**)malloc(*n * sizeof(char*));
    if (!*nuclides) {
      set_errmsg("Failed to allocate memory for nuclides array");
      return OPENMC_E_ALLOCATE;
    }

    // Copy each nuclide name
    for (int i = 0; i < *n; ++i) {
      (*nuclides)[i] = strdup(nuc_list[i].c_str());
      if (!(*nuclides)[i]) {
        // Free previously allocated strings
        for (int j = 0; j < i; ++j) {
          free((*nuclides)[j]);
        }
        free(*nuclides);
        set_errmsg("Failed to allocate memory for nuclide name");
        return OPENMC_E_ALLOCATE;
      }
    }

    return 0;

  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
}

int openmc_gendf_get_xs(int32_t lib_id, const char* nuclide, int32_t mt,
  int n_energy_bounds, const double* energy_bounds, double** xs_data,
  int* n_groups)
{
  if (!nuclide || !energy_bounds || !xs_data || !n_groups) {
    set_errmsg("Null pointer argument to openmc_gendf_get_xs");
    return OPENMC_E_INVALID_ARGUMENT;
  }

  GENDFLibrary* lib = get_library(lib_id);
  if (!lib)
    return OPENMC_E_INVALID_ID;

  try {
    // Create energy bounds vector for validation
    vector<double> bounds(energy_bounds, energy_bounds + n_energy_bounds);

    // Get cross-section data
    auto xs = lib->get_xs(nuclide, mt, bounds);
    *n_groups = xs.size();

    // Allocate output array
    *xs_data = (double*)malloc(*n_groups * sizeof(double));
    if (!*xs_data) {
      set_errmsg("Failed to allocate memory for cross-section data");
      return OPENMC_E_ALLOCATE;
    }

    // Copy data
    std::copy(xs.begin(), xs.end(), *xs_data);

    return 0;

  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
}

void openmc_gendf_free_xs(double* xs_data)
{
  if (xs_data) {
    free(xs_data);
  }
}

void openmc_gendf_free_nuclides(char** nuclides, int n)
{
  if (nuclides == nullptr) {
    return;  // Safe no-op for null pointer
  }

  // Free each individual string
  for (int i = 0; i < n; ++i) {
    if (nuclides[i] != nullptr) {
      free(nuclides[i]);
      nuclides[i] = nullptr;  // Prevent double-free
    }
  }

  // Free the array itself
  free(nuclides);
}

} // extern "C"

} // namespace openmc
