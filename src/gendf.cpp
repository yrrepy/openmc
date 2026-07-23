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
// Group-grid alignment (shared by MF=3 and MF=10 lanes)
//==============================================================================

namespace {

//! Align a GENDF section onto the full library group grid.
//!
//! Mirrors the Python backend's _align_to_group_grid. Coverage is decided from
//! the stored ENERGY grid, never from the value count: a full-range section has
//! one energy point per group boundary (size == n_groups + 1), and its last
//! cross section is the TAB1 top-boundary dummy y(NP) which is dropped. A
//! threshold section stores n_real = energies.size() - 1 real values plus that
//! dummy; the reals are placed starting at the group whose boundary matches the
//! section's first energy. Sections with more points than a full grid, or a
//! threshold section with no energy grid to place it, throw (never silently
//! yield an all-zero cross section).
//!
//! \param[in] energies Stored per-section energy grid (size == raw_xs.size())
//! \param[in] raw_xs Stored per-section cross sections (includes dummy y(NP))
//! \param[in] n_groups Number of library energy groups
//! \param[in] library_bounds Library group boundaries (size == n_groups + 1)
//! \param[in] context Nuclide/MT[/LFS] label for warnings and errors
//! \return Cross sections aligned to the library grid (length n_groups)
vector<double> align_to_group_grid(const vector<double>& energies,
  const vector<double>& raw_xs, int n_groups,
  const vector<double>& library_bounds, const std::string& context)
{
  auto ng = static_cast<size_t>(n_groups);

  // Full energy range: NP == n_groups + 1. Return the first n_groups values,
  // dropping the top-boundary dummy.
  if (energies.size() == ng + 1) {
    return vector<double>(raw_xs.begin(), raw_xs.begin() + n_groups);
  }

  // Malformed: more points than a full grid. Fail loudly (K20/R1-10) rather
  // than push an all-zero level; get_xs and get_production_xs behave alike.
  if (energies.size() > ng + 1) {
    throw std::runtime_error(
      "Cross-section size mismatch for " + context + ": expected at most " +
      std::to_string(n_groups + 1) + " energy points, got " +
      std::to_string(energies.size()));
  }

  // Threshold reaction: energy-aware placement onto a zero-filled grid.
  vector<double> aligned(n_groups, 0.0);

  // The Python backend always stores per-section energies; an empty grid (or
  // absent library bounds) leaves no way to place the section, so throw.
  if (energies.empty() || library_bounds.empty()) {
    throw std::runtime_error("Cannot align " + context +
                             ": threshold section has no energy grid for "
                             "group placement");
  }

  double start_energy = energies.front();

  // Find the library boundary matching the section start (relative tolerance).
  int start_idx = -1;
  for (size_t i = 0; i < library_bounds.size(); ++i) {
    double diff = std::abs(library_bounds[i] - start_energy);
    double threshold = GENDF_RTOL_MATCH * std::abs(start_energy) + GENDF_ATOL;
    if (diff <= threshold) {
      start_idx = static_cast<int>(i);
      break;
    }
  }

  // No exact match: snap to the nearest boundary, warn if the gap is large.
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
    if (rel_diff > GENDF_RTOL_WARN) {
      warning("Energy alignment uncertainty for " + context +
              ": GENDF starts at " + std::to_string(start_energy) +
              " eV, using nearest boundary at " +
              std::to_string(library_bounds[start_idx]) +
              " eV (rel diff: " + std::to_string(rel_diff) + ")");
    }
  }

  // Real group values = energy points minus the top-boundary dummy.
  int n_real = static_cast<int>(energies.size()) - 1;
  int end_idx = std::min(start_idx + n_real, n_groups);

  // Validate the section end boundary against the library grid (contiguity
  // sanity check, mirrors the Python backend; warn only in this lenient path).
  if (end_idx < n_groups &&
      static_cast<size_t>(end_idx) < library_bounds.size()) {
    double end_energy = energies.back();
    double expected_end = library_bounds[end_idx];
    double end_rtol = GENDF_RTOL_MATCH * 10.0; // 1e-5, matches Python
    if (std::abs(expected_end - end_energy) >
        end_rtol * std::abs(end_energy) + GENDF_ATOL) {
      double rel_diff_end = std::abs(end_energy - expected_end) /
                            std::max(std::abs(end_energy), 1.0e-10);
      warning("End energy mismatch for " + context + ": GENDF " +
              std::to_string(end_energy) + " eV vs expected " +
              std::to_string(expected_end) +
              " eV (rel diff: " + std::to_string(rel_diff_end) + ")");
    }
  }

  // Copy only the real values (never the dummy), clipped at the grid top.
  int n_copy = std::min(end_idx - start_idx, n_real);
  for (int i = 0; i < n_copy; ++i) {
    aligned[start_idx + i] = raw_xs[i];
  }
  return aligned;
}

} // namespace

//==============================================================================
// GENDFMaterial implementation
//==============================================================================

GENDFMaterial::GENDFMaterial(const std::string& filename)
{
  load_from_file(filename);
}

void GENDFMaterial::load_from_file(const std::string& filename)
{
  // Use full validated parser to get both MF=3 and MF=10 data
  std::filesystem::path p(filename);
  nuclide_name_ = convert_gendf_to_openmc_name(p.stem().string());

  GENDFParserOptions opts;
  opts.warn_short_lines = false;
  opts.parse_mf10 = true;

  GENDFParseResult result = parse_gendf_validated(filename, opts);
  if (!result.success) {
    throw std::runtime_error(result.error_message);
  }

  // Surface parser diagnostics (negative-XS clamps, energy/XS mismatches,
  // bad ZA), capped and deduplicated to avoid flooding on pathological files
  emit_gendf_warnings(result.warnings, nuclide_name_);

  za_ = result.za;
  xs_data_ = std::move(result.xs_data);
  energy_data_ = std::move(result.energy_data);
  prod_xs_data_ = std::move(result.prod_xs_data);
  prod_energy_data_ = std::move(result.prod_energy_data);
  prod_izap_data_ = std::move(result.prod_izap_data);
}

vector<double> GENDFMaterial::get_xs(
  int mt, int n_groups, const vector<double>& library_bounds) const
{
  auto xs_it = xs_data_.find(mt);
  if (xs_it == xs_data_.end()) {
    // No MF=3 section for this MT. EAF-2010 stores any reaction whose residual
    // has a tabulated isomeric state as MF=10 per-final-state partials only
    // (no MF=3). Each event yields exactly one final state, so the sum over the
    // aligned MF=10 levels IS the total reaction cross section. Serve it
    // transparently; fall through to the throw only when neither MF=3 nor MF=10
    // exists (unchanged behavior -> OPENMC_E_UNASSIGNED at the C-API boundary).
    if (has_mf10(mt)) {
      auto levels = get_production_xs(mt, n_groups, library_bounds);
      vector<double> total(static_cast<size_t>(n_groups), 0.0);
      for (const auto& level : levels) {
        for (size_t g = 0; g < total.size(); ++g) {
          total[g] += level.xs[g];
        }
      }
      return total;
    }
    throw std::runtime_error("MT=" + std::to_string(mt) +
                             " not found in GENDF material " + nuclide_name_);
  }

  const auto& xs = xs_it->second;

  // Coverage is driven by the stored per-section energy grid, not the value
  // count (mirrors the Python backend and keeps this lane identical to
  // get_production_xs). energy_data_ is populated alongside xs_data_ by the
  // parser, so an absent grid means malformed data and the helper throws.
  static const vector<double> empty_energies;
  auto energy_it = energy_data_.find(mt);
  const vector<double>& energies =
    (energy_it != energy_data_.end()) ? energy_it->second : empty_energies;

  return align_to_group_grid(energies, xs, n_groups, library_bounds,
    nuclide_name_ + " MT=" + std::to_string(mt));
}

bool GENDFMaterial::has_mt(int mt) const
{
  // A reaction is served by get_xs when MF=3 exists, OR (the EAF-2010 case)
  // when only MF=10 per-final-state partials exist and get_xs returns their
  // sum. Kept consistent with get_xs so has_mt(mt) predicts get_xs success.
  return xs_data_.find(mt) != xs_data_.end() || has_mf10(mt);
}

const vector<double>& GENDFMaterial::get_energies(int mt) const
{
  auto it = energy_data_.find(mt);
  if (it == energy_data_.end()) {
    throw std::runtime_error("MT=" + std::to_string(mt) +
                             " energy data not found in GENDF material " +
                             nuclide_name_);
  }
  return it->second;
}

bool GENDFMaterial::has_energies(int mt) const
{
  return energy_data_.find(mt) != energy_data_.end();
}

bool GENDFMaterial::has_mf10(int mt) const
{
  // Check if any key in range [MT*1000, MT*1000+999] exists
  int base = mt * 1000;
  for (const auto& kv : prod_xs_data_) {
    if (kv.first >= base && kv.first < base + 1000)
      return true;
  }
  return false;
}

vector<ProductionLevel> GENDFMaterial::get_production_xs(
  int mt, int n_groups, const vector<double>& library_bounds) const
{
  vector<ProductionLevel> levels;
  int base = mt * 1000;
  static const vector<double> empty_energies;

  for (const auto& kv : prod_xs_data_) {
    if (kv.first < base || kv.first >= base + 1000)
      continue;

    int lfs = kv.first - base;
    const auto& raw_xs = kv.second;

    // Look up IZAP
    int izap = 0;
    auto izap_it = prod_izap_data_.find(kv.first);
    if (izap_it != prod_izap_data_.end()) {
      izap = izap_it->second;
    }

    // Align to the library energy grid using the shared helper — identical
    // full-range/threshold logic and identical loud failure on malformed
    // sizes as get_xs (K20/R1-10: no silent all-zero production level).
    auto energy_it = prod_energy_data_.find(kv.first);
    const vector<double>& energies = (energy_it != prod_energy_data_.end())
                                       ? energy_it->second
                                       : empty_energies;

    vector<double> aligned_xs =
      align_to_group_grid(energies, raw_xs, n_groups, library_bounds,
        "MF=10 " + nuclide_name_ + " MT=" + std::to_string(mt) +
          " LFS=" + std::to_string(lfs));

    levels.push_back({lfs, izap, std::move(aligned_xs)});
  }

  // Sort by LFS ascending
  std::sort(levels.begin(), levels.end(),
    [](const ProductionLevel& a, const ProductionLevel& b) {
      return a.lfs < b.lfs;
    });

  return levels;
}

void GENDFMaterial::parse_mf3_only(const std::string& filename)
{
  std::filesystem::path p(filename);
  nuclide_name_ = convert_gendf_to_openmc_name(p.stem().string());
  parse_gendf_mf3_only(filename, xs_data_, energy_data_, za_);
}

//==============================================================================
// GENDFLibrary implementation
//==============================================================================

GENDFLibrary::GENDFLibrary(const std::string& library_path,
  const vector<double>& energy_bounds, const std::string& energy_structure_name)
  : library_path_(library_path), energy_bounds_(energy_bounds),
    energy_structure_(energy_structure_name),
    n_groups_(static_cast<int>(energy_bounds.size()) - 1)
{
  // Validate energy bounds
  if (energy_bounds_.size() < 2) {
    throw std::runtime_error("Energy bounds must have at least 2 elements");
  }

  // Check library path exists
  if (!dir_exists(library_path_)) {
    throw std::runtime_error(
      "GENDF library path does not exist: " + library_path_);
  }

  // Build file index
  build_file_index();
}

//==============================================================================
// Helper function: Strip leading zeros from nuclide name mass
//==============================================================================

std::string strip_mass_leading_zeros(const std::string& name)
{
  // Parse nuclide name: Element + Mass + optional metastable state
  // Example: "Al027" -> "Al27"

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

//==============================================================================
// Helper function: Convert GENDF filename stem to OpenMC nuclide name
//==============================================================================

std::string convert_gendf_to_openmc_name(const std::string& stem)
{
  if (stem.empty())
    return stem;

  // Check for 2-char metastable suffixes (mg, ng, og, pg, qg)
  // GENDF convention: mg=m1, ng=m2, og=m3, pg=m4, qg=m5
  if (stem.length() >= 2) {
    std::string last_two = stem.substr(stem.length() - 2);

    static const std::pair<const char*, const char*> meta_map[] = {
      {"mg", "_m1"}, {"ng", "_m2"}, {"og", "_m3"}, {"pg", "_m4"},
      {"qg", "_m5"}};

    for (const auto& [gendf_suffix, openmc_suffix] : meta_map) {
      if (last_two == gendf_suffix) {
        return strip_mass_leading_zeros(stem.substr(0, stem.length() - 2)) +
               openmc_suffix;
      }
    }
  }

  // Ground state: single 'g' suffix
  if (!stem.empty() && stem.back() == 'g') {
    return strip_mass_leading_zeros(stem.substr(0, stem.length() - 1));
  }

  // No recognized suffix
  return strip_mass_leading_zeros(stem);
}

void GENDFLibrary::build_file_index()
{
  file_index_.clear();

  for (const auto& entry : std::filesystem::directory_iterator(library_path_)) {
    if (!entry.is_regular_file())
      continue;

    std::string filename = entry.path().filename().string();

    // Skip macOS metadata files
    if (filename.length() >= 2 && filename.substr(0, 2) == "._")
      continue;

    // Check for .asc extension
    if (filename.length() <= 4 ||
        filename.substr(filename.length() - 4) != ".asc")
      continue;

    // Extract stem and convert to OpenMC nuclide name
    std::string stem = filename.substr(0, filename.length() - 4);
    file_index_[convert_gendf_to_openmc_name(stem)] = entry.path().string();
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
  // one thread can write (unique lock). This is more scalable than omp
  // critical.

  // First, try read-only cache lookup with shared lock (allows concurrent
  // readers)
  {
    std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
    auto it = material_cache_.find(nuclide);
    if (it != material_cache_.end()) {
      return *it->second;
    }
  }

  // Check if nuclide exists before acquiring write lock
  // (avoid throwing exceptions while holding locks)
  std::string filepath = get_file_path(nuclide); // May throw if not found

  // Acquire exclusive write lock for cache insertion
  std::unique_lock<std::shared_mutex> write_lock(cache_mutex_);

  // Double-check pattern: another thread may have loaded while we waited for
  // lock
  auto it = material_cache_.find(nuclide);
  if (it == material_cache_.end()) {
    auto material = make_unique<GENDFMaterial>(filepath);
    material_cache_[nuclide] = std::move(material);
    it = material_cache_.find(nuclide);
  }

  return *it->second;
}

vector<double> GENDFLibrary::get_xs(
  const std::string& nuclide, int mt, const vector<double>& energy_bounds)
{
  // Size check provides sufficient validation - callers always pass library
  // bounds back
  if (energy_bounds.size() != energy_bounds_.size()) {
    throw std::runtime_error("Energy bounds size mismatch: expected " +
                             std::to_string(energy_bounds_.size()) + ", got " +
                             std::to_string(energy_bounds.size()));
  }

  auto& material = load_material(nuclide);
  return material.get_xs(mt, n_groups_, energy_bounds_);
}

vector<ProductionLevel> GENDFLibrary::get_production_xs(
  const std::string& nuclide, int mt)
{
  auto& material = load_material(nuclide);
  return material.get_production_xs(mt, n_groups_, energy_bounds_);
}

//==============================================================================
// C API IMPLEMENTATION
//==============================================================================

extern "C" {

// Global registry for GENDF library instances (managed via integer IDs)
namespace {
std::unordered_map<int, unique_ptr<GENDFLibrary>> g_gendf_libs;
int g_next_lib_id = 1;
constexpr int MAX_GENDF_LIB_ID = 2000000000;

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
    std::string energy_name = energy_structure_name
                                ? std::string(energy_structure_name)
                                : std::string();
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
  try {
    auto it = g_gendf_libs.find(lib_id);
    if (it == g_gendf_libs.end()) {
      set_errmsg("Invalid GENDF library ID");
      return OPENMC_E_INVALID_ID;
    }

    g_gendf_libs.erase(it);
    return 0;

  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
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

  try {
    *n_groups = lib->n_groups();
    return 0;

  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
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

  try {
    *bounds = lib->energy_bounds().data();
    *n = lib->energy_bounds().size();
    return 0;

  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
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

  try {
    *has = lib->has_nuclide(nuclide);
    return 0;

  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
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
    *nuclides = static_cast<char**>(malloc(*n * sizeof(char*)));
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

  // Validate the caller-supplied bound count before constructing a vector from
  // the raw pointer. get_xs requires the count to match the library grid
  // exactly (n_groups + 1); checking here avoids an out-of-bounds read (a
  // negative count is read as a huge unsigned range, an oversized count over-
  // reads the buffer) that would otherwise occur before get_xs's size check.
  if (n_energy_bounds != lib->n_groups() + 1) {
    set_errmsg("n_energy_bounds must equal the GENDF library group count + 1");
    return OPENMC_E_INVALID_SIZE;
  }

  try {
    // Create energy bounds vector for validation
    vector<double> bounds(energy_bounds, energy_bounds + n_energy_bounds);

    // Get cross-section data
    auto xs = lib->get_xs(nuclide, mt, bounds);
    *n_groups = xs.size();

    // Allocate output array
    *xs_data = static_cast<double*>(malloc(*n_groups * sizeof(double)));
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
    return; // Safe no-op for null pointer
  }

  // Free each individual string
  for (int i = 0; i < n; ++i) {
    if (nuclides[i] != nullptr) {
      free(nuclides[i]);
      nuclides[i] = nullptr; // Prevent double-free
    }
  }

  // Free the array itself
  free(nuclides);
}

int openmc_gendf_get_production_xs(int32_t lib_id, const char* nuclide,
  int32_t mt, int* n_levels, int* n_groups, int** lfs_out, int** izap_out,
  double** xs_out)
{
  if (!nuclide || !n_levels || !n_groups || !lfs_out || !izap_out || !xs_out) {
    set_errmsg("Null pointer argument to openmc_gendf_get_production_xs");
    return OPENMC_E_INVALID_ARGUMENT;
  }

  GENDFLibrary* lib = get_library(lib_id);
  if (!lib)
    return OPENMC_E_INVALID_ID;

  try {
    auto levels = lib->get_production_xs(nuclide, mt);
    *n_levels = levels.size();
    *n_groups = lib->n_groups();

    if (levels.empty()) {
      *lfs_out = nullptr;
      *izap_out = nullptr;
      *xs_out = nullptr;
      return 0;
    }

    int nl = levels.size();
    int ng = *n_groups;

    *lfs_out = static_cast<int*>(malloc(nl * sizeof(int)));
    *izap_out = static_cast<int*>(malloc(nl * sizeof(int)));
    *xs_out = static_cast<double*>(malloc(nl * ng * sizeof(double)));

    if (!*lfs_out || !*izap_out || !*xs_out) {
      free(*lfs_out);
      free(*izap_out);
      free(*xs_out);
      set_errmsg("Failed to allocate memory for production XS data");
      return OPENMC_E_ALLOCATE;
    }

    for (int i = 0; i < nl; ++i) {
      (*lfs_out)[i] = levels[i].lfs;
      (*izap_out)[i] = levels[i].izap;
      std::copy(levels[i].xs.begin(), levels[i].xs.end(), *xs_out + i * ng);
    }

    return 0;

  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
}

void openmc_gendf_free_production_xs(int* lfs, int* izap, double* xs)
{
  if (lfs)
    free(lfs);
  if (izap)
    free(izap);
  if (xs)
    free(xs);
}

} // extern "C"

} // namespace openmc
