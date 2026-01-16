//! \file gendf.h
//! \brief GENDF (Group-averaged ENDF) cross-section library for depletion

#ifndef OPENMC_GENDF_H
#define OPENMC_GENDF_H

#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "openmc/memory.h" // for unique_ptr
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
// GENDF tolerance constants (H1 fix - unified across C++ and Python)
//==============================================================================

//! Relative tolerance for energy boundary matching
constexpr double GENDF_RTOL_MATCH = 1.0e-6;

//! Relative tolerance for energy boundary mismatch warnings
constexpr double GENDF_RTOL_WARN = 1.0e-4;

//! Absolute tolerance for energy matching (only used when values are near zero)
constexpr double GENDF_ATOL = 0.0;

//==============================================================================
// Parser validation options and results (G9 fix)
//==============================================================================

//! Validation options for GENDF parsing
struct GENDFParserOptions {
  bool validate_za {true};           //!< Check ZA is physically valid
  bool validate_xs_positive {true};  //!< Check XS values are non-negative
  bool warn_short_lines {true};      //!< Warn about skipped short lines
  bool require_mf1_header {true};    //!< Require MF=1, MT=451 header
  int min_file_lines {10};           //!< Minimum expected lines in file
};

//! Result from parsing with diagnostics
struct GENDFParseResult {
  int za {0};                                           //!< Z*1000 + A
  int zam {0};                                          //!< Z*1000 + A + isomeric state
  std::unordered_map<int, vector<double>> xs_data;      //!< MT -> cross-sections
  std::unordered_map<int, vector<double>> energy_data;  //!< MT -> energy boundaries
  int lines_read {0};                                   //!< Total lines read
  int lines_skipped {0};                                //!< Lines skipped (too short)
  int negative_xs_count {0};                            //!< Count of negative XS clamped
  vector<std::string> warnings;                         //!< Warning messages
  bool success {false};                                 //!< Parsing succeeded
  std::string error_message;                            //!< Error message if failed
};

//==============================================================================
//! Material data from a single GENDF file (MF=3 cross-sections only)
//==============================================================================

class GENDFMaterial {
public:
  // Constructors
  GENDFMaterial() = default;

  //! Construct material from GENDF file
  //! \param[in] filename Path to GENDF .asc file
  explicit GENDFMaterial(const std::string& filename);

  // Methods

  //! Load material data from GENDF file
  //! \param[in] filename Path to GENDF .asc file
  void load_from_file(const std::string& filename);

  //! Get cross-section data for specific MT number with energy-aware alignment
  //! \param[in] mt ENDF MT reaction number
  //! \param[in] n_groups Number of energy groups
  //! \param[in] library_bounds Library energy group boundaries for threshold alignment
  //! \return Vector of cross-section values (one per group)
  vector<double> get_xs(int mt, int n_groups, const vector<double>& library_bounds) const;

  //! Check if MT reaction exists in this material
  //! \param[in] mt ENDF MT reaction number
  //! \return True if reaction exists
  bool has_mt(int mt) const;

  //! Get energy boundaries for specific MT number
  //! \param[in] mt ENDF MT reaction number
  //! \return Vector of energy boundaries for this reaction
  const vector<double>& get_energies(int mt) const;

  //! Check if MT has energy data
  //! \param[in] mt ENDF MT reaction number
  //! \return True if energy data exists
  bool has_energies(int mt) const;

  // Accessors
  const std::string& nuclide_name() const { return nuclide_name_; }
  int za() const { return za_; }   //!< Z*1000 + A
  int zam() const { return zam_; } //!< Z*1000 + A + isomeric state

private:
  // Data members
  std::string nuclide_name_;           //!< Nuclide name (e.g., "U235")
  int za_ {0};                         //!< Z*1000 + A
  int zam_ {0};                        //!< Z*1000 + A + isomeric state

  //! Cross-section data: map MT -> vector<double> (one value per group)
  std::unordered_map<int, vector<double>> xs_data_;

  //! Energy data: map MT -> vector<double> (energy boundaries for threshold alignment)
  std::unordered_map<int, vector<double>> energy_data_;

  // Parsing methods

  //! Parse GENDF file (MF=3 only for speed)
  //! \param[in] filename Path to GENDF .asc file
  void parse_mf3_only(const std::string& filename);
};

//==============================================================================
//! GENDF library manager with caching
//==============================================================================

class GENDFLibrary {
public:
  // Constructors

  //! Construct GENDF library
  //! \param[in] library_path Path to directory containing GENDF .asc files
  //! \param[in] energy_bounds Energy group boundaries in [eV]
  //! \param[in] energy_structure_name Optional name of energy structure
  explicit GENDFLibrary(
    const std::string& library_path,
    const vector<double>& energy_bounds,
    const std::string& energy_structure_name = "");

  // Methods

  //! Get cross-section for nuclide and MT
  //! \param[in] nuclide Nuclide name (e.g., "U235")
  //! \param[in] mt ENDF MT reaction number
  //! \param[in] energy_bounds Energy group boundaries in [eV]
  //! \return Vector of cross-section values (one per group)
  vector<double> get_xs(
    const std::string& nuclide,
    int mt,
    const vector<double>& energy_bounds);

  //! Check if nuclide is available in library
  //! \param[in] nuclide Nuclide name
  //! \return True if nuclide file exists
  bool has_nuclide(const std::string& nuclide) const;

  //! Get list of available nuclides
  //! \return Vector of nuclide names
  vector<std::string> available_nuclides() const;

  // Accessors
  const vector<double>& energy_bounds() const { return energy_bounds_; }
  int n_groups() const { return n_groups_; }
  const std::string& energy_structure() const { return energy_structure_; }
  const std::string& library_path() const { return library_path_; }

private:
  // Data members
  std::string library_path_;                 //!< Path to GENDF library directory
  std::string energy_structure_;             //!< Energy group structure name
  vector<double> energy_bounds_;             //!< Energy group boundaries in [eV]
  int n_groups_ {0};                         //!< Number of energy groups

  //! Material cache: map nuclide name -> GENDFMaterial
  std::unordered_map<std::string, unique_ptr<GENDFMaterial>> material_cache_;

  //! Mutex for thread-safe cache access (mutable for const method compatibility)
  //! Uses shared_mutex for read/write locking: multiple readers OR single writer
  mutable std::shared_mutex cache_mutex_;

  //! File index: map nuclide name -> file path
  std::unordered_map<std::string, std::string> file_index_;

  // Private methods

  //! Load material (with caching)
  //! \param[in] nuclide Nuclide name
  //! \return Reference to cached material
  GENDFMaterial& load_material(const std::string& nuclide);

  //! Build file index by scanning library directory
  void build_file_index();

  //! Get file path for nuclide
  //! \param[in] nuclide Nuclide name
  //! \return Full path to GENDF file
  std::string get_file_path(const std::string& nuclide) const;
};

//==============================================================================
// Global variables
//==============================================================================

namespace data {

//! Map of library ID -> GENDFLibrary instance
extern std::unordered_map<int, unique_ptr<GENDFLibrary>> gendf_libraries;

//! Next available library ID
extern int n_gendf_libraries;

} // namespace data

//==============================================================================
// Non-member functions
//==============================================================================

//! Strip leading zeros from mass number in nuclide name
//! \param[in] name Nuclide name with possible leading zeros (e.g., "Al027")
//! \return Name with leading zeros stripped (e.g., "Al27")
std::string strip_mass_leading_zeros(const std::string& name);

//! Convert GENDF filename stem to OpenMC nuclide name
//! Handles metastable suffixes: mg->_m1, ng->_m2, og->_m3, pg->_m4, qg->_m5, g->ground
//! \param[in] stem GENDF filename stem (e.g., "U235g", "Am242mg")
//! \return OpenMC nuclide name (e.g., "U235", "Am242_m1")
std::string convert_gendf_to_openmc_name(const std::string& stem);

//! Parse GENDF file using MF=3-only parser (7.6x faster)
//! \param[in] filename Path to GENDF .asc file
//! \param[out] xs_data Map of MT -> vector<double> (cross-sections)
//! \param[out] energy_data Map of MT -> vector<double> (energy boundaries)
//! \param[out] za Z*1000 + A
//! \param[out] zam Z*1000 + A + isomeric state
void parse_gendf_mf3_only(
  const std::string& filename,
  std::unordered_map<int, vector<double>>& xs_data,
  std::unordered_map<int, vector<double>>& energy_data,
  int& za,
  int& zam);

//! Parse GENDF file with validation and diagnostics (G9 fix)
//! \param[in] filename Path to GENDF .asc file
//! \param[in] options Parser validation options
//! \return Parse result with diagnostics
GENDFParseResult parse_gendf_validated(
  const std::string& filename,
  const GENDFParserOptions& options = GENDFParserOptions{});

//! Validate ZA value is physically reasonable
//! \param[in] za Z*1000 + A value
//! \param[out] error Error message if invalid
//! \return True if valid
bool validate_za(int za, std::string& error);

} // namespace openmc

#endif // OPENMC_GENDF_H
