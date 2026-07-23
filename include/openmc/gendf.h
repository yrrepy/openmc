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
  bool validate_za {true};          //!< Check ZA is physically valid
  bool validate_xs_positive {true}; //!< Check XS values are non-negative
  bool warn_short_lines {true};     //!< Warn about skipped short lines
  bool require_mf1_header {true};   //!< Require MF=1, MT=451 header
  bool parse_mf10 {true};           //!< Also parse MF=10 production XS
  int min_file_lines {10};          //!< Minimum expected lines in file
};

//! Result from parsing with diagnostics
struct GENDFParseResult {
  int za {0};                                      //!< Z*1000 + A
  std::unordered_map<int, vector<double>> xs_data; //!< MT -> cross-sections
  std::unordered_map<int, vector<double>>
    energy_data; //!< MT -> energy boundaries

  //! MF=10 production XS: key = MT*1000 + LFS. Unique per MT for
  //! single-product MTs; multi-product MT=5 is skipped at parse.
  std::unordered_map<int, vector<double>> prod_xs_data;
  //! MF=10 energy boundaries: key = MT*1000 + LFS
  std::unordered_map<int, vector<double>> prod_energy_data;
  //! MF=10 metadata: key = MT*1000 + LFS -> IZAP (Z*1000+A of product)
  std::unordered_map<int, int> prod_izap_data;

  int lines_read {0};           //!< Total lines read
  int lines_skipped {0};        //!< Lines skipped (too short)
  int negative_xs_count {0};    //!< Count of negative XS clamped
  vector<std::string> warnings; //!< Warning messages
  bool success {false};         //!< Parsing succeeded
  std::string error_message;    //!< Error message if failed
};

//==============================================================================
//! Single MF=10 production level (LFS, IZAP, cross-sections)
//==============================================================================

struct ProductionLevel {
  int lfs;           //!< Level flag (0=ground, >0=metastable)
  int izap;          //!< Product Z*1000 + A
  vector<double> xs; //!< Production cross-sections per group
};

//==============================================================================
//! Material data from a single GENDF file
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
  //! \param[in] library_bounds Library energy group boundaries for threshold
  //! alignment \return Vector of cross-section values (one per group)
  vector<double> get_xs(
    int mt, int n_groups, const vector<double>& library_bounds) const;

  //! Get MF=10 production XS for all levels of a given MT
  //! \param[in] mt ENDF MT reaction number
  //! \param[in] n_groups Number of energy groups in library
  //! \param[in] library_bounds Library energy group boundaries for alignment
  //! \return Vector of ProductionLevel sorted by LFS ascending
  vector<ProductionLevel> get_production_xs(
    int mt, int n_groups, const vector<double>& library_bounds) const;

  //! Check if MT reaction is served by get_xs (MF=3, or MF=10-only via the
  //! Sigma-partials fallback)
  bool has_mt(int mt) const;

  //! Check if MT has MF=10 production data
  bool has_mf10(int mt) const;

  //! Get energy boundaries for specific MT number
  const vector<double>& get_energies(int mt) const;

  //! Check if MT has energy data
  bool has_energies(int mt) const;

  // Accessors
  const std::string& nuclide_name() const { return nuclide_name_; }
  int za() const { return za_; } //!< Z*1000 + A

private:
  // Data members
  std::string nuclide_name_; //!< Nuclide name (e.g., "U235")
  int za_ {0};               //!< Z*1000 + A

  //! MF=3 cross-section data: map MT -> vector<double> (one value per group)
  std::unordered_map<int, vector<double>> xs_data_;

  //! MF=3 energy data: map MT -> vector<double> (energy boundaries)
  std::unordered_map<int, vector<double>> energy_data_;

  //! MF=10 production XS: key = MT*1000 + LFS -> production XS per group
  std::unordered_map<int, vector<double>> prod_xs_data_;

  //! MF=10 energy boundaries: key = MT*1000 + LFS
  std::unordered_map<int, vector<double>> prod_energy_data_;

  //! MF=10 metadata: key = MT*1000 + LFS -> IZAP
  std::unordered_map<int, int> prod_izap_data_;

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
  explicit GENDFLibrary(const std::string& library_path,
    const vector<double>& energy_bounds,
    const std::string& energy_structure_name = "");

  // Methods

  //! Get cross-section for nuclide and MT
  //! \param[in] nuclide Nuclide name (e.g., "U235")
  //! \param[in] mt ENDF MT reaction number
  //! \param[in] energy_bounds Energy group boundaries in [eV]
  //! \return Vector of cross-section values (one per group)
  vector<double> get_xs(
    const std::string& nuclide, int mt, const vector<double>& energy_bounds);

  //! Get MF=10 production XS for all levels of a given MT
  //! \param[in] nuclide Nuclide name
  //! \param[in] mt ENDF MT reaction number
  //! \return Vector of ProductionLevel sorted by LFS ascending
  vector<ProductionLevel> get_production_xs(const std::string& nuclide, int mt);

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
  std::string library_path_;     //!< Path to GENDF library directory
  std::string energy_structure_; //!< Energy group structure name
  vector<double> energy_bounds_; //!< Energy group boundaries in [eV]
  int n_groups_ {0};             //!< Number of energy groups

  //! Material cache: map nuclide name -> GENDFMaterial
  std::unordered_map<std::string, unique_ptr<GENDFMaterial>> material_cache_;

  //! Mutex for thread-safe cache access (mutable for const method
  //! compatibility) Uses shared_mutex for read/write locking: multiple readers
  //! OR single writer
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
// Non-member functions
//==============================================================================

//! Strip leading zeros from mass number in nuclide name
//! \param[in] name Nuclide name with possible leading zeros (e.g., "Al027")
//! \return Name with leading zeros stripped (e.g., "Al27")
std::string strip_mass_leading_zeros(const std::string& name);

//! Convert GENDF filename stem to OpenMC nuclide name
//! Handles metastable suffixes: mg->_m1, ng->_m2, og->_m3, pg->_m4, qg->_m5,
//! g->ground \param[in] stem GENDF filename stem (e.g., "U235g", "Am242mg")
//! \return OpenMC nuclide name (e.g., "U235", "Am242_m1")
std::string convert_gendf_to_openmc_name(const std::string& stem);

//! Parse GENDF file using MF=3-only parser (7.6x faster)
//! \param[in] filename Path to GENDF .asc file
//! \param[out] xs_data Map of MT -> vector<double> (cross-sections)
//! \param[out] energy_data Map of MT -> vector<double> (energy boundaries)
//! \param[out] za Z*1000 + A
void parse_gendf_mf3_only(const std::string& filename,
  std::unordered_map<int, vector<double>>& xs_data,
  std::unordered_map<int, vector<double>>& energy_data, int& za);

//! Parse GENDF file with validation and diagnostics (G9 fix)
//! \param[in] filename Path to GENDF .asc file
//! \param[in] options Parser validation options
//! \return Parse result with diagnostics
GENDFParseResult parse_gendf_validated(const std::string& filename,
  const GENDFParserOptions& options = GENDFParserOptions {});

//! Validate ZA value is physically reasonable
//! \param[in] za Z*1000 + A value
//! \param[out] error Error message if invalid
//! \return True if valid
bool validate_za(int za, std::string& error);

//! Emit GENDF parser warnings via warning(), deduplicated and capped
//! \param[in] warnings Warning messages from a GENDFParseResult
//! \param[in] context Nuclide name or filename for the summary line
void emit_gendf_warnings(
  const vector<std::string>& warnings, const std::string& context);

} // namespace openmc

#endif // OPENMC_GENDF_H
