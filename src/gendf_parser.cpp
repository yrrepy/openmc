//! \file gendf_parser.cpp
//! \brief GENDF file parser for MF=3 cross-sections
//!
//! ENDF-6 Format Reference (ENDF-102):
//! - Columns 1-66: Data fields (six 11-character fields)
//! - Columns 67-70: MAT (material number)
//! - Columns 71-72: MF (file number)
//! - Columns 73-75: MT (section number)
//! - Each data field is 11 characters wide

#include "openmc/gendf.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "openmc/error.h"
#include "openmc/file_utils.h"

namespace openmc {

//==============================================================================
// Helper functions for ENDF-6 format parsing
//==============================================================================

//! Trim whitespace from both ends of string
//! \param[in] str String to trim
//! \return Trimmed string
std::string trim(const std::string& str) {
  auto start = str.begin();
  while (start != str.end() && std::isspace(*start)) {
    ++start;
  }

  auto end = str.end();
  do {
    --end;
  } while (std::distance(start, end) > 0 && std::isspace(*end));

  return std::string(start, end + 1);
}

//==============================================================================
// ZA validation
//==============================================================================

bool validate_za(int za, std::string& error) {
  if (za <= 0) {
    error = "ZA value is zero or negative: " + std::to_string(za);
    return false;
  }

  int Z = za / 1000;
  int A = za % 1000;

  if (Z < 1 || Z > 118) {
    error = "Invalid atomic number Z=" + std::to_string(Z) +
            " (must be 1-118)";
    return false;
  }

  if (A < Z) {
    error = "Invalid mass number A=" + std::to_string(A) +
            " < Z=" + std::to_string(Z);
    return false;
  }

  // Rough upper bound for stable/metastable nuclei
  if (A > 3 * Z + 10) {
    error = "Suspicious mass number A=" + std::to_string(A) +
            " for Z=" + std::to_string(Z) + " (expected A <= 3*Z+10)";
    return false;
  }

  return true;
}

//==============================================================================
// Safe extraction functions with error tracking
//==============================================================================

namespace {

//! Result from extraction with success/error tracking
template<typename T>
struct ExtractResult {
  T value;
  bool success;
  std::string error;
};

//! Extract integer with error tracking
ExtractResult<int> extract_int_safe(
    const std::string& line, size_t start, size_t length) {
  ExtractResult<int> result{0, false, ""};

  if (line.length() <= start) {
    result.error = "Line too short for column " + std::to_string(start);
    return result;
  }

  size_t end = std::min(start + length, line.length());
  std::string substr = trim(line.substr(start, end - start));

  if (substr.empty()) {
    result.value = 0;
    result.success = true;  // Empty field is valid (means 0)
    return result;
  }

  try {
    result.value = std::stoi(substr);
    result.success = true;
  } catch (const std::exception& e) {
    result.error = "Cannot parse integer from '" + substr + "': " + e.what();
  }

  return result;
}

//! Extract double with error tracking
ExtractResult<double> extract_double_safe(
    const std::string& line, size_t start, size_t length) {
  ExtractResult<double> result{0.0, false, ""};

  if (line.length() <= start) {
    result.error = "Line too short for column " + std::to_string(start);
    return result;
  }

  size_t end = std::min(start + length, line.length());
  std::string substr = trim(line.substr(start, end - start));

  if (substr.empty()) {
    result.value = 0.0;
    result.success = true;
    return result;
  }

  // Replace ENDF format (e.g., "1.23+4" -> "1.23e+4")
  size_t plus_pos = substr.find_last_of("+-");
  if (plus_pos != std::string::npos && plus_pos > 0 &&
      (substr[plus_pos - 1] != 'e' && substr[plus_pos - 1] != 'E')) {
    substr.insert(plus_pos, "e");
  }

  try {
    result.value = std::stod(substr);
    result.success = true;
  } catch (const std::exception& e) {
    result.error = "Cannot parse double from '" + substr + "': " + e.what();
  }

  return result;
}

//! Store validated section data into result
//! \param[in] basename Filename for warning messages
//! \param[in] mt Current MT number
//! \param[in,out] xs Cross-section data (moved out)
//! \param[in,out] energies Energy boundary data (moved out)
//! \param[out] result Parse result to store into
void store_section_data(
  const std::string& basename,
  int mt,
  vector<double>& xs,
  vector<double>& energies,
  GENDFParseResult& result)
{
  if (xs.empty()) return;

  // Validate energy-XS count consistency (H3 fix)
  if (energies.size() != xs.size()) {
    result.warnings.push_back(
      "Energy-XS count mismatch in " + basename + " (MF=3, MT=" +
      std::to_string(mt) + "): " +
      std::to_string(energies.size()) + " energies vs " +
      std::to_string(xs.size()) + " XS values");
    size_t min_size = std::min(energies.size(), xs.size());
    energies.resize(min_size);
    xs.resize(min_size);
  }

  result.xs_data[mt] = std::move(xs);
  result.energy_data[mt] = std::move(energies);
}

} // anonymous namespace

//==============================================================================
// Validated GENDF parser
//==============================================================================

GENDFParseResult parse_gendf_validated(
  const std::string& filename,
  const GENDFParserOptions& options)
{
  GENDFParseResult result;

  // Check file exists
  if (!file_exists(filename)) {
    result.error_message = "GENDF file not found: " + filename;
    return result;
  }

  std::ifstream infile(filename);
  if (!infile.is_open()) {
    result.error_message = "Failed to open GENDF file: " + filename;
    return result;
  }

  std::string line;
  int line_number = 0;
  int current_mf = 0;
  int current_mt = 0;
  vector<double> current_xs;
  vector<double> current_energies;  // Energy boundaries for threshold alignment
  int n_groups = 0;
  bool in_data_section = false;
  int nr_lines_to_skip = 0;
  bool found_mf1_header = false;
  int max_short_line_warnings = 5;

  // MF=10 state tracking
  int mf10_current_lfs = 0;
  int mf10_current_izap = 0;
  vector<double> mf10_current_xs;
  vector<double> mf10_current_energies;
  int mf10_n_groups = 0;
  bool mf10_in_data = false;
  int mf10_nr_skip = 0;

  // Extract basename from filename for cleaner warnings
  std::string basename = filename;
  size_t pos = filename.find_last_of("/\\");
  if (pos != std::string::npos) {
    basename = filename.substr(pos + 1);
  }

  while (std::getline(infile, line)) {
    ++line_number;
    ++result.lines_read;

    // Track short lines
    if (line.length() < 70) {
      ++result.lines_skipped;
      if (options.warn_short_lines && result.lines_skipped <= max_short_line_warnings) {
        result.warnings.push_back(
          "Short line in " + basename + " at line " +
          std::to_string(line_number) + " (" +
          std::to_string(line.length()) + " chars), skipped");
      }
      continue;
    }

    // Extract MF/MT with safe parsing
    auto mf_result = extract_int_safe(line, 70, 2);
    auto mt_result = extract_int_safe(line, 72, 3);

    if (!mf_result.success || !mt_result.success) {
      result.warnings.push_back(
        "Line " + std::to_string(line_number) +
        ": Could not parse MF/MT fields");
      continue;
    }

    int mf = mf_result.value;
    int mt = mt_result.value;

    // Filter to MF=1, MF=3, and optionally MF=10
    if (mf != 1 && mf != 3 && (mf != 10 || !options.parse_mf10)) {
      continue;
    }

    // Track section changes
    if (mf != 0 && mt != 0) {
      if (mf != current_mf || mt != current_mt) {
        // Save previous MF=3 section
        if (current_mf == 3) {
          store_section_data(basename, current_mt, current_xs, current_energies, result);
          current_xs.clear();
          current_energies.clear();
        }
        // Save previous MF=10 subsection
        if (current_mf == 10 && !mf10_current_xs.empty()) {
          int key = current_mt * 1000 + mf10_current_lfs;
          result.prod_xs_data[key] = std::move(mf10_current_xs);
          result.prod_izap_data[key] = mf10_current_izap;
          mf10_current_xs.clear();
          mf10_current_energies.clear();
          mf10_in_data = false;
        }
        current_mf = mf;
        current_mt = mt;
        in_data_section = false;
        nr_lines_to_skip = 0;
        mf10_in_data = false;
        mf10_nr_skip = 0;
      }
    }

    // Parse MF=1, MT=451 header (only first record is HEAD, rest are TEXT)
    // Per ENDF-6 format: only the first record contains actual ZA data,
    // subsequent records are documentation where columns 1-66 are free-form text
    if (mf == 1 && mt == 451 && !found_mf1_header) {
      found_mf1_header = true;

      auto za_result = extract_int_safe(line, 0, 11);
      if (za_result.success && za_result.value > 0) {
        result.za = za_result.value;

        // Validate ZA (only on HEAD record, not TEXT records)
        if (options.validate_za) {
          std::string za_error;
          if (!validate_za(result.za, za_error)) {
            result.warnings.push_back(
              "ZA validation in " + basename + " (MF=1, MT=451): " + za_error);
          }
        }

        if (result.zam == 0) {
          result.zam = result.za;
        }
      }
    }

    // Parse MF=3 cross-section data
    if (mf == 3) {
      if (!in_data_section && line.length() >= 55) {
        auto nr_result = extract_int_safe(line, 44, 11);
        auto np_result = extract_int_safe(line, 55, 11);

        if (np_result.success && np_result.value > 0) {
          n_groups = np_result.value;
          in_data_section = true;
          nr_lines_to_skip = nr_result.success ? nr_result.value : 0;
          current_xs.clear();
          current_xs.reserve(n_groups);
          current_energies.clear();
          current_energies.reserve(n_groups + 1);  // n_groups + 1 energy boundaries
          continue;
        }
      }

      if (in_data_section && nr_lines_to_skip > 0) {
        --nr_lines_to_skip;
        continue;
      }

      // Extract energy and XS values with validation
      // ENDF TAB1 format: alternating Energy, XS pairs
      // Even indices (0, 2, 4) = energy values
      // Odd indices (1, 3, 5) = cross-section values
      if (in_data_section && nr_lines_to_skip == 0 && line.length() >= 66) {
        for (int i = 0; i < 6 && current_xs.size() < static_cast<size_t>(n_groups); ++i) {
          int col_start = i * 11;
          if (col_start + 11 <= 66) {
            auto val_result = extract_double_safe(line, col_start, 11);

            if (i % 2 == 0) {  // Energy values at even indices
              if (val_result.success) {
                current_energies.push_back(val_result.value);
              } else {
                // Could not parse energy - use 0 but track
                current_energies.push_back(0.0);
                result.warnings.push_back(
                  "Parse error (energy) in " + basename + " (MF=3, MT=" +
                  std::to_string(current_mt) + ") at line " +
                  std::to_string(line_number) + ": " + val_result.error);
              }
            } else {  // XS values at odd indices
              if (val_result.success) {
                double xs_val = val_result.value;

                // Validate non-negative
                if (options.validate_xs_positive && xs_val < 0) {
                  ++result.negative_xs_count;
                  if (result.negative_xs_count <= 3) {
                    // Use scientific notation to show actual value
                    std::ostringstream oss;
                    oss << std::scientific << xs_val;
                    result.warnings.push_back(
                      "Negative XS in " + basename + " (MF=3, MT=" +
                      std::to_string(current_mt) + "): value=" +
                      oss.str() + " at line " + std::to_string(line_number));
                  }
                  xs_val = 0.0;  // Clamp to zero
                }

                current_xs.push_back(xs_val);
              } else {
                // Could not parse - use 0 but track
                current_xs.push_back(0.0);
                result.warnings.push_back(
                  "Parse error in " + basename + " (MF=3, MT=" +
                  std::to_string(current_mt) + ") at line " +
                  std::to_string(line_number) + ": " + val_result.error);
              }
            }
          }
        }
      }
    }

    // Parse MF=10 production cross-section data
    // Same TAB1 format as MF=3 but multiple subsections per MT (one per LFS).
    // Level HEAD record: [QM, QI, IZAP, LFS, NR, NP]
    if (mf == 10) {
      if (!mf10_in_data && line.length() >= 55) {
        // Check for subsection HEAD record (has NP > 0 in cols 55-65)
        auto np_result = extract_int_safe(line, 55, 11);
        if (np_result.success && np_result.value > 0) {
          // Save previous subsection if any
          if (!mf10_current_xs.empty()) {
            int key = current_mt * 1000 + mf10_current_lfs;
            result.prod_xs_data[key] = std::move(mf10_current_xs);
            result.prod_izap_data[key] = mf10_current_izap;
            mf10_current_xs.clear();
            mf10_current_energies.clear();
          }

          // Parse HEAD fields: IZAP (cols 22-32), LFS (cols 33-43)
          auto izap_result = extract_int_safe(line, 22, 11);
          auto lfs_result = extract_int_safe(line, 33, 11);
          auto nr_result = extract_int_safe(line, 44, 11);

          mf10_current_izap = izap_result.success ? izap_result.value : 0;
          mf10_current_lfs = lfs_result.success ? lfs_result.value : 0;
          mf10_n_groups = np_result.value;
          mf10_nr_skip = nr_result.success ? nr_result.value : 0;

          // Skip subsections with IZAP=0 (known data quality issue)
          if (mf10_current_izap == 0) {
            result.warnings.push_back(
              "Skipping MF=10 level in " + basename + " MT=" +
              std::to_string(current_mt) + " LFS=" +
              std::to_string(mf10_current_lfs) + ": IZAP=0");
            continue;
          }

          // Validate IZAP
          if (options.validate_za) {
            std::string izap_error;
            if (!validate_za(mf10_current_izap, izap_error)) {
              result.warnings.push_back(
                "MF=10 IZAP validation in " + basename + " MT=" +
                std::to_string(current_mt) + " LFS=" +
                std::to_string(mf10_current_lfs) + ": " + izap_error);
            }
          }

          mf10_in_data = true;
          mf10_current_xs.clear();
          mf10_current_xs.reserve(mf10_n_groups);
          mf10_current_energies.clear();
          mf10_current_energies.reserve(mf10_n_groups + 1);
          continue;
        }
      }

      if (mf10_in_data && mf10_nr_skip > 0) {
        --mf10_nr_skip;
        continue;
      }

      // Parse energy/XS pairs (same format as MF=3)
      if (mf10_in_data && mf10_nr_skip == 0 && line.length() >= 66) {
        for (int i = 0; i < 6 && mf10_current_xs.size() < static_cast<size_t>(mf10_n_groups); ++i) {
          int col_start = i * 11;
          if (col_start + 11 <= 66) {
            auto val_result = extract_double_safe(line, col_start, 11);
            if (i % 2 == 0) {
              // Energy
              mf10_current_energies.push_back(
                  val_result.success ? val_result.value : 0.0);
            } else {
              // Production XS
              double xs_val = val_result.success ? val_result.value : 0.0;
              if (options.validate_xs_positive && xs_val < 0) {
                xs_val = 0.0;
              }
              mf10_current_xs.push_back(xs_val);
            }
          }
        }

        // When subsection is complete, reset so next HEAD is detected
        if (mf10_current_xs.size() >= static_cast<size_t>(mf10_n_groups)) {
          mf10_in_data = false;
        }
      }
    }
  }

  // Save last section
  if (current_mf == 3) {
    store_section_data(basename, current_mt, current_xs, current_energies, result);
  }

  // Save last MF=10 subsection
  if (current_mf == 10 && !mf10_current_xs.empty()) {
    int key = current_mt * 1000 + mf10_current_lfs;
    result.prod_xs_data[key] = std::move(mf10_current_xs);
    result.prod_izap_data[key] = mf10_current_izap;
  }

  infile.close();

  // Final validation
  if (result.lines_read < options.min_file_lines) {
    result.error_message = "File too small: only " +
      std::to_string(result.lines_read) + " lines (minimum: " +
      std::to_string(options.min_file_lines) + ")";
    return result;
  }

  if (options.require_mf1_header && !found_mf1_header) {
    result.error_message = "No MF=1, MT=451 header found in file";
    return result;
  }

  if (result.xs_data.empty()) {
    result.error_message = "No MF=3 cross-section data found";
    return result;
  }

  if (result.lines_skipped > result.lines_read / 2) {
    result.warnings.push_back(
      "More than 50% of lines were skipped (" +
      std::to_string(result.lines_skipped) + "/" +
      std::to_string(result.lines_read) + ")");
  }

  if (result.negative_xs_count > 3) {
    result.warnings.push_back(
      "Total " + std::to_string(result.negative_xs_count) +
      " negative XS values clamped to zero");
  }

  result.success = true;
  return result;
}

//==============================================================================
// MF=3-only GENDF parser (uses validated parser with optimized options)
//==============================================================================

void parse_gendf_mf3_only(
  const std::string& filename,
  std::unordered_map<int, vector<double>>& xs_data,
  std::unordered_map<int, vector<double>>& energy_data,
  int& za,
  int& zam)
{
  // Use validated parser with minimal warnings for performance
  // This ensures consistent behavior and validation across both entry points
  GENDFParserOptions options;
  options.warn_short_lines = false;  // Don't accumulate short line warnings
  options.validate_za = true;        // Keep ZA validation
  options.validate_xs_positive = true; // Keep XS validation
  options.require_mf1_header = true;
  options.parse_mf10 = false;        // MF=3 only for speed
  options.min_file_lines = 10;

  GENDFParseResult result = parse_gendf_validated(filename, options);

  if (!result.success) {
    throw std::runtime_error(result.error_message);
  }

  // Log warnings if any
  for (const auto& warn : result.warnings) {
    warning(warn);
  }

  // Move results to output parameters
  xs_data = std::move(result.xs_data);
  energy_data = std::move(result.energy_data);
  za = result.za;
  zam = result.zam;
}

} // namespace openmc
