#include "openmc/tallies/tally.h"

#include "openmc/array.h"
#include "openmc/capi.h"
#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/container_util.h"
#include "openmc/error.h"
#include "openmc/file_utils.h"
#include "openmc/mesh.h"
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/reaction.h"
#include "openmc/reaction_product.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/source.h"
#include "openmc/tallies/derivative.h"
#include "openmc/tallies/filter.h"
#include "openmc/tallies/filter_cell.h"
#include "openmc/tallies/filter_cellborn.h"
#include "openmc/tallies/filter_cellfrom.h"
#include "openmc/tallies/filter_collision.h"
#include "openmc/tallies/filter_delayedgroup.h"
#include "openmc/tallies/filter_energy.h"
#include "openmc/tallies/filter_legendre.h"
#include "openmc/tallies/filter_mesh.h"
#include "openmc/tallies/filter_meshborn.h"
#include "openmc/tallies/filter_meshmaterial.h"
#include "openmc/tallies/filter_meshsurface.h"
#include "openmc/tallies/filter_particle.h"
#include "openmc/tallies/filter_sph_harm.h"
#include "openmc/tallies/filter_surface.h"
#include "openmc/tallies/filter_time.h"
#include "openmc/xml_interface.h"

#include "openmc/tensor.h"
#include <fmt/core.h>

#include <algorithm> // for max, set_union, clamp
#include <cassert>
#include <cstddef>  // for size_t
#include <cstdint>  // for uintptr_t
#include <cstdlib>  // for getenv, atoll
#include <iterator> // for back_inserter
#include <string>

namespace openmc {

//==============================================================================
// Global variable definitions
//==============================================================================

namespace model {
//! a mapping of tally ID to index in the tallies vector
std::unordered_map<int, int> tally_map;
vector<unique_ptr<Tally>> tallies;
vector<int> active_tallies;
vector<int> active_analog_tallies;
vector<int> active_tracklength_tallies;
vector<int> active_timed_tracklength_tallies;
vector<int> active_collision_tallies;
vector<int> active_meshsurf_tallies;
vector<int> active_surface_tallies;
vector<int> active_pulse_height_tallies;
vector<int32_t> pulse_height_cells;
vector<double> time_grid;
} // namespace model

namespace simulation {
tensor::StaticTensor2D<double, N_GLOBAL_TALLIES, 3> global_tallies;
int32_t n_realizations {0};
} // namespace simulation

double global_tally_absorption;
double global_tally_collision;
double global_tally_tracklength;
double global_tally_leakage;

//==============================================================================
// Tally object implementation
//==============================================================================

Tally::Tally(int32_t id)
{
  index_ = model::tallies.size(); // Avoids warning about narrowing
  this->set_id(id);
  this->set_filters({});
}

Tally::Tally(pugi::xml_node node)
{
  index_ = model::tallies.size(); // Avoids warning about narrowing

  // Copy and set tally id
  if (!check_for_node(node, "id")) {
    throw std::runtime_error {"Must specify id for tally in tally XML file."};
  }
  int32_t id = std::stoi(get_node_value(node, "id"));
  this->set_id(id);

  if (check_for_node(node, "name"))
    name_ = get_node_value(node, "name");

  if (check_for_node(node, "multiply_density")) {
    multiply_density_ = get_node_value_bool(node, "multiply_density");
  }

  if (check_for_node(node, "higher_moments")) {
    higher_moments_ = get_node_value_bool(node, "higher_moments");
  }
  // =======================================================================
  // READ DATA FOR FILTERS

  // Check if user is using old XML format and throw an error if so
  if (check_for_node(node, "filter")) {
    throw std::runtime_error {
      "Tally filters must be specified independently of "
      "tallies in a <filter> element. The <tally> element itself should "
      "have a list of filters that apply, e.g., <filters>1 2</filters> "
      "where 1 and 2 are the IDs of filters specified outside of "
      "<tally>."};
  }

  // Determine number of filters
  vector<int> filter_ids;
  if (check_for_node(node, "filters")) {
    filter_ids = get_node_array<int>(node, "filters");
  }

  // Allocate and store filter user ids
  vector<Filter*> filters;
  for (int filter_id : filter_ids) {
    // Determine if filter ID is valid
    auto it = model::filter_map.find(filter_id);
    if (it == model::filter_map.end()) {
      throw std::runtime_error {fmt::format(
        "Could not find filter {} specified on tally {}", filter_id, id_)};
    }

    // Store the index of the filter
    filters.push_back(model::tally_filters[it->second].get());
  }

  // Set the filters
  this->set_filters(filters);

  // Check for the presence of certain filter types
  bool has_energyout = energyout_filter_ >= 0;
  int particle_filter_index = C_NONE;
  for (int64_t j = 0; j < filters_.size(); ++j) {
    int i_filter = filters_[j];
    const auto& f = model::tally_filters[i_filter].get();

    auto pf = dynamic_cast<ParticleFilter*>(f);
    if (pf)
      particle_filter_index = i_filter;

    // Change the tally estimator if a filter demands it
    FilterType filt_type = f->type();
    if (filt_type == FilterType::ENERGY_OUT ||
        filt_type == FilterType::LEGENDRE) {
      estimator_ = TallyEstimator::ANALOG;
    } else if (filt_type == FilterType::SPHERICAL_HARMONICS) {
      auto sf = dynamic_cast<SphericalHarmonicsFilter*>(f);
      if (sf->cosine() == SphericalHarmonicsCosine::scatter) {
        estimator_ = TallyEstimator::ANALOG;
      }
    } else if (filt_type == FilterType::SPATIAL_LEGENDRE ||
               filt_type == FilterType::ZERNIKE ||
               filt_type == FilterType::ZERNIKE_RADIAL) {
      estimator_ = TallyEstimator::COLLISION;
    } else if (filt_type == FilterType::PARTICLE_PRODUCTION) {
      estimator_ = TallyEstimator::ANALOG;
    } else if (filt_type == FilterType::REACTION) {
      if (estimator_ == TallyEstimator::TRACKLENGTH) {
        estimator_ = TallyEstimator::COLLISION;
      }
    }
  }

  // =======================================================================
  // READ DATA FOR NUCLIDES

  this->set_nuclides(node);

  // =======================================================================
  // READ DATA FOR SCORES

  this->set_scores(node);

  if (!check_for_node(node, "scores")) {
    fatal_error(fmt::format("No scores specified on tally {}.", id_));
  }

  // Set IFP if needed
  if (!settings::ifp_on) {
    // Determine if this tally has an IFP score
    bool has_ifp_score = false;
    for (int score : scores_) {
      if (score == SCORE_IFP_TIME_NUM || score == SCORE_IFP_BETA_NUM ||
          score == SCORE_IFP_DENOM) {
        has_ifp_score = true;
        break;
      }
    }

    // Check for errors
    if (has_ifp_score) {
      if (settings::run_mode == RunMode::EIGENVALUE) {
        if (settings::ifp_n_generation < 0) {
          settings::ifp_n_generation = DEFAULT_IFP_N_GENERATION;
          warning(fmt::format(
            "{} generations will be used for IFP (default value). It can be "
            "changed using the 'ifp_n_generation' settings.",
            settings::ifp_n_generation));
        }
        if (settings::ifp_n_generation > settings::n_inactive) {
          fatal_error("'ifp_n_generation' must be lower than or equal to the "
                      "number of inactive cycles.");
        }
        settings::ifp_on = true;
      } else if (settings::run_mode == RunMode::FIXED_SOURCE) {
        fatal_error(
          "Iterated Fission Probability can only be used in an eigenvalue "
          "calculation.");
      }
    }
  }

  // Set IFP parameters if needed
  if (settings::ifp_on) {
    for (int score : scores_) {
      switch (score) {
      case SCORE_IFP_TIME_NUM:
        if (settings::ifp_parameter == IFPParameter::None) {
          settings::ifp_parameter = IFPParameter::GenerationTime;
        } else if (settings::ifp_parameter == IFPParameter::BetaEffective) {
          settings::ifp_parameter = IFPParameter::Both;
        }
        break;
      case SCORE_IFP_BETA_NUM:
      case SCORE_IFP_DENOM:
        if (settings::ifp_parameter == IFPParameter::None) {
          settings::ifp_parameter = IFPParameter::BetaEffective;
        } else if (settings::ifp_parameter == IFPParameter::GenerationTime) {
          settings::ifp_parameter = IFPParameter::Both;
        }
        break;
      }
    }
  }

  // Check if tally is compatible with particle type
  if (!settings::photon_transport) {
    for (int score : scores_) {
      switch (score) {
      case SCORE_PULSE_HEIGHT:
        fatal_error("For pulse-height tallies, photon transport needs to be "
                    "activated.");
        break;
      }
    }
  }
  if (settings::photon_transport) {
    if (particle_filter_index == C_NONE) {
      for (int score : scores_) {
        switch (score) {
        case SCORE_INVERSE_VELOCITY:
          fatal_error("Particle filter must be used with photon "
                      "transport on and inverse velocity score");
          break;
        case SCORE_FLUX:
        case SCORE_TOTAL:
        case SCORE_SCATTER:
        case SCORE_NU_SCATTER:
        case SCORE_ABSORPTION:
        case SCORE_FISSION:
        case SCORE_NU_FISSION:
        case SCORE_CURRENT:
        case SCORE_EVENTS:
        case SCORE_DELAYED_NU_FISSION:
        case SCORE_PROMPT_NU_FISSION:
        case SCORE_DECAY_RATE:
          warning("You are tallying the '" + reaction_name(score) +
                  "' score and haven't used a particle filter. This score will "
                  "include contributions from all particles.");
          break;
        }
      }
    }
  } else {
    if (particle_filter_index >= 0) {
      const auto& f = model::tally_filters[particle_filter_index].get();
      auto pf = dynamic_cast<ParticleFilter*>(f);
      for (auto p : pf->particles()) {
        if (!p.is_neutron()) {
          warning(fmt::format(
            "Particle filter other than NEUTRON used with "
            "photon transport turned off. All tallies for particle type {}"
            " will have no scores",
            p.str()));
        }
      }
    }
  }

  // Check for a tally derivative.
  if (check_for_node(node, "derivative")) {
    int deriv_id = std::stoi(get_node_value(node, "derivative"));

    // Find the derivative with the given id, and store it's index.
    auto it = model::tally_deriv_map.find(deriv_id);
    if (it == model::tally_deriv_map.end()) {
      fatal_error(fmt::format(
        "Could not find derivative {} specified on tally {}", deriv_id, id_));
    }

    deriv_ = it->second;

    // Only analog or collision estimators are supported for differential
    // tallies.
    if (estimator_ == TallyEstimator::TRACKLENGTH) {
      estimator_ = TallyEstimator::COLLISION;
    }

    const auto& deriv = model::tally_derivs[deriv_];
    if (deriv.variable == DerivativeVariable::NUCLIDE_DENSITY ||
        deriv.variable == DerivativeVariable::TEMPERATURE) {
      for (int i_nuc : nuclides_) {
        if (has_energyout && i_nuc == -1) {
          fatal_error(fmt::format(
            "Error on tally {}: Cannot use a "
            "'nuclide_density' or 'temperature' derivative on a tally with "
            "an "
            "outgoing energy filter and 'total' nuclide rate. Instead, tally "
            "each nuclide in the material individually.",
            id_));
          // Note that diff tallies with these characteristics would work
          // correctly if no tally events occur in the perturbed material
          // (e.g. pertrubing moderator but only tallying fuel), but this
          // case would be hard to check for by only reading inputs.
        }
      }
    }
  }

  // If settings.xml trigger is turned on, create tally triggers
  if (settings::trigger_on) {
    this->init_triggers(node);
  }

  // =======================================================================
  // SET TALLY ESTIMATOR

  // Check if user specified estimator
  if (check_for_node(node, "estimator")) {
    std::string est = get_node_value(node, "estimator");
    if (est == "analog") {
      estimator_ = TallyEstimator::ANALOG;
    } else if (est == "tracklength" || est == "track-length" ||
               est == "pathlength" || est == "path-length") {
      // If the estimator was set to an analog estimator, this means the
      // tally needs post-collision information
      if (estimator_ == TallyEstimator::ANALOG ||
          estimator_ == TallyEstimator::COLLISION) {
        throw std::runtime_error {fmt::format("Cannot use track-length "
                                              "estimator for tally {}",
          id_)};
      }

      // Set estimator to track-length estimator
      estimator_ = TallyEstimator::TRACKLENGTH;

    } else if (est == "collision") {
      // If the estimator was set to an analog estimator, this means the
      // tally needs post-collision information
      if (estimator_ == TallyEstimator::ANALOG) {
        throw std::runtime_error {fmt::format("Cannot use collision estimator "
                                              "for tally ",
          id_)};
      }

      // Set estimator to collision estimator
      estimator_ = TallyEstimator::COLLISION;

    } else {
      throw std::runtime_error {
        fmt::format("Invalid estimator '{}' on tally {}", est, id_)};
    }
  }

  // Resolve the storage mode: a per-tally <storage> element overrides the
  // global settings::tally_storage default (the same idiom as <estimator>).
  storage_ = settings::tally_storage;
  if (check_for_node(node, "storage")) {
    std::string storage = get_node_value(node, "storage", true, true);
    if (storage == "replicated") {
      storage_ = TallyStorage::REPLICATED;
    } else if (storage == "shared") {
      storage_ = TallyStorage::SHARED;
    } else if (storage == "rma") {
      storage_ = TallyStorage::RMA;
    } else {
      throw std::runtime_error {
        fmt::format("Invalid storage mode '{}' on tally {}", storage, id_)};
    }
  }

  // Validate the storage mode. The distributed modes carry hard prerequisites.
  if (storage_ != TallyStorage::REPLICATED) {
#ifndef OPENMC_MPI
    fatal_error(fmt::format("Tally {} requests a non-replicated storage mode, "
                            "which requires an MPI-enabled build.",
      id_));
#endif
#ifndef _OPENMP
    // Shared scoring updates one plane from several ranks through omp atomic;
    // with OpenMP disabled that pragma is a no-op and the updates would race.
    fatal_error(fmt::format("Tally {} requests a non-replicated storage mode, "
                            "which requires an OpenMP-enabled build.",
      id_));
#endif
    if (!settings::reduce_tallies) {
      fatal_error(
        fmt::format("Tally {} cannot combine a non-replicated storage mode "
                    "with the no-reduction (no_reduce) option.",
          id_));
    }
    if (settings::solver_type != SolverType::MONTE_CARLO) {
      fatal_error(
        fmt::format("Tally {} requests a non-replicated storage mode, which is "
                    "only supported by the Monte Carlo solver.",
          id_));
    }
    if (storage_ == TallyStorage::RMA) {
#if defined(OPENMC_MPI) && defined(_OPENMP)
      // The remote-scoring arm issues MPI from inside the OpenMP scoring
      // region. Until MPI is initialized with MPI_THREAD_SERIALIZED, reject a
      // threaded rma run whose MPI cannot serialize those calls; a
      // single-thread run is safe regardless of the provided level.
      int provided;
      MPI_Query_thread(&provided);
      if (provided < MPI_THREAD_SERIALIZED && num_threads() > 1) {
        fatal_error(fmt::format(
          "Tally {} requests storage mode 'rma' with more than one OpenMP "
          "thread, but MPI does not provide MPI_THREAD_SERIALIZED. Run with a "
          "single thread or an MPI build providing thread serialization.",
          id_));
      }
#endif
      // Event-based transport reorders scoring; the rma coalescer assumes the
      // history-based order.
      if (settings::event_based) {
        fatal_error(
          fmt::format("Tally {} requests storage mode 'rma', which is "
                      "not supported in event-based mode.",
            id_));
      }
    }
  }

#ifdef OPENMC_LIBMESH_ENABLED
  // ensure a tracklength tally isn't used with a libMesh filter
  for (auto i : this->filters_) {
    auto df = dynamic_cast<MeshFilter*>(model::tally_filters[i].get());
    if (df) {
      auto lm = dynamic_cast<LibMesh*>(model::meshes[df->mesh()].get());
      if (lm && estimator_ == TallyEstimator::TRACKLENGTH) {
        fatal_error("A tracklength estimator cannot be used with "
                    "an unstructured LibMesh tally.");
      }
    }
  }
#endif
}

Tally::~Tally()
{
  model::tally_map.erase(id_);
#ifdef OPENMC_MPI
  free_accum_window();
#endif
}

Tally* Tally::create(int32_t id)
{
  model::tallies.push_back(make_unique<Tally>(id));
  return model::tallies.back().get();
}

void Tally::set_id(int32_t id)
{
  assert(id >= 0 || id == C_NONE);

  // Clear entry in tally map if an ID was already assigned before
  if (id_ != C_NONE) {
    model::tally_map.erase(id_);
    id_ = C_NONE;
  }

  // Make sure no other tally has the same ID
  if (model::tally_map.find(id) != model::tally_map.end()) {
    throw std::runtime_error {
      fmt::format("Two tallies have the same ID: {}", id)};
  }

  // If no ID specified, auto-assign next ID in sequence
  if (id == C_NONE) {
    id = 0;
    for (const auto& t : model::tallies) {
      id = std::max(id, t->id_);
    }
    ++id;
  }

  // Update ID and entry in tally map
  id_ = id;
  model::tally_map[id] = index_;
}

std::vector<FilterType> Tally::filter_types() const
{
  std::vector<FilterType> filter_types;
  for (auto idx : this->filters())
    filter_types.push_back(model::tally_filters[idx]->type());
  return filter_types;
}

std::unordered_map<FilterType, int32_t> Tally::filter_indices() const
{
  std::unordered_map<FilterType, int32_t> filter_indices;
  for (int i = 0; i < this->filters().size(); i++) {
    const auto& f = model::tally_filters[this->filters(i)];

    filter_indices[f->type()] = i;
  }
  return filter_indices;
}

bool Tally::has_filter(FilterType filter_type) const
{
  for (auto idx : this->filters()) {
    if (model::tally_filters[idx]->type() == filter_type)
      return true;
  }
  return false;
}

void Tally::set_filters(span<Filter*> filters)
{
  // Clear old data.
  filters_.clear();
  strides_.clear();

  // Copy in the given filter indices.
  auto n = filters.size();
  filters_.reserve(n);

  for (auto* filter : filters) {
    add_filter(filter);
  }
}

void Tally::add_filter(Filter* filter)
{
  int32_t filter_idx = model::filter_map.at(filter->id());
  // if this filter is already present, do nothing and return
  if (std::find(filters_.begin(), filters_.end(), filter_idx) != filters_.end())
    return;

  // Keep track of indices for special filters
  if (filter->type() == FilterType::ENERGY_OUT) {
    energyout_filter_ = filters_.size();
  } else if (filter->type() == FilterType::DELAYED_GROUP) {
    delayedgroup_filter_ = filters_.size();
  }
  filters_.push_back(filter_idx);
}

void Tally::set_strides()
{
  // Set the strides.  Filters are traversed in reverse so that the last
  // filter has the shortest stride in memory and the first filter has the
  // longest stride.
  auto n = filters_.size();
  strides_.resize(n, 0);
  // int64 accumulator: the product of per-filter bin counts can exceed 2^31
  // for large tallies (e.g. a fine mesh crossed with a fine energy filter).
  int64_t stride = 1;
  for (int i = n - 1; i >= 0; --i) {
    strides_[i] = stride;
    stride *= model::tally_filters[filters_[i]]->n_bins();
  }
  n_filter_bins_ = stride;
}

void Tally::set_scores(pugi::xml_node node)
{
  if (!check_for_node(node, "scores"))
    fatal_error(fmt::format("No scores specified on tally {}", id_));

  auto scores = get_node_array<std::string>(node, "scores");
  set_scores(scores);
}

void Tally::set_scores(const vector<std::string>& scores)
{
  // Reset state and prepare for the new scores.
  scores_.clear();
  scores_.reserve(scores.size());

  // Check for the presence of certain restrictive filters.
  bool energyout_present = energyout_filter_ != C_NONE;
  bool legendre_present = false;
  bool cell_present = false;
  bool cellfrom_present = false;
  bool material_present = false;
  bool materialfrom_present = false;
  bool surface_present = false;
  bool meshsurface_present = false;
  bool non_cell_energy_present = false;
  for (auto i_filt : filters_) {
    const auto* filt {model::tally_filters[i_filt].get()};
    // Checking for only cell and energy filters for pulse-height tally
    if (!(filt->type() == FilterType::CELL ||
          filt->type() == FilterType::ENERGY)) {
      non_cell_energy_present = true;
    }
    if (filt->type() == FilterType::LEGENDRE) {
      legendre_present = true;
    } else if (filt->type() == FilterType::CELLFROM) {
      cellfrom_present = true;
    } else if (filt->type() == FilterType::CELL) {
      cell_present = true;
    } else if (filt->type() == FilterType::MATERIALFROM) {
      materialfrom_present = true;
    } else if (filt->type() == FilterType::MATERIAL) {
      material_present = true;
    } else if (filt->type() == FilterType::SURFACE) {
      surface_present = true;
    } else if (filt->type() == FilterType::MESH_SURFACE) {
      meshsurface_present = true;
    }
  }
  bool surface_types_present =
    (surface_present || cellfrom_present || materialfrom_present);
  bool non_meshsurface_types_present =
    (surface_present || cell_present || cellfrom_present || material_present ||
      materialfrom_present);

  // Iterate over the given scores.
  for (auto score_str : scores) {
    // Make sure a delayed group filter wasn't used with an incompatible
    // score.
    if (delayedgroup_filter_ != C_NONE) {
      if (score_str != "delayed-nu-fission" && score_str != "decay-rate" &&
          score_str != "ifp-beta-numerator")
        fatal_error("Cannot tally " + score_str + "with a delayedgroup filter");
    }

    // Determine integer code for score
    int score = reaction_tally_mt(score_str);

    switch (score) {
    case SCORE_FLUX:
      if (!nuclides_.empty())
        if (!(nuclides_.size() == 1 && nuclides_[0] == -1))
          fatal_error("Cannot tally flux for an individual nuclide.");
      if (energyout_present)
        fatal_error("Cannot tally flux with an outgoing energy filter.");
      if (surface_types_present) {
        if (meshsurface_present)
          fatal_error("OpenMC does not support mesh surface fluxes yet");
        type_ = TallyType::SURFACE;
        estimator_ = TallyEstimator::ANALOG;
      }
      break;

    case SCORE_TOTAL:
    case SCORE_ABSORPTION:
    case SCORE_FISSION:
      if (energyout_present)
        fatal_error("Cannot tally " + score_str +
                    " reaction rate with an "
                    "outgoing energy filter");
      break;

    case SCORE_SCATTER:
      if (legendre_present)
        estimator_ = TallyEstimator::ANALOG;
    case SCORE_NU_FISSION:
    case SCORE_DELAYED_NU_FISSION:
    case SCORE_PROMPT_NU_FISSION:
      if (energyout_present)
        estimator_ = TallyEstimator::ANALOG;
      break;

    case SCORE_NU_SCATTER:
      if (settings::run_CE) {
        estimator_ = TallyEstimator::ANALOG;
      } else {
        if (energyout_present || legendre_present)
          estimator_ = TallyEstimator::ANALOG;
      }
      break;

    case SCORE_CURRENT:
      // Check which type of current is desired: mesh or surface currents.
      if (meshsurface_present) {
        if (non_meshsurface_types_present)
          fatal_error("Cannot tally mesh surface currents in the same tally as "
                      "normal surface currents");
        type_ = TallyType::MESH_SURFACE;
      } else {
        type_ = TallyType::SURFACE;
        estimator_ = TallyEstimator::ANALOG;
      }
      break;

    case HEATING:
      if (settings::photon_transport) {
        // Photon heating requires a collision estimator (analog energy
        // balance). However, if the tally only scores neutrons, we can keep the
        // tracklength estimator since neutron heating uses kerma coefficients
        // that support tracklength scoring.
        bool neutron_only = false;
        for (auto i_filt : filters_) {
          auto pf =
            dynamic_cast<ParticleFilter*>(model::tally_filters[i_filt].get());
          if (pf && pf->particles().size() == 1 &&
              pf->particles()[0].is_neutron()) {
            neutron_only = true;
            break;
          }
        }
        if (!neutron_only)
          estimator_ = TallyEstimator::COLLISION;
      }
      break;

    case SCORE_PULSE_HEIGHT: {
      if (non_cell_energy_present) {
        fatal_error("Pulse-height tallies are not compatible with filters "
                    "other than CellFilter and EnergyFilter");
      }
      type_ = TallyType::PULSE_HEIGHT;
      // Collect all unique cell indices covered by this tally.
      // If no CellFilter is present, all cells in the geometry are scored.
      const auto* cell_filter_ptr = get_filter<CellFilter>();
      int n = cell_filter_ptr ? cell_filter_ptr->n_bins()
                              : static_cast<int>(model::cells.size());
      for (int i = 0; i < n; ++i) {
        int32_t cell_index = cell_filter_ptr ? cell_filter_ptr->cells()[i] : i;
        if (!contains(model::pulse_height_cells, cell_index))
          model::pulse_height_cells.push_back(cell_index);
      }
      break;
    }

    case SCORE_IFP_TIME_NUM:
    case SCORE_IFP_BETA_NUM:
    case SCORE_IFP_DENOM:
      estimator_ = TallyEstimator::COLLISION;
      break;
    }

    scores_.push_back(score);
  }

  // Make sure that no duplicate scores exist.
  for (auto it1 = scores_.begin(); it1 != scores_.end(); ++it1) {
    for (auto it2 = it1 + 1; it2 != scores_.end(); ++it2) {
      if (*it1 == *it2)
        fatal_error(
          fmt::format("Duplicate score of type \"{}\" found in tally {}",
            reaction_name(*it1), id_));
    }
  }

  // Make sure all scores are compatible with multigroup mode.
  if (!settings::run_CE) {
    for (auto sc : scores_)
      if (sc > 0)
        fatal_error("Cannot tally " + reaction_name(sc) +
                    " reaction rate "
                    "in multi-group mode");
  }

  // Make sure mesh surface tallies contain only current score.
  if (meshsurface_present) {
    if ((scores_[0] != SCORE_CURRENT) || (scores_.size() > 1))
      fatal_error("Cannot tally score other than 'current' when using a "
                  "mesh-surface filter.");
  }

  // Make sure surface tallies contain only surface type scores score.
  if (type_ == TallyType::SURFACE) {
    for (auto sc : scores_)
      if ((sc != SCORE_CURRENT) && (sc != SCORE_FLUX))
        fatal_error("Cannot tally scores other than 'current' or 'flux' "
                    "when using surface filters.");
  }
}

void Tally::set_nuclides(pugi::xml_node node)
{
  nuclides_.clear();

  // By default, we tally just the total material rates.
  if (!check_for_node(node, "nuclides")) {
    nuclides_.push_back(-1);
    return;
  }

  // The user provided specifics nuclides.  Parse it as an array with either
  // "total" or a nuclide name like "U235" in each position.
  auto words = get_node_array<std::string>(node, "nuclides");
  this->set_nuclides(words);
}

void Tally::set_nuclides(const vector<std::string>& nuclides)
{
  nuclides_.clear();

  for (const auto& nuc : nuclides) {
    if (nuc == "total") {
      nuclides_.push_back(-1);
    } else {
      auto search = data::nuclide_map.find(nuc);
      if (search == data::nuclide_map.end()) {
        int err = openmc_load_nuclide(nuc.c_str(), nullptr, 0);
        if (err < 0)
          throw std::runtime_error {openmc_err_msg};
      }
      nuclides_.push_back(data::nuclide_map.at(nuc));
    }
  }
}

void Tally::init_triggers(pugi::xml_node node)
{
  for (auto trigger_node : node.children("trigger")) {
    // Read the trigger type.
    TriggerMetric metric;
    if (check_for_node(trigger_node, "type")) {
      auto type_str = get_node_value(trigger_node, "type");
      if (type_str == "std_dev") {
        metric = TriggerMetric::standard_deviation;
      } else if (type_str == "variance") {
        metric = TriggerMetric::variance;
      } else if (type_str == "rel_err") {
        metric = TriggerMetric::relative_error;
      } else {
        fatal_error(fmt::format(
          "Unknown trigger type \"{}\" in tally {}", type_str, id_));
      }
    } else {
      fatal_error(fmt::format(
        "Must specify trigger type for tally {} in tally XML file", id_));
    }

    // Read the trigger threshold.
    double threshold;
    if (check_for_node(trigger_node, "threshold")) {
      threshold = std::stod(get_node_value(trigger_node, "threshold"));
      if (threshold <= 0) {
        fatal_error("Tally trigger threshold must be positive");
      }
    } else {
      fatal_error(fmt::format(
        "Must specify trigger threshold for tally {} in tally XML file", id_));
    }

    // Read whether to allow zero-tally bins to be ignored.
    bool ignore_zeros = false;
    if (check_for_node(trigger_node, "ignore_zeros")) {
      ignore_zeros = get_node_value_bool(trigger_node, "ignore_zeros");
    }

    // Read the trigger scores.
    vector<std::string> trigger_scores;
    if (check_for_node(trigger_node, "scores")) {
      trigger_scores = get_node_array<std::string>(trigger_node, "scores");
    } else {
      trigger_scores.push_back("all");
    }

    // Parse the trigger scores and populate the triggers_ vector.
    for (auto score_str : trigger_scores) {
      if (score_str == "all") {
        triggers_.reserve(triggers_.size() + this->scores_.size());
        for (auto i_score = 0; i_score < this->scores_.size(); ++i_score) {
          triggers_.push_back({metric, threshold, ignore_zeros, i_score});
        }
      } else {
        int i_score = 0;
        for (; i_score < this->scores_.size(); ++i_score) {
          if (this->scores_[i_score] == reaction_tally_mt(score_str))
            break;
        }
        if (i_score == this->scores_.size()) {
          fatal_error(
            fmt::format("Could not find the score \"{}\" in tally "
                        "{} but it was listed in a trigger on that tally",
              score_str, id_));
        }
        triggers_.push_back({metric, threshold, ignore_zeros, i_score});
      }
    }
  }
}

#ifdef OPENMC_MPI
void Tally::rma_score_add(int64_t filter_index, int score_index, double val)
{
  // Local arm: a bin this rank owns is scored into the private owned-rows plane
  // (accum_), locally indexed. The window block itself receives only
  // MPI_Accumulate, so a NIC-side remote atomic never races a CPU store on the
  // same cell (the two classes target disjoint address ranges by construction).
  const int owner = rma_owner(filter_index);
  if (owner == mpi::rank) {
    int64_t local =
      (filter_index - rma_first_row_) * n_score_bins_ + score_index;
    atomic_score_add(&accum_[local], val);
    return;
  }

  // Remote arm (C2): coalesce consecutive contributions to a single filter bin
  // into a thread-local row, then stage rows for a batched MPI_Accumulate to
  // the owning rank. Design follows Dun et al. (2015): origin-side buffering of
  // one-sided accumulates cuts the per-score RMA cost. A remote bin implies
  // n_procs > 1, so init_rma_staging() has allocated a slot for every thread.
  // The state is per-thread, so nothing here races another thread; only the
  // accumulate itself is serialized (a critical).
  RmaThreadStaging& ts = rma_staging_[thread_num()];
  if (filter_index != ts.coalesce_bin) {
    if (ts.coalesce_bin >= 0)
      rma_coalescer_flush(ts);
    ts.coalesce_bin = filter_index;
    ts.coalesce_owner = owner;
    std::fill(ts.row.begin(), ts.row.end(), 0.0);
  }
  ts.row[score_index] += val;
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
    // reuse still owns an in-flight payload. flush_local_all retires every
    // origin buffer this rank has issued at once; clear this thread's flags to
    // match (other threads stay conservatively marked -- a redundant flush at
    // worst, never a reused-in-flight payload).
#pragma omp critical(openmc_rma)
    {
      MPI_Win_flush_local_all(accum_win_);
    }
    std::fill(ts.in_flight.begin(), ts.in_flight.end(), 0);
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
  for (int i = 0; i < f; ++i)
    ts.perm[i] = i;
  std::sort(ts.perm.begin(), ts.perm.begin() + f,
    [disp](int a, int b) { return disp[a] < disp[b]; });

  int u = 0;
  for (int i = 0; i < f; ++i) {
    const int src = ts.perm[i];
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
  MPI_Win_allocate_shared(
    bytes, sizeof(double), MPI_INFO_NULL, mpi::node_comm, &base, &accum_win_);

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
      "Shared tally storage requires an MPI_WIN_UNIFIED shared-memory "
      "window (non-cache-coherent hardware is not supported).");
  }

  // Open a persistent passive-target epoch for the window's whole life so the
  // win_sync memory barriers used at every batch boundary are legal MPI.
  MPI_Win_lock_all(MPI_MODE_NOCHECK, accum_win_);

  // Zero the plane on the leader and publish it so every rank sees all-zeros
  // before the first score.
  if (mpi::node_leader)
    std::fill(accum_, accum_ + accum_size_, 0.0);
  MPI_Win_sync(accum_win_);
  MPI_Barrier(mpi::node_comm);
  MPI_Win_sync(accum_win_);
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
  // folds it directly. World rank 0 is rank 0 of internode_comm, so
  // reduce_in_place_chunked roots the reduction on the master.
  if (mpi::n_nodes > 1 && mpi::node_leader) {
    reduce_in_place_chunked(accum_, accum_size_, mpi::internode_comm);
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

  // Distributed window: each rank contributes its owned block. same_op_no_op +
  // no ordering lets the implementation use hardware atomics for the SUM-only
  // accumulates and drops ordering overhead (we never mix ops or rely on
  // order).
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "accumulate_ops", "same_op_no_op");
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
    fatal_error("rma tally storage requires an MPI_WIN_UNIFIED window "
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
  // otherwise K comes from a fixed per-rank budget split across threads,
  // targets, and the two buffers, so total staging stays near the budget.
  const int nt = num_threads();
  if (const char* env = std::getenv("OPENMC_RMA_STAGING_ROWS")) {
    rma_k_ = std::max<int64_t>(1, std::atoll(env));
  } else {
    constexpr int64_t BUDGET = int64_t {64} << 20; // 64 MiB / rank
    const int64_t row_bytes =
      static_cast<int64_t>(n_score_bins_) * sizeof(double);
    const int64_t denom = static_cast<int64_t>(nt) * mpi::n_procs * 2 *
                          std::max<int64_t>(1, row_bytes);
    rma_k_ = std::clamp<int64_t>(BUDGET / std::max<int64_t>(1, denom), 8, 4096);
  }

  // One slot per thread; self-target slots are allocated but never used (the
  // local arm handles owned bins). Every per-thread buffer starts empty.
  const int64_t np = mpi::n_procs;
  rma_merged_rows_ = 0;
  rma_staging_.resize(nt);
  for (auto& ts : rma_staging_) {
    ts.row.assign(n_score_bins_, 0.0);
    ts.data.assign(np * 2 * rma_k_ * n_score_bins_, 0.0);
    ts.disp.assign(np * 2 * rma_k_, 0);
    ts.fill.assign(np * 2, 0);
    ts.in_flight.assign(np * 2, 0);
    ts.cur.assign(np, 0);
    // Per-thread compaction scratch (one buffer's worth), reused at each issue.
    ts.perm.assign(rma_k_, 0);
    ts.comp_disp.assign(rma_k_, 0);
    ts.comp_data.assign(rma_k_ * n_score_bins_, 0.0);
    ts.merged = 0;
  }

  write_message(8,
    "rma tally {}: staging K={} rows/buffer, {:.1f} MiB/rank "
    "({} threads x {} targets x 2 buffers)",
    id_, rma_k_,
    static_cast<double>(nt) * np * 2 * rma_k_ * n_score_bins_ * sizeof(double) /
      (1024.0 * 1024.0),
    nt, np);
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
    ts.merged = 0;
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
#endif

void Tally::init_results()
{
  n_score_bins_ = scores_.size() * nuclides_.size();
  accum_size_ = n_filter_bins_ * n_score_bins_;

#ifdef OPENMC_MPI
  // A tally read on all ranks (weight-window generation) cannot use a
  // distributed mode, which homes moments on the master alone (shared) or on
  // the owning rank (rma). This flag is set after construction, so the check
  // lives here rather than in the XML ctor.
  if (storage_ != TallyStorage::REPLICATED && moments_all_ranks_) {
    fatal_error(fmt::format("Tally {} cannot use a distributed storage mode "
                            "because its moments are required on all ranks.",
      id_));
  }

  if (storage_ == TallyStorage::SHARED) {
    // The per-batch accumulator is one shared plane per node rather than a
    // per-rank buffer.
    init_shared_accum();
  } else if (storage_ == TallyStorage::RMA) {
    // The accumulator and moments are block-distributed across ranks; each rank
    // allocates only its owned window block plus a private owned-rows plane.
    init_rma_accum();
  } else
#endif
  {
    // Per-batch accumulator plane. In replicated mode every rank owns a private
    // contiguous buffer.
    accum_buffer_.assign(accum_size_, 0.0);
    accum_ = accum_buffer_.data();
  }

  // Cross-batch moments, shaped [n_filter_bins, n_score_bins, n_moments]. The
  // moment count is 2 (SUM, SUM_SQ) or 4 (adding SUM_THIRD, SUM_FOURTH). This
  // is exactly the on-disk statepoint layout, so no VALUE column is stored.
  //
  // In reduced mode only the master folds and holds the moments during the run;
  // the other ranks receive a copy at the final broadcast. They therefore skip
  // the allocation here, cutting their per-tally footprint to just the
  // accumulator plane. The moments live on every rank when tallies are not
  // reduced (each rank is its own owner) or when a consumer reads them off the
  // master mid-run (moments_all_ranks_).
#ifdef OPENMC_MPI
  if (storage_ == TallyStorage::RMA) {
    // Under rma every rank folds and holds the moment rows for the filter bins
    // it owns; a rank owning zero rows gets an empty (size-0) array.
    moments_ = tensor::Tensor<double>({static_cast<size_t>(rma_n_rows_),
      static_cast<size_t>(n_score_bins_), static_cast<size_t>(n_moments())});
  } else
#endif
    if (mpi::master || !settings::reduce_tallies || moments_all_ranks_) {
    moments_ = tensor::Tensor<double>({static_cast<size_t>(n_filter_bins_),
      static_cast<size_t>(n_score_bins_), static_cast<size_t>(n_moments())});
  } else {
    moments_ = tensor::Tensor<double>();
  }
}

void Tally::reset()
{
  n_realizations_ = 0;
#ifdef OPENMC_MPI
  if (storage_ == TallyStorage::RMA) {
    // Both owned-rows planes are invariantly zero between batches (the fold
    // zeroes them and the window is born zeroed), so this is defensive. Zero
    // the private plane and this rank's window block; MPI_Win_sync is a local
    // memory barrier (not collective), so a single-tally reset stays valid.
    if (rma_plane_size_ > 0) {
      std::fill(accum_, accum_ + rma_plane_size_, 0.0);
      std::fill(rma_win_base_, rma_win_base_ + rma_plane_size_, 0.0);
      MPI_Win_sync(accum_win_);
    }
    if (moments_.size() != 0) {
      moments_.fill(0.0);
    }
    return;
  }
#endif
  if (accum_ != nullptr && accum_size_ != 0) {
#ifdef OPENMC_MPI
    // The shared plane is invariantly all-zeros between batches (each batch
    // ends with a fold/zero + publish, and the window is born zeroed), so this
    // fill is a no-op in every reachable state. Only the node leader writes it,
    // to avoid a cross-process data race on the shared window.
    if (storage_ == TallyStorage::SHARED) {
      if (mpi::node_leader)
        std::fill(accum_, accum_ + accum_size_, 0.0);
    } else
#endif
    {
      std::fill(accum_, accum_ + accum_size_, 0.0);
    }
  }
  if (moments_.size() != 0) {
    moments_.fill(0.0);
  }
}

double Tally::tally_normalization() const
{
  // Total source strength (fixed source) or unity (eigenvalue) per generation,
  // divided by the number of contributing particles.
  double total_source = 1.0;
  if (settings::run_mode == RunMode::FIXED_SOURCE) {
    total_source = model::external_sources_probability.integral();
  }
  double contributing_particles = settings::reduce_tallies
                                    ? settings::n_particles
                                    : simulation::work_per_rank;
  double norm =
    total_source / (contributing_particles * settings::gen_per_batch);
  if (settings::solver_type == SolverType::RANDOM_RAY) {
    norm = 1.0;
  }
  return norm;
}

void Tally::accumulate()
{
  // Increment number of realizations
  n_realizations_ += settings::reduce_tallies ? 1 : mpi::n_procs;

#ifdef OPENMC_MPI
  if (storage_ == TallyStorage::RMA) {
    // Every rank folds its owned rows. The batch sum for a bin is its window
    // block (remote contributions, zero while scoring is local) plus the
    // private plane (this rank's own scores); both planes are then zeroed. norm
    // is rank-identical: it depends only on global model data and rma requires
    // reduce_tallies. n_realizations_ incremented on all ranks above.
    const double norm = tally_normalization();
    if (higher_moments_) {
#pragma omp parallel for
      for (int64_t i = 0; i < rma_n_rows_; ++i) {
        for (int j = 0; j < n_score_bins_; ++j) {
          const int64_t k = i * n_score_bins_ + j;
          double val = (rma_win_base_[k] + accum_[k]) * norm;
          rma_win_base_[k] = 0.0;
          accum_[k] = 0.0;
          double val2 = val * val;
          moments_(i, j, TallyMoment::SUM) += val;
          moments_(i, j, TallyMoment::SUM_SQ) += val2;
          moments_(i, j, TallyMoment::SUM_THIRD) += val2 * val;
          moments_(i, j, TallyMoment::SUM_FOURTH) += val2 * val2;
        }
      }
    } else {
#pragma omp parallel for
      for (int64_t i = 0; i < rma_n_rows_; ++i) {
        for (int j = 0; j < n_score_bins_; ++j) {
          const int64_t k = i * n_score_bins_ + j;
          double val = (rma_win_base_[k] + accum_[k]) * norm;
          rma_win_base_[k] = 0.0;
          accum_[k] = 0.0;
          moments_(i, j, TallyMoment::SUM) += val;
          moments_(i, j, TallyMoment::SUM_SQ) += val * val;
        }
      }
    }
    // Publish the zeroed window block before the next batch accumulates.
    rma_resume();
    return;
  }
#endif

  if (mpi::master || !settings::reduce_tallies) {
    double norm = tally_normalization();

    // Fold the per-batch accumulator into the moments, then zero it. accum_ is
    // stored contiguously as [n_filter_bins, n_score_bins].
    if (higher_moments_) {
#pragma omp parallel for
      // filter bins (specific cell, energy bins)
      for (int64_t i = 0; i < n_filter_bins_; ++i) {
        // score bins (flux, total reaction rate, fission reaction rate, etc.)
        for (int j = 0; j < n_score_bins_; ++j) {
          double& acc = accum_[i * n_score_bins_ + j];
          double val = acc * norm;
          double val2 = val * val;
          acc = 0.0;
          moments_(i, j, TallyMoment::SUM) += val;
          moments_(i, j, TallyMoment::SUM_SQ) += val2;
          moments_(i, j, TallyMoment::SUM_THIRD) += val2 * val;
          moments_(i, j, TallyMoment::SUM_FOURTH) += val2 * val2;
        }
      }
    } else {
#pragma omp parallel for
      // filter bins (specific cell, energy bins)
      for (int64_t i = 0; i < n_filter_bins_; ++i) {
        // score bins (flux, total reaction rate, fission reaction rate, etc.)
        for (int j = 0; j < n_score_bins_; ++j) {
          double& acc = accum_[i * n_score_bins_ + j];
          double val = acc * norm;
          acc = 0.0;
          moments_(i, j, TallyMoment::SUM) += val;
          moments_(i, j, TallyMoment::SUM_SQ) += val * val;
        }
      }
    }
  }
#ifdef OPENMC_MPI
  else if (storage_ == TallyStorage::SHARED && mpi::node_leader) {
    // A non-master node leader does not fold, but its node plane was the source
    // of the internode reduce and still holds this node's batch sum. Zero it so
    // it does not carry into the next batch. (The master zeroes its own plane
    // in the fold above.)
#pragma omp parallel for
    for (int64_t i = 0; i < accum_size_; ++i)
      accum_[i] = 0.0;
  }

  // Publish the zeroed plane before the next batch scores into it. Every node
  // rank participates in the barrier inside shared_resume().
  if (storage_ == TallyStorage::SHARED)
    shared_resume();
#endif
}

int Tally::score_index(const std::string& score) const
{
  for (int i = 0; i < scores_.size(); i++) {
    if (this->score_name(i) == score)
      return i;
  }
  return -1;
}

tensor::Tensor<double> Tally::get_reshaped_data() const
{
  vector<size_t> shape;
  for (auto f : filters()) {
    shape.push_back(model::tally_filters[f]->n_bins());
  }

  // add number of scores and nuclides to tally
  shape.push_back(moments_.shape(1));
  shape.push_back(moments_.shape(2));

  tensor::Tensor<double> reshaped_results = moments_;
  reshaped_results.reshape(shape);
  return reshaped_results;
}

std::string Tally::score_name(int score_idx) const
{
  if (score_idx < 0 || score_idx >= scores_.size()) {
    fatal_error("Index in scores array is out of bounds.");
  }
  return reaction_name(scores_[score_idx]);
}

std::vector<std::string> Tally::scores() const
{
  std::vector<std::string> score_names;
  for (int score : scores_)
    score_names.push_back(reaction_name(score));
  return score_names;
}

std::string Tally::nuclide_name(int nuclide_idx) const
{
  if (nuclide_idx < 0 || nuclide_idx >= nuclides_.size()) {
    fatal_error("Index in nuclides array is out of bounds");
  }

  int nuclide = nuclides_.at(nuclide_idx);
  if (nuclide == -1) {
    return "total";
  }
  return data::nuclides.at(nuclide)->name_;
}

//==============================================================================
// Non-member functions
//==============================================================================

void read_tallies_xml()
{
  // Check if tallies.xml exists. If not, just return since it is optional
  std::string filename = settings::path_input + "tallies.xml";
  if (!file_exists(filename))
    return;

  write_message("Reading tallies XML file...", 5);

  // Parse tallies.xml file
  pugi::xml_document doc;
  doc.load_file(filename.c_str());
  pugi::xml_node root = doc.document_element();

  read_tallies_xml(root);
}

void read_tallies_xml(pugi::xml_node root)
{
  // Check for <assume_separate> setting
  if (check_for_node(root, "assume_separate")) {
    settings::assume_separate = get_node_value_bool(root, "assume_separate");
  }

  // Check for user meshes and allocate
  read_meshes(root);

  // We only need the mesh info for plotting
  if (settings::run_mode == RunMode::PLOTTING)
    return;

  // Read data for tally derivatives
  read_tally_derivatives(root);

  // ==========================================================================
  // READ FILTER DATA

  // Check for user filters and allocate
  for (auto node_filt : root.children("filter")) {
    auto f = Filter::create(node_filt);
  }

  // ==========================================================================
  // READ TALLY DATA

  // Check for user tallies
  int n = 0;
  for (auto node : root.children("tally"))
    ++n;
  if (n == 0 && mpi::master) {
    warning("No tallies present in tallies.xml file.");
  }

  for (auto node_tal : root.children("tally")) {
    model::tallies.push_back(make_unique<Tally>(node_tal));
  }
}

#ifdef OPENMC_MPI
void reduce_in_place_chunked(double* data, int64_t n, MPI_Comm comm)
{
  // 2^27 doubles = 1 GiB per call; keeps the int MPI count below 2^31 for the
  // very large planes (up to ~4.3e9 elements) these tallies can reach.
  constexpr int64_t MAX_CHUNK = int64_t {1} << 27;
  for (int64_t offset = 0; offset < n; offset += MAX_CHUNK) {
    int count = static_cast<int>(std::min<int64_t>(MAX_CHUNK, n - offset));
    if (mpi::master) {
      MPI_Reduce(
        MPI_IN_PLACE, data + offset, count, MPI_DOUBLE, MPI_SUM, 0, comm);
    } else {
      MPI_Reduce(data + offset, nullptr, count, MPI_DOUBLE, MPI_SUM, 0, comm);
    }
  }
}

void reduce_tally_results()
{
  // Don't reduce tallies if the no_reduce option is on
  if (settings::reduce_tallies) {
    for (int i_tally : model::active_tallies) {
      // Skip any tallies that are not active
      auto& tally {model::tallies[i_tally]};

      if (tally->storage_ == TallyStorage::SHARED) {
        // Publish this node's scores and combine the node planes onto the
        // master, replacing the intracomm reduce. The plane then belongs to the
        // node leader until accumulate() folds/zeroes it and resumes.
        tally->shared_publish();
        continue;
      }

      if (tally->storage_ == TallyStorage::RMA) {
        // Complete every rank's accumulates into the distributed window and
        // barrier so nothing is in flight; accumulate() then folds each rank's
        // owned rows. No intracomm reduce -- the data is already at its owner.
        tally->rma_publish();
        continue;
      }

      // The accumulator is contiguous, so it reduces in place onto the master
      // with no scratch buffer. Chunked to respect the int MPI count limit.
      reduce_in_place_chunked(
        tally->accum_, tally->accum_size_, mpi::intracomm);

      // The fold that zeroes accum_ only runs on the master, so non-master
      // ranks must clear their (now already-summed) accumulator here to avoid
      // double-counting into the next batch.
      if (!mpi::master) {
        std::fill(tally->accum_, tally->accum_ + tally->accum_size_, 0.0);
      }
    }
  }

  // Note that global tallies are *always* reduced even when no_reduce option
  // is on.

  // Get reference to global tallies
  auto& gt = simulation::global_tallies;
  const int val_col = static_cast<int>(TallyResult::VALUE);

  // Copy VALUE column into contiguous array for MPI reduction
  tensor::Tensor<double> gt_values(gt.slice(tensor::all, val_col));
  tensor::Tensor<double> gt_values_reduced({size_t {N_GLOBAL_TALLIES}});

  // Reduce contiguous data
  MPI_Reduce(gt_values.data(), gt_values_reduced.data(), N_GLOBAL_TALLIES,
    MPI_DOUBLE, MPI_SUM, 0, mpi::intracomm);

  // Transfer values on master and reset on other ranks
  if (mpi::master) {
    gt.slice(tensor::all, val_col) = gt_values_reduced;
  } else {
    gt.slice(tensor::all, val_col) = 0.0;
  }

  // We also need to determine the total starting weight of particles from the
  // last realization
  double weight_reduced;
  MPI_Reduce(&simulation::total_weight, &weight_reduced, 1, MPI_DOUBLE, MPI_SUM,
    0, mpi::intracomm);
  if (mpi::master)
    simulation::total_weight = weight_reduced;
}
#endif

void accumulate_tallies()
{
#ifdef OPENMC_MPI
  // Combine tally results onto master process
  if (mpi::n_procs > 1 && settings::solver_type == SolverType::MONTE_CARLO) {
    reduce_tally_results();
  }
#endif

  // Increase number of realizations (only used for global tallies)
  simulation::n_realizations += 1;

  // Accumulate on master only unless run is not reduced then do it on all
  if (mpi::master || !settings::reduce_tallies) {
    auto& gt = simulation::global_tallies;

    if (settings::run_mode == RunMode::EIGENVALUE) {
      if (simulation::current_batch > settings::n_inactive) {
        // Accumulate products of different estimators of k
        double k_col = gt(GlobalTally::K_COLLISION, TallyResult::VALUE) /
                       simulation::total_weight;
        double k_abs = gt(GlobalTally::K_ABSORPTION, TallyResult::VALUE) /
                       simulation::total_weight;
        double k_tra = gt(GlobalTally::K_TRACKLENGTH, TallyResult::VALUE) /
                       simulation::total_weight;
        simulation::k_col_abs += k_col * k_abs;
        simulation::k_col_tra += k_col * k_tra;
        simulation::k_abs_tra += k_abs * k_tra;
      }
    }

    // Accumulate results for global tallies
    for (int i = 0; i < N_GLOBAL_TALLIES; ++i) {
      double val = gt(i, TallyResult::VALUE) / simulation::total_weight;
      gt(i, TallyResult::VALUE) = 0.0;
      gt(i, TallyResult::SUM) += val;
      gt(i, TallyResult::SUM_SQ) += val * val;
    }
  }

  // Accumulate results for each tally
  for (int i_tally : model::active_tallies) {
    auto& tally {model::tallies[i_tally]};
    tally->accumulate();
  }
}

double distance_to_time_boundary(double time, double speed)
{
  if (model::time_grid.empty()) {
    return INFTY;
  } else if (time >= model::time_grid.back()) {
    return INFTY;
  } else {
    double next_time =
      *std::upper_bound(model::time_grid.begin(), model::time_grid.end(), time);
    return (next_time - time) * speed;
  }
}

//! Add new points to the global time grid
//
//! \param grid Vector of new time points to add
void add_to_time_grid(vector<double> grid)
{
  if (grid.empty())
    return;

  // Create new vector with enough space to hold old and new grid points
  vector<double> merged;
  merged.reserve(model::time_grid.size() + grid.size());

  // Merge and remove duplicates
  std::set_union(model::time_grid.begin(), model::time_grid.end(), grid.begin(),
    grid.end(), std::back_inserter(merged));

  // Swap in the new grid
  model::time_grid.swap(merged);
}

void setup_active_tallies()
{
  model::active_tallies.clear();
  model::active_analog_tallies.clear();
  model::active_tracklength_tallies.clear();
  model::active_timed_tracklength_tallies.clear();
  model::active_collision_tallies.clear();
  model::active_meshsurf_tallies.clear();
  model::active_surface_tallies.clear();
  model::active_pulse_height_tallies.clear();
  model::time_grid.clear();

  for (auto i = 0; i < model::tallies.size(); ++i) {
    const auto& tally {*model::tallies[i]};

    if (tally.active_) {
      model::active_tallies.push_back(i);
      bool mesh_present = (tally.get_filter<MeshFilter>() ||
                           tally.get_filter<MeshMaterialFilter>());
      auto time_filter = tally.get_filter<TimeFilter>();
      switch (tally.type_) {

      case TallyType::VOLUME:
        switch (tally.estimator_) {
        case TallyEstimator::ANALOG:
          model::active_analog_tallies.push_back(i);
          break;
        case TallyEstimator::TRACKLENGTH:
          if (time_filter && mesh_present) {
            model::active_timed_tracklength_tallies.push_back(i);
            add_to_time_grid(time_filter->bins());
          } else {
            model::active_tracklength_tallies.push_back(i);
          }
          break;
        case TallyEstimator::COLLISION:
          model::active_collision_tallies.push_back(i);
        }
        break;

      case TallyType::MESH_SURFACE:
        model::active_meshsurf_tallies.push_back(i);
        break;

      case TallyType::SURFACE:
        model::active_surface_tallies.push_back(i);
        break;

      case TallyType::PULSE_HEIGHT:
        model::active_pulse_height_tallies.push_back(i);
        break;
      }
    }
  }
}

void free_memory_tally()
{
  model::tally_derivs.clear();
  model::tally_deriv_map.clear();

  model::tally_filters.clear();
  model::filter_map.clear();

  model::tallies.clear();

  model::active_tallies.clear();
  model::active_analog_tallies.clear();
  model::active_tracklength_tallies.clear();
  model::active_timed_tracklength_tallies.clear();
  model::active_collision_tallies.clear();
  model::active_meshsurf_tallies.clear();
  model::active_surface_tallies.clear();
  model::active_pulse_height_tallies.clear();
  model::time_grid.clear();

  model::tally_map.clear();
}

//==============================================================================
// C-API functions
//==============================================================================

extern "C" int openmc_extend_tallies(
  int32_t n, int32_t* index_start, int32_t* index_end)
{
  if (index_start)
    *index_start = model::tallies.size();
  if (index_end)
    *index_end = model::tallies.size() + n - 1;
  for (int i = 0; i < n; ++i) {
    model::tallies.push_back(make_unique<Tally>(-1));
  }
  return 0;
}

extern "C" int openmc_get_tally_index(int32_t id, int32_t* index)
{
  auto it = model::tally_map.find(id);
  if (it == model::tally_map.end()) {
    set_errmsg(fmt::format("No tally exists with ID={}.", id));
    return OPENMC_E_INVALID_ID;
  }

  *index = it->second;
  return 0;
}

extern "C" void openmc_get_tally_next_id(int32_t* id)
{
  int32_t largest_tally_id = 0;
  for (const auto& t : model::tallies) {
    largest_tally_id = std::max(largest_tally_id, t->id_);
  }
  *id = largest_tally_id + 1;
}

extern "C" int openmc_tally_get_estimator(int32_t index, int* estimator)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  *estimator = static_cast<int>(model::tallies[index]->estimator_);
  return 0;
}

extern "C" int openmc_tally_set_estimator(int32_t index, const char* estimator)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  auto& t {model::tallies[index]};

  std::string est = estimator;
  if (est == "analog") {
    t->estimator_ = TallyEstimator::ANALOG;
  } else if (est == "collision") {
    t->estimator_ = TallyEstimator::COLLISION;
  } else if (est == "tracklength") {
    t->estimator_ = TallyEstimator::TRACKLENGTH;
  } else {
    set_errmsg("Unknown tally estimator: " + est);
    return OPENMC_E_INVALID_ARGUMENT;
  }
  return 0;
}

extern "C" int openmc_tally_get_id(int32_t index, int32_t* id)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  *id = model::tallies[index]->id_;
  return 0;
}

extern "C" int openmc_tally_set_id(int32_t index, int32_t id)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  model::tallies[index]->set_id(id);
  return 0;
}

extern "C" int openmc_tally_get_type(int32_t index, int32_t* type)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  *type = static_cast<int>(model::tallies[index]->type_);

  return 0;
}

extern "C" int openmc_tally_set_type(int32_t index, const char* type)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  if (strcmp(type, "volume") == 0) {
    model::tallies[index]->type_ = TallyType::VOLUME;
  } else if (strcmp(type, "mesh-surface") == 0) {
    model::tallies[index]->type_ = TallyType::MESH_SURFACE;
  } else if (strcmp(type, "surface") == 0) {
    model::tallies[index]->type_ = TallyType::SURFACE;
  } else if (strcmp(type, "pulse-height") == 0) {
    model::tallies[index]->type_ = TallyType::PULSE_HEIGHT;
  } else {
    set_errmsg(fmt::format("Unknown tally type: {}", type));
    return OPENMC_E_INVALID_ARGUMENT;
  }

  return 0;
}

extern "C" int openmc_tally_get_active(int32_t index, bool* active)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  *active = model::tallies[index]->active_;

  return 0;
}

extern "C" int openmc_tally_set_active(int32_t index, bool active)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  model::tallies[index]->active_ = active;

  return 0;
}

extern "C" int openmc_tally_get_writable(int32_t index, bool* writable)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  *writable = model::tallies[index]->writable();

  return 0;
}

extern "C" int openmc_tally_set_writable(int32_t index, bool writable)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  model::tallies[index]->set_writable(writable);

  return 0;
}

extern "C" int openmc_tally_get_multiply_density(int32_t index, bool* value)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  *value = model::tallies[index]->multiply_density();

  return 0;
}

extern "C" int openmc_tally_set_multiply_density(int32_t index, bool value)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  model::tallies[index]->set_multiply_density(value);

  return 0;
}

extern "C" int openmc_tally_get_scores(int32_t index, int** scores, int* n)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  *scores = model::tallies[index]->scores_.data();
  *n = model::tallies[index]->scores_.size();
  return 0;
}

extern "C" int openmc_tally_set_scores(
  int32_t index, int n, const char** scores)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  vector<std::string> scores_str(scores, scores + n);
  try {
    model::tallies[index]->set_scores(scores_str);
  } catch (const std::invalid_argument& ex) {
    set_errmsg(ex.what());
    return OPENMC_E_INVALID_ARGUMENT;
  }

  return 0;
}

extern "C" int openmc_tally_get_nuclides(int32_t index, int** nuclides, int* n)
{
  // Make sure the index fits in the array bounds.
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  *n = model::tallies[index]->nuclides_.size();
  *nuclides = model::tallies[index]->nuclides_.data();

  return 0;
}

extern "C" int openmc_tally_set_nuclides(
  int32_t index, int n, const char** nuclides)
{
  // Make sure the index fits in the array bounds.
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  vector<std::string> words(nuclides, nuclides + n);
  vector<int> nucs;
  for (auto word : words) {
    if (word == "total") {
      nucs.push_back(-1);
    } else {
      auto search = data::nuclide_map.find(word);
      if (search == data::nuclide_map.end()) {
        int err = openmc_load_nuclide(word.c_str(), nullptr, 0);
        if (err < 0) {
          set_errmsg(openmc_err_msg);
          return OPENMC_E_DATA;
        }
      }
      nucs.push_back(data::nuclide_map.at(word));
    }
  }

  model::tallies[index]->nuclides_ = nucs;

  return 0;
}

extern "C" int openmc_tally_get_filters(
  int32_t index, const int32_t** indices, size_t* n)
{
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  *indices = model::tallies[index]->filters().data();
  *n = model::tallies[index]->filters().size();
  return 0;
}

extern "C" int openmc_tally_set_filters(
  int32_t index, size_t n, const int32_t* indices)
{
  // Make sure the index fits in the array bounds.
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  // Set the filters.
  try {
    // Convert indices to filter pointers
    vector<Filter*> filters;
    for (int64_t i = 0; i < n; ++i) {
      int32_t i_filt = indices[i];
      filters.push_back(model::tally_filters.at(i_filt).get());
    }
    model::tallies[index]->set_filters(filters);
  } catch (const std::out_of_range& ex) {
    set_errmsg("Index in tally filter array out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  return 0;
}

//! Reset tally results and number of realizations
extern "C" int openmc_tally_reset(int32_t index)
{
  // Make sure the index fits in the array bounds.
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  model::tallies[index]->reset();
  return 0;
}

extern "C" int openmc_tally_get_n_realizations(int32_t index, int32_t* n)
{
  // Make sure the index fits in the array bounds.
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  *n = model::tallies[index]->n_realizations_;
  return 0;
}

//! \brief Returns a pointer to a tally results array along with its shape.
//! This allows a user to obtain in-memory tally results from Python directly.
extern "C" int openmc_tally_results(
  int32_t index, double** results, size_t* shape)
{
  // Make sure the index fits in the array bounds.
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  const auto& t {model::tallies[index]};
#ifdef OPENMC_MPI
  if (t->storage_ == TallyStorage::RMA) {
    // Under rma each rank holds only its owned moment rows, so there is no
    // global in-memory array to return. Consume rma results from the
    // statepoint.
    set_errmsg("In-memory results are not available for a tally using 'rma' "
               "storage; read them from the statepoint file instead.");
    return OPENMC_E_ALLOCATE;
  }
#endif
  if (!t->has_moments()) {
    set_errmsg("Tally results have not been allocated yet.");
    return OPENMC_E_ALLOCATE;
  }

  // Set pointer to the moments array and copy its shape. The innermost
  // dimension is now n_moments (SUM=0, SUM_SQ=1, ...), matching the on-disk
  // statepoint layout -- there is no leading VALUE column.
  *results = t->moments().data();
  auto s = t->moments().shape();
  shape[0] = s[0];
  shape[1] = s[1];
  shape[2] = s[2];
  return 0;
}

extern "C" int openmc_global_tallies(double** ptr)
{
  *ptr = simulation::global_tallies.data();
  return 0;
}

extern "C" size_t tallies_size()
{
  return model::tallies.size();
}

// given a tally ID, remove it from the tallies vector. For a shared-storage
// tally this destroys an MPI window and is therefore collective on node_comm --
// every rank must call it for the same index.
extern "C" int openmc_remove_tally(int32_t index)
{
  // check that id is in the map
  if (index < 0 || index >= model::tallies.size()) {
    set_errmsg("Index in tallies array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  // delete the tally via iterator pointing to correct position
  // this calls the Tally destructor, removing the tally from the map as well
  model::tallies.erase(model::tallies.begin() + index);

  return 0;
}

} // namespace openmc
