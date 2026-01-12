get_filename_component(OpenMC_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)

# Compute the install prefix from this file's location
get_filename_component(_OPENMC_PREFIX "${OpenMC_CMAKE_DIR}/../../.." ABSOLUTE)

find_package(fmt CONFIG REQUIRED HINTS ${_OPENMC_PREFIX})
find_package(pugixml CONFIG REQUIRED HINTS ${_OPENMC_PREFIX})
find_package(xtl CONFIG REQUIRED HINTS ${_OPENMC_PREFIX})
find_package(xtensor CONFIG REQUIRED HINTS ${_OPENMC_PREFIX})
if(OFF)
  find_package(DAGMC REQUIRED HINTS )
endif()

if(OFF)
  include(FindPkgConfig)
  list(APPEND CMAKE_PREFIX_PATH )
  set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH True)
  pkg_check_modules(LIBMESH REQUIRED >=1.7.0 IMPORTED_TARGET)
endif()

find_package(PNG)

if(NOT TARGET OpenMC::libopenmc)
  include("${OpenMC_CMAKE_DIR}/OpenMCTargets.cmake")
endif()

if(on)
  find_package(MPI REQUIRED)
endif()

if(ON)
  find_package(OpenMP REQUIRED)
endif()

if(OFF AND NOT ${DAGMC_BUILD_UWUW})
  message(FATAL_ERROR "UWUW is enabled in OpenMC but the DAGMC installation discovered was not configured with UWUW.")
endif()
