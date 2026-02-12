#ifndef OPENMC_VERSION_H
#define OPENMC_VERSION_H

#include "openmc/array.h"

namespace openmc {

// OpenMC major, minor, and release numbers
// clang-format off
constexpr int VERSION_MAJOR {0};
constexpr int VERSION_MINOR {0};
constexpr int VERSION_RELEASE {0};
constexpr bool VERSION_DEV {false};
constexpr const char* VERSION_COMMIT_COUNT = "";
constexpr const char* VERSION_COMMIT_HASH = "e8d61dd7e9ce918e98ec0b024b5a8bdf46f6bc36";
constexpr std::array<int, 3> VERSION {VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE};
// clang-format on

} // namespace openmc

#endif // OPENMC_VERSION_H
