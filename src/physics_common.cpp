#include "openmc/physics_common.h"

#include "openmc/random_lcg.h"
#include "openmc/settings.h"

namespace openmc {

//==============================================================================
// RUSSIAN_ROULETTE
//==============================================================================
int counts = 1;
void russian_roulette(Particle& p, double weight_survive)
{
  std::cout<<"Number of Roulette: " + counts <<std::endl; 
  if (weight_survive * prn(p.current_seed()) < p.wgt()) {
    p.wgt() = weight_survive;
  } else {
    p.wgt() = 0.;
  }
  counts += 1;
}

} // namespace openmc
