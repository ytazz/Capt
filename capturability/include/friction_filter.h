#ifndef __FRICTION_FILTER_H__
#define __FRICTION_FILTER_H__

#include "capturability.h"
#include "input.h"
#include "pendulum.h"
#include "swing_foot.h"
#include "vector.h"
#include <iostream>

namespace CA {

class FrictionFilter {
public:
  FrictionFilter(Capturability capturability, Pendulum pendulum);
  ~FrictionFilter();

  void setCaptureRegion(std::vector<CaptureSet> capture_set);
  std::vector<CaptureSet>
  getCaptureRegion(vec2_t com, vec2_t com_vel,
                   float mu); // mu = friction coefficient

private:
  Capturability capturability;
  Pendulum pendulum;

  // current target capture region
  std::vector<CaptureSet> capture_set;
};

} // namespace CA

#endif // __FRICTION_FILTER_H____