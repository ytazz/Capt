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
  FrictionFilter(Capturability capturability);
  ~FrictionFilter();

  void setCaptureRegion(std::vector<Input> capture_region);
  std::vector<Input> getCaptureRegion(State state, vec2_t com,
                                      float mu); // mu = coefficient of friction

private:
  Capturability capturability;
  std::vector<Input> capture_region; // current target capture region
};

} // namespace CA

#endif // __FRICTION_FILTER_H__