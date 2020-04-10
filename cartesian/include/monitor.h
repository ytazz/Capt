#ifndef __MONITOR_H__
#define __MONITOR_H__

#include <iostream>
#include <vector>
#include "swing.h"
#include "pendulum.h"
#include "grid.h"
#include "capturability.h"
#include "tree.h"

namespace Capt {

class Monitor {
public:
  Monitor(Capturability *capturability);
  ~Monitor();

  Status check(const EnhancedState& state, EnhancedInput& input);
  //std::vector<CaptData> getCaptureRegion();

private:
  Capturability *capturability;

  //std::vector<CaptureRegion> region_1step;
};

} // namespace Capt

#endif // __MONITOR_H__