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
  Monitor(Model *model, Param *param, Grid *grid, Capturability *capturability);
  ~Monitor();

  Status                check(EnhancedState state, Footstep footstep);
  //std::vector<CaptData> getCaptureRegion();

private:
  Grid          *grid;
  Capturability *capturability;

  //std::vector<CaptureRegion> region_1step;
};

} // namespace Capt

#endif // __MONITOR_H__