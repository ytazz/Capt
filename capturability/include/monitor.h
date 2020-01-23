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
  Monitor(Model *model, Grid *grid, Capturability *capturability);
  ~Monitor();

  Status                check(EnhancedState state, Footstep footstep);
  EnhancedInput         get();
  std::vector<CaptData> getCaptureRegion();

private:
  Grid          *grid;
  Capturability *capturability;
  Swing         *swing;
  Pendulum      *pendulum;

  EnhancedState state;
  EnhancedInput input;

  // capture region
  arr2_t captureRegion;

  double dt_min;

  double min;
  int    minId;
};

} // namespace Capt

#endif // __MONITOR_H__