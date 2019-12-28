#ifndef __MONITOR_H__
#define __MONITOR_H__

#include <iostream>
#include <vector>
#include "swing.h"
#include "grid.h"
#include "capturability.h"

namespace Capt {

class Monitor {
public:
  Monitor(Model *model, Grid *grid, Capturability *capturability);
  ~Monitor();

  bool          check(EnhancedState state, Footstep footstep);
  EnhancedInput get();

private:
  Grid          *grid;
  Capturability *capturability;
  Swing         *swing;

  EnhancedState state;
  EnhancedInput input;

  double dt_min;

  double min;
  int    minId;
};

} // namespace Capt

#endif // __MONITOR_H__