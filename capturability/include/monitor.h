#ifndef __MONITOR_H__
#define __MONITOR_H__

#include <iostream>
#include <chrono>
#include <vector>

namespace Capt {

class Monitor {
public:
  Monitor(Grid *grid, Capturability *capturability);
  ~Monitor();

  bool check(EnhancedState state, vec2_t nextLandingPos);

private:
  Grid          *grid;
  Capturability *capturability;

  double min;
  int    minId;
};

} // namespace Capt

#endif // __MONITOR_H__