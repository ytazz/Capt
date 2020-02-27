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
  EnhancedInput         get();
  std::vector<CaptData> getCaptureRegion();
  arr3_t                getFootstepR();
  arr3_t                getFootstepL();

private:
  Grid          *grid;
  Capturability *capturability;
  Swing         *swing;
  Pendulum      *pendulum;

  EnhancedState state;
  EnhancedInput input;

  Foot supportFoot;

  // capture region
  arr2_t                   captureRegion;
  std::vector<CaptureSet*> nstepCaptureRegion;
  std::vector<CaptData>    captData;

  vec2_t nextLandingPos;

  double dt_min;

  double min;
  int    minId;
};

} // namespace Capt

#endif // __MONITOR_H__