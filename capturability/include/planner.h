#ifndef __PLANNER_H__
#define __PLANNER_H__

#include "model.h"
#include "param.h"
#include "config.h"
#include "grid.h"
#include "capturability.h"
#include "tree.h"
#include "search.h"
#include "swing.h"
#include "pendulum.h"
#include <iostream>
#include <chrono>
#include <vector>

namespace Capt {

class Planner {
public:
  Planner(Model *model, Param *param, Config *config, Grid *grid, Capturability *capt);
  ~Planner();

  void                  set(EnhancedState state);
  EnhancedInput         get();
  std::vector<Sequence> getSequence();
  arr3_t                getFootstepR();
  arr3_t                getFootstepL();
  std::vector<CaptData> getCaptureRegion();

  bool plan();

private:
  void calculateStart();
  void calculateGoal();
  bool runSearch();

  Tree     *tree;
  Search   *search;
  Swing    *swing;
  Pendulum *pendulum;

  EnhancedState state;
  EnhancedInput input;

  // start
  Foot   s_suf;
  vec2_t rfoot, lfoot, icp;
  vec3_t suf;
  double elapsed;
  // goal
  Foot   g_suf;
  arr2_t goal;

  double dt, dt_min;
  int    preview;

  bool found;
};

} // namespace Capt

#endif // __PLANNER_H__