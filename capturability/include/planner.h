#ifndef __PLANNER_H__
#define __PLANNER_H__

#include "model.h"
#include "param.h"
#include "config.h"
#include "grid.h"
#include "capturability.h"
#include "tree.h"
#include "search.h"
#include "swing_foot.h"
#include "pendulum.h"
#include <iostream>
#include <chrono>
#include <vector>

namespace planner {

struct Input {
  Capt::Footstep footstep;
  Capt::vec3_t   icp;
  Capt::vec3_t   rfoot;
  Capt::vec3_t   lfoot;
  double         elapsed;
  Capt::Foot     s_suf;

  void operator=(const Input &input) {
    this->footstep = input.footstep;
    this->icp      = input.icp;
    this->rfoot    = input.rfoot;
    this->lfoot    = input.lfoot;
    this->elapsed  = input.elapsed;
    this->s_suf    = input.s_suf;
  }
};

struct Output {
  double       duration; // remained step duration
  Capt::vec3_t cop;
  Capt::vec3_t icp;
  Capt::vec3_t suf;
  Capt::vec3_t swf;
  Capt::vec3_t land;

  void operator=(const Output &output) {
    this->duration = output.duration;
    this->cop      = output.cop;
    this->icp      = output.icp;
    this->suf      = output.suf;
    this->swf      = output.swf;
    this->land     = output.land;
  }
};

}

namespace Capt {

class Planner {
public:
  Planner(Model *model, Param *param, Config *config, Grid *grid, Capturability *capt);
  ~Planner();

  void                  set(planner::Input input);
  planner::Output       get();
  std::vector<Sequence> getSequence();
  arr3_t                getFootstepR();
  arr3_t                getFootstepL();
  std::vector<CaptData> getCaptureRegion();

  void plan();
  void replan();

private:
  void calculateStart();
  void calculateGoal();
  void runSearch();

  Tree      *tree;
  Search    *search;
  SwingFoot *swingFoot;
  Pendulum  *pendulum;

  planner::Input  input;
  planner::Output output;

  // start
  Foot   s_suf;
  vec2_t rfoot, lfoot, icp;
  double elapsed;
  // goal
  Foot   g_suf;
  vec2_t goal;

  double dt, dt_min;
  int    preview;

  bool found;
};

} // namespace Capt

#endif // __PLANNER_H__