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
  double         elapsed;
  Capt::Footstep footstep;
  Capt::Foot     s_suf;
  Capt::vec3_t   rfoot;
  Capt::vec3_t   lfoot;
  Capt::vec3_t   icp;

  void operator=(const Input &input) {
    this->elapsed  = input.elapsed;
    this->footstep = input.footstep;
    this->s_suf    = input.s_suf;
    this->rfoot    = input.rfoot;
    this->lfoot    = input.lfoot;
    this->icp      = input.icp;
  }
};

struct Output {
  double       planning_time;
  double       duration; // step duration
  Capt::vec3_t rfoot;
  Capt::vec3_t lfoot;
  Capt::vec3_t icp;
  Capt::vec3_t cop;

  void operator=(const Output &output) {
    this->planning_time = output.planning_time;
    this->duration      = output.duration;
    this->rfoot         = output.rfoot;
    this->lfoot         = output.lfoot;
    this->icp           = output.icp;
    this->cop           = output.cop;
  }
};

}

namespace Capt {

class Planner {
public:
  Planner(Model *model, Param *param, Config *config, Grid *grid, Capturability *capt);
  ~Planner();

  void                  set(planner::Input input);
  planner::Output       get(double time);
  std::vector<Sequence> getSequence();
  arr3_t                getFootstepR();
  arr3_t                getFootstepL();
  double                getPreviewRadius();
  std::vector<CaptData> getCaptureRegion();

private:
  void plan();
  void replan();

  void calculateGoal();
  void runSearch();
  void generatePath(double time);

  Tree      *tree;
  Search    *search;
  SwingFoot *swingFoot;
  Pendulum  *pendulum;

  planner::Input  input;
  planner::Output output;

  Foot   g_suf;
  vec3_t g_foot;

  double dt;
  int    preview;

  bool found;
};

} // namespace Capt

#endif // __PLANNER_H__