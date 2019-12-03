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
  double       elapsed;
  Capt::Foot   suf;
  Capt::vec3_t rfoot;
  Capt::vec3_t lfoot;
  Capt::vec3_t com;
  Capt::vec3_t com_vel;
  Capt::vec3_t goal;
  double       stance;

  void operator=(const Input &input) {
    this->elapsed = input.elapsed;
    this->suf     = input.suf;
    this->rfoot   = input.rfoot;
    this->lfoot   = input.lfoot;
    this->com     = input.com;
    this->com_vel = input.com_vel;
    this->goal    = input.goal;
    this->stance  = input.stance;
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

  void            set(planner::Input input);
  planner::Output get();
  arr3_t          getFootstepR();
  arr3_t          getFootstepL();

private:
  void run();
  void selectSupportFoot();
  void runSearch();
  void generatePath();

  Tree      *tree;
  Search    *search;
  SwingFoot *swingFoot;
  Pendulum  *pendulum;

  planner::Input  input;
  planner::Output output;

  Foot   suf;
  double dt;
  double omega;
};

} // namespace Capt

#endif // __PLANNER_H__