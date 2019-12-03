#ifndef __PLANNER_H__
#define __PLANNER_H__

#include "model.h"
#include "param.h"
#include "grid.h"
#include "capturability.h"
#include <iostream>
#include <chrono>
#include <vector>

namespace planner {

struct Input {
  double       elapsed_time;
  Capt::Foot   suf;
  Capt::vec3_t rfoot;
  Capt::vec3_t lfoot;
  Capt::vec3_t icp;
  Capt::vec3_t goal;

  void operator=(const Input &input) {
    this->cop          = input.cop;
    this->swf          = input.swf;
    this->elapsed_time = input.elapsed_time;
    this->suf          = input.suf;
    this->rfoot        = input.rfoot;
    this->lfoot        = input.lfoot;
    this->icp          = input.icp;
    this->goal         = input.goal;
  }
};

struct Output {
  double       planning_time;
  Capt::vec3_t rfoot;
  Capt::vec3_t lfoot;
  Capt::vec3_t icp;
  Capt::vec3_t cop;

  void operator=(const Output &output) {
    this->planning_time = output.planning_time;
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
  Planner(Model *model, Param *param, Grid *grid, Capturability *capt);
  ~Planner();

  void            set(planner::Input input);
  planner::Output get();

private:
  void run();
  void selectSupportFoot();
  void runSearch();
  void generatePath();

  Tree   *tree;
  Search *search;

  planner::Input  input;
  planner::Output output;
};

} // namespace Capt

#endif // __PLANNER_H__