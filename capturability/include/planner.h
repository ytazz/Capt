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
  Capt::vec3_t icp;
  Capt::vec3_t goal;
  double       stance;

  void print(){
    printf("elapsed: %1.4lf\n", elapsed);
    switch (suf) {
    case Capt::Foot::FOOT_NONE:
      printf("suf: %s\n", "FOOT_NONE");
      break;
    case Capt::Foot::FOOT_R:
      printf("suf: %s\n", "FOOT_R");
      break;
    case Capt::Foot::FOOT_L:
      printf("suf: %s\n", "FOOT_L");
      break;
    }
    printf("rfoot : [ %+1.6lf, %+1.6lf, %+1.6lf ]\n", rfoot.x(), rfoot.y(), rfoot.z() );
    printf("lfoot : [ %+1.6lf, %+1.6lf, %+1.6lf ]\n", lfoot.x(), lfoot.y(), lfoot.z() );
    printf("icp   : [ %+1.6lf, %+1.6lf, %+1.6lf ]\n", icp.x(), icp.y(), icp.z() );
    printf("goal  : [ %+1.6lf, %+1.6lf, %+1.6lf ]\n", goal.x(), goal.y(), goal.z() );
    printf("stance: %+1.6lf\n", stance );
  }

  void operator=(const Input &input) {
    this->elapsed = input.elapsed;
    this->suf     = input.suf;
    this->rfoot   = input.rfoot;
    this->lfoot   = input.lfoot;
    this->icp     = input.icp;
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

  void print(){
    printf("planning_time: %1.4lf\n", planning_time);
    printf("duration: %1.4lf\n", duration);
    printf("rfoot : [ %+1.6lf, %+1.6lf, %+1.6lf ]\n", rfoot.x(), rfoot.y(), rfoot.z() );
    printf("lfoot : [ %+1.6lf, %+1.6lf, %+1.6lf ]\n", lfoot.x(), lfoot.y(), lfoot.z() );
    printf("icp   : [ %+1.6lf, %+1.6lf, %+1.6lf ]\n", icp.x(), icp.y(), icp.z() );
    printf("cop   : [ %+1.6lf, %+1.6lf, %+1.6lf ]\n", cop.x(), cop.y(), cop.z() );
  }

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
  arr3_t                getFootstepR();
  arr3_t                getFootstepL();
  std::vector<CaptData> getCaptureRegion();

private:
  void run();
  void selectSupportFoot();
  void runSearch();
  void generatePath(double time);

  Tree      *tree;
  Search    *search;
  SwingFoot *swingFoot;
  Pendulum  *pendulum;

  planner::Input  input;
  planner::Output output;

  Foot   suf;
  double dt;

  bool found;
};

} // namespace Capt

#endif // __PLANNER_H__