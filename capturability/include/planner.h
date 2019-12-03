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
  double       time;
  Capt::vec3_t rfoot;
  Capt::vec3_t lfoot;
  Capt::vec3_t icp;
};

struct Output {
  Capt::vec3_t rfoot;
  Capt::vec3_t lfoot;
  Capt::vec3_t icp;
  Capt::vec3_t cop;
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
};

} // namespace Capt

#endif // __PLANNER_H__