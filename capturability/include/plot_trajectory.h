#ifndef __PLOT_TRAJECTORY_H__
#define __PLOT_TRAJECTORY_H__

#include "gnuplot.h"

#include "capturability.h"
#include "input.h"
#include "param.h"
#include "planning.h"
#include "state.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

namespace CA {

class PlotTrajectory {
public:
  PlotTrajectory(Model model, Param param, Capturability capturability,
                 Grid grid, float timestep);
  ~PlotTrajectory();

  void plan(vec2_t icp, vec3_t com, vec3_t com_vel, vec3_t rleg, vec3_t lleg);

  void plotXY(float t);

private:
  void fileOutput(vec2_t vec);
  void fileOutput(vec3_t vec);

  Gnuplot p;
  Model model;
  Param param;
  Planning planning;
  FootPlanner foot_planner;

  FILE *fp;

  vec2_t world_p_icp;
  vec2_t world_p_cop;
  vec3_t world_p_com;
  vec3_t world_p_com_vel;
  vec3_t world_p_rleg;
  vec3_t world_p_lleg;

  std::vector<vec2_t> foot;
};
}

#endif // __PLOT_TRAJECTORY_H__
