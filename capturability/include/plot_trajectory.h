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
  PlotTrajectory(Model model, Param param);
  ~PlotTrajectory();

  void setCom(vec3_t com);
  void setComVel(vec3_t com_vel);
  void setRLeg(vec3_t rleg);
  void setLLeg(vec3_t lleg);
  void calcRef();

  void plot(float t);

private:
  Gnuplot p;
  Model model;
  Param param;
  Planning planning;

  vec2_t world_p_icp;
  vec2_t world_p_cop;
  vec3_t world_p_com;
  vec3_t world_p_com_vel;
  vec3_t world_p_rleg;
  vec3_t world_p_lleg;
};
}

#endif // __PLOT_TRAJECTORY_H__
