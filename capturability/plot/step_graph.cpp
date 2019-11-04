#include "grid.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "base.h"
#include "search.h"
#include "step_plot.h"
#include <iostream>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model *model = new Model("data/valkyrie.xml");
  Param *param = new Param("data/footstep.xml");
  Grid  *grid  = new Grid(param);

  Capturability *capturability = new Capturability(grid);
  capturability->load("gpu/Basin.csv", DataType::BASIN);
  capturability->load("gpu/Nstep.csv", DataType::NSTEP);

  Search *search = new Search(grid, capturability);

  vec2_t s_rfoot(0.0, 0.0);
  vec2_t s_lfoot(0.0, 0.4);
  vec2_t s_icp(0.0, 0.15);
  vec2_t g_rfoot(1.0, 0.0);
  vec2_t g_lfoot(1.0, 0.4);

  search->setStart(s_rfoot, s_lfoot, s_icp, Foot::FOOT_R);
  search->setGoal(g_rfoot, g_lfoot);

  bool flag = true;
  search->exe();
  // while(flag) {
  //   search->step();
  // }

  StepPlot *plot = new StepPlot(model, param, grid);
  arr2_t    foot_r, foot_l, icp;

  foot_r.resize(3);
  foot_r[0] << 0.25, -0.5;
  foot_r[1] << 0.75, -0.5;
  foot_r[2] << 1.25, -0.5;

  foot_l.resize(2);
  foot_l[0] << 0.50, 0.5;
  foot_l[1] << 1.00, 0.5;

  icp.resize(6);
  icp[0] << 0, 0;
  icp[1] << 0.25, -0.5;
  icp[2] << 0.50, +0.5;
  icp[3] << 0.75, -0.5;
  icp[4] << 1.00, +0.5;
  icp[5] << 1.25, -0.5;

  plot->setFootR(foot_r);
  plot->setFootL(foot_l);
  plot->setIcp(icp);
  plot->plot();

  delete model;
  delete param;
  delete grid;
  delete capturability;
  delete search;
  delete plot;

  return 0;
}