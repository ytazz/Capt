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
  Grid  *grid  = new Grid(*param);

  Capturability* capturability = new Capturability(*model, *param);
  capturability->load("cpu/Basin.csv", DataType::BASIN);
  capturability->load("cpu/Nstep.csv", DataType::NSTEP);

  delete model;
  delete param;
  delete grid;
  delete capturability;

  // Search *search = new Search(grid, capturability);

  // StepPlot plot(model, param);
  // arr2_t   foot_r, foot_l, icp;
  //
  // foot_r.resize(3);
  // foot_r[0].setCartesian(0.25, -0.5);
  // foot_r[1].setCartesian(0.75, -0.5);
  // foot_r[2].setCartesian(1.25, -0.5);
  //
  // foot_l.resize(2);
  // foot_l[0].setCartesian(0.50, 0.5);
  // foot_l[1].setCartesian(1.00, 0.5);
  //
  // icp.resize(6);
  // icp[0].setCartesian(0, 0);
  // icp[1].setCartesian(0.25, -0.5);
  // icp[2].setCartesian(0.50, +0.5);
  // icp[3].setCartesian(0.75, -0.5);
  // icp[4].setCartesian(1.00, +0.5);
  // icp[5].setCartesian(1.25, -0.5);
  //
  // plot.setFootR(foot_r);
  // plot.setFootL(foot_l);
  // plot.setIcp(icp);
  // plot.plot();

  return 0;
}