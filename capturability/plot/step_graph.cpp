#include "model.h"
#include "param.h"
#include "config.h"
#include "grid.h"
#include "grid_map.h"
#include "timer.h"
#include "tree.h"
#include "search.h"
#include "step_plot.h"
#include "monitor.h"
#include "planner.h"
#include <chrono>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model *model = new Model("data/valkyrie.xml");
  Param *param = new Param("data/valkyrie_xy.xml");
  Grid  *grid  = new Grid(param);

  // capturability
  Capturability *capturability = new Capturability(grid);
  capturability->loadBasin("cpu/Basin.csv");
  capturability->loadNstep("cpu/Nstep.csv");

  Tree   *tree   = new Tree(grid, capturability);
  Search *search = new Search(grid, tree);

  vec2_t rfoot = vec2_t(0, -0.2);
  vec2_t lfoot = vec2_t(0, +0.2);
  vec2_t icp   = vec2_t(0, 0);
  Foot   s_suf = Foot::FOOT_L;

  vec2_t step[11];
  step[0]  = vec2_t(0.00, -0.20);
  step[1]  = vec2_t(0.05, +0.25);
  step[2]  = vec2_t(0.25, +0.05);
  step[3]  = vec2_t(0.45, +0.25);
  step[4]  = vec2_t(0.65, +0.05);
  step[5]  = vec2_t(0.85, +0.25);
  step[6]  = vec2_t(1.15, +0.05);
  step[7]  = vec2_t(1.45, +0.25);
  step[8]  = vec2_t(1.75, +0.05);
  step[9]  = vec2_t(2.05, +0.25);
  step[10] = vec2_t(2.00, -0.20);
  arr2_t goal;
  for(int i = 0; i < 10; i++) {
    goal.push_back(step[i]);
  }

  Timer timer;
  timer.start();
  search->setStart(rfoot, lfoot, icp, s_suf);
  search->setReference(goal);
  search->calc();
  timer.end();
  timer.print();

  // draw path
  StepPlot *plt = new StepPlot(model, param, grid);
  plt->setSequence(search->getSequence() );
  plt->plot();

  return 0;
}