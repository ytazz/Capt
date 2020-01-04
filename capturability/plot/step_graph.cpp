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
  Foot   s_suf = Foot::FOOT_R;
  vec2_t goal  = vec2_t(0.3, -0.2);
  Foot   g_suf = Foot::FOOT_R;

  Timer timer;
  timer.start();
  search->setStart(rfoot, lfoot, icp, s_suf);
  search->setGoal(goal, g_suf);
  search->calc();
  timer.end();
  timer.print();

  // draw path
  StepPlot *plt = new StepPlot(model, param, grid);
  plt->setSequence(search->getSequence() );
  plt->plot();

  return 0;
}