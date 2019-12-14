#include "model.h"
#include "param.h"
#include "config.h"
#include "grid.h"
#include "grid_map.h"
#include "timer.h"
#include "tree.h"
#include "search.h"
#include "step_plot.h"
#include "planner.h"
#include <chrono>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model  *model  = new Model("data/valkyrie.xml");
  Param  *param  = new Param("data/footstep.xml");
  Config *config = new Config("data/valkyrie_config.xml");
  Grid   *grid   = new Grid(param);

  // capturability
  Capturability *capturability = new Capturability(grid);
  capturability->load("gpu/Basin.csv", DataType::BASIN);
  capturability->load("gpu/Nstep.csv", DataType::NSTEP);

  Planner *planner = new Planner(model, param, config, grid, capturability);

  planner::Input input;
  input.elapsed = 0.0;
  input.suf     = Foot::FOOT_R;
  input.rfoot   = vec3_t(0, -0.2, 0);
  input.lfoot   = vec3_t(0, +0.2, 0);
  input.icp     = vec3_t(0, 0, 0);
  input.goal    = vec3_t(1, 0, 0);
  input.stance  = 0.4;

  // Timer timer;
  // timer.start();
  planner::Output output;
  planner->set(input);
  // timer.end();
  // timer.print();

  // draw path
  StepPlot *plt = new StepPlot(model, param, grid);
  plt->setSequence(planner->getSequence() );
  plt->plot();

  return 0;
}