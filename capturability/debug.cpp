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
  Model  *model  = new Model("data/valkyrie.xml");
  Param  *param  = new Param("data/valkyrie_xy.xml");
  Config *config = new Config("data/valkyrie_config.xml");
  Grid   *grid   = new Grid(param);

  // capturability
  Capturability *capturability = new Capturability(grid);
  capturability->loadBasin("cpu/Basin.csv");
  capturability->loadNstep("cpu/Nstep.csv");

  Monitor *monitor = new Monitor(model, grid, capturability);
  Planner *planner = new Planner(model, param, config, grid, capturability);

  // footstep
  Step step[10];
  step[0].pos = vec3_t(0.025, 0.225, 0.000);
  step[0].suf = Foot::FOOT_L;
  step[1].pos = vec3_t(0.275, 0.025, 0.000);
  step[1].suf = Foot::FOOT_R;
  step[2].pos = vec3_t(0.525, 0.225, 0.000);
  step[2].suf = Foot::FOOT_L;
  step[3].pos = vec3_t(0.775, 0.025, 0.000);
  step[3].suf = Foot::FOOT_R;
  step[4].pos = vec3_t(1.025, 0.225, 0.000);
  step[4].suf = Foot::FOOT_L;
  step[5].pos = vec3_t(1.275, 0.025, 0.000);
  step[5].suf = Foot::FOOT_R;
  step[6].pos = vec3_t(1.525, 0.225, 0.000);
  step[6].suf = Foot::FOOT_L;
  step[7].pos = vec3_t(1.775, 0.025, 0.000);
  step[7].suf = Foot::FOOT_R;
  step[8].pos = vec3_t(2.025, 0.225, 0.000);
  step[8].suf = Foot::FOOT_L;
  step[9].pos = vec3_t(2.000, -0.200, 0.000);
  step[9].suf = Foot::FOOT_R;
  Footstep footstep;
  for(int i = 0; i < 10; i++) {
    footstep.push_back(step[i]);
  }

  EnhancedState state;
  state.footstep = footstep;
  state.rfoot    = vec3_t(0, -0.2, 0);
  state.lfoot    = vec3_t(0, +0.2, 0);
  state.icp      = vec3_t(0, 0, 0);
  state.elapsed  = 0.0;
  state.s_suf    = Foot::FOOT_R;

  Timer timer;
  timer.start();
  // planner->set(state);
  // planner->plan();
  bool safe = monitor->check(state, vec3Tovec2(step[0].pos) );
  timer.end();
  timer.print();
  printf("safe: %d\n", safe);

  // draw path
  // StepPlot *plt = new StepPlot(model, param, grid);
  // plt->setSequence(planner->getSequence() );
  // plt->plot();

  return 0;
}