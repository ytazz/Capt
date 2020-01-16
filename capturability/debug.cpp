#include "Capt.h"
#include "step_plot.h"
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
  // generator
  Generator *generator = new Generator(model);

  Planner *planner = new Planner(model, param, config, grid, capturability);

  // footstep
  Step step[12];
  step[0].pos  = vec3_t(0.00, +0.20, 0);
  step[0].suf  = Foot::FOOT_L;
  step[1].pos  = vec3_t(0.00, -0.20, 0);
  step[1].suf  = Foot::FOOT_R;
  step[2].pos  = vec3_t(0.05, +0.25, 0);
  step[2].suf  = Foot::FOOT_L;
  step[3].pos  = vec3_t(0.25, +0.05, 0);
  step[3].suf  = Foot::FOOT_R;
  step[4].pos  = vec3_t(0.45, +0.25, 0);
  step[4].suf  = Foot::FOOT_L;
  step[5].pos  = vec3_t(0.65, +0.05, 0);
  step[5].suf  = Foot::FOOT_R;
  step[6].pos  = vec3_t(0.85, +0.25, 0);
  step[6].suf  = Foot::FOOT_L;
  step[7].pos  = vec3_t(1.15, +0.05, 0);
  step[7].suf  = Foot::FOOT_R;
  step[8].pos  = vec3_t(1.45, +0.25, 0);
  step[8].suf  = Foot::FOOT_L;
  step[9].pos  = vec3_t(1.75, +0.05, 0);
  step[9].suf  = Foot::FOOT_R;
  step[10].pos = vec3_t(2.05, +0.25, 0);
  step[10].suf = Foot::FOOT_L;
  step[11].pos = vec3_t(2.00, -0.20, 0);
  step[11].suf = Foot::FOOT_R;
  Footstep footstep;
  for(int i = 0; i < 12; i++) {
    footstep.push_back(step[i]);
  }

  generator->calc(&footstep);

  EnhancedState state;
  state.footstep = footstep;
  state.rfoot    = vec3_t(0, -0.2, 0);
  state.lfoot    = vec3_t(0, +0.2, 0);
  state.icp      = vec3_t(0, 0, 0);
  state.elapsed  = 0.0;
  state.s_suf    = Foot::FOOT_R;

  Timer timer;
  timer.start();
  planner->set(state);
  planner->plan();
  timer.end();
  timer.print();

  // draw path
  StepPlot *plt = new StepPlot(model, param, grid);
  plt->setSequence(planner->getSequence() );
  plt->plot();

  return 0;
}