#include "model.h"
#include "param.h"
#include "grid.h"
#include "grid_map.h"
#include "timer.h"
#include "tree.h"
#include "search.h"
#include "step_plot.h"
#include <chrono>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model *model = new Model("data/valkyrie.xml");
  Param *param = new Param("data/footstep.xml");
  Grid  *grid  = new Grid(param);

  // capturability
  Capturability *capturability = new Capturability(grid);
  capturability->load("gpu/Basin.csv", DataType::BASIN);
  capturability->load("gpu/Nstep.csv", DataType::NSTEP);

  Tree *tree = new Tree(capturability, grid, param);
  // tree->setPreviewStep(10);
  // tree->generate();

  // calc path
  vec2_t  rfoot(0.0, -0.2);
  vec2_t  lfoot(0.0, 0.2);
  vec2_t  icp(0.0, 0.0);
  vec2_t  gfoot(1.0, -0.5);
  Search* search = new Search(grid, tree);
  Timer   timer;
  timer.start();
  search->setStart(rfoot, lfoot, icp, Foot::FOOT_R);
  search->setGoal(gfoot);
  search->calc();
  timer.end();
  timer.print();

  // draw path
  StepPlot *plt = new StepPlot(model, param, grid);
  plt->setFootstep(search->getFootstep() );
  plt->plot();

  return 0;
}