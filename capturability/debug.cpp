#include "model.h"
#include "param.h"
#include "grid.h"
#include "grid_map.h"
#include "timer.h"
#include "tree.h"
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

  GridMap *gridmap = new GridMap(param);
  // gridmap->plot();

  Tree *tree = new Tree(capturability, gridmap);
  tree->setStepMax(10);
  tree->generate();

  // draw path
  // StepPlot *plt = new StepPlot(model, param, grid);
  // plt->setFootstep(search->getFootstep() );
  // plt->plot();

  return 0;
}