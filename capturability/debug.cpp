#include "model.h"
#include "param.h"
#include "grid.h"
#include "grid_map.h"
#include "search.h"
#include "timer.h"
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

  for(int i = 0; i <= 5; i++) {
    for(int j = 0; j <= 5; j++) {
      vec2_t pos(0.5 + 0.05 * i, -0.5 + 0.05 * j);
      // gridmap->setObstacle(pos);
    }
  }

  // footstep search
  Search *search = new Search(gridmap, grid, capturability);

  vec2_t s_rfoot(0.0, -0.2);
  vec2_t s_lfoot(0.0, 0.2);
  vec2_t s_icp(0.0, 0.0);
  vec2_t g_foot(1.0, 0.0);
  double stance = 0.4;

  search->setStanceWidth(stance);
  search->setStart(s_rfoot, s_lfoot, s_icp, Foot::FOOT_L);
  search->setGoal(g_foot);

  search->exe();

  // draw path
  Trans trans = search->getTrans();

  StepPlot *plt = new StepPlot(model, param, grid);
  plt->setTransition(trans.states, trans.inputs, Foot::FOOT_L);
  plt->plot();

  return 0;
}