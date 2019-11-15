#include "model.h"
#include "param.h"
#include "grid.h"
#include "grid_map.h"
#include "search.h"
#include "timer.h"
#include <chrono>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model *model = new Model("data/valkyrie.xml");
  Param *param = new Param("data/footstep.xml");
  Grid  *grid  = new Grid(param);

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

  Timer timer;
  timer.start();

  Search *search = new Search(gridmap, grid, capturability);

  vec2_t s_rfoot(0.0, -0.2);
  vec2_t s_lfoot(0.0, 0.2);
  vec2_t s_icp(0.0, 0.15);
  vec2_t g_foot(1.0, 0.0);
  double stance = 0.4;

  search->setStanceWidth(stance);
  search->setStart(s_rfoot, s_lfoot, s_icp, Foot::FOOT_L);
  search->setGoal(g_foot);

  search->init();
  while(search->step() ) {
    sleep(0.5);
  }
  // search->exe();
  // search->step();

  timer.end();
  timer.print();

  return 0;
}