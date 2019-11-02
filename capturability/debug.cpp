#include "model.h"
#include "param.h"
#include "grid.h"
#include "capturability.h"
#include "search.h"
#include <chrono>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model *model = new Model("data/valkyrie.xml");
  Param *param = new Param("data/valkyrie_xy.xml");
  Grid  *grid  = new Grid(param);

  Capturability *capturability = new Capturability(grid);
  capturability->load("_cpu/Basin.csv", DataType::BASIN);
  capturability->load("_cpu/Nstep.csv", DataType::NSTEP);

  Search *search = new Search(grid, capturability);
  vec2_t  s_rfoot(0.0, 0.0);
  vec2_t  s_lfoot(0.0, 0.4);
  vec2_t  s_icp(0.0, 0.15);
  vec2_t  g_rfoot(1.0, 0.0);
  vec2_t  g_lfoot(1.0, 0.4);

  search->setStart(s_rfoot, s_lfoot, s_icp, Foot::FOOT_R);
  search->setGoal(g_rfoot, g_lfoot);

  bool flag = true;
  while(flag) {
    search->exe();
    search->step();
  }

  return 0;
}