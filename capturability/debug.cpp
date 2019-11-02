#include "model.h"
#include "param.h"
#include "grid.h"
#include "capturability.h"
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

  return 0;
}