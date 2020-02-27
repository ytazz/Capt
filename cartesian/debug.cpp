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
  // param->print();

  vec2_t pos(0.2, 0.1);

  if(grid->isSteppable(pos) ) {
    printf("ok\n");
  }else{
    printf("ng\n");
  }

  // capturability
  // Capturability *capturability = new Capturability(grid);
  // capturability->loadBasin("cpu/Basin.csv");
  // capturability->loadNstep("cpu/Nstep.csv");

  return 0;
}