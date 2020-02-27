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

  vec3_t pos(0, 0, 0);
  vec3_t pos_(0.2, 0.1, 0);

  Swing *swing = new Swing(model, param);
  swing->set(pos, pos_);

  for(int i = 0; i < 100; i++) {
    double t = 0.01 * i;
    vec3_t p = swing->getTraj(t);
    printf("%lf,%lf,%lf,%lf\n", t, p.x(), p.y(), p.z() );
  }

  // capturability
  // Capturability *capturability = new Capturability(grid);
  // capturability->loadBasin("cpu/Basin.csv");
  // capturability->loadNstep("cpu/Nstep.csv");

  return 0;
}