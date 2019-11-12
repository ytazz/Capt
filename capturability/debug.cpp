#include "model.h"
#include "param.h"
#include "grid.h"
#include "grid_map.h"
#include <chrono>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  // Model *model = new Model("data/valkyrie.xml");
  Param *param = new Param("data/footstep.xml");

  GridMap *gridmap = new GridMap(param);

  for(int i = 0; i <= 5; i++) {
    for(int j = 0; j <= 5; j++) {
      vec2_t pos(0.5 + 0.05 * i, -0.5 + 0.05 * j);
      gridmap->setObstacle(pos);
    }
  }

  gridmap->plot();

  return 0;
}