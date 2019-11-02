#include "model.h"
#include "param.h"
#include "grid.h"
#include <chrono>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model *model = new Model("data/valkyrie.xml");

  Param *param = new Param("data/valkyrie_xy.xml");
  param->print();

  Grid *grid = new Grid(param);

  return 0;
}