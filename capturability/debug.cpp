#include "analysis.h"
#include "grid.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "vector.h"
#include <iostream>

using namespace std;
using namespace CA;

int main(int argc, char const *argv[]) {
  Model model("nao.xml");
  model.parse();

  Param param("analysis.xml");
  param.parse();

  Grid grid(param);
  Vector2 icp, swft;
  icp.setPolar(0.06, 2.094);
  swft.setPolar(0.14, 0.698);
  State state;
  GridState grid_state;
  state.icp = icp;
  state.swft = swft;
  grid_state = grid.roundState(state);

  state.printPolar();
  printf("%d\n", grid_state.id);

  return 0;
}