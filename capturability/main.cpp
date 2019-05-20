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
  Param param("analysis.xml");
  param.parse();
  // param.print();

  Model model("nao.xml");
  model.parse();
  // model.print();

  Grid grid(param);
  // grid.getState(1, 1, 1, 1).printPolar();

  Pendulum pendulum(model);
}