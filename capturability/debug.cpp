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

  Vector2 icp, cop, icp_;
  icp.setPolar(0.06, 2.094);
  cop.setPolar(0.04, 2.094);
  icp.printCartesian("icp : ");
  cop.printCartesian("cop : ");

  Pendulum pendulum(model);
  pendulum.setIcp(icp);
  pendulum.setCop(cop);
  icp_ = pendulum.getIcp(0.28978);
  icp_.printCartesian("icp_: ");

  return 0;
}