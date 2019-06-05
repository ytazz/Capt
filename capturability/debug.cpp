#include "analysis.h"
#include "capturability.h"
#include "grid.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "polygon.h"
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
  Capturability capturability(model, param);

  Analysis analysis(model, param);
  // analysis.exe(1);
  // analysis.save("1step.csv", 1);

  Vector2 point;
  point.setCartesian(0.05, 0.05);
  Polygon polygon;
  polygon.setVertex(model.getVec("link", "foot_r"));

  polygon.getClosestPoint(point, polygon.getConvexHull());

  return 0;
}