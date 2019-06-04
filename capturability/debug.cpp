#include "analysis.h"
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

  State state, state_;
  Input input;
  GridState grid_state;
  GridInput grid_input;

  Analysis analysis(model, param);

  FILE *fp = fopen("test.csv", "w");
  fprintf(fp,
          "state_id,input_id,state.icp.r,state.icp.th,state.swft.r,state.swft."
          "th\n");

  Polygon polygon;
  polygon.setVertex(model.getVec("link", "foot_r"));
  polygon.printVertex();
  printf("------\n");
  polygon.getConvexHull();
  polygon.printConvex();
  Vector2 point;
  printf("------\n");
  for (int i = 0; i <= 20; i++) {
    for (int j = 0; j <= 20; j++) {
      point.setCartesian(-0.1 + 0.01 * i, -0.1 + 0.01 * j);
      if (polygon.inPolygon(point, polygon.getConvexHull())) {
        printf("%lf, %lf\n", -0.1 + 0.01 * i, -0.1 + 0.01 * j);
      }
    }
  }

  return 0;
}