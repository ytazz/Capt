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

  Model model("nao.xml");
  model.parse();

  Grid grid(param);

  Pendulum pendulum(model);
  Vector2 icp, icp_, cop;
  icp.setPolar(0.056, 2.64);
  cop.setPolar(0.04, 2.64);
  pendulum.setIcp(icp);
  pendulum.setCop(cop);

  float v_max = 1.0;
  float t_min = 0.1;
  float dt = 0.01;
  float r_foot = 0.04;
  float l_max = 0.22;
  Vector2 sw0;
  sw0.setPolar(0.158, 1.12);

  FILE *fp = fopen("capture_region.csv", "w");
  fprintf(fp, "time, icp_x, icp_y, 1step_x, 1step_y, 2step_x, 2step_y\n");
  for (int i = 0; i < 20; i++) {
    float t = t_min + dt * i;
    // float t = 0.3;
    icp_ = pendulum.getIcp(t);
    for (int j = 0; j < 360; j++) {
      float sw_r = v_max * (t - t_min);
      float sw_th = j * M_PI / 180.0;
      float sw_x = sw0.x + sw_r * cos(sw_th);
      float sw_y = sw0.y + sw_r * sin(sw_th);
      Vector2 swft;
      printf("x = %lf, y = %lf ", sw_x, sw_y);
      swft.setCartesian(sw_x, sw_y);
      float norm = (swft - icp_).norm();
      printf("deg = \t%3.0d, \tnorm = \t%lf\n", j, norm);
      // fprintf(fp, "%lf,%lf,%lf,%lf,%lf\n", icp_.x, icp_.y, swft.x, swft.y);
      if (0.09 <= swft.r && swft.r <= 0.22) {
        if (norm <= r_foot) {
          printf("in!\n");
          fprintf(fp, "%lf,%lf,%lf,%lf,%lf\n", t, icp_.x, icp_.y, swft.x,
                  swft.y);
        } else if (norm <= l_max * exp(-5.718 * t) + r_foot) {
          fprintf(fp, "%lf,%lf,%lf,,,%lf,%lf\n", t, icp_.x, icp_.y, swft.x,
                  swft.y);
        }
      }
    }
  }

  return 0;
}