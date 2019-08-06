#include "Capt.h"
#include "plot_trajectory.h"
#include <stdlib.h>
#include <vector>

float PI = 3.1415;

using namespace std;
using namespace Capt;

int main() {
  string root_dir = getenv("HOME");
  root_dir += "/study/capturability";
  string data_dir = root_dir + "/result/1step_slowfoot.csv";
  string model_dir = root_dir + "/data/nao.xml";
  string param_dir = root_dir + "/data/analysis.xml";

  Param param(param_dir);
  Model model(model_dir);
  Grid grid(param);
  Capturability capturability(model, param);
  capturability.load("1step.csv");

  vec2_t icp;
  vec3_t com, com_vel;
  vec3_t rleg, lleg;
  com << 0, 0, 0;
  icp.setCartesian(-0.035, 0.035);
  com_vel.x() = (icp.x - com.x()) * sqrt(9.81 / 0.25);
  com_vel.y() = (icp.y - com.y()) * sqrt(9.81 / 0.25);
  com_vel.z() = 0.0;
  rleg << 0, -0.055, 0;
  lleg << 0, 0.055, 0;

  PlotTrajectory plot(model, param, capturability, grid, 0.01);
  plot.plan(icp, com, com_vel, rleg, lleg);
  double t = 0.0;
  while (t <= 1.0) {
    plot.plotXY(t);
    // plot.plotYZ(t);
    // usleep(500 * 1000); // usleep takes sleep time in us
    t += 0.01;
  }

  return 0;
}
