#include "model.h"
#include "param.h"
#include "plot_trajectory.h"
#include <stdlib.h>
#include <vector>

float PI = 3.1415;

using namespace std;
using namespace CA;

int main() {
  string root_dir = getenv("HOME");
  root_dir += "/study/capturability";
  string data_dir = root_dir + "/result/1step_new.csv";
  string model_dir = root_dir + "/data/nao.xml";
  string param_dir = root_dir + "/data/analysis.xml";

  Param param(param_dir);
  Model model(model_dir);

  vec2_t icp;
  icp.setCartesian(-0.013626, 0.030283);
  vec3_t com;
  com << -0.0136255, -0.0262312, 0.251226;
  vec3_t com_vel;
  com_vel << -5.81843e-07, 0.353957, -0.00553961;
  vec3_t torso;
  torso << -0.022417, -0.038482, 0.293978;
  vec3_t rleg;
  rleg << -0.00441721, -0.0530343, 0.000301669;
  vec3_t lleg;
  lleg << -0.00278837, 0.0571003, 0.0198305;

  PlotTrajectory plot(model, param, 0.01);
  plot.setIcp(icp);
  plot.setCom(com);
  plot.setComVel(com_vel);
  plot.setRLeg(rleg);
  plot.setLLeg(lleg);
  plot.calcDes();

  double t = 0.0;
  while (t <= 1.0) {
    plot.plotXY(t);
    // plot.plotYZ(t);
    // usleep(500 * 1000); // usleep takes sleep time in us
    t += 0.01;
  }

  return 0;
}
