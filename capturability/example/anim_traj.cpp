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

  vec3_t com;
  com << -0.0136254, -0.028355, 0.250988;
  vec3_t com_vel;
  com_vel << 0, 0.707916, -0.00822986;
  vec3_t torso;
  torso << -0.022432, -0.0407954, 0.293682;
  vec3_t rleg;
  rleg << -0.0040939, -0.0541447, 0;
  vec3_t lleg;
  lleg << -0.00274675, 0.0559155, 0.0199344;

  PlotTrajectory plot(model, param);
  plot.setCom(com);
  plot.setComVel(com_vel);
  plot.setRLeg(rleg);
  plot.setLLeg(lleg);
  plot.calcRef();

  double t = 0.0;
  while (t <= 0.3) {
    plot.plot(t);
    // usleep(500 * 1000); // usleep takes sleep time in us
    t += 0.001;
    std::cout << t << '\n';
  }

  return 0;
}
