#include "capturability.h"
#include "cr_plot.h"
#include "friction_filter.h"
#include "grid.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "state.h"
#include <stdlib.h>
#include <vector>

float PI = 3.1415;

using namespace std;
using namespace CA;

int main() {
  string root_dir = getenv("HOME");
  root_dir += "/study/capturability";
  string data_dir = root_dir + "/result/1step.csv";
  string model_dir = root_dir + "/data/nao.xml";
  string param_dir = root_dir + "/data/analysis.xml";

  Param param(param_dir);
  Model model(model_dir);

  Pendulum pendulum(model);

  Grid grid(param);
  Capturability capturability(model, param);
  capturability.load("1step.csv");

  State state;
  state.icp.setCartesian(0.0, 0.08);
  state.swft.setPolar(0.14, 3.14159 * 3 / 4);

  CRPlot cr_plot(model, param, "svg");
  GridState gstate;
  std::vector<CaptureSet> region;

  // cr_plot.plot();
  gstate = grid.roundState(state);
  region = capturability.getCaptureRegion(gstate.id, 1);

  FrictionFilter friction_filter(capturability, pendulum);
  friction_filter.setCaptureRegion(region);
  vec2_t icp, com, com_vel;
  icp = state.icp;
  com.setPolar(0.03, 100 * 3.14159 / 180);
  com.printCartesian();
  com_vel = (icp - com) * 5.71741;
  std::vector<CaptureSet> modified_region =
      friction_filter.getCaptureRegion(com, com_vel, 0.2);

  cr_plot.plot(gstate.state, modified_region);

  return 0;
}
