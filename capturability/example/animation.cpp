#include "capturability.h"
#include "cr_plot.h"
#include "grid.h"
#include "model.h"
#include "param.h"
#include "state.h"
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

  Grid grid(param);
  Capturability capturability(model, param);
  capturability.load("1step.csv");

  State state;
  state.icp.setCartesian(-0.013626, 0.1388);
  state.swft.setPolar(0.11, 3.14159 / 2);

  CRPlot cr_plot(model, param, "gif");
  GridState gstate;
  std::vector<CaptureSet> region;

  // cr_plot.plot();
  while (true) {
    state.icp.th += param.getVal("icp_th", "step");
    if (state.icp.th > 3.14)
      break;
    gstate = grid.roundState(state);
    region = capturability.getCaptureRegion(gstate.id, 1);

    cr_plot.plot(gstate.state, region);
    usleep(500 * 1000); // usleep takes sleep time in us
  }

  return 0;
}
