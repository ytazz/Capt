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
  Param param("analysis.xml");
  Model model("nao.xml");

  Grid grid(param);
  Capturability capturability(model, param);
  capturability.load("csv/1step_ssp.csv");

  State state;
  state.icp.setPolar(0.06, 0);
  state.swft.setPolar(0.1, 120 * 3.14159 / 180);

  CRPlot cr_plot(model, param, "gif");
  GridState gstate;
  std::vector<CaptureSet> region;

  while (true) {
    state.icp.th += param.getVal("icp_th", "step");
    if (state.icp.th > 3.14)
      break;
    gstate = grid.roundState(state);
    region = capturability.getCaptureRegion(gstate.id, 1);
    std::cout << region.size() << '\n';

    if (!region.empty()) {
      cr_plot.plot(gstate.state, region);
    }
  }

  return 0;
}
