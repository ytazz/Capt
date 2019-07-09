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
  std::cout << "0" << '\n';

  Grid grid(param);
  Capturability capturability(model, param);
  std::cout << "1" << '\n';
  capturability.load("csv/0step_ssp.csv");
  std::cout << "2" << '\n';

  State state;
  state.icp.setPolar(0.03, 0);
  state.swft.setPolar(0.10, 3.14159 / 2);

  std::cout << "3" << '\n';
  CRPlot cr_plot(model, param, "gif");
  GridState gstate;
  std::vector<CaptureSet> region;
  std::cout << "4" << '\n';

  while (true) {
    std::cout << "5" << '\n';
    state.icp.th += param.getVal("icp_th", "step");
    if (state.icp.th > 3.14)
      break;
    std::cout << "6" << '\n';
    gstate = grid.roundState(state);
    std::cout << "7" << '\n';
    region = capturability.getCaptureRegion(gstate.id, 0);
    std::cout << "8" << '\n';

    if (!region.empty()) {
      std::cout << "9" << '\n';
      cr_plot.plot(gstate.state, region);
    }

    std::cout << "10" << '\n';
  }

  return 0;
}
