#include "capturability.h"
#include "cr_plot.h"
#include "grid.h"
#include "model.h"
#include "param.h"
#include "state.h"
#include <stdlib.h>
#include <vector>

using namespace std;
using namespace Capt;

int main() {
  Param param("analysis.xml");
  Model model("nao.xml");
  Grid  grid(param);

  State state;
  // state.icp.setPolar(0.08, 30 * 3.14159 / 180);
  // state.swft.setPolar(0.11, 90 * 3.14159 / 180);
  state.icp.setCartesian(0.073724, 0.051622);
  state.swft.setCartesian(0.001018, 0.109979);

  int state_id = grid.getStateIndex(state);
  std::cout << "state_id: " << state_id << '\n';
  state = grid.getState(state_id);
  state.icp.printCartesian();
  state.swft.printCartesian();

  CRPlot cr_plot(model, param);
  cr_plot.setInput("Basin.csv", DataType::ZERO_STEP);
  cr_plot.setInput("Nstep.csv", DataType::N_STEP);

  cr_plot.setOutput("gif");
  cr_plot.animCaptureRegion(state);

  cr_plot.setOutput("svg");
  cr_plot.plotCaptureRegion(state);

  // cr_plot.setOutput("svg");
  // cr_plot.plotCaptureIcp(state);

  return 0;
}