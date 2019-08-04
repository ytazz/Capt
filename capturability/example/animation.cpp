#include "capturability.h"
#include "cr_plot.h"
#include "grid.h"
#include "model.h"
#include "param.h"
#include "state.h"
#include <stdlib.h>
#include <vector>

using namespace std;
using namespace CA;

int main() {
  Param param("analysis.xml");
  Model model("nao.xml");

  State state;
  state.icp.setPolar(0.06, 90 * 3.14159 / 180);
  state.swft.setPolar(0.1, 120 * 3.14159 / 180);

  CRPlot cr_plot(model, param);
  cr_plot.setInput("result.csv");

  cr_plot.setOutput("gif");
  cr_plot.animCaptureRegion(state);

  cr_plot.setOutput("svg");
  cr_plot.plotCaptureRegion(state);

  return 0;
}