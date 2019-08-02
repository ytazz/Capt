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

  CRPlot cr_plot(model, param);
  cr_plot.setInput("result.csv");

  // cr_plot.setOutput("gif");
  // cr_plot.plotCaptureRegion();

  cr_plot.setOutput("svg");
  cr_plot.plotCaptureIcp();

  return 0;
}