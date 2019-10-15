#include "grid.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "vector.h"
#include "cr_plot.h"
#include <iostream>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model model("nao.xml");
  // model.print();
  Param param("nao_xy.xml");
  // param.print();

  Grid grid(param);

  State state = grid.getState(0);

  CRPlot cr_plot(model, param);
  // cr_plot.setOutput("eps");
  // cr_plot.setCaptureRegion();
  cr_plot.setZerostep(state);
  cr_plot.setFoot(state);
  cr_plot.plot();

  return 0;
}