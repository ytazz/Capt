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
  Model model("val.xml");
  // model.print();
  Param param("param_val_xy.xml");
  // param.print();

  Grid   grid(param);
  CRPlot cr_plot(model, param);
  // cr_plot.setOutput("eps");
  cr_plot.plot();

  return 0;
}