#include "analysis.h"
#include "grid.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "vector.h"
#include <iostream>

using namespace std;
using namespace CA;

int main(int argc, char const *argv[]) {
  Model model("nao.xml");

  Param param("analysis.xml");

  Grid grid(param);

  Analysis analysis(model, param);
  analysis.exe(0);
  // analysis.exe(1);
  analysis.save("csv/0step_dsp.csv", 0);
  // analysis.save("csv/1step_dsp.csv", 1);
  // analysis.save("2step.csv", 2);

  return 0;
}