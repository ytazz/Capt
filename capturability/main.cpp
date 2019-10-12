#include "analysis_cpu.h"
#include "grid.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "vector.h"
#include <iostream>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model model("nao.xml");
  Param param("analysis.xml");
  Grid  grid(param);

  Analysis analysis(model, param);
  analysis.exe(0);
  analysis.save("csv/0step_ssp.csv", 0);
  analysis.exe(1);
  // analysis.save("csv/1step_ssp.csv", 1);
  // analysis.exe(2);
  // analysis.save("csv/2step_ssp.csv", 2);
  // analysis.exe(3);
  // analysis.save("csv/3step_ssp.csv", 3);

  return 0;
}
