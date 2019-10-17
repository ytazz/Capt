#include "analysis_cpu.h"
#include <iostream>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model model("data/nao.xml");
  Param param("data/nao_rt.xml");
  Grid  grid(param);

  Analysis analysis(model, param);
  for(int N = 1; N <= NUM_STEP_MAX; N++) {
    analysis.exe(N);
  }
  analysis.saveBasin("BasinCpu.csv", false);
  analysis.saveNstep("NstepCpu.csv", false);

  return 0;
}
