#include "analysis_cpu.h"
#include <iostream>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model model("data/nao.xml");
  Param param("data/nao_xy.xml");

  Analysis analysis(model, param);
  for(int N = 1; N <= NUM_STEP_MAX; N++) {
    analysis.exe(N);
  }
  analysis.saveCop("csv/Cop.csv");
  analysis.saveStepTime("csv/StepTime.csv");
  analysis.saveBasin("csv/BasinCpu.csv");
  analysis.saveNstep("csv/NstepCpu.csv");

  return 0;
}