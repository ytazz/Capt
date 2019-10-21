#include "analysis_cpu.h"
#include <iostream>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model model("data/valkyrie.xml");
  Param param("data/valkyrie_xy.xml");

  Analysis analysis(model, param);
  for(int N = 1; N <= NUM_STEP_MAX; N++) {
    analysis.exe(N);
  }
  analysis.saveCop("CopList.csv");
  analysis.saveBasin("BasinCpu.csv");
  analysis.saveNstep("NstepCpu.csv");

  return 0;
}