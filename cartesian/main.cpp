#include "analysis_cpu.h"
#include <iostream>
#include <chrono>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  // 時間計測用変数
  std::chrono::system_clock::time_point start, end_exe, end_save;
  start = std::chrono::system_clock::now();

  Model *model = new Model("data/valkyrie.xml");
  Param *param = new Param("data/valkyrie_xy.xml");
  Grid  *grid  = new Grid(param);

  Analysis analysis(model, param, grid);
  analysis.exe();
  end_exe = std::chrono::system_clock::now();

  printf("*** Result ***\n");
  analysis.saveBasin("cpu/Basin.csv");
  analysis.saveNstep("cpu/1step.csv", 1);
  analysis.saveNstep("cpu/2step.csv", 2);
  analysis.saveNstep("cpu/3step.csv", 3);
  analysis.saveNstep("cpu/4step.csv", 4);
  analysis.saveNstep("cpu/5step.csv", 5);
  analysis.saveNstep("cpu/6step.csv", 6);
  end_save = std::chrono::system_clock::now();

  printf("*** Time ***\n");
  int time_exe  = std::chrono::duration_cast<std::chrono::milliseconds>(end_exe - start).count();
  int time_save = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - end_exe).count();
  int time_sum  = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start).count();
  printf("  exe : %7d [ms]\n", time_exe);
  printf("  save: %7d [ms]\n", time_save);
  printf("  sum : %7d [ms]\n", time_sum);

  return 0;
}