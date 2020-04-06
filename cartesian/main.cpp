#include "capturability.h"
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
  Swing *swing = new Swing(model, param);

  Capturability cap(model, param, grid, swing);
  cap.analyze();
  end_exe = std::chrono::system_clock::now();

  printf("*** Result ***\n");
  //cap.saveBasin("cpu/0basin.csv", 0, false);
  //cap.saveBasin("cpu/1basin.csv", 1, false);
  //cap.saveBasin("cpu/2basin.csv", 2, false);
  //cap.saveBasin("cpu/3basin.csv", 3, false);
  //cap.saveBasin("cpu/4basin.csv", 4, false);
  //cap.saveBasin("cpu/5basin.csv", 5, false);
  cap.saveBasin("cpu/basin0.bin", 0, true);
  cap.saveBasin("cpu/basin1.bin", 1, true);
  cap.saveBasin("cpu/basin2.bin", 2, true);
  cap.saveBasin("cpu/basin3.bin", 3, true);
  cap.saveBasin("cpu/basin4.bin", 4, true);
  cap.saveBasin("cpu/basin5.bin", 5, true);

  //cap.saveTrans("cpu/0trans.csv", 0, false);
  //cap.saveTrans("cpu/1trans.csv", 1, false);
  //cap.saveTrans("cpu/2trans.csv", 2, false);
  //cap.saveTrans("cpu/3trans.csv", 3, false);
  //cap.saveTrans("cpu/4trans.csv", 4, false);
  //cap.saveTrans("cpu/5trans.csv", 5, false);
  cap.saveTrans("cpu/trans0.bin", 0, true);
  cap.saveTrans("cpu/trans1.bin", 1, true);
  cap.saveTrans("cpu/trans2.bin", 2, true);
  cap.saveTrans("cpu/trans3.bin", 3, true);
  cap.saveTrans("cpu/trans4.bin", 4, true);
  cap.saveTrans("cpu/trans5.bin", 5, true);

  end_save = std::chrono::system_clock::now();

  printf("*** Time ***\n");
  int time_exe  = std::chrono::duration_cast<std::chrono::milliseconds>(end_exe - start).count();
  int time_save = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - end_exe).count();
  int time_sum  = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start).count();
  printf("  exe : %7d [ms]\n", time_exe);
  printf("  save: %7d [ms]\n", time_save);
  printf("  sum : %7d [ms]\n", time_sum);

  // save calculation result
  FILE *fp = fopen("log.csv", "w");
  fprintf(fp, "state,exe,save,sum\n");
  fprintf(fp, "%d,", grid->getNumState() );
  fprintf(fp, "%d,%d,%d\n", time_exe, time_save, time_sum );
  fclose(fp);

  return 0;
}