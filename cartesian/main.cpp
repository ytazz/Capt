#include "capturability.h"
#include <iostream>
#include <chrono>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  // 時間計測用変数
  std::chrono::system_clock::time_point start, end_exe, end_save;
  start = std::chrono::system_clock::now();

  Model *model = new Model("/home/dl-box/Capturability/cartesian/data/valkyrie.xml");
  Param *param = new Param("/home/dl-box/Capturability/cartesian/data/valkyrie_xy.xml");

  Capturability cap(model, param);
  cap.analyze();
  end_exe = std::chrono::system_clock::now();

  printf("*** Result ***\n");
  cap.saveBasin("/home/dl-box/Capturability/cartesian/cpu/basin0.csv", 0, false);
  cap.saveBasin("/home/dl-box/Capturability/cartesian/cpu/basin1.csv", 1, false);
  cap.saveBasin("/home/dl-box/Capturability/cartesian/cpu/basin2.csv", 2, false);
  cap.saveBasin("/home/dl-box/Capturability/cartesian/cpu/basin3.csv", 3, false);
  cap.saveBasin("/home/dl-box/Capturability/cartesian/cpu/basin4.csv", 4, false);
  cap.saveBasin("/home/dl-box/Capturability/cartesian/cpu/basin5.csv", 5, false);
  cap.saveBasin("/home/dl-box/Capturability/cartesian/cpu/basin0.bin", 0, true);
  cap.saveBasin("/home/dl-box/Capturability/cartesian/cpu/basin1.bin", 1, true);
  cap.saveBasin("/home/dl-box/Capturability/cartesian/cpu/basin2.bin", 2, true);
  cap.saveBasin("/home/dl-box/Capturability/cartesian/cpu/basin3.bin", 3, true);
  cap.saveBasin("/home/dl-box/Capturability/cartesian/cpu/basin4.bin", 4, true);
  cap.saveBasin("/home/dl-box/Capturability/cartesian/cpu/basin5.bin", 5, true);

  //cap.saveTrans("cpu/0trans.csv", 0, false);
  //cap.saveTrans("cpu/1trans.csv", 1, false);
  //cap.saveTrans("cpu/2trans.csv", 2, false);
  //cap.saveTrans("cpu/3trans.csv", 3, false);
  //cap.saveTrans("cpu/4trans.csv", 4, false);
  //cap.saveTrans("cpu/5trans.csv", 5, false);
  //cap.saveTrans("/home/dl-box/Capturability/cartesian/cpu/trans0.bin", 0, true);
  //cap.saveTrans("/home/dl-box/Capturability/cartesian/cpu/trans1.bin", 1, true);
  //cap.saveTrans("/home/dl-box/Capturability/cartesian/cpu/trans2.bin", 2, true);
  //cap.saveTrans("/home/dl-box/Capturability/cartesian/cpu/trans3.bin", 3, true);
  //cap.saveTrans("/home/dl-box/Capturability/cartesian/cpu/trans4.bin", 4, true);
  //cap.saveTrans("/home/dl-box/Capturability/cartesian/cpu/trans5.bin", 5, true);

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
  fprintf(fp, "%d,%d,%d\n", time_exe, time_save, time_sum );
  fclose(fp);

  return 0;
}