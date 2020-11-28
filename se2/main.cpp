#define _CRT_SECURE_NO_WARNINGS

#include "capturability.h"
#include <iostream>
#include <chrono>

using namespace std;
using namespace Capt;

const int nmax = 10;

int main(int argc, char const *argv[]) {
  // 時間計測用変数
  std::chrono::system_clock::time_point start, end_exe, end_save;
  start = std::chrono::system_clock::now();

  Scenebuilder::XML xml;
  xml.Load("conf/capt.xml");
  
  Capturability cap;
  cap.Read(xml.GetRootNode());
  cap.Analyze();
  end_exe = std::chrono::system_clock::now();

  printf("*** Result ***\n");
  cap.Save("../data/");
  end_save = std::chrono::system_clock::now();

  printf("*** Time ***\n");
  int time_exe  = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end_exe  - start  ).count();
  int time_save = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end_save - end_exe).count();
  int time_sum  = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start  ).count();
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