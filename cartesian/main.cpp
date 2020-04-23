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

  Model *model = new Model("/home/dl-box/Capturability/cartesian/data/valkyrie.xml");
  Param *param = new Param("/home/dl-box/Capturability/cartesian/data/valkyrie_xy.xml");

  Capturability cap(model, param);
  cap.analyze();
  end_exe = std::chrono::system_clock::now();

  printf("*** Result ***\n");
  string path  = "/home/dl-box/Capturability/cartesian/cpu/";
  stringstream ss;
  for(int n = 0; n < cap.cap_basin.size(); n++){
    ss.str("");
    ss << path << "basin" << n << ".csv";
    cap.saveBasin(ss.str(), n, false);

    ss.str("");
    ss << path << "basin" << n << ".bin";
    cap.saveBasin(ss.str(), n, true );

    ss.str("");
    ss << path << "trans" << n << ".bin";
    cap.saveTrans(ss.str(), n, true );

    ss.str("");
    ss << path << "index" << n << ".bin";
    cap.saveTransIndex(ss.str(), n, true );
  }
  ss.str("");
  ss << path << "duration_map.bin";
  cap.saveDurationMap(ss.str(), true);

  ss.str("");
  ss << path << "icp_map.bin";
  cap.saveIcpMap(ss.str(), true);

  ss.str("");
  ss << path << "mu_map.bin";
  cap.saveMuMap(ss.str(), true);

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