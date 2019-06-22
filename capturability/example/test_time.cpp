#include "analysis.h"
#include "capturability.h"
#include "friction_filter.h"
#include "grid.h"
#include "kinematics.h"
#include "loader.h"
#include "model.h"
#include "monitor.h"
#include "param.h"
#include "pendulum.h"
#include "planning.h"
#include "polygon.h"
#include "trajectory.h"
#include "vector.h"
#include <chrono>
#include <iostream>

using namespace std;
using namespace CA;

int main(int argc, char const *argv[]) {
  Model model("nao.xml");
  Polygon polygon;

  auto start = std::chrono::system_clock::now(); // 計測スタート時刻を保存
  std::vector<vec2_t> region;
  region = model.getVec("foot", "foot_l");
  polygon.setVertex(region);
  region = polygon.getConvexHull();
  vec2_t vec;
  vec.setCartesian(0.1, 0.1);
  polygon.getClosestPoint(vec, region);
  auto end = std::chrono::system_clock::now(); // 計測終了時刻を保存
  auto dur = end - start;                      // 要した時間を計算
  auto msec =
      std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
  // 要した時間をミリ秒（1/1000秒）に変換して表示
  std::cout << msec << " milli sec \n";

  return 0;
}