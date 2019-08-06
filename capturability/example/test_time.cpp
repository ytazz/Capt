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
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model model("nao.xml");

  for (int i = 0; i < 100; i++) {
    auto start = std::chrono::system_clock::now(); // 計測スタート時刻を保存
    Polygon polygon;
    std::vector<vec2_t> region;
    polygon.clear();
    vec2_t l_foot;
    l_foot.setCartesian(0.0, 0.11);
    region = model.getVec("foot", "foot_r");
    region = model.getVec("foot", "foot_l", l_foot);
    polygon.setVertex(region);
    region = polygon.getConvexHull();
    vec2_t p;
    p.setCartesian(0.1, 0.1);
    polygon.inPolygon(p, region);
    auto end = std::chrono::system_clock::now(); // 計測終了時刻を保存
    auto dur = end - start;                      // 要した時間を計算
    auto msec =
        std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
    // 要した時間をミリ秒（1/1000秒）に変換して表示
    std::cout << i << ": " << msec << " micro sec \n";
  }

  return 0;
}