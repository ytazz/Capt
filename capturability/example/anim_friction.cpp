#include "capturability.h"
#include "cr_plot.h"
#include "friction_filter.h"
#include "grid.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "state.h"
#include <chrono>
#include <stdlib.h>
#include <vector>

float PI = 3.1415;

using namespace std;
using namespace Capt;

int main() {
  string root_dir = getenv("HOME");
  root_dir += "/study/capturability";
  string data_dir = root_dir + "/result/1step.csv";
  string model_dir = root_dir + "/data/nao.xml";
  string param_dir = root_dir + "/data/analysis.xml";

  Param param(param_dir);
  Model model(model_dir);

  Pendulum pendulum(model);

  Grid grid(param);
  Capturability capturability(model, param);
  capturability.load("1step.csv");

  State state;
  // state.icp.setCartesian(-0.010953, 0.083149);
  state.icp.setCartesian(-0.012082, 0.081432);
  state.swft.setCartesian(-0.100742, 0.109221);

  CRPlot cr_plot(model, param, "svg");
  GridState gstate;
  std::vector<CaptureSet> region;
  std::vector<CaptureSet> modified_region;

  FrictionFilter friction_filter(capturability, pendulum);
  vec2_t icp, com, com_vel;

  auto start = std::chrono::system_clock::now(); // 計測スタート時刻を保存
  gstate = grid.roundState(state);
  gstate.state.printCartesian();
  region = capturability.getCaptureRegion(gstate.id, 1);

  friction_filter.setCaptureRegion(region);
  icp = state.icp;
  com.setCartesian(-0.0206967 + 0.00966615, -0.0266499 + 0.0515595);
  com.printCartesian();
  com_vel = (icp - com) * sqrt(9.81 / 0.25);
  modified_region = friction_filter.getCaptureRegion(com, com_vel, 0.2);

  auto end = std::chrono::system_clock::now(); // 計測終了時刻を保存
  auto dur = end - start;                      // 要した時間を計算
  auto msec =
      std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
  // 要した時間をマイクロ秒（1/1000ms）に変換して表示
  std::cout << msec << " micro sec \n";
  // cr_plot.plot(gstate.state, region, com);
  cr_plot.plot(gstate.state, region, com);
  std::cout << "region: " << region.size() << '\n';
  // std::cout << "modified region: " << modified_region.size() << '\n';

  return 0;
}
