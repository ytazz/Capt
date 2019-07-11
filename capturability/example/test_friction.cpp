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
#include <iostream>

using namespace std;
using namespace CA;

int main(int argc, char const *argv[]) {
  Model model("nao.xml");
  // model.print();
  Param param("analysis.xml");
  // param.print();

  Grid grid(param);
  Pendulum pendulum(model);

  // Capturability capturability(model, param);
  // capturability.load("1step.csv");

  // std::vector<CaptureSet> region = capturability.getCaptureRegion(44457, 1);
  //
  // std::cout << "state" << '\n';
  // State state = grid.getState(44457);
  // state.printCartesian();
  // std::cout << "--------------------------" << '\n';
  // std::cout << "region: " << (int)region.size() << '\n';
  // for (size_t i = 0; i < region.size(); i++) {
  //   region[i].swft.printCartesian();
  // }
  //
  // FrictionFilter friction_filter(capturability, pendulum);
  // friction_filter.setCaptureRegion(region);
  // vec2_t icp, com, com_vel;
  // icp = state.icp;
  // com.setPolar(0.03, 100 * 3.14159 / 180);
  // com.printCartesian();
  // com_vel = (icp - com) * 5.71741;
  // std::vector<CaptureSet> modified_region =
  //     friction_filter.getCaptureRegion(com, com_vel, 0.2);
  // std::cout << "modified region: " << (int)modified_region.size() << '\n';
  // for (size_t i = 0; i < modified_region.size(); i++) {
  //   modified_region[i].swft.printCartesian();
  // }

  return 0;
}