#include "CA.h"
#include <chrono>
#include <iostream>

using namespace std;
using namespace CA;

int main(int argc, char const *argv[]) {
  Model model("nao.xml");
  Param param("analysis.xml");
  Grid grid(param);
  Capturability capturability(model, param);
  capturability.load("1step.csv");

  FootPlanner foot_planner(&model, &capturability, &grid);
  vec3_t com, com_vel;
  vec3_t rleg, lleg;
  com << -0.0206913, -0.0266439, 0.250184;
  com_vel << -0.000980377, 0.359949, 0.00233465;
  rleg << -0.00966615, -0.0515594, -9.12238e-05;
  lleg << -0.110408, 0.0576617, 0;
  foot_planner.setComState(com, com_vel);
  foot_planner.setRLeg(rleg);
  foot_planner.setLLeg(lleg);
  foot_planner.plan();
  foot_planner.show();

  return 0;
}