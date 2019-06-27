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

  FootPlanner foot_planner(&model, capturability, &grid);
  vec3_t com, com_vel;
  vec3_t rleg, lleg;
  com << 0, 0, 0;
  vec2_t icp;
  icp.setCartesian(0, -0.025);
  com_vel.x() = (icp.x - com.x()) * sqrt(9.81 / 0.25);
  com_vel.y() = (icp.y - com.y()) * sqrt(9.81 / 0.25);
  com_vel.z() = 0.0;
  rleg << 0, -0.055, 0;
  lleg << 0, 0.055, 0;
  foot_planner.setComState(com, com_vel);
  foot_planner.setRLeg(rleg);
  foot_planner.setLLeg(lleg);
  foot_planner.plan();
  foot_planner.show();

  return 0;
}