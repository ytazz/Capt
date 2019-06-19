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
#include <iostream>

using namespace std;
using namespace CA;

int main(int argc, char const *argv[]) {
  Model model("nao.xml");
  // model.print();
  Param param("analysis.xml");
  // param.print();

  Kinematics kinematics(model);
  std::vector<float> joint;
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(1.7453); // right
  joint.push_back(0.0);
  joint.push_back(1.5708);
  joint.push_back(1.2217);
  joint.push_back(0.0);
  joint.push_back(1.7453); // left
  joint.push_back(0.0);
  joint.push_back(-1.5708);
  joint.push_back(-1.2217);
  joint.push_back(0.0);
  joint.push_back(0.0); // rleg
  joint.push_back(0.0);
  joint.push_back(-0.7448);
  joint.push_back(1.2568);
  joint.push_back(-0.5120);
  joint.push_back(0.0);
  joint.push_back(0.0); // lleg
  joint.push_back(0.0);
  joint.push_back(-0.7448);
  joint.push_back(1.2568);
  joint.push_back(-0.5120);
  joint.push_back(0.0);

  kinematics.forward(joint, CHAIN_BODY);
  std::vector<float> joints = kinematics.getJoints(CHAIN_BODY);
  for (size_t i = 0; i < joints.size(); i++) {
    std::cout << i << ": " << joints[i] << '\n';
  }
  // kinematics.getCom(CHAIN_BODY);
  //
  // Trajectory trajectory(model);
  // trajectory.setJoints(joint);
  // vec3_t world_p_com_ref;
  // vec3_t world_p_rleg_ref;
  // vec3_t world_p_lleg_ref;
  // vec3_t world_p_torso_ref;
  // world_p_com_ref << 0.0, 0.04, 0.25;
  // world_p_rleg_ref << 0.0173 + 0.025, -0.05 - 0.005, 0.045 - 0.04519;
  // world_p_lleg_ref << 0.0173 + 0.025, +0.05 + 0.005, 0.045 - 0.04519;
  // world_p_torso_ref << 0, 0, 0.2933;
  // trajectory.setRLegRef(world_p_rleg_ref);
  // trajectory.setLLegRef(world_p_lleg_ref);
  // trajectory.setComRef(world_p_com_ref);
  //
  // if (trajectory.calc())
  //   std::cout << "success" << '\n';
  // else
  //   std::cout << "failed" << '\n';
  //
  // world_p_torso_ref = trajectory.getTorsoRef();
  // world_p_rleg_ref = trajectory.getRLegRef();
  // world_p_lleg_ref = trajectory.getLLegRef();
  //
  // vec3_t torso_p_rleg = world_p_rleg_ref - world_p_torso_ref;
  vec3_t world_p_torso;
  world_p_torso << 0, 0, 0;
  vec3_t world_p_lleg;
  world_p_lleg << 0.0306181, 0.0750597, -0.294408;
  vec3_t torso_p_lleg = world_p_lleg - world_p_torso;
  //
  std::vector<float> joints_r, joints_l;
  // if (kinematics.inverse(torso_p_rleg, CHAIN_RLEG))
  //   joints_r = kinematics.getJoints(CHAIN_RLEG);
  // else
  //   std::cout << "Right side IK failed" << '\n';
  if (kinematics.inverse(torso_p_lleg, CHAIN_LLEG))
    joints_l = kinematics.getJoints(CHAIN_LLEG);
  else
    std::cout << "Left side IK failed" << '\n';
  for (size_t i = 0; i < joints_l.size(); i++) {
    std::cout << joints_l[i] << '\n';
  }
  // kinematics.forward(joints_r, CHAIN_RLEG);
  // kinematics.forward(joints_l, CHAIN_LLEG);
  // vec3_t torso_p_com = kinematics.getCom(CHAIN_BODY);
  // std::cout << "world_p_torso_ref" << '\n';
  // std::cout << world_p_torso_ref << '\n';
  // std::cout << "world_p_com" << '\n';
  // std::cout << world_p_torso_ref + torso_p_com << '\n';

  return 0;
}