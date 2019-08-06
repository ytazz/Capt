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
using namespace Capt;

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

  Trajectory trajectory(model);
  trajectory.setJoints(joint);
  vec3_t world_p_torso;
  vec3_t world_p_com;
  vec3_t world_p_lleg, world_p_rleg;
  vec3_t world_p_com_ref;
  vec3_t world_p_rleg_ref, world_p_lleg_ref;
  vec3_t world_p_torso_ref;

  // setting
  world_p_com << -0.00999698, -0.0104361, 0.249976;
  world_p_torso << -0.018111, -0.0053466, 0.295333;
  world_p_lleg << 0, 0.055, -0.000170774;
  world_p_rleg << 0, -0.055, -0.000170774;

  trajectory.setTorso(world_p_torso);
  trajectory.setRLegRef(world_p_rleg);
  trajectory.setLLegRef(world_p_lleg);
  trajectory.setComRef(world_p_com);

  if (trajectory.calc())
    std::cout << "success" << '\n';
  else
    std::cout << "failed" << '\n';

  world_p_torso_ref = trajectory.getTorsoRef();
  world_p_rleg_ref = trajectory.getRLegRef();
  world_p_lleg_ref = trajectory.getLLegRef();

  vec3_t torso_p_rleg = world_p_rleg_ref - world_p_torso_ref;
  vec3_t torso_p_lleg = world_p_lleg_ref - world_p_torso_ref;
  std::vector<float> joints_r, joints_l;
  if (kinematics.inverse(torso_p_rleg, CHAIN_RLEG)) {
    joints_r = kinematics.getJoints(CHAIN_RLEG);
    for (size_t i = 0; i < joints_r.size(); i++) {
      std::cout << joints_r[i] << '\n';
    }
  } else {
    std::cout << "Right side IK failed" << '\n';
  }

  std::cout << "----------------" << '\n';

  if (kinematics.inverse(torso_p_lleg, CHAIN_LLEG)) {
    joints_l = kinematics.getJoints(CHAIN_LLEG);
    for (size_t i = 0; i < joints_l.size(); i++) {
      std::cout << joints_l[i] << '\n';
    }
  } else {
    std::cout << "Left side IK failed" << '\n';
  }

  std::cout << "----------------" << '\n';
  std::cout << "world_p_torso_ref" << '\n';
  std::cout << world_p_torso_ref << '\n';
  std::cout << "world_p_rleg_ref" << '\n';
  std::cout << world_p_rleg_ref << '\n';
  std::cout << "world_p_lleg_ref" << '\n';
  std::cout << world_p_lleg_ref << '\n';
  std::cout << "torso_p_rleg" << '\n';
  std::cout << torso_p_rleg << '\n';
  std::cout << "torso_p_lleg" << '\n';
  std::cout << torso_p_lleg << '\n';

  return 0;
}