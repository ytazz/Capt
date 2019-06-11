#include "analysis.h"
#include "capturability.h"
#include "grid.h"
#include "kinematics.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
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
  Capturability capturability(model, param);

  Analysis analysis(model, param);
  // analysis.exe(1);
  // analysis.save("1step.csv", 1);

  // capturability.load("1step.csv");
  // std::vector<Input> region = capturability.getCaptureRegion(44392, 1);
  //
  // for (size_t i = 0; i < region.size(); i++) {
  //   region[i].swft.printCartesian();
  // }

  Kinematics kinematics(model);
  std::vector<float> joint;
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0); // rleg
  joint.push_back(0.0);
  joint.push_back(-0.1745);
  joint.push_back(+0.1745);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0); // lleg
  joint.push_back(0.0);
  joint.push_back(-0.1745);
  joint.push_back(+0.1745);
  joint.push_back(0.0);
  joint.push_back(0.0);

  kinematics.forward(joint, CHAIN_BODY);
  kinematics.getCom(CHAIN_BODY);

  Trajectory trajectory(model);
  trajectory.setJoints(joint);
  vec3_t world_p_com_ref;
  vec3_t world_p_rleg_ref;
  vec3_t world_p_lleg_ref;
  vec3_t world_p_torso_ref;
  world_p_com_ref << 0.05, 0.0, 0.25;
  world_p_rleg_ref << 0.0173, -0.05, 0.045;
  world_p_lleg_ref << 0.0173, +0.05, 0.045;
  world_p_torso_ref << 0, 0, 0;
  trajectory.setRLegRef(world_p_rleg_ref);
  trajectory.setLLegRef(world_p_lleg_ref);
  trajectory.setComRef(world_p_com_ref);

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
  if (kinematics.inverse(torso_p_rleg, CHAIN_RLEG))
    joints_r = kinematics.getJoints(CHAIN_RLEG);
  else
    std::cout << "Right side IK failed" << '\n';
  if (kinematics.inverse(torso_p_lleg, CHAIN_LLEG))
    joints_l = kinematics.getJoints(CHAIN_LLEG);
  else
    std::cout << "Left side IK failed" << '\n';
  kinematics.forward(joints_r, CHAIN_RLEG);
  kinematics.forward(joints_l, CHAIN_LLEG);
  vec3_t torso_p_com = kinematics.getCom(CHAIN_BODY);
  std::cout << "world_p_torso_ref" << '\n';
  std::cout << world_p_torso_ref << '\n';
  std::cout << "world_p_com" << '\n';
  std::cout << world_p_torso_ref + torso_p_com << '\n';

  return 0;
}