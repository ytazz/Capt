#include "trajectory.h"

namespace CA {

Trajectory::Trajectory(Model model)
    : kinematics(model), lambda(0.1), accuracy(0.00001) {
  torso_ref = Eigen::Vector3f::Zero();
}

Trajectory::~Trajectory() {}

void Trajectory::setJoints(std::vector<float> joints) {
  kinematics.forward(joints, CHAIN_BODY);
}

void Trajectory::setRLegRef(vec3_t rleg_ref) { this->rleg_ref = rleg_ref; }

void Trajectory::setLLegRef(vec3_t lleg_ref) { this->lleg_ref = lleg_ref; }

void Trajectory::setComRef(vec3_t com_ref) { this->com_ref = com_ref; }

bool Trajectory::calc() {
  torso_ref = com_ref;
  vec3_t err = Eigen::Vector3f::Zero();
  std::vector<float> joints_r, joints_l;

  vec3_t torso_p_rleg;
  vec3_t torso_p_lleg;

  bool find_traj = false;
  while (!find_traj) {
    torso_p_rleg = rleg_ref - torso_ref;
    torso_p_lleg = lleg_ref - torso_ref;
    if (kinematics.inverse(torso_p_rleg, CHAIN_RLEG))
      joints_r = kinematics.getJoints(CHAIN_RLEG);
    if (kinematics.inverse(torso_p_lleg, CHAIN_LLEG))
      joints_l = kinematics.getJoints(CHAIN_LLEG);
    kinematics.forward(joints_r, CHAIN_RLEG);
    kinematics.forward(joints_l, CHAIN_LLEG);
    err = com_ref - kinematics.getCom(CHAIN_BODY);
    if (err.norm() <= accuracy)
      find_traj = true;
    std::cout << "------------" << '\n';
    std::cout << err(2) << '\n';
    torso_ref = lambda * err;
  }

  return find_traj;
}

vec3_t Trajectory::getRLegRef() { return rleg_ref; }

vec3_t Trajectory::getLLegRef() { return lleg_ref; }

vec3_t Trajectory::getTorsoRef() { return torso_ref; }
}