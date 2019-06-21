#include "trajectory.h"

namespace CA {

Trajectory::Trajectory(Model model)
    : kinematics(model), lambda(0.5), accuracy(0.0001) {
  this->torso_ref = Eigen::Vector3f::Zero();
}

Trajectory::~Trajectory() {}

void Trajectory::setTorso(vec3_t torso) { this->torso_ref = torso; }

void Trajectory::setJoints(std::vector<float> joints) {
  kinematics.forward(joints, CHAIN_BODY);
}

void Trajectory::setRLegRef(vec3_t rleg_ref) { this->rleg_ref = rleg_ref; }

void Trajectory::setLLegRef(vec3_t lleg_ref) { this->lleg_ref = lleg_ref; }

void Trajectory::setComRef(vec3_t com_ref) { this->com_ref = com_ref; }

bool Trajectory::calc() {
  vec3_t err = Eigen::Vector3f::Zero();
  std::vector<float> joints_r, joints_l;

  vec3_t torso_p_rleg;
  vec3_t torso_p_lleg;

  bool find_traj = false;
  int iteration = 0;
  while (!find_traj) {
    torso_p_rleg = rleg_ref - torso_ref;
    torso_p_lleg = lleg_ref - torso_ref;
    if (kinematics.inverse(torso_p_rleg, CHAIN_RLEG)) {
      joints_r = kinematics.getJoints(CHAIN_RLEG);
    } else {
      std::cout << "ik false right" << '\n';
      torso_ref -= lambda * err;
      break;
    }
    if (kinematics.inverse(torso_p_lleg, CHAIN_LLEG)) {
      joints_l = kinematics.getJoints(CHAIN_LLEG);
    } else {
      std::cout << "ik false left" << '\n';
      torso_ref -= lambda * err;
      break;
    }
    kinematics.forward(joints_r, CHAIN_RLEG);
    kinematics.forward(joints_l, CHAIN_LLEG);
    err = com_ref - (torso_ref + kinematics.getCom(CHAIN_BODY));
    torso_ref += lambda * err;
    if (err.norm() <= accuracy)
      find_traj = true;
    iteration++;
  }
  if (iteration == 0) {
    std::cout << " 最初から無理" << '\n';
    std::cout << "torso_ref" << '\n';
    std::cout << torso_ref << '\n';
    std::cout << "rleg_ref" << '\n';
    std::cout << rleg_ref << '\n';
    std::cout << "lleg_ref" << '\n';
    std::cout << lleg_ref << '\n';
    std::cout << "com_ref" << '\n';
    std::cout << com_ref << '\n';
  }

  return find_traj;
}

vec3_t Trajectory::getRLegRef() { return this->rleg_ref; }

vec3_t Trajectory::getLLegRef() { return this->lleg_ref; }

vec3_t Trajectory::getTorsoRef() { return this->torso_ref; }
} // namespace CA