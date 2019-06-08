#include "kinematics.h"

namespace CA {

Kinematics::Kinematics(Model model) : model(model), pi(M_PI) {
  for (int i = 0; i < NUM_LINK; i++) {
    link[i].trans = model.getLinkVec(i, "trn");
    link[i].axis = model.getLinkVec(i, "axis");
    link[i].com = model.getLinkVec(i, "com");
    link[i].mass = model.getLinkVal(i, "mass");
  }
}

Kinematics::~Kinematics() {}

bool Kinematics::forward(std::vector<float> joint_angle, Chain chain) {
  bool success = false;
  std::vector<float> joint_angle_;

  switch (chain) {
  case CHAIN_BODY:
    if (joint_angle.size() == 24) {
      joint_angle_.clear();
      for (int i = 0; i < 2; i++) {
        joint_angle_.push_back(joint_angle[i]);
      }
      forwardHead(joint_angle_);
      printf("ok\n");
      joint_angle_.clear();
      for (int i = 0; i < 5; i++) {
        joint_angle_.push_back(joint_angle[i + 2]);
      }
      forwardRArm(joint_angle_);
      joint_angle_.clear();
      for (int i = 0; i < 5; i++) {
        joint_angle_.push_back(joint_angle[i + 7]);
      }
      forwardLArm(joint_angle_);
      joint_angle_.clear();
      for (int i = 0; i < 6; i++) {
        joint_angle_.push_back(joint_angle[i + 12]);
      }
      forwardRLeg(joint_angle_);
      joint_angle_.clear();
      for (int i = 0; i < 6; i++) {
        joint_angle_.push_back(joint_angle[i + 18]);
      }
      forwardLLeg(joint_angle_);
      success = true;
    }
    break;
  case CHAIN_HEAD:
    if (joint_angle.size() == 2) {
      forwardHead(joint_angle);
      success = true;
    }
    break;
  case CHAIN_RARM:
    if (joint_angle.size() == 5) {
      forwardRArm(joint_angle);
      success = true;
    }
    break;
  case CHAIN_LARM:
    if (joint_angle.size() == 5) {
      forwardLArm(joint_angle);
      success = true;
    }
    break;
  case CHAIN_RLEG:
    if (joint_angle.size() == 6) {
      forwardRLeg(joint_angle);
      success = true;
    }
    break;
  case CHAIN_LLEG:
    if (joint_angle.size() == 6) {
      forwardLLeg(joint_angle);
      success = true;
    }
    break;
  default:
    break;
  }

  return success;
}

void Kinematics::forwardHead(std::vector<float> joint_angle) {
  mat4_t m = Eigen::Matrix4f::Identity();
  int id = 0;
  for (int i = static_cast<int>(HEAD_YAW); i <= static_cast<int>(HEAD_PITCH);
       i++) {
    m *= homogeneous(rodrigues(link[i].axis, joint_angle[id]), link[i].trans);
    link[i].homo = m;
    id++;
  }
}

void Kinematics::forwardRArm(std::vector<float> joint_angle) {
  mat4_t m = Eigen::Matrix4f::Identity();
  int id = 0;
  for (int i = static_cast<int>(R_SHOULDER_PITCH);
       i <= static_cast<int>(R_WRIST_YAW); i++) {
    m *= homogeneous(rodrigues(link[i].axis, joint_angle[id]), link[i].trans);
    link[i].homo = m;
    id++;
  }
}

void Kinematics::forwardLArm(std::vector<float> joint_angle) {
  mat4_t m = Eigen::Matrix4f::Identity();
  int id = 0;
  for (int i = static_cast<int>(L_SHOULDER_PITCH);
       i <= static_cast<int>(L_WRIST_YAW); i++) {
    m *= homogeneous(rodrigues(link[i].axis, joint_angle[id]), link[i].trans);
    link[i].homo = m;
    id++;
  }
}

void Kinematics::forwardRLeg(std::vector<float> joint_angle) {
  mat4_t m = Eigen::Matrix4f::Identity();
  int id = 0;
  for (int i = static_cast<int>(R_HIP_YAWPITCH);
       i <= static_cast<int>(R_ANKLE_ROLL); i++) {
    m *= homogeneous(rodrigues(link[i].axis, joint_angle[id]), link[i].trans);
    link[i].homo = m;
    id++;
  }
}

void Kinematics::forwardLLeg(std::vector<float> joint_angle) {
  mat4_t m = Eigen::Matrix4f::Identity();
  int id = 0;
  for (int i = static_cast<int>(L_HIP_YAWPITCH);
       i <= static_cast<int>(L_ANKLE_ROLL); i++) {
    m *= homogeneous(rodrigues(link[i].axis, joint_angle[id]), link[i].trans);
    link[i].homo = m;
    id++;
  }
}

vec3_t Kinematics::getLinkPos(ELink elink) {
  vec3_t pos;
  pos = link[elink].homo.block(0, 3, 3, 1);
  return pos;
}

mat3_t Kinematics::getLinkRot(ELink elink) {
  mat3_t mat;
  mat = link[elink].homo.block(0, 0, 3, 3);
  return mat;
}

vec3_t Kinematics::getLinkEuler(ELink elink) {
  mat3_t mat = getLinkRot(elink);

  float roll = 0, pitch = 0, yaw = 0;
  float epsilon = 0.001;

  if (fabs(mat(2, 0) - 1.0) < epsilon) { // 正しく計算できていない
    roll = pi / 2;
    pitch = 0;
    yaw = atan2(mat(1, 0), mat(0, 0));
  } else if (fabs(mat(2, 0) + 1.0) < epsilon) { // 正しく計算できていない
    roll = -pi / 2;
    pitch = 0;
    yaw = atan2(mat(1, 0), mat(0, 0));
  } else {
    roll = atan2(mat(2, 1), mat(2, 2));
    pitch = asin(-mat(2, 0));
    yaw = atan2(mat(1, 0), mat(0, 0));
  }

  vec3_t vec;
  vec << roll, pitch, yaw;

  return vec;
}

mat3_t Kinematics::rodrigues(vec3_t axis, float angle) {
  mat3_t axis_wedge = Eigen::Matrix3f::Zero(); // = a^
  axis_wedge(0, 1) = -axis(2);
  axis_wedge(0, 2) = axis(1);
  axis_wedge(1, 0) = axis(2);
  axis_wedge(1, 2) = -axis(0);
  axis_wedge(2, 0) = -axis(1);
  axis_wedge(2, 1) = axis(0);

  mat3_t R = Eigen::Matrix3f::Zero();
  R = Eigen::Matrix3f::Identity() + axis_wedge * sin(angle) +
      axis_wedge * axis_wedge * (1 - cos(angle));

  return R;
}

mat4_t Kinematics::homogeneous(mat3_t matrix, vec3_t vector) {
  mat4_t m = Eigen::Matrix4f::Zero();
  m.block(0, 0, 3, 3) = matrix;
  m.block(0, 3, 3, 1) = vector;
  m(3, 3) = 1;
  return m;
}

} // namespace CA