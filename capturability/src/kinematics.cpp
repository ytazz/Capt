#include "kinematics.h"

namespace CA {

Kinematics::Kinematics(Model model)
    : model(model), pi(M_PI), lambda(0.5), accuracy(0.00001), weight_p(1),
      weight_R(1) {
  for (int i = 0; i < NUM_LINK; i++) {
    link[i].trans = model.getLinkVec(i, "trn");
    link[i].axis = model.getLinkVec(i, "axis");
    link[i].com = model.getLinkVec(i, "com");
    link[i].mass = model.getLinkVal(i, "mass");
  }

  link[TORSO].homo =
      homogeneous(Eigen::Matrix3f::Identity(), link[TORSO].trans);
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
    link[i].joint = joint_angle[id];
    m *= homogeneous(rodrigues(link[i].axis, link[i].joint), link[i].trans);
    link[i].homo = m;
    id++;
  }
}

void Kinematics::forwardRArm(std::vector<float> joint_angle) {
  mat4_t m = Eigen::Matrix4f::Identity();
  int id = 0;
  for (int i = static_cast<int>(R_SHOULDER_PITCH);
       i <= static_cast<int>(R_WRIST_YAW); i++) {
    link[i].joint = joint_angle[id];
    m *= homogeneous(rodrigues(link[i].axis, link[i].joint), link[i].trans);
    link[i].homo = m;
    id++;
  }
}

void Kinematics::forwardLArm(std::vector<float> joint_angle) {
  mat4_t m = Eigen::Matrix4f::Identity();
  int id = 0;
  for (int i = static_cast<int>(L_SHOULDER_PITCH);
       i <= static_cast<int>(L_WRIST_YAW); i++) {
    link[i].joint = joint_angle[id];
    m *= homogeneous(rodrigues(link[i].axis, link[i].joint), link[i].trans);
    link[i].homo = m;
    id++;
  }
}

void Kinematics::forwardRLeg(std::vector<float> joint_angle) {
  mat4_t m = Eigen::Matrix4f::Identity();
  int id = 0;
  for (int i = static_cast<int>(R_HIP_YAWPITCH);
       i <= static_cast<int>(R_ANKLE_ROLL); i++) {
    link[i].joint = joint_angle[id];
    m *= homogeneous(rodrigues(link[i].axis, link[i].joint), link[i].trans);
    link[i].homo = m;
    id++;
  }
}

void Kinematics::forwardLLeg(std::vector<float> joint_angle) {
  mat4_t m = Eigen::Matrix4f::Identity();
  int id = 0;
  for (int i = static_cast<int>(L_HIP_YAWPITCH);
       i <= static_cast<int>(L_ANKLE_ROLL); i++) {
    link[i].joint = joint_angle[id];
    m *= homogeneous(rodrigues(link[i].axis, link[i].joint), link[i].trans);
    link[i].homo = m;
    id++;
  }
}

bool Kinematics::inverse(mat4_t link_trans, Chain chain) {
  vec6_t q;
  for (int i = static_cast<int>(R_HIP_YAWPITCH);
       i <= static_cast<int>(R_ANKLE_ROLL); i++) {
    ELink elink = static_cast<ELink>(i);
    q(i - static_cast<int>(R_HIP_YAWPITCH), 0) = getJointAngle(elink);
  }

  vec3_t p_ref = link_trans.block(0, 3, 3, 1);
  mat3_t R_ref = link_trans.block(0, 0, 3, 3);

  bool find_solution = false;
  while (!find_solution) {
    std::vector<float> joint;
    for (int j = 0; j < 6; j++) {
      joint.push_back(q(j, 0));
    }
    forward(joint, CHAIN_RLEG);
    vec3_t err_p = p_ref - getLinkPos(R_ANKLE_ROLL);
    mat3_t err_R = getLinkRot(R_ANKLE_ROLL).transpose() * R_ref;
    vec3_t err_r = log_func(err_R);
    // vec3_t err_r = Eigen::Vector3f::Zero();

    vec6_t err;
    err.block(0, 0, 3, 1) = err_p;
    err.block(3, 0, 3, 1) = err_r;
    if (err.norm() <= accuracy)
      find_solution = true;

    vec6_t dq;
    // printf("det = %1.10lf\n", fabs(jacobian(chain).determinant()));
    if (fabs(jacobian(chain).determinant()) <= FLT_EPSILON)
      break;
    // std::cout << jacobian(chain) << '\n';
    dq = lambda * jacobian(chain).inverse() * err;
    q = q + dq;
  }

  for (int i = static_cast<int>(R_HIP_YAWPITCH);
       i <= static_cast<int>(R_ANKLE_ROLL); i++) {
    ELink elink = static_cast<ELink>(i);
    if (model.getLinkVal(elink, "lower_limit") > getJointAngle(elink))
      find_solution *= false;
    if (model.getLinkVal(elink, "upper_limit") < getJointAngle(elink))
      find_solution *= false;
  }

  return find_solution;
}

bool Kinematics::inverse(vec3_t link_pos, Chain chain) {
  mat4_t T_ref;
  mat3_t E = Eigen::Matrix3f::Identity();
  T_ref.block(0, 0, 3, 3) = E;
  T_ref.block(0, 3, 3, 1) = link_pos;

  return inverse(T_ref, chain);
}

mat6_t Kinematics::jacobian(Chain chain) {
  mat6_t jacobi;

  switch (chain) {
  case CHAIN_RLEG:
    jacobi = jacobianRLeg();
    break;
  case CHAIN_LLEG:
    jacobi = jacobianLLeg();
    break;
  default:
    break;
  }

  return jacobi;
}

mat6_t Kinematics::jacobianRLeg() {
  mat6_t jacobi;

  int column = 0;
  for (int i = static_cast<int>(R_HIP_YAWPITCH);
       i <= static_cast<int>(R_ANKLE_ROLL); i++) {
    ELink elink = static_cast<ELink>(i);
    link[i].world_axis = getLinkRot(elink) * link[i].axis;

    jacobi.block(0, column, 3, 1) =
        link[i].world_axis.cross(getLinkPos(R_ANKLE_ROLL) - getLinkPos(elink));
    jacobi.block(3, column, 3, 1) = link[i].world_axis;
    column++;
  }

  return jacobi;
}

mat6_t Kinematics::jacobianLLeg() {
  mat6_t jacobi;

  int column = 0;
  for (int i = static_cast<int>(L_HIP_YAWPITCH);
       i <= static_cast<int>(L_ANKLE_ROLL); i++) {
    ELink elink = static_cast<ELink>(i);
    link[i].world_axis = getLinkRot(elink) * link[i].axis;

    jacobi.block(0, column, 3, 1) =
        link[i].world_axis.cross(getLinkPos(L_ANKLE_ROLL) - getLinkPos(elink));
    jacobi.block(3, column, 3, 1) = link[i].world_axis;
    column++;
  }

  return jacobi;
}

vec3_t Kinematics::getCom(Chain chain) {
  vec3_t com = Eigen::Vector3f::Zero();
  float sum_mass = 0.0;

  switch (chain) {
  case CHAIN_BODY:
    for (int i = static_cast<int>(TORSO); i <= static_cast<int>(L_ANKLE_ROLL);
         i++) {
      ELink elink = static_cast<ELink>(i);

      com += link[elink].mass *
             (getLinkPos(elink) + getLinkRot(elink) * link[elink].com);
      sum_mass += link[elink].mass;
    }
    com = com / sum_mass;
    break;
  default:
    break;
  }

  return com;
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

  if (fabs(mat(2, 0) - 1.0) < FLT_EPSILON) { // 正しく計算できていない
    roll = pi / 2;
    pitch = 0;
    yaw = atan2(mat(1, 0), mat(0, 0));
  } else if (fabs(mat(2, 0) + 1.0) < FLT_EPSILON) { // 正しく計算できていない
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

float Kinematics::getJointAngle(ELink elink) { return link[elink].joint; }

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

vec3_t Kinematics::log_func(mat3_t matrix) {
  vec3_t vec = Eigen::Vector3f::Zero();

  float theta = acos((matrix(0, 0) + matrix(1, 1) + matrix(2, 2) - 1) / 2);
  if (fabs(theta) <= FLT_EPSILON) {
    vec(0) = 0.0;
    vec(1) = 0.0;
    vec(2) = 0.0;
  } else {
    vec(0) = (matrix(2, 1) - matrix(1, 2)) * theta / (2 * sin(theta));
    vec(1) = (matrix(0, 2) - matrix(2, 0)) * theta / (2 * sin(theta));
    vec(2) = (matrix(1, 0) - matrix(0, 1)) * theta / (2 * sin(theta));
  }

  return vec;
}

} // namespace CA