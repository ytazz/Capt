#ifndef __KINEMATICS_H__
#define __KINEMATICS_H__

#include "model.h"
#include <Eigen/LU>
#include <float.h>
#include <iostream>
#include <vector>

namespace Capt {

typedef Eigen::Matrix3f mat3_t;
typedef Eigen::Matrix4f mat4_t;
typedef Eigen::Matrix<float, 6, 6> mat6_t;
typedef Eigen::Matrix<float, 6, 1> vec6_t;

enum Chain {
  CHAIN_BODY,
  CHAIN_HEAD,
  CHAIN_RARM,
  CHAIN_LARM,
  CHAIN_RLEG,
  CHAIN_LLEG,
  NUM_CHAIN
};

struct Link {
  mat4_t homo; // result of FK
  float joint; // result of IK

  vec3_t trans; // const
  vec3_t axis;  // const
  vec3_t world_axis;
  vec3_t com; // const
  float mass; // const
};

class Kinematics {
public:
  Kinematics(Model model);
  ~Kinematics();

  bool forward(std::vector<float> joint_angle, Chain chain);

  bool inverse(vec3_t link_pos, Chain chain);
  bool inverse(mat4_t link_trans, Chain chain);

  mat6_t jacobian(Chain chain);

  vec3_t getCom(Chain chain);

  vec3_t getLinkPos(ELink elink);
  mat3_t getLinkRot(ELink elink);
  vec3_t getLinkEuler(ELink elink); // [roll pitch yaw]
  float getJointAngle(ELink elink);
  std::vector<float> getJoints(Chain chain);

private:
  void forwardHead(std::vector<float> joint_angle);
  void forwardRArm(std::vector<float> joint_angle);
  void forwardLArm(std::vector<float> joint_angle);
  void forwardRLeg(std::vector<float> joint_angle);
  void forwardLLeg(std::vector<float> joint_angle);

  bool inverseRLeg(mat4_t link_trans);
  bool inverseLLeg(mat4_t link_trans);

  mat6_t jacobianRLeg();
  mat6_t jacobianLLeg();

  // Rodrigues' rotation formula
  // axis should be normarized vector
  mat3_t rodrigues(vec3_t axis, float angle);

  // create 4x4 matrix from 3x3 matrix and 3x1 vector
  mat4_t homogeneous(mat3_t matrix, vec3_t vector);

  // 行列対数関数
  vec3_t log_func(mat3_t matrix);

  Model model;

  const float pi;
  const float lambda; // 0 < lambda <= 1 stabilize calculation
  const float accuracy;
  const int iteration_max;
  const float weight_p;
  const float weight_R;

  Link link[NUM_LINK];

  // AnkleRoll -> Foot
  vec3_t offset_rfoot, offset_lfoot;
};

} // namespace Capt

#endif // __KINEMATICS_H____