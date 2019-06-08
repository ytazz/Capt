#ifndef __KINEMATICS_H__
#define __KINEMATICS_H__

// #include ".h"
#include "model.h"
#include <iostream>
#include <vector>

namespace CA {

typedef Eigen::Matrix3f mat3_t;
typedef Eigen::Matrix4f mat4_t;

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
  vec3_t com;   // const
  float mass;   // const
};

class Kinematics {
public:
  Kinematics(Model model);
  ~Kinematics();

  bool forward(std::vector<float> joint_angle, Chain chain);
  void forwardHead(std::vector<float> joint_angle);
  void forwardRArm(std::vector<float> joint_angle);
  void forwardLArm(std::vector<float> joint_angle);
  void forwardRLeg(std::vector<float> joint_angle);
  void forwardLLeg(std::vector<float> joint_angle);

  bool inverse(vec3_t link_pos, Chain chain);
  bool inverse(mat4_t link_trans, Chain chain);

  vec3_t getLinkPos(ELink elink);
  mat3_t getLinkRot(ELink elink);
  vec3_t getLinkEuler(ELink elink); // [roll pitch yaw]
  float getJointAngle(ELink elink);
  vec3_t getCom(ELink elink);

private:
  // Rodrigues' rotation formula
  // axis should be normarized vector
  mat3_t rodrigues(vec3_t axis, float angle);
  // create 4x4 matrix from 3x3 matrix and 3x1 vector
  mat4_t homogeneous(mat3_t matrix, vec3_t vector);

  Model model;

  const float pi;

  Link link[NUM_LINK];
};

} // namespace CA

#endif // __KINEMATICS_H__