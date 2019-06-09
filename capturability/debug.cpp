#include "analysis.h"
#include "capturability.h"
#include "grid.h"
#include "kinematics.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "polygon.h"
#include "vector.h"
#include <iostream>

using namespace std;
using namespace CA;

int main(int argc, char const *argv[]) {
  Model model("nao.xml");
  model.parse();

  Param param("analysis.xml");
  param.parse();

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
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0); // lleg
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);

  kinematics.forward(joint, CHAIN_BODY);
  kinematics.getCom(CHAIN_BODY);

  // mat4_t T_ref = Eigen::Matrix4f::Identity();
  // mat3_t R_ref;
  // float theta = 3.1415926 / 6;
  // R_ref << cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1;
  // T_ref.block(0, 0, 3, 3) = R_ref;
  // T_ref(0, 3) = 0;
  // T_ref(1, 3) = -0.05;
  // T_ref(2, 3) = -0.2;
  // if (kinematics.inverse(T_ref, CHAIN_RLEG)) {
  //   printf("%d: %lf\n", static_cast<int>(R_HIP_YAWPITCH),
  //          kinematics.getJointAngle(R_HIP_YAWPITCH));
  //   printf("%d: %lf\n", static_cast<int>(R_HIP_ROLL),
  //          kinematics.getJointAngle(R_HIP_ROLL));
  //   printf("%d: %lf\n", static_cast<int>(R_HIP_PITCH),
  //          kinematics.getJointAngle(R_HIP_PITCH));
  //   printf("%d: %lf\n", static_cast<int>(R_KNEE_PITCH),
  //          kinematics.getJointAngle(R_KNEE_PITCH));
  //   printf("%d: %lf\n", static_cast<int>(R_ANKLE_PITCH),
  //          kinematics.getJointAngle(R_ANKLE_PITCH));
  //   printf("%d: %lf\n", static_cast<int>(R_ANKLE_ROLL),
  //          kinematics.getJointAngle(R_ANKLE_ROLL));
  // }

  return 0;
}