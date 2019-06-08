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
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.0);
  joint.push_back(0.1);
  joint.push_back(0.2);
  joint.push_back(0.3);
  joint.push_back(0.4);
  joint.push_back(0.5);
  joint.push_back(0.6);

  if (kinematics.forward(joint, CHAIN_BODY)) {
    for (int i = static_cast<int>(HEAD_YAW);
         i <= static_cast<int>(L_ANKLE_ROLL); i++) {
      ELink elink = static_cast<ELink>(i);
      vec3_t pos = kinematics.getLinkPos(elink);
      mat3_t mat = kinematics.getLinkRot(elink);
      vec3_t euler = kinematics.getLinkEuler(elink);
      std::cout << "--------------------------" << '\n';
      std::cout << "pos" << '\n';
      std::cout << pos << '\n';
      std::cout << "mat" << '\n';
      std::cout << mat << '\n';
      std::cout << "euler" << '\n';
      std::cout << euler << '\n';
    }
  }

  return 0;
}