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
  // model.parse();
  // model.print();

  Param param("analysis.xml");
  // param.parse();
  // param.print();

  std::cout << "/* message */" << '\n';
  Grid grid(param);
  std::cout << "/* message */" << '\n';
  Capturability capturability(model, param);
  std::cout << "/* message */" << '\n';

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
  joint.push_back(0.1); // rleg
  joint.push_back(0.1);
  joint.push_back(0.1);
  joint.push_back(0.1);
  joint.push_back(0.1);
  joint.push_back(0.1);
  joint.push_back(0.1); // lleg
  joint.push_back(0.1);
  joint.push_back(0.1);
  joint.push_back(0.1);
  joint.push_back(0.1);
  joint.push_back(0.1);

  kinematics.forward(joint, CHAIN_BODY);
  kinematics.getCom(CHAIN_BODY);

  Trajectory trajectory(model);
  trajectory.setJoints(joint);
  vec3_t com_ref;
  com_ref << 0, 0, 0.25;
  vec3_t rleg_ref;
  rleg_ref << 0, -0.05, 0.0467;
  vec3_t lleg_ref;
  lleg_ref << 0, 0.05, 0.0467;
  trajectory.setRLegRef(rleg_ref);
  trajectory.setLLegRef(lleg_ref);
  trajectory.setComRef(com_ref);

  trajectory.calc();

  return 0;
}