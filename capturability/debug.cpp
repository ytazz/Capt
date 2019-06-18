#include "analysis.h"
#include "capturability.h"
#include "friction_filter.h"
#include "grid.h"
#include "kinematics.h"
#include "loader.h"
#include "model.h"
#include "monitor.h"
#include "param.h"
#include "pendulum.h"
#include "planning.h"
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
  Pendulum pendulum(model);

  // Capturability capturability(model, param);
  // capturability.load("1step.csv");

  Planning planning(model);
  std::vector<StepSeq> step_seq;
  FILE *fp_walk = fopen("step_param.csv", "r");
  StepSeq step;
  float buf[6];
  int ibuf;
  while (fscanf(fp_walk, "%f,%f,%f,%f,%f,%f,%d", &buf[0], &buf[1], &buf[2],
                &buf[3], &buf[4], &buf[5], &ibuf) != EOF) {
    step.footstep << buf[1], buf[2], 0;
    step.cop.setCartesian(buf[3], buf[4]);
    step.step_time = buf[5];
    if (ibuf == 0)
      step.e_suft = L_FOOT;
    else
      step.e_suft = R_FOOT;
    step_seq.push_back(step);
  }
  vec2_t com_pos, com_vel;
  com_pos.setCartesian(0.0, 0.0);
  com_vel.setCartesian(0.0, 0.0);

  planning.setStepSeq(step_seq);
  planning.setComState(com_pos, com_vel);
  planning.printStepSeq();
  planning.calcRef();
  for (int i = 0; i < 10; i++) {
    vec2_t foot = planning.getFootstep(i);
    foot.printCartesian();
  }
  // float planning_time = planning.getPlanningTime();
  // FILE *fp = fopen("com_traj.csv", "w");
  // float t = 0.0;
  // while (t < planning_time) {
  //   com_pos = planning.getCom(t);
  //
  //   fprintf(fp, "%lf,%lf,%lf\n", t, com_pos.x, com_pos.y);
  //   t += 0.001;
  // }

  // std::vector<CaptureSet> region = capturability.getCaptureRegion(44457, 1);
  //
  // std::cout << "state" << '\n';
  // State state = grid.getState(44457);
  // state.printCartesian();
  // std::cout << "--------------------------" << '\n';
  // std::cout << "region: " << (int)region.size() << '\n';
  // for (size_t i = 0; i < region.size(); i++) {
  //   region[i].swft.printCartesian();
  // }
  //
  // FrictionFilter friction_filter(capturability, pendulum);
  // friction_filter.setCaptureRegion(region);
  // vec2_t icp, com, com_vel;
  // icp = state.icp;
  // com.setPolar(0.03, 100 * 3.14159 / 180);
  // com.printCartesian();
  // com_vel = (icp - com) * 5.71741;
  // std::vector<CaptureSet> modified_region =
  //     friction_filter.getCaptureRegion(com, com_vel, 0.2);
  // std::cout << "modified region: " << (int)modified_region.size() << '\n';
  // for (size_t i = 0; i < modified_region.size(); i++) {
  //   modified_region[i].swft.printCartesian();
  // }

  // Kinematics kinematics(model);
  // std::vector<float> joint;
  // joint.push_back(0.0);
  // joint.push_back(0.0);
  // joint.push_back(0.0);
  // joint.push_back(0.0);
  // joint.push_back(0.0);
  // joint.push_back(0.0);
  // joint.push_back(0.0);
  // joint.push_back(0.0);
  // joint.push_back(0.0);
  // joint.push_back(0.0);
  // joint.push_back(0.0);
  // joint.push_back(0.0);
  // joint.push_back(0.0); // rleg
  // joint.push_back(0.0);
  // joint.push_back(-0.1745);
  // joint.push_back(+0.1745);
  // joint.push_back(0.0);
  // joint.push_back(0.0);
  // joint.push_back(0.0); // lleg
  // joint.push_back(0.0);
  // joint.push_back(-0.1745);
  // joint.push_back(+0.1745);
  // joint.push_back(0.0);
  // joint.push_back(0.0);
  //
  // kinematics.forward(joint, CHAIN_BODY);
  // kinematics.getCom(CHAIN_BODY);
  //
  // Trajectory trajectory(model);
  // trajectory.setJoints(joint);
  // vec3_t world_p_com_ref;
  // vec3_t world_p_rleg_ref;
  // vec3_t world_p_lleg_ref;
  // vec3_t world_p_torso_ref;
  // world_p_com_ref << 0.05, 0.0, 0.25;
  // world_p_rleg_ref << 0.0173, -0.05, 0.045;
  // world_p_lleg_ref << 0.0173, +0.05, 0.045;
  // world_p_torso_ref << 0, 0, 0;
  // trajectory.setRLegRef(world_p_rleg_ref);
  // trajectory.setLLegRef(world_p_lleg_ref);
  // trajectory.setComRef(world_p_com_ref);
  //
  // if (trajectory.calc())
  //   std::cout << "success" << '\n';
  // else
  //   std::cout << "failed" << '\n';
  //
  // world_p_torso_ref = trajectory.getTorsoRef();
  // world_p_rleg_ref = trajectory.getRLegRef();
  // world_p_lleg_ref = trajectory.getLLegRef();
  //
  // vec3_t torso_p_rleg = world_p_rleg_ref - world_p_torso_ref;
  // vec3_t torso_p_lleg = world_p_lleg_ref - world_p_torso_ref;
  //
  // std::vector<float> joints_r, joints_l;
  // if (kinematics.inverse(torso_p_rleg, CHAIN_RLEG))
  //   joints_r = kinematics.getJoints(CHAIN_RLEG);
  // else
  //   std::cout << "Right side IK failed" << '\n';
  // if (kinematics.inverse(torso_p_lleg, CHAIN_LLEG))
  //   joints_l = kinematics.getJoints(CHAIN_LLEG);
  // else
  //   std::cout << "Left side IK failed" << '\n';
  // kinematics.forward(joints_r, CHAIN_RLEG);
  // kinematics.forward(joints_l, CHAIN_LLEG);
  // vec3_t torso_p_com = kinematics.getCom(CHAIN_BODY);
  // std::cout << "world_p_torso_ref" << '\n';
  // std::cout << world_p_torso_ref << '\n';
  // std::cout << "world_p_com" << '\n';
  // std::cout << world_p_torso_ref + torso_p_com << '\n';

  return 0;
}