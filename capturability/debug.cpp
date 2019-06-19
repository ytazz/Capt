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
      step.e_suft = FOOT_L;
    else
      step.e_suft = FOOT_R;
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

  return 0;
}