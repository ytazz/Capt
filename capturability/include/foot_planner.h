#ifndef __FOOT_PLANNER_H__
#define __FOOT_PLANNER_H__

#include "CA.h"

namespace CA {

enum SupportFoot { SUPPORT_R, SUPPORT_L };

struct FootStep {
  vec3_t foot_r_ini;
  vec3_t foot_l_ini;

  size_t size() { return foot.size(); }

  std::vector<vec3_t> foot;
  std::vector<vec3_t> cop;
  std::vector<float> step_time;
  std::vector<SupportFoot> suft;
};

class FootPlanner {
public:
  FootPlanner(Model *model, Capturability *capturability, Grid *grid);
  ~FootPlanner();

  void setComState(vec3_t com, vec3_t com_vel);
  void setRLeg(vec3_t rleg);
  void setLLeg(vec3_t lleg);

  GridState calcCurrentState(SupportFoot suft);

  bool plan();

  FootStep getFootStep();

  void show();

private:
  void getInitState();

  Model *model;
  Capturability *capturability;
  Grid *grid;

  GridState gstate;

  // physics
  float g;
  float h;
  float omega;

  // current value
  vec2_t icp_cr;
  vec3_t com_cr, com_vel_cr;

  int num_step;
  std::vector<CaptureSet> capture_set;
  FootStep foot_step;
};

} // namespace CA

#endif // __FOOT_PLANNER_H________