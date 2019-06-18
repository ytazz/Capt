#ifndef __PLANNING_H__
#define __PLANNING_H__

#include "kinematics.h"
#include "model.h"
#include "monitor.h"
#include "pendulum.h"
#include "polygon.h"
#include "vector.h"
#include <vector>

namespace CA {

struct StepSeq {
  vec3_t footstep;
  vec2_t cop;
  float step_time;
  EFoot e_suft;
};

struct ComState {
  vec2_t pos;
  vec2_t vel;
};

class Planning {
public:
  Planning(Model model);
  ~Planning();

  void setStepSeq(std::vector<StepSeq> step_seq);
  void setComState(vec2_t com, vec2_t com_vel);

  void calcRef();

  void printStepSeq();

  float getPlanningTime();
  vec2_t getFootstep(int num_step);
  vec2_t getCom(float time);
  vec2_t getComVel(float time);
  vec3_t getIcp(float time);
  vec6_t getJoints(float time, std::string right_or_left);

private:
  Polygon polygon;
  Pendulum pendulum;

  int getNumStep(float time);

  std::vector<StepSeq> step_seq;
  std::vector<ComState> com_state; // at t=0, n-step
};

} // namespace CA

#endif // __PLANNING_H__