#ifndef __PLANNING_H__
#define __PLANNING_H__

#include "capturability.h"
#include "grid.h"
#include "kinematics.h"
#include "model.h"
#include "monitor.h"
#include "pendulum.h"
#include "polygon.h"
#include "swing_foot.h"
#include "vector.h"
#include <vector>

namespace CA {

class Planning {
public:
  Planning(Model model, Param param);
  ~Planning();

  // world coordinate
  void setCom(vec3_t com);
  void setComVel(vec3_t com_vel);
  void setRLeg(vec3_t rleg);
  void setLLeg(vec3_t lleg);

  void calcRef();

  vec3_t getCom(float time);
  vec3_t getRLeg(float time);
  vec3_t getLLeg(float time);

private:
  Polygon polygon;
  Pendulum pendulum;
  Capturability capturability;
  Grid grid;
  SwingFoot swing_foot;

  float step_time;
  float g;
  float h;
  float omega;

  vec3_t vec2tovec3(vec2_t vec2);
  vec2_t vec3tovec2(vec3_t vec3);

  vec2_t cop, cop_;
  vec3_t com, com_vel;
  vec3_t com_, com_vel_;
  vec3_t rleg, lleg;

  vec3_t com_ref, com_vel_ref;
  vec3_t rleg_ref, lleg_ref;
};

} // namespace CA

#endif // __PLANNING_H__