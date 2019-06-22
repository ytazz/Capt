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
  Planning(Model model, Param param, float timestep);
  ~Planning();

  /* world coordinate */
  // set current value
  void setIcp(vec2_t icp);
  void setCom(vec3_t com);
  void setComVel(vec3_t com_vel);
  void setRLeg(vec3_t rleg);
  void setLLeg(vec3_t lleg);

  // calculate desired trajectory
  void calcDes();

  // get desired value
  vec2_t getIcpDes(float time);
  vec2_t getIcpVelDes(float time);

  // get command value
  vec2_t getCop(float time);
  vec2_t getIcp(float time);
  vec3_t getCom(float time);
  vec3_t getComVel(float time);
  vec3_t getRLeg(float time);
  vec3_t getLLeg(float time);

private:
  Model model;
  Polygon polygon;
  Pendulum pendulum;
  Capturability capturability;
  Grid grid;
  SwingFoot swing_foot;

  // gain
  const float k;

  // physics
  float g;
  float h;
  float omega;

  vec3_t vec2tovec3(vec2_t vec2);
  vec2_t vec3tovec2(vec3_t vec3);

  // timestep
  float dt;

  // step time
  float step_time;

  // current value
  vec2_t icp_cr;
  vec3_t com_cr, com_vel_cr;

  // command value
  vec2_t icp_cmd, cop_cmd;
  vec3_t com_cmd, com_vel_cmd;
  vec3_t rleg_cmd, lleg_cmd;

  // desired value
  vec2_t icp_des, icp_vel_des;
  std::vector<vec2_t> cop_0_des, icp_0_des;
  std::vector<vec3_t> com_0_des, com_vel_0_des;
  std::vector<vec3_t> rleg_des, lleg_des;
};

} // namespace CA

#endif // __PLANNING_H__