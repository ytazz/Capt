#ifndef __PLANNING_H__
#define __PLANNING_H__

#include "capturability.h"
#include "foot_planner.h"
#include "grid.h"
#include "kinematics.h"
#include "model.h"
#include "monitor.h"
#include "pendulum.h"
#include "polygon.h"
#include "swing_foot.h"
#include "vector.h"
#include <vector>

namespace Capt {

class Planning {
public:
  Planning(Model model, Param param, float timestep);
  ~Planning();

  /* world coordinate */
  // set current value
  void setIcp(vec2_t icp);
  void setCom(vec3_t com);
  void setComVel(vec3_t com_vel);
  void setFootstep(Footstep footstep);

  // calculate desired trajectory
  void plan();

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
  Model                  model;
  Polygon                polygon;
  Grid                   grid;
  std::vector<Pendulum>  pendulum_des;
  Pendulum               pendulum_cr;
  std::vector<SwingFoot> swf;

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

  // footstep
  Footstep footstep;

  // current value
  vec2_t icp_cr;
  vec3_t com_cr, com_vel_cr;

  // command value
  vec2_t icp_cmd, cop_cmd;
  vec3_t com_cmd, com_vel_cmd;
  vec3_t rleg_cmd, lleg_cmd;
};

} // namespace Capt

#endif // __PLANNING_H__