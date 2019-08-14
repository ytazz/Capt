#include "planning.h"

namespace Capt {

Planning::Planning(Model model, Param param, float timestep)
  : model(model), polygon(), pendulum_cr(model),
    grid(param), k(0.5), dt(timestep) {
  this->cop_cmd.clear();
  this->com_cmd  = vec3_t::Zero();
  this->rleg_cmd = vec3_t::Zero();
  this->lleg_cmd = vec3_t::Zero();

  this->pendulum_des.clear();
  this->swft.clear();

  g     = model.getVal("environment", "gravity");
  h     = model.getVal("physics", "com_height");
  omega = sqrt(g / h);
}

Planning::~Planning() {
}

void Planning::setCom(vec3_t com) {
  this->com_cr = com;
}

void Planning::setComVel(vec3_t com_vel) {
  this->com_vel_cr = com_vel;
}

void Planning::setIcp(vec2_t icp) {
  this->icp_cr = icp;
}

void Planning::setFootstep(Footstep footstep) {
  this->footstep = footstep;
}

vec3_t Planning::vec2tovec3(vec2_t vec2) {
  vec3_t vec3;
  vec3.x() = vec2.x;
  vec3.y() = vec2.y;
  vec3.z() = 0.0;
  return vec3;
}

vec2_t Planning::vec3tovec2(vec3_t vec3) {
  vec2_t vec2;
  vec2.setCartesian(vec3.x(), vec3.y() );
  return vec2;
}

void Planning::plan() {
  // memory
  for (size_t i = 0; i < footstep.size(); i++) {
    pendulum_des.push_back(Pendulum(model) );
    swft.push_back(SwingFoot(model) );
  }

  pendulum_des[0].setCop(footstep.cop[0]);
  pendulum_des[0].setIcp(icp_cr);
  pendulum_des[0].setCom(com_cr);
  pendulum_des[0].setComVel(com_vel_cr);

  pendulum_des[1].setCop(pendulum_des[0].getIcp(footstep.step_time[0]) );
  pendulum_des[1].setIcp(pendulum_des[0].getIcp(footstep.step_time[0]) );
  pendulum_des[1].setCom(pendulum_des[0].getCom(footstep.step_time[0]) );
  pendulum_des[1].setComVel(pendulum_des[0].getComVel(footstep.step_time[0]) );

  swft[0].set(footstep.foot_l_ini, footstep.foot[1]);
}

vec2_t Planning::getCop(float time) {
  vec2_t cop_cmd_;

  vec2_t icp_des, icp_vel_des;
  if (time < footstep.step_time[0]) {
    icp_des     = pendulum_des[0].getIcp(time);
    icp_vel_des = pendulum_des[0].getIcpVel(time);
  } else {
    icp_des     = pendulum_des[1].getIcp(time - footstep.step_time[0]);
    icp_vel_des = pendulum_des[1].getIcpVel(time - footstep.step_time[0]);
  }
  cop_cmd_ = icp_cr + k * ( icp_cr - icp_des ) / omega - icp_vel_des / omega;

  std::vector<vec2_t> region;
  if (time < footstep.step_time[0]) {
    region = model.getVec("foot", "foot_r_convex", footstep.foot[0]);
  } else {
    std::vector<vec2_t> region_r, region_l;
    region_r = model.getVec("foot", "foot_r_convex", footstep.foot[0]);
    region_l = model.getVec("foot", "foot_l_convex", footstep.foot[1]);

    polygon.clear();
    polygon.setVertex(region_r);
    polygon.setVertex(region_l);
    region = polygon.getConvexHull();
  }
  cop_cmd = polygon.getClosestPoint(cop_cmd_, region);
  // cop_cmd = cop_cmd_;

  return cop_cmd;
}

vec3_t Planning::getCom(float time) {
  pendulum_cr.setCom(com_cr);
  pendulum_cr.setComVel(com_vel_cr);
  pendulum_cr.setCop(cop_cmd);
  com_cmd.x() = pendulum_cr.getCom(dt).x;
  com_cmd.y() = pendulum_cr.getCom(dt).y;
  com_cmd.z() = h;
  return com_cmd;
}

vec3_t Planning::getComVel(float time) {
  pendulum_cr.setCom(com_cr);
  pendulum_cr.setComVel(com_vel_cr);
  pendulum_cr.setCop(cop_cmd);
  com_vel_cmd.x() = pendulum_cr.getComVel(dt).x;
  com_vel_cmd.y() = pendulum_cr.getComVel(dt).y;
  com_vel_cmd.z() = 0.0;
  return com_vel_cmd;
}

vec2_t Planning::getIcp(float time) {
  pendulum_cr.setIcp(icp_cr);
  pendulum_cr.setCop(cop_cmd);
  icp_cmd = pendulum_cr.getIcp(dt);
  return icp_cmd;
}

vec3_t Planning::getRLeg(float time) {
  rleg_cmd = footstep.foot[0];
  return rleg_cmd;
}

vec3_t Planning::getLLeg(float time) {
  if (time < footstep.step_time[0]) {
    lleg_cmd = swft[0].getTraj(time);
  } else {
    lleg_cmd = footstep.foot[1];
  }
  return lleg_cmd;
}

} // namespace Capt