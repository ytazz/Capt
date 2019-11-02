#include "pendulum.h"

namespace Capt {

Pendulum::Pendulum(Model model) {
  com.clear();
  com_vel.clear();
  icp.clear();
  cop.clear();

  g = model.getVal("environment", "gravity");
  h = model.getVal("physics", "com_height");
  omega = sqrt(g / h);
}

Pendulum::~Pendulum(){};

void Pendulum::setCom(const vec2_t com) { this->com = com; }

void Pendulum::setCom(vec3_t com) { this->com.setCartesian(com.x(), com.y()); }

void Pendulum::setComVel(const vec2_t com_vel) { this->com_vel = com_vel; }

void Pendulum::setComVel(vec3_t com_vel) {
  this->com_vel.setCartesian(com_vel.x(), com_vel.y());
}

void Pendulum::setIcp(const vec2_t icp) { this->icp = icp; }

void Pendulum::setIcp(vec3_t icp) { this->icp.setCartesian(icp.x(), icp.y()); }

void Pendulum::setCop(const vec2_t cop) { this->cop = cop; }

void Pendulum::setCop(vec3_t cop) { this->cop.setCartesian(cop.x(), cop.y()); }

vec2_t Pendulum::getCom(double dt) {
  vec2_t com_;
  com_ =
      cop + (com - cop) * cosh(omega * dt) + com_vel * sinh(omega * dt) / omega;
  return com_;
}

vec2_t Pendulum::getComVel(double dt) {
  vec2_t com_vel_;
  com_vel_ =
      (com - cop) * omega * sinh(omega * dt) + com_vel * cosh(omega * dt);
  return com_vel_;
}

vec2_t Pendulum::getIcp(double dt) {
  vec2_t icp_;
  icp_ = (icp - cop) * exp(omega * dt) + cop;
  return icp_;
}

vec2_t Pendulum::getIcpVel(double dt) {
  vec2_t icp_vel_;
  icp_vel_ = omega * (icp - cop) * exp(omega * dt);
  return icp_vel_;
}

} // namespace Capt
