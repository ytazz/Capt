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

void Pendulum::setCom(const Vector2 com) { this->com = com; }

void Pendulum::setCom(vec3_t com) { this->com.setCartesian(com.x(), com.y()); }

void Pendulum::setComVel(const Vector2 com_vel) { this->com_vel = com_vel; }

void Pendulum::setComVel(vec3_t com_vel) {
  this->com_vel.setCartesian(com_vel.x(), com_vel.y());
}

void Pendulum::setIcp(const Vector2 icp) { this->icp = icp; }

void Pendulum::setIcp(vec3_t icp) { this->icp.setCartesian(icp.x(), icp.y()); }

void Pendulum::setCop(const Vector2 cop) { this->cop = cop; }

void Pendulum::setCop(vec3_t cop) { this->cop.setCartesian(cop.x(), cop.y()); }

Vector2 Pendulum::getCom(float dt) {
  Vector2 com_;
  com_ =
      cop + (com - cop) * cosh(omega * dt) + com_vel * sinh(omega * dt) / omega;
  return com_;
}

Vector2 Pendulum::getComVel(float dt) {
  Vector2 com_vel_;
  com_vel_ =
      (com - cop) * omega * sinh(omega * dt) + com_vel * cosh(omega * dt);
  return com_vel_;
}

Vector2 Pendulum::getIcp(float dt) {
  Vector2 icp_;
  icp_ = (icp - cop) * exp(omega * dt) + cop;
  return icp_;
}

Vector2 Pendulum::getIcpVel(float dt) {
  Vector2 icp_vel_;
  icp_vel_ = omega * (icp - cop) * exp(omega * dt);
  return icp_vel_;
}

} // namespace Capt
