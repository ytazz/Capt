#include "pendulum.h"

namespace CA {

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

void Pendulum::setComVel(const Vector2 com_vel) { this->com_vel = com_vel; }

void Pendulum::setIcp(const Vector2 icp) { this->icp = icp; }

void Pendulum::setCop(const Vector2 cop) { this->cop = cop; }

Vector2 Pendulum::getCom(float dt) {
  Vector2 com_;
  com_ = com * cosh(omega * dt) + com_vel * sinh(omega * dt) / omega;
  return com_;
}

Vector2 Pendulum::getComVel(float dt) {
  Vector2 com_vel_;
  com_vel_ = com * omega * sinh(omega * dt) + com_vel * cosh(omega * dt);
  return com_vel_;
}

Vector2 Pendulum::getIcp(float dt) {
  Vector2 icp_;
  icp_ = (icp - cop) * exp(omega * dt) + cop;
  return icp_;
}

} // namespace CA
