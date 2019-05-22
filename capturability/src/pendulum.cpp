#include "pendulum.h"

namespace CA {

Pendulum::Pendulum(Model model) {
  // com = {0.0, 0.0};
  // com_vel = {0.0, 0.0};
  icp.init();
  cop.init();

  g = model.getVal("environment", "gravity");
  h = model.getVal("physics", "com_height");
  omega = sqrt(g / h);
}

Pendulum::~Pendulum(){};

void Pendulum::setIcp(const Vector2 icp) { this->icp = icp; }

void Pendulum::setCop(const Vector2 cop) { this->cop = cop; }

Vector2 Pendulum::getIcp(float dt) {
  Vector2 icp_;
  icp_ = (icp - cop) * exp(omega * dt) + cop;
  return icp_;
}

} // namespace CA
