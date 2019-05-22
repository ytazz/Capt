#ifndef __PENDULUM_H__
#define __PENDULUM_H__

#include "model.h"
#include "vector.h"

namespace CA {

class Pendulum {

public:
  Pendulum(Model model);
  ~Pendulum();

  // void setCom(const Vector2 &com, const Vector2 &com_vel);
  void setIcp(const Vector2 icp);
  void setCop(const Vector2 cop);

  // Vector2 getCom(float dt);
  // Vector2 getComVel(float dt);
  Vector2 getIcp(float dt);

private:
  // Vector2 com, com_vel;
  Vector2 icp;
  Vector2 cop;
  float g;
  float h;
  float omega;
};

} // namespace CA

#endif // __PENDULUM_H__