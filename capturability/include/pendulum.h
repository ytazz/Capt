#ifndef __PENDULUM_H__
#define __PENDULUM_H__

#include "loader.h"
#include "model.h"
#include "vector.h"

namespace Capt {

class Pendulum {

public:
  Pendulum(Model model);
  ~Pendulum();

  void setCom(const Vector2 com);
  void setCom(vec3_t com);
  void setComVel(const Vector2 com_vel);
  void setComVel(vec3_t com_vel);
  void setIcp(const Vector2 icp);
  void setIcp(vec3_t icp);
  void setCop(const Vector2 cop);
  void setCop(vec3_t cop);

  Vector2 getCom(float dt);
  Vector2 getComVel(float dt);
  Vector2 getIcp(float dt);
  Vector2 getIcpVel(float dt);

private:
  Vector2 com, com_vel;
  Vector2 icp;
  Vector2 cop;
  float g;
  float h;
  float omega;
};

} // namespace Capt

#endif // __PENDULUM_H__