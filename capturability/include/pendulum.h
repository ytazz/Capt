#ifndef __PENDULUM_H__
#define __PENDULUM_H__

#include "loader.h"
#include "model.h"
#include "base.h"

namespace Capt {

class Pendulum {

public:
  Pendulum(Model *model);
  ~Pendulum();

  void setCom(const vec2_t com);
  void setCom(const vec3_t com);
  void setComVel(const vec2_t com_vel);
  void setComVel(const vec3_t com_vel);
  void setIcp(const vec2_t icp);
  void setIcp(const vec3_t icp);
  void setCop(const vec2_t cop);
  void setCop(const vec3_t cop);

  vec2_t getCom(double dt);
  vec2_t getComVel(double dt);
  vec2_t getIcp(double dt);
  vec2_t getIcpVel(double dt);

private:
  vec2_t com, com_vel;
  vec2_t icp;
  vec2_t cop;
  double g;
  double h;
  double omega;
};

} // namespace Capt

#endif // __PENDULUM_H__