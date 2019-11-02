#ifndef __SWING_FOOT_H__
#define __SWING_FOOT_H__

// #include "interpolation.h"
#include "cycloid.h"
#include "model.h"
#include "base.h"

namespace Capt {

class SwingFoot {
public:
  SwingFoot(Model *model);
  ~SwingFoot();

  void   set(vec2_t foot, vec2_t foot_des);
  void   set(vec3_t foot, vec3_t foot_des);
  double getTime();
  vec3_t getTraj(double dt);

private:
  Cycloid cycloid;

  vec3_t foot, foot_des;
  double step_time_min, step_time;
  double step_height;
  double foot_vel;
};

} // namespace Capt

#endif // __SWING_FOOT_H__
