#ifndef __SWING_H__
#define __SWING_H__

#include "interpolation.h"
// #include "cycloid.h"
#include "model.h"
#include "param.h"
#include "base.h"

namespace Capt {

class Swing {
public:
  Swing(Model *model, Param *param);
  ~Swing();

  void set(vec2_t foot, vec2_t foot_des);
  void set(vec3_t foot, vec3_t foot_des);

  // get step time (duration)
  double getTime();
  // get desired swing foot position
  vec3_t getTraj(double dt); // dt = time from support foot exchange

private:
  vec3_t foot, foot_des;
  double dist, dist_x, dist_y, tau;
  double v_max, z_max;
  double dt_min;
};

} // namespace Capt

#endif // __SWING_H__
