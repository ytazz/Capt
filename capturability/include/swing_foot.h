#ifndef __SWING_FOOT_H__
#define __SWING_FOOT_H__

#include "interpolation.h"
#include "model.h"
#include "vector.h"

namespace CA {

class SwingFoot {
public:
  SwingFoot(Model model);
  ~SwingFoot();

  void set(vec2_t foot, vec2_t foot_des);
  void set(vec3_t foot, vec3_t foot_des);
  float getTime();
  vec3_t getTraj(float dt);

private:
  Interpolation interpolation[4]; // x, y, z_first_half, z_last_half

  vec3_t foot, foot_des;
  float step_time_min;
  float step_height;
  float foot_vel, foot_vel_x, foot_vel_y;
};

} // namespace CA

#endif // __SWING_FOOT_H__
