#ifndef __SWING_FOOT_H__
#define __SWING_FOOT_H__

#include "model.h"
#include "vector.h"

namespace CA {

class SwingFoot {
public:
  SwingFoot(Model model);
  ~SwingFoot();

  void set(Vector2 foot, Vector2 foot_des);
  float getTime();
  Vector2 getTraj(float dt);

private:
  Vector2 foot, foot_des;
  float step_time_min;
  float foot_vel, foot_vel_x, foot_vel_y;
};

} // namespace CA

#endif // __SWING_FOOT_H__
