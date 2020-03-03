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

  void set(vec2_t foot, vec2_t land);
  void set(vec3_t foot, vec3_t land);

  // get step duration
  double getDuration();
  // get desired swing foot position
  // dt = elapsed time from set() is called
  vec3_t getTraj(double dt);

  // swing foot is swinging down or not
  bool isSwingDown(double dt);

private:
  vec3_t foot, land;
  double dist, dist_x, dist_y, tau, tau_offset;
  double v_max, z_max;
  double dt_min;
};

} // namespace Capt

#endif // __SWING_H__
