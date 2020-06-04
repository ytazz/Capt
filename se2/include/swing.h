#ifndef __SWING_H__
#define __SWING_H__

//#include "interpolation.h"
// #include "cycloid.h"
#include "model.h"
#include "param.h"
#include "base.h"

namespace Capt {

class Swing {
public:
  Swing(Model *model);
  ~Swing();

  // set swing foot position and landing position
  // set() should not be called after swing foot starts descending
  void set(vec3_t foot, vec3_t land);

  // get step duration
  float getDuration();

  // get swing foot position
  // t is elapsed time after set() is called
  vec3_t getTraj(float t);

  // swing foot is swinging down or not
  //bool isSwingDown(float dt);

private:
  vec3_t foot, land;
  float  dist, dist_x, dist_y;
  float  tau_ascend, tau_travel, tau_descend;
  float  v_max, z_max;
};

} // namespace Capt

#endif // __SWING_H__
