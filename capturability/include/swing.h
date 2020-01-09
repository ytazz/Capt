#ifndef __SWING_H__
#define __SWING_H__

#include "interpolation.h"
// #include "cycloid.h"
#include "model.h"
#include "base.h"

namespace Capt {

enum SwingPhase { OFF, SWING, ON, LAND };

class Swing {
public:
  Swing(Model *model);
  ~Swing();

  void   set(vec2_t foot, vec2_t foot_des, double elapsed);
  void   set(vec3_t foot, vec3_t foot_des, double elapsed);
  double getTime();
  vec3_t getTraj(double dt);

private:
  // Cycloid cycloid;
  Interp3 swingUp;
  Interp3 swingDown;

  vec3_t foot, foot_des;
  double dt_min, tau, tau_swing;
  double h;
  double v_max;
};

} // namespace Capt

#endif // __SWING_H__
