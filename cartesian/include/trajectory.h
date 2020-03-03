#ifndef __TRAJECTORY_H__
#define __TRAJECTORY_H__

#include <iostream>
#include <vector>
#include "base.h"
#include "state.h"
#include "input.h"
#include "pendulum.h"
#include "swing.h"
#include "planner.h"

namespace Capt {

class Trajectory {
public:
  Trajectory(Model *model, Param *param);
  ~Trajectory();

  void set(EnhancedInput input, Foot suf);

  vec3_t getCop(double dt);
  vec3_t getIcp(double dt);
  vec3_t getFootR(double dt);
  vec3_t getFootL(double dt);

private:
  EnhancedInput input;

  Pendulum pendulum;
  Swing    swing;

  double h, dt_min;
  Foot   suf;
  double time;
};

} // namespace Capt

#endif // __TRAJECTORY_H__