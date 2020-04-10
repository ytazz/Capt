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
  Trajectory(Pendulum* pendulum, Swing* swing);
  ~Trajectory();

  void set(const EnhancedState& state, const EnhancedInput& input);

  vec3_t getCop  (float t);
  vec3_t getIcp  (float t);
  vec3_t getFootR(float t);
  vec3_t getFootL(float t);

private:
  EnhancedState state;
  EnhancedInput input;

  Pendulum* pendulum;
  Swing*    swing;

  //float  h, dt_min;
  //Foot   suf;
  //float  time;
};

} // namespace Capt

#endif // __TRAJECTORY_H__