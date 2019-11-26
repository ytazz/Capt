#ifndef __TRAJECTORY_H__
#define __TRAJECTORY_H__

#include <iostream>
#include <vector>
#include "base.h"
#include "state.h"
#include "input.h"
#include "pendulum.h"
#include "swing_foot.h"

namespace Capt {

class Trajectory {
public:
  Trajectory(Model *model);
  ~Trajectory();

  void set(State state, Input input, vec3_t world_p_com, vec3_t world_p_suf, Foot suf);

  double getTime();
  vec3_t getCom(double elapsed_time);
  vec3_t getIcp(double elapsed_time);
  vec3_t getFootR(double elapsed_time);
  vec3_t getFootL(double elapsed_time);

private:
  Pendulum  pendulum;
  SwingFoot swing_foot;

  double h;
  Foot   suf;
  vec3_t world_p_suf;
};

} // namespace Capt

#endif // __TRAJECTORY_H__