#ifndef __MONITOR_H__
#define __MONITOR_H__

#include "model.h"
#include "pendulum.h"
#include "polygon.h"
#include "vector.h"
#include <string.h>

namespace Capt {

class Monitor {
public:
  Monitor(Model model);
  ~Monitor();

  void setSuft(vec2_t suft, EFoot which_suft);
  void setIcp(vec2_t icp);
  void setLandingPos(vec2_t landing_pos);
  void setStepTime(float step_time);

  bool judge();

private:
  Model model;
  Pendulum pendulum;
  Polygon polygon;

  vec2_t cop, icp;
  vec2_t suft;
  vec2_t landing_pos;
  float step_time;
  EFoot e_suft;
};

} // namespace Capt

#endif // __MONITOR_H__