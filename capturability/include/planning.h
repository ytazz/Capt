#ifndef __PLANNING_H__
#define __PLANNING_H__

#include "model.h"
#include "polygon.h"
#include <vector>

namespace CA {

struct FootStep {
  vec3_t footstep;
  vec2_t cop;
  float step_time;
};

class Planning {
public:
  Planning(Model model);
  ~Planning();

  setFootStep(std::vector<FootStep> footstep);
  setCom(vec3_t com);
  setComVel(vec3_t com_vel);

private:
  Polygon polygon;

  std::vector<FootStep> footstep;
  vec3_t com;
  vec3_t com_vel;
};

} // namespace CA

#endif // __PLANNING_H__