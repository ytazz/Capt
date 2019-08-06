#ifndef __TRAJECTORY_H__
#define __TRAJECTORY_H__

#include "kinematics.h"
#include "model.h"
#include <iostream>

namespace Capt {

class Trajectory {
public:
  Trajectory(Model model);
  ~Trajectory();

  void setTorso(vec3_t torso);
  void setJoints(std::vector<float> joints);
  void setRLegRef(vec3_t rleg_ref);
  void setLLegRef(vec3_t lleg_ref);
  void setComRef(vec3_t com_ref);

  vec3_t getTorsoRef();
  vec3_t getRLegRef();
  vec3_t getLLegRef();

  bool calc();

private:
  Kinematics kinematics;

  vec3_t com_ref;
  vec3_t rleg_ref;
  vec3_t lleg_ref;
  vec3_t torso_ref;

  const float lambda; // 0 < lambda <= 1 stabilize calculation
  const float accuracy;
};
} // namespace Capt

#endif // __TRAJECTORY_H__