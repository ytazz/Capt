#ifndef __FOOTSTEP_H__
#define __FOOTSTEP_H__

#include "CA.h"

namespace CA {
class FootStep {
public:
  FootStep(Capturability *capturability, Grid *grid);
  ~FootStep();

  void setState(State state);
  void setRLeg(vec3_t rleg);
  void setLLeg(vec3_t lleg);

  bool plan();

  int getNumStep();
  vec3_t getLanding(int step);
  Phase getPhase(int step);

  void show();

private:
  Capturability *capturability;
  Grid *grid;

  GridState gstate;

  int num_step;
  vec3_t rleg_ini, lleg_ini;
  std::vector<Phase> phase;
};

} // namespace CA

#endif // __FOOTSTEP_H____