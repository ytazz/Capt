#include "footstep.h"

namespace CA {

FootStep::FootStep(Capturability *capturability, Grid *grid)
    : capturability(capturability), grid(grid) {
  num_step = 0;
}

FootStep::~FootStep() {}

void FootStep::setState(State state) {
  if (grid->existState(state))
    gstate = grid->roundState(state);
}

void FootStep::setRLeg(vec3_t rleg) { this->rleg_ini = rleg; }

void FootStep::setLLeg(vec3_t lleg) { this->lleg_ini = lleg; }

bool FootStep::plan() {
  bool flag = false;

  return flag;
}

int FootStep::getNumStep() { return num_step; }

vec3_t FootStep::getLanding(int step) {
  if (step <= num_step) {
  }
}

Phase FootStep::getPhase(int step) {
  if (step <= num_step) {
  }
}

void FootStep::show() {
  printf("|no.\t|leg\t|x\t|y\t|z\t|t\t|\n");
  printf("|no.\t|leg\t|x\t|y\t|z\t|t\t|\n");
}
} // namespace CA