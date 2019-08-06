#ifndef __STATE_H__
#define __STATE_H__

#include "vector.h"
#include <iostream>
#include <string>

namespace Capt {

struct State {
  Vector2 icp;
  Vector2 swft;

  void printPolar() {
    printf("icp  = [ %lf, %lf ]\n", icp.r, icp.th);
    printf("swft = [ %lf, %lf ]\n", swft.r, swft.th);
  }
  void printCartesian() {
    printf("icp  = [ %lf, %lf ]\n", icp.x, icp.y);
    printf("swft = [ %lf, %lf ]\n", swft.x, swft.y);
  }

  void operator=(const State &state) {
    this->icp = state.icp;
    this->swft = state.swft;
  }
};

} // namespace Capt

#endif // __STATE_H__