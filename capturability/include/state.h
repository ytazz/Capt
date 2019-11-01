#ifndef __STATE_H__
#define __STATE_H__

#include "base.h"
#include <iostream>
#include <string>

namespace Capt {

struct State {
  Vector2 icp;
  Vector2 swf;

  void printPolar() {
    printf("icp  = [ %lf, %lf ]\n", icp.r, icp.th);
    printf("swf = [ %lf, %lf ]\n", swf.r, swf.th);
  }
  void printCartesian() {
    printf("icp  = [ %lf, %lf ]\n", icp.x, icp.y);
    printf("swf = [ %lf, %lf ]\n", swf.x, swf.y);
  }

  void operator=(const State &state) {
    this->icp = state.icp;
    this->swf = state.swf;
  }
};

} // namespace Capt

#endif // __STATE_H__