#ifndef __STATE_H__
#define __STATE_H__

#include "vector.h"
#include <iostream>
#include <string>

namespace CA {

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
};

} // namespace CA

#endif // __STATE_H__