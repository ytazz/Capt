#ifndef __INPUT_H__
#define __INPUT_H__

#include "vector.h"
#include <iostream>
#include <string>

namespace Capt {

struct Input {
  Vector2 swft;

  void printPolar() { printf("swft = [ %lf, %lf ]\n", swft.r, swft.th); }
  void printCartesian() { printf("swft = [ %lf, %lf ]\n", swft.x, swft.y); }

  void operator=(const Input &input) { this->swft = input.swft; }
};

} // namespace Capt

#endif // __INPUT_H__