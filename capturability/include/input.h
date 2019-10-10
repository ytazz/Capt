#ifndef __INPUT_H__
#define __INPUT_H__

#include "vector.h"
#include <iostream>
#include <string>

namespace Capt {

struct Input {
  Vector2 swf;

  void printPolar() { printf("swf = [ %lf, %lf ]\n", swf.r, swf.th); }
  void printCartesian() { printf("swf = [ %lf, %lf ]\n", swf.x, swf.y); }

  void operator=(const Input &input) { this->swf = input.swf; }
};

} // namespace Capt

#endif // __INPUT_H__