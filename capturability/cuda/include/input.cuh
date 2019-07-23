#ifndef __INPUT_CUH__
#define __INPUT_CUH__

#include "vector.cuh"
#include <iostream>
#include <string>

struct Input {
  Vector2 swf;

  __device__ void operator=(const Input &input) { this->swf = input.swf; }
};

#endif // __INPUT_CUH__