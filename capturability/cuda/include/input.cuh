#ifndef __INPUT_CUH__
#define __INPUT_CUH__

#include "vector.cuh"
#include <iostream>
#include <string>

namespace GPGPU {

struct Input {
  Vector2 swft;

  __device__ void operator=(const Input &input) { this->swft = input.swft; }
};

} // namespace GPGPU

#endif // __INPUT_CUH__