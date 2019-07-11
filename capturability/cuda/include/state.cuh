#ifndef __STATE_CUH__
#define __STATE_CUH__

#include "vector.cuh"
#include <iostream>
#include <string>

namespace GPGPU {

struct State {
  Vector2 icp;
  Vector2 swft;

  __device__ void operator=(const State &state) {
    this->icp = state.icp;
    this->swft = state.swft;
  }
};

} // namespace GPGPU

#endif // __STATE_CUH__