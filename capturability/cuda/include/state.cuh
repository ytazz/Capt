#ifndef __STATE_CUH__
#define __STATE_CUH__

#include "vector.cuh"
#include <iostream>
#include <string>

struct State {
  Vector2 icp;
  Vector2 swf;

  __device__ void operator=(const State &state) {
    this->icp = state.icp;
    this->swf = state.swf;
  }
};

#endif // __STATE_CUH__