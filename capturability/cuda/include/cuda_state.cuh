#ifndef __CUDA_STATE_CUH__
#define __CUDA_STATE_CUH__

#include "cuda_vector.cuh"
#include <iostream>
#include <string>

struct CudaState {
  CudaVector2 icp;
  CudaVector2 swf;

  __device__ void operator=(const CudaState &state) {
    this->icp = state.icp;
    this->swf = state.swf;
  }
};

#endif // __CUDA_STATE_CUH__