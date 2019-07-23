#ifndef __CUDA_INPUT_CUH__
#define __CUDA_INPUT_CUH__

#include "cuda_vector.cuh"
#include <iostream>
#include <string>

struct CudaInput {
  CudaVector2 swf;

  __device__ void operator=(const CudaInput &input) { this->swf = input.swf; }
};

#endif // __CUDA_INPUT_CUH__