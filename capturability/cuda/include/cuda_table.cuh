#ifndef __CUDA_TABLE_CUH__
#define __CUDA_TABLE_CUH__

#include <iostream>

struct CudaTable {
  CudaState *state;
  CudaInput *input;
  int *nstep;
};

#endif // __CUDA_TABLE_CUH__