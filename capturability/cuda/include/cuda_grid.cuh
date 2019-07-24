#ifndef __CUDA_GRID_CUH__
#define __CUDA_GRID_CUH__

#include "cuda_input.cuh"
#include "cuda_state.cuh"
#include "cuda_vector.cuh"
#include <iostream>
#include <string>
#include <vector>

struct CudaGrid {
  // CudaState *state;
  // CudaInput *input;
  // int *nstep;

  int num_state;
  int num_input;
  int num_nstep;

  int num_icp_r;
  int num_icp_th;
  int num_swf_r;
  int num_swf_th;

  double icp_r_min, icp_r_max, icp_r_step;
  double icp_th_min, icp_th_max, icp_th_step;
  double swf_r_min, swf_r_max, swf_r_step;
  double swf_th_min, swf_th_max, swf_th_step;

  __device__ int getStateIndex(CudaState state);
  __device__ int round(double value);
};

#endif // __CUDA_GRID_CUH__