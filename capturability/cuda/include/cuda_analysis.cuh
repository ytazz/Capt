#ifndef __CUDA_ANALYSIS_CUH__
#define __CUDA_ANALYSIS_CUH__

#include "cuda_vector.cuh"
#include "grid.h"
#include "model.h"
#include "nvidia.cuh"
#include "param.h"
#include <chrono>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <vector>

/* struct */

struct CudaGrid {
  int num_state;
  int num_input;
  int num_foot; // vertex of feet
  int num_nstep;

  int icp_r_num;
  int icp_th_num;
  int swf_r_num;
  int swf_th_num;

  double icp_r_min, icp_r_max, icp_r_step;
  double icp_th_min, icp_th_max, icp_th_step;
  double swf_r_min, swf_r_max, swf_r_step;
  double swf_th_min, swf_th_max, swf_th_step;
};

struct CudaState {
  CudaVector2 icp;
  CudaVector2 swf;

  __device__ void operator=(const CudaState &state);
};

struct CudaInput {
  CudaVector2 swf;

  __device__ void operator=(const CudaInput &input);
};

struct CudaPhysics {
  double g;
  double h;
};

/* host function */

void setNstep(CA::Grid grid, int *cnstep);
void setFoot(CudaVector2 *cfoot, CudaVector2 *cfoot_r, CudaVector2 *cfoot_l,
             const int num_foot);

void setState(CA::Grid grid, CudaState *cstate);
void setInput(CA::Grid grid, CudaInput *cinput);
void setGrid(CA::Grid grid, CA::Model model, CA::Param param, CudaGrid *cgrid);

void init(int *next_state_id, int size);

__host__ void exeZeroStep(CA::Grid grid, CA::Model model, int *nstep,
                          int *next_state_id);

/* device function */

__device__ int getStateIndex(CudaState state, CudaGrid grid);

__device__ int roundValue(double value);

/* global function */

// __global__ void exeZeroStep(CudaState *state, CudaInput *input, int *nstep,
//                             CudaVector2 *foot, CudaGrid *grid);

#endif // __CUDA_ANALYSIS_CUH__