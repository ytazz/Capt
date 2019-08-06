#ifndef __CUDA_ANALYSIS_CUH__
#define __CUDA_ANALYSIS_CUH__

#include "cuda_vector.cuh"
#include "grid.h"
#include "model.h"
#include "nvidia.cuh"
#include "param.h"
#include "state.h"
#include <chrono>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <vector>

/* struct */

struct CudaGrid {
  int num_state;
  int num_input;
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
  double g;     // gravity
  double h;     // com_height
  double v;     // foot_vel_max
  double dt;    // step_time_min
  double omega; // LIPM omega
};

struct Condition {
  CA::Model *model;
  CA::Param *param;
  CA::Grid *grid;
};

/* host function */

__host__ void initState(CudaState *cstate, int *next_state_id, Condition cond);
__host__ void initInput(CudaInput *cinput, Condition cond);
__host__ void initNstep(int *cnstep, Condition cond);
__host__ void initGrid(CudaGrid *cgrid, Condition cond);
__host__ void initCop(CudaVector2 *cop, Condition cond);
__host__ void initPhysics(CudaPhysics *physics, Condition cond);

__host__ void output(std::string file_name, Condition cond, int *cnstep,
                     int *next_state_id);

__host__ void exeZeroStep(CA::Grid grid, CA::Model model, int *nstep,
                          int *next_state_id);

/* device function */

__device__ CudaState step(CudaState state, CudaInput input);

__device__ bool existState(CudaState state, CudaGrid grid);
__device__ int getStateIndex(CudaState state, CudaGrid grid);

__device__ int roundValue(double value);

/* global function */

__global__ void exeNStep(CudaState *state, CudaInput *input, int *nstep,
                         int *next_state_id, CudaGrid *grid, CudaVector2 *cop,
                         CudaPhysics *physics);

#endif // __CUDA_ANALYSIS_CUH__