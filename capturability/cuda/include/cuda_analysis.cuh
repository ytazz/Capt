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

struct Condition {
  CA::Model *model;
  CA::Param *param;
  CA::Grid *grid;
};

/* host function */

void initState(CudaState *cstate, int *next_state_id, Condition cond);
void initInput(CudaInput *cinput, Condition cond);
void initNstep(int *cnstep, Condition cond);
void initGrid(CudaGrid *cgrid, Condition cond);
void initCop(CudaVector2 *cop, Condition cond);

void output(std::string file_name, Condition cond, int *cnstep,
            int *next_state_id);

__host__ void exeZeroStep(CA::Grid grid, CA::Model model, int *nstep,
                          int *next_state_id);

/* device function */

__device__ int getStateIndex(CudaState state, CudaGrid grid);

__device__ int roundValue(double value);

/* global function */

// __global__ void exeZeroStep(CudaState *state, CudaInput *input, int *nstep,
//                             CudaVector2 *foot, CudaGrid *grid);

#endif // __CUDA_ANALYSIS_CUH__