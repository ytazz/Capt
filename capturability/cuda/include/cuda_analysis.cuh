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

namespace Cuda {

/* struct */

struct Grid {
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

struct State {
  Vector2 icp;
  Vector2 swf;

  __device__ void operator=(const State &state);
};

struct Input {
  Vector2 swf;

  __device__ void operator=(const Input &input);
};

struct Physics {
  double g;     // gravity
  double h;     // com_height
  double v;     // foot_vel_max
  double dt;    // step_time_min
  double omega; // LIPM omega
};

struct Condition {
  Capt::Model *model;
  Capt::Param *param;
  Capt::Grid *grid;
};

/* host function */

__host__ void initState(State *cstate, int *next_state_id, Condition cond);
__host__ void initInput(Input *cinput, Condition cond);
__host__ void initNstep(int *cnstep, Condition cond);
__host__ void initGrid(Grid *cgrid, Condition cond);
__host__ void initCop(Vector2 *cop, Condition cond);
__host__ void initPhysics(Physics *physics, Condition cond);

__host__ void output(std::string file_name, Condition cond, int *cnstep,
                     int *next_state_id);

__host__ void exeZeroStep(Capt::Grid grid, Capt::Model model, int *nstep,
                          int *next_state_id);

/* device function */

__device__ State step(State state, Input input);

__device__ bool existState(State state, Grid grid);
__device__ int getStateIndex(State state, Grid grid);

__device__ int roundValue(double value);

/* global function */

__global__ void exeNStep(State *state, Input *input, int *nstep,
                         int *next_state_id, Grid *grid, Vector2 *cop,
                         Physics *physics);
} // namespace Cuda

#endif // __CUDA_ANALYSIS_CUH__