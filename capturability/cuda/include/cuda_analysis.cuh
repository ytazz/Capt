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

struct GridSize {
  int num_state;
  int num_input;
  int num_grid;
};

struct GridCartesian {
  int num_state;
  int num_input;
  int num_grid;

  int icp_x_num;
  int icp_y_num;
  int swf_x_num;
  int swf_y_num;

  double icp_x_min, icp_x_max, icp_x_step;
  double icp_y_min, icp_y_max, icp_y_step;
  double swf_x_min, swf_x_max, swf_x_step;
  double swf_y_min, swf_y_max, swf_y_step;

  void operator=(const GridCartesian &grid);
};

struct GridPolar {
  int num_state;
  int num_input;
  int num_grid;

  int icp_r_num;
  int icp_th_num;
  int swf_r_num;
  int swf_th_num;

  double icp_r_min, icp_r_max, icp_r_step;
  double icp_th_min, icp_th_max, icp_th_step;
  double swf_r_min, swf_r_max, swf_r_step;
  double swf_th_min, swf_th_max, swf_th_step;

  void operator=(const GridPolar &grid);
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
  Capt::Grid  *grid;
};

/* host function */

// メモリ確保や初期値代入を行うクラス
class MemoryManager {
public:
  void setGrid(GridCartesian* grid);
  void setGrid(GridPolar* grid);

  __host__ void initHostState(State *state, Condition cond);
  __host__ void initHostInput(Input *input, Condition cond);
  __host__ void initHostTrans(int   *trans, Condition cond);
  __host__ void initHostBasin(int   *basin, Condition cond);
  __host__ void initHostNstep(int   *nstep, Condition cond);
  __host__ void initHostGrid(GridCartesian *grid, Condition cond);
  __host__ void initHostGrid(GridPolar *grid, Condition cond);
  __host__ void initCop(Vector2 *cop, Condition cond);
  __host__ void initPhysics(Physics *physics, Condition cond);

  __host__ void initDevState(Cuda::State *dev_state);
  __host__ void initDevInput(Cuda::Input *dev_input);
  __host__ void initDevTrans(int   *dev_trans);
  __host__ void initDevBasin(int   *dev_basin);
  __host__ void initDevNstep(int   *dev_nstep);
  __host__ void initDevGrid(Cuda::GridCartesian *dev_grid);
  __host__ void initDevGrid(Cuda::GridPolar *dev_grid);
  __host__ void initDevCop(Vector2 *dev_cop);
  __host__ void initDevPhysics(Physics *dev_physics);

  __host__ void copyHostToDevState(State *state, Cuda::State *dev_state);
  __host__ void copyHostToDevInput(Input *input, Cuda::Input *dev_input);
  __host__ void copyHostToDevTrans(int   *trans, int *dev_trans);
  __host__ void copyHostToDevBasin(int   *basin, int *dev_basin);
  __host__ void copyHostToDevNstep(int   *nstep, int *dev_nstep);
  __host__ void copyHostToDevGrid(GridCartesian *grid, Cuda::GridCartesian *dev_grid);
  __host__ void copyHostToDevGrid(GridPolar *grid, Cuda::GridPolar *dev_grid);
  __host__ void copyHostToDevCop(Vector2 *cop, Vector2 *dev_cop);
  __host__ void copyHostToDevPhysics(Physics *physics, Physics *dev_physics);

  __host__ void copyDevToHostState(Cuda::State *dev_state, State *state);
  __host__ void copyDevToHostInput(Cuda::Input *dev_input, Input *input);
  __host__ void copyDevToHostTrans(int *dev_trans, int *trans);
  __host__ void copyDevToHostBasin(int *dev_basin, int *basin);
  __host__ void copyDevToHostNstep(int *dev_nstep, int *nstep);
  __host__ void copyDevToHostGrid(Cuda::GridCartesian *dev_grid, GridCartesian *grid);
  __host__ void copyDevToHostGrid(Cuda::GridPolar *dev_grid, GridPolar *grid);

private:
  GridSize grid;
};

__host__ void outputBasin(std::string file_name, bool header, Condition cond,
                          int *basin);
__host__ void outputNStep(std::string file_name, bool header, Condition cond,
                          int *nstep, int *trans);

__host__ void exeZeroStep(Capt::Grid grid, Capt::Model model, int *basin);

/* device function */

// メモリ確保と初期値代入を行う初期化関数
__device__ State step(State state, Input input, Vector2 cop, Physics *physics);
__device__ bool  existState(State state, GridPolar *grid);
__device__ int   getStateIndex(State state, GridPolar *grid);
__device__ int   roundValue(double value);

/* global function */

__global__ void calcStateTrans(State *state, Input *input, int *trans, GridPolar *grid,
                               Vector2 *cop, Physics *physics);

__global__ void exeNStep(int N, int *basin, int *nstep, int *trans, GridPolar *grid);

} // namespace Cuda

#endif // __CUDA_ANALYSIS_CUH__