#ifndef __CUDA_MEMORY_MANAGER_CUH__
#define __CUDA_MEMORY_MANAGER_CUH__

#include "cuda_vector.cuh"
#include "cuda_config.cuh"
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
  int num_foot_vertex;
};

struct GridCartesian {
  int num_state;
  int num_input;
  int num_grid;
  int num_foot_vertex;

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
  int num_foot_vertex;

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
  __host__ void initHostFootR(Vector2 *foot, Condition cond);
  __host__ void initHostFootL(Vector2 *foot, Condition cond);
  __host__ void initHostConvex(Vector2 *convex, Condition cond);
  __host__ void initHostCop(Vector2 *cop, Condition cond);
  __host__ void initHostStepTime(double *step_time, Condition cond);
  __host__ void initHostPhysics(Physics *physics, Condition cond);

  __host__ void initDevState(Cuda::State *dev_state);
  __host__ void initDevInput(Cuda::Input *dev_input);
  __host__ void initDevTrans(int   *dev_trans);
  __host__ void initDevBasin(int   *dev_basin);
  __host__ void initDevNstep(int   *dev_nstep);
  __host__ void initDevGrid(Cuda::GridCartesian *dev_grid);
  __host__ void initDevGrid(Cuda::GridPolar *dev_grid);
  __host__ void initDevFootR(Vector2 *dev_foot_r);
  __host__ void initDevFootL(Vector2 *dev_foot_l);
  __host__ void initDevConvex(Vector2 *dev_convex);
  __host__ void initDevCop(Vector2 *dev_cop);
  __host__ void initDevStepTime(double *dev_step_time);
  __host__ void initDevPhysics(Physics *dev_physics);

  __host__ void copyHostToDevState(State *state, Cuda::State *dev_state);
  __host__ void copyHostToDevInput(Input *input, Cuda::Input *dev_input);
  __host__ void copyHostToDevTrans(int   *trans, int *dev_trans);
  __host__ void copyHostToDevBasin(int   *basin, int *dev_basin);
  __host__ void copyHostToDevNstep(int   *nstep, int *dev_nstep);
  __host__ void copyHostToDevGrid(GridCartesian *grid, Cuda::GridCartesian *dev_grid);
  __host__ void copyHostToDevGrid(GridPolar *grid, Cuda::GridPolar *dev_grid);
  __host__ void copyHostToDevFootR(Vector2 *foot_r, Vector2 *dev_foot_r);
  __host__ void copyHostToDevFootL(Vector2 *foot_l, Vector2 *dev_foot_l);
  __host__ void copyHostToDevConvex(Vector2 *convex, Vector2 *dev_convex);
  __host__ void copyHostToDevCop(Vector2 *cop, Vector2 *dev_cop);
  __host__ void copyHostToDevStepTime(double *step_time, double *dev_step_time);
  __host__ void copyHostToDevPhysics(Physics *physics, Physics *dev_physics);

  __host__ void copyDevToHostState(Cuda::State *dev_state, State *state);
  __host__ void copyDevToHostInput(Cuda::Input *dev_input, Input *input);
  __host__ void copyDevToHostTrans(int *dev_trans, int *trans);
  __host__ void copyDevToHostBasin(int *dev_basin, int *basin);
  __host__ void copyDevToHostNstep(int *dev_nstep, int *nstep);
  __host__ void copyDevToHostGrid(Cuda::GridCartesian *dev_grid, GridCartesian *grid);
  __host__ void copyDevToHostGrid(Cuda::GridPolar *dev_grid, GridPolar *grid);
  __host__ void copyDevToHostFootR(Vector2 *dev_foot_r, Vector2 *foot_r);
  __host__ void copyDevToHostFootL(Vector2 *dev_foot_l, Vector2 *foot_l);
  __host__ void copyDevToHostConvex(Vector2 *dev_convex, Vector2 *convex);
  __host__ void copyDevToHostCop(Vector2 *dev_cop, Vector2 *cop);
  __host__ void copyDevToHostStepTime(double *dev_step_time, double *step_time);
  __host__ void copyDevToHostPhysics(Physics *dev_physics, Physics *physics);

private:
  GridSize grid;
};

} // namespace Cuda

#endif // __CUDA_MEMORY_MANAGER_CUH__