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

struct Grid {
  int state_num;
  int input_num;
  int grid_num;

  int foot_r_num;
  int foot_l_num;

  int icp_x_num;
  int icp_y_num;
  int swf_x_num;
  int swf_y_num;
  int swf_z_num;
  int cop_x_num;
  int cop_y_num;

  double icp_x_min, icp_x_max, icp_x_stp;
  double icp_y_min, icp_y_max, icp_y_stp;
  double swf_x_min, swf_x_max, swf_x_stp;
  double swf_y_min, swf_y_max, swf_y_stp;
  double swf_z_min, swf_z_max, swf_z_stp;
  double cop_x_min, cop_x_max, cop_x_stp;
  double cop_y_min, cop_y_max, cop_y_stp;

  void operator=(const Grid &grid);
};

struct State {
  vec2_t icp;
  vec3_t swf;

  __device__ void operator=(const State &state);
};

struct Input {
  vec2_t cop;
  vec2_t swf;

  __device__ void operator=(const Input &input);
};

struct Physics {
  double g;     // gravity
  double h;     // com height
  double v_max; // max. foot velocity
  double z_max; // max. step height
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
  __host__ void set(Capt::Model* model, Capt::Param* param, Capt::Grid* grid);

  __host__ void initHostState(State *state);
  __host__ void initHostInput(Input *input);
  __host__ void initHostTrans(int   *trans);
  __host__ void initHostBasin(int   *basin);
  __host__ void initHostNstep(int   *nstep);
  __host__ void initHostGrid(Grid *grid);
  __host__ void initHostFootR(vec2_t *foot);
  __host__ void initHostFootL(vec2_t *foot);
  __host__ void initHostConvex(vec2_t *convex);
  __host__ void initHostStepTime(double *step_time);
  __host__ void initHostPhysics(Physics *physics);

  __host__ void initDevState(State *dev_state);
  __host__ void initDevInput(Input *dev_input);
  __host__ void initDevTrans(int   *dev_trans);
  __host__ void initDevBasin(int   *dev_basin);
  __host__ void initDevNstep(int   *dev_nstep);
  __host__ void initDevGrid(Grid *dev_grid);
  __host__ void initDevFootR(vec2_t *dev_foot_r);
  __host__ void initDevFootL(vec2_t *dev_foot_l);
  __host__ void initDevConvex(vec2_t *dev_convex);
  __host__ void initDevStepTime(double *dev_step_time);
  __host__ void initDevPhysics(Physics *dev_physics);

  __host__ void copyHostToDevState(State *state, State *dev_state);
  __host__ void copyHostToDevInput(Input *input, Input *dev_input);
  __host__ void copyHostToDevTrans(int   *trans, int *dev_trans);
  __host__ void copyHostToDevBasin(int   *basin, int *dev_basin);
  __host__ void copyHostToDevNstep(int   *nstep, int *dev_nstep);
  __host__ void copyHostToDevGrid(Grid *grid, Grid *dev_grid);
  __host__ void copyHostToDevFootR(vec2_t *foot_r, vec2_t *dev_foot_r);
  __host__ void copyHostToDevFootL(vec2_t *foot_l, vec2_t *dev_foot_l);
  __host__ void copyHostToDevConvex(vec2_t *convex, vec2_t *dev_convex);
  __host__ void copyHostToDevStepTime(double *step_time, double *dev_step_time);
  __host__ void copyHostToDevPhysics(Physics *physics, Physics *dev_physics);

  __host__ void copyDevToHostState(State *dev_state, State *state);
  __host__ void copyDevToHostInput(Input *dev_input, Input *input);
  __host__ void copyDevToHostTrans(int *dev_trans, int *trans);
  __host__ void copyDevToHostBasin(int *dev_basin, int *basin);
  __host__ void copyDevToHostNstep(int *dev_nstep, int *nstep);
  __host__ void copyDevToHostGrid(Grid *dev_grid, Grid *grid);
  __host__ void copyDevToHostFootR(vec2_t *dev_foot_r, vec2_t *foot_r);
  __host__ void copyDevToHostFootL(vec2_t *dev_foot_l, vec2_t *foot_l);
  __host__ void copyDevToHostConvex(vec2_t *dev_convex, vec2_t *convex);
  __host__ void copyDevToHostStepTime(double *dev_step_time, double *step_time);
  __host__ void copyDevToHostPhysics(Physics *dev_physics, Physics *physics);

  __host__ void saveBasin(std::string file_name, int *basin, bool header = false);
  __host__ void saveNstep(std::string file_name, int *nstep, int *trans, int n, bool header = false);
  __host__ void saveStepTime(std::string file_name, double *step_time, bool header = false);

private:
  Capt::Model *model;
  Capt::Param *param;
  Capt::Grid  *cgrid;
  Grid         grid;
};

} // namespace Cuda

#endif // __CUDA_MEMORY_MANAGER_CUH__