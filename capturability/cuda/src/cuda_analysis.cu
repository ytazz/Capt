#include "cuda_analysis.cuh"

namespace Cuda {

__host__ void outputBasin(std::string file_name, Condition cond, int *basin,
                          bool header) {
  FILE     *fp        = fopen(file_name.c_str(), "w");
  const int num_state = cond.grid->getNumState();

  // Header
  if (header) {
    fprintf(fp, "%s,", "state_id");
    fprintf(fp, "%s", "nstep");
    fprintf(fp, "\n");
  }

  // Data
  for (int state_id = 0; state_id < num_state; state_id++) {
    fprintf(fp, "%d,", state_id);
    fprintf(fp, "%d", basin[state_id]);
    fprintf(fp, "\n");
  }

  fclose(fp);
}

__host__ void outputNStep(std::string file_name, Condition cond, int *nstep, int *trans,
                          bool header) {
  FILE     *fp        = fopen(file_name.c_str(), "w");
  const int num_state = cond.grid->getNumState();
  const int num_input = cond.grid->getNumInput();
  int       max       = 0;

  // Header
  if (header) {
    fprintf(fp, "%s,", "state_id");
    fprintf(fp, "%s,", "input_id");
    fprintf(fp, "%s,", "trans");
    fprintf(fp, "%s", "nstep");
    fprintf(fp, "\n");
  }

  // Data
  for (int state_id = 0; state_id < num_state; state_id++) {
    for (int input_id = 0; input_id < num_input; input_id++) {
      int id = state_id * num_input + input_id;
      fprintf(fp, "%d,", state_id);
      fprintf(fp, "%d,", input_id);
      fprintf(fp, "%d,", trans[id]);
      fprintf(fp, "%d", nstep[id]);
      fprintf(fp, "\n");
      if (max < nstep[id])
        max = nstep[id];
    }
  }

  printf("max(nstep) = %d\n", max);

  fclose(fp);
}

__host__ void exeZeroStep(Capt::Grid grid, Capt::Model model, int *basin) {
  for (int state_id = 0; state_id < grid.getNumState(); state_id++) {
    Capt::State state = grid.getState(state_id);

    Capt::Polygon polygon;
    polygon.setVertex(model.getVec("foot", "foot_r_convex") );
    polygon.setVertex(model.getVec("foot", "foot_l_convex", state.swf) );

    bool flag = false;
    flag = polygon.inPolygon(state.icp, polygon.getConvexHull() );

    if (flag) {
      basin[state_id] = 0;
    }
  }
}

/* device function */

__device__ State step(State state, Input input, Vector2 cop, Physics *physics) {
  State state_;

  // 踏み出し時間
  Vector2 foot_dist = state.swf - input.swf;
  double  dist
    = sqrt(foot_dist.x() * foot_dist.x() + foot_dist.y() * foot_dist.y() );
  double t = dist / physics->v + physics->dt;

  // LIPM
  Vector2 icp = state.icp;
  icp = ( icp - cop ) * exp(physics->omega * t) + cop;

  // 状態変換
  state_.icp.setCartesian(-input.swf.x() + icp.x(), input.swf.y() - icp.y() );
  state_.swf.setCartesian(-input.swf.x(), input.swf.y() );

  return state_;
}

__device__ int roundValue(double value) {
  int result = (int)value;

  double decimal = value - (int)value;
  if (decimal >= 0.5) {
    result += 1;
  }

  return result;
}

__device__ bool existState(State state, GridCartesian *grid) {
  bool flag_icp_x = false, flag_icp_y = false;
  bool flag_swf_x = false, flag_swf_y = false;

  // icp_x
  if (state.icp.r_ >= grid->icp_x_min - grid->icp_x_step / 2.0 &&
      state.icp.r_ < grid->icp_x_max + grid->icp_x_step / 2.0) {
    flag_icp_x = true;
  }
  // icp_y
  if (state.icp.th_ >= grid->icp_y_min - grid->icp_y_step / 2.0 &&
      state.icp.th_ < grid->icp_y_max + grid->icp_y_step / 2.0) {
    flag_icp_y = true;
  }
  // swf_x
  if (state.swf.r_ >= grid->swf_x_min - grid->swf_x_step / 2.0 &&
      state.swf.r_ < grid->swf_x_max + grid->swf_x_step / 2.0) {
    flag_swf_x = true;
  }
  // swf_y
  if (state.swf.th_ >= grid->swf_y_min - grid->swf_y_step / 2.0 &&
      state.swf.th_ < grid->swf_y_max + grid->swf_y_step / 2.0) {
    flag_swf_y = true;
  }

  bool flag = flag_icp_x * flag_icp_y * flag_swf_x * flag_swf_y;
  return flag;
}

__device__ bool existState(State state, GridPolar *grid) {
  bool flag_icp_r = false, flag_icp_th = false;
  bool flag_swf_r = false, flag_swf_th = false;

  // icp_r
  if (state.icp.r_ >= grid->icp_r_min - grid->icp_r_step / 2.0 &&
      state.icp.r_ < grid->icp_r_max + grid->icp_r_step / 2.0) {
    flag_icp_r = true;
  }
  // icp_th
  if (state.icp.th_ >= grid->icp_th_min - grid->icp_th_step / 2.0 &&
      state.icp.th_ < grid->icp_th_max + grid->icp_th_step / 2.0) {
    flag_icp_th = true;
  }
  // swf_r
  if (state.swf.r_ >= grid->swf_r_min - grid->swf_r_step / 2.0 &&
      state.swf.r_ < grid->swf_r_max + grid->swf_r_step / 2.0) {
    flag_swf_r = true;
  }
  // swf_th
  if (state.swf.th_ >= grid->swf_th_min - grid->swf_th_step / 2.0 &&
      state.swf.th_ < grid->swf_th_max + grid->swf_th_step / 2.0) {
    flag_swf_th = true;
  }

  bool flag = flag_icp_r * flag_icp_th * flag_swf_r * flag_swf_th;
  return flag;
}

__device__ int getStateIndex(State state, GridCartesian *grid) {
  int icp_x_id = 0, icp_y_id = 0;
  int swf_x_id = 0, swf_y_id = 0;

  icp_x_id = roundValue( ( state.icp.r() - grid->icp_x_min ) / grid->icp_x_step);
  icp_y_id = roundValue( ( state.icp.th() - grid->icp_y_min ) / grid->icp_y_step);
  swf_x_id = roundValue( ( state.swf.r() - grid->swf_x_min ) / grid->swf_x_step);
  swf_y_id = roundValue( ( state.swf.th() - grid->swf_y_min ) / grid->swf_y_step);

  int state_id = 0;
  state_id = grid->swf_y_num * grid->swf_x_num * grid->icp_y_num * icp_x_id +
             grid->swf_y_num * grid->swf_x_num * icp_y_id +
             grid->swf_y_num * swf_x_id + swf_y_id;

  return state_id;
}

__device__ int getStateIndex(State state, GridPolar *grid) {
  int icp_r_id = 0, icp_th_id = 0;
  int swf_r_id = 0, swf_th_id = 0;

  icp_r_id  = roundValue( ( state.icp.r() - grid->icp_r_min ) / grid->icp_r_step);
  icp_th_id = roundValue( ( state.icp.th() - grid->icp_th_min ) / grid->icp_th_step);
  swf_r_id  = roundValue( ( state.swf.r() - grid->swf_r_min ) / grid->swf_r_step);
  swf_th_id = roundValue( ( state.swf.th() - grid->swf_th_min ) / grid->swf_th_step);

  int state_id = 0;
  state_id = grid->swf_th_num * grid->swf_r_num * grid->icp_th_num * icp_r_id +
             grid->swf_th_num * grid->swf_r_num * icp_th_id +
             grid->swf_th_num * swf_r_id + swf_th_id;

  return state_id;
}

/* global function */

__global__ void calcStateTrans(State *state, Input *input, int *trans, GridCartesian *grid,
                               Vector2 *cop, Physics *physics){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < grid->num_state * grid->num_input) {
    int state_id = tid / grid->num_input;
    int input_id = tid % grid->num_input;

    State state_ = step(state[state_id], input[input_id], cop[state_id], physics);
    if (existState(state_, grid) )
      trans[tid] = getStateIndex(state_, grid);
    else
      trans[tid] = -1;

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void calcStateTrans(State *state, Input *input, int *trans, GridPolar *grid,
                               Vector2 *cop, Physics *physics){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < grid->num_state * grid->num_input) {
    int state_id = tid / grid->num_input;
    int input_id = tid % grid->num_input;

    State state_ = step(state[state_id], input[input_id], cop[state_id], physics);
    if (existState(state_, grid) )
      trans[tid] = getStateIndex(state_, grid);
    else
      trans[tid] = -1;

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void exeNStep(int N, int *basin,
                         int *nstep, int *trans, GridCartesian *grid) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < grid->num_state * grid->num_input) {
    int state_id = tid / grid->num_input;
    int input_id = tid % grid->num_input;

    if (trans[tid] >= 0) {
      if (basin[trans[tid]] == ( N - 1 ) ) {
        nstep[tid] = N;
        if (basin[state_id] < 0) {
          basin[state_id] = N;
        }
      }
    }

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void exeNStep(int N, int *basin,
                         int *nstep, int *trans, GridPolar *grid) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < grid->num_state * grid->num_input) {
    int state_id = tid / grid->num_input;
    int input_id = tid % grid->num_input;

    if (trans[tid] >= 0) {
      if (basin[trans[tid]] == ( N - 1 ) ) {
        nstep[tid] = N;
        if (basin[state_id] < 0) {
          basin[state_id] = N;
        }
      }
    }

    tid += blockDim.x * gridDim.x;
  }
}

} // namespace Cuda