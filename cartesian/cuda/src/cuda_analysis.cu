#include "cuda_analysis.cuh"

namespace Cuda {

/* device function */

__device__ bool inPolygon(vec2_t point, vec2_t *convex, const int max_size, int swf_id){
  int num_vertex = 0;
  for(int i = 0; i < max_size; i++) {
    int convex_id = swf_id * max_size + i;
    if(convex[convex_id].x > -1) {
      num_vertex++;
    }
  }

  vec2_t *vertex = new vec2_t[num_vertex];
  for(int i = 0; i < num_vertex; i++) {
    int convex_id = swf_id * max_size + i;
    vertex[i] = convex[convex_id];
  }

  bool flag = inPolygon(point, vertex, num_vertex);

  delete vertex;

  return flag;
}

__device__ bool inPolygon(vec2_t point, vec2_t *vertex, int num_vertex){
  bool         flag    = false;
  double       product = 0.0;
  int          sign    = 0, on_line = 0;
  const double epsilon = 0.00001;

  for (size_t i = 0; i < num_vertex - 1; i++) {
    product = ( point - vertex[i] ) % ( vertex[i + 1] - vertex[i] );
    if (-epsilon <= product && product <= epsilon) {
      on_line += 1;
    } else if (product > 0) {
      sign += 1;
    } else if (product < 0) {
      sign -= 1;
    }
  }

  if (sign == int(num_vertex - 1 - on_line) ||
      sign == -int(num_vertex - 1 - on_line) ) {
    flag = true;
  }

  return flag;
}

__device__ vec2_t getClosestPoint(vec2_t point, vec2_t* vertex, int num_vertex) {
  vec2_t closest;
  vec2_t v1, v2, v3, v4; // vector
  vec2_t n1, n2;         // normal vector

  if (inPolygon(point, vertex, num_vertex) ) {
    closest = point;
  } else {
    for (int i = 0; i < num_vertex - 1; i++) {
      //最近点が角にあるとき
      if (i == 0) {
        n1 = ( vertex[1] - vertex[i] ).normal();
        n2 = ( vertex[i] - vertex[num_vertex - 2] ).normal();
      } else {
        n1 = ( vertex[i + 1] - vertex[i] ).normal();
        n2 = ( vertex[i] - vertex[i - 1] ).normal();
      }
      v1 = point - vertex[i];
      if ( ( n1 % v1 ) < 0 && ( n2 % v1 ) > 0) {
        closest = vertex[i];
      }
      // 最近点が辺にあるとき
      n1 = ( vertex[i + 1] - vertex[i] ).normal();
      v1 = point - vertex[i];
      v2 = vertex[i + 1] - vertex[i];
      v3 = point - vertex[i + 1];
      v4 = vertex[i] - vertex[i + 1];
      if ( ( n1 % v1 ) > 0 && ( v2 % v1 ) < 0 && ( n1 % v3 ) < 0 && ( v4 % v3 ) > 0) {
        double k = v1 * v2 / ( v2.norm() * v2.norm() );
        closest = vertex[i] + k * v2;
      }
    }
  }

  return closest;
}

__device__ State step(State state, Input input, double step_time, Physics *physics) {
  // LIPM
  vec2_t icp;
  icp = ( state.icp - input.cop ) * exp(physics->omega * step_time) + input.cop;

  // 状態変換
  State state_;
  state_.icp.set(-input.swf.x + icp.x, input.swf.y - icp.y );
  state_.swf.set(-input.swf.x, input.swf.y );
  state_.elp = 0;

  return state_;
}

__device__ int roundValue(double value) {
  int integer = (int)value;

  double decimal = value - integer;
  if(decimal > 0) {
    if (decimal >= 0.5) {
      integer += 1;
    }
  }else{
    if (decimal <= -0.5) {
      integer -= 1;
    }
  }

  return integer;
}

__device__ bool existState(State state, Grid *grid) {
  bool flag_icp_x = false, flag_icp_y = false;
  bool flag_swf_x = false, flag_swf_y = false;

  // icp_x
  if (state.icp.x >= grid->icp_x_min - grid->icp_x_stp / 2.0 &&
      state.icp.x < grid->icp_x_max + grid->icp_x_stp / 2.0) {
    flag_icp_x = true;
  }
  // icp_y
  if (state.icp.y >= grid->icp_y_min - grid->icp_y_stp / 2.0 &&
      state.icp.y < grid->icp_y_max + grid->icp_y_stp / 2.0) {
    flag_icp_y = true;
  }
  // swf_x
  if (state.swf.x >= grid->swf_x_min - grid->swf_x_stp / 2.0 &&
      state.swf.x < grid->swf_x_max + grid->swf_x_stp / 2.0) {
    flag_swf_x = true;
  }
  // swf_y
  if (state.swf.y >= grid->swf_y_min - grid->swf_y_stp / 2.0 &&
      state.swf.y < grid->swf_y_max + grid->swf_y_stp / 2.0) {
    flag_swf_y = true;
  }

  bool flag = flag_icp_x * flag_icp_y * flag_swf_x * flag_swf_y;
  return flag;
}

__device__ int getStateIndex(State state, Grid *grid) {
  int icp_x_id = 0, icp_y_id = 0;
  int swf_x_id = 0, swf_y_id = 0;
  int elp_t_id = 0;

  icp_x_id = roundValue( ( state.icp.x - grid->icp_x_min ) / grid->icp_x_stp);
  icp_y_id = roundValue( ( state.icp.y - grid->icp_y_min ) / grid->icp_y_stp);
  swf_x_id = roundValue( ( state.swf.x - grid->swf_x_min ) / grid->swf_x_stp);
  swf_y_id = roundValue( ( state.swf.y - grid->swf_y_min ) / grid->swf_y_stp);
  elp_t_id = 0;

  int state_id = 0;
  state_id = grid->elp_t_num * grid->swf_y_num * grid->swf_x_num * grid->icp_y_num * icp_x_id +
             grid->elp_t_num * grid->swf_y_num * grid->swf_x_num * icp_y_id +
             grid->elp_t_num * grid->swf_y_num * swf_x_id +
             grid->elp_t_num * swf_y_id +
             elp_t_id;

  return state_id;
}

/* global function */

__global__ void calcStepTime(State *state, Input *input, Grid *grid, double *step_time, Physics *physics){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  vec2_t foot_dist;
  while (tid < grid->state_num * grid->input_num) {
    int state_id = tid / grid->input_num;
    int input_id = tid % grid->input_num;

    // double tau  = max(0, step_time_min / 2 - state[state_id].elp) + dist / foot_vel_max + step_time_min / 2;
    foot_dist      = state[state_id].swf - input[input_id].swf;
    step_time[tid] = foot_dist.norm() / physics->v + physics->dt / 2;
    if(physics->dt / 2 - state[state_id].elp > 0) {
      step_time[tid] += physics->dt / 2 - state[state_id].elp;
    }

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void calcBasin(State *state, int *basin, Grid *grid, vec2_t *foot_r, vec2_t *foot_l, vec2_t *convex){
  int       tid      = threadIdx.x + blockIdx.x * blockDim.x;
  const int max_size = grid->foot_r_num + grid->foot_l_num;

  if(enableDoubleSupport) {
    while (tid < grid->state_num) {
      int swf_id = tid % grid->input_num;
      // state[tid].elp < 0.001 means landing state, not swing phase
      if(inPolygon(state[tid].icp, convex, max_size, swf_id) && state[tid].elp < 0.001 )
        basin[tid] = 0;
      tid += blockDim.x * gridDim.x;
    }
  }else{
    while (tid < grid->state_num) {
      if(inPolygon(state[tid].icp, foot_r, grid->foot_r_num) && state[tid].elp < 0.001 )
        basin[tid] = 0;
      tid += blockDim.x * gridDim.x;
    }
  }
}

__global__ void calcTrans(State *state, Input *input, int *trans, Grid *grid,
                          double *step_time, Physics *physics){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < grid->state_num * grid->input_num) {
    int state_id = tid / grid->input_num;
    int input_id = tid % grid->input_num;

    State state_ = step(state[state_id], input[input_id], step_time[tid], physics);
    if (existState(state_, grid) )
      trans[tid] = getStateIndex(state_, grid);
    else
      trans[tid] = -1;

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void exeNstep(int N, int *basin,
                         int *nstep, int *trans, Grid *grid) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < grid->state_num * grid->input_num) {
    int state_id = tid / grid->input_num;
    // int input_id = tid % grid->input_num;

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