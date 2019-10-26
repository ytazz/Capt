#include "cuda_analysis.cuh"

namespace Cuda {

__host__ void saveBasin(std::string file_name, Condition cond, int *basin,
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

__host__ void saveNStep(std::string file_name, Condition cond, int *nstep, int *trans,
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
  int num_step[NUM_STEP_MAX + 1]; // 最大踏み出し歩数を10とする
  for(int i = 0; i < NUM_STEP_MAX + 1; i++) {
    num_step[i] = 0;
  }
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
      if(nstep[id] > 0)
        num_step[nstep[id]]++;
    }
  }

  printf("max(nstep) = %d\n", max);
  for(int i = 1; i <= max; i++) {
    printf("%d-step capture point: %d\n", i, num_step[i]);
  }

  fclose(fp);
}

__host__ void saveCop(std::string file_name, Condition cond, Vector2 *cop,
                      bool header){
  FILE     *fp        = fopen(file_name.c_str(), "w");
  const int num_state = cond.grid->getNumState();

  // Header
  if (header) {
    fprintf(fp, "%s,", "state_id");
    fprintf(fp, "%s,", "cop_x");
    fprintf(fp, "%s", "cop_y");
    fprintf(fp, "\n");
  }

  // Data
  for(int state_id = 0; state_id < num_state; state_id++) {
    fprintf(fp, "%d,", state_id);
    fprintf(fp, "%1.4lf,", cop[state_id].x_);
    fprintf(fp, "%1.4lf", cop[state_id].y_);
    fprintf(fp, "\n");
  }

  fclose(fp);
}

__host__ void saveStepTime(std::string file_name, Condition cond, double *step_time,
                           bool header){
  FILE     *fp        = fopen(file_name.c_str(), "w");
  const int num_state = cond.grid->getNumState();
  const int num_input = cond.grid->getNumInput();

  // Header
  if (header) {
    fprintf(fp, "%s,", "state_id");
    fprintf(fp, "%s,", "input_id");
    fprintf(fp, "%s", "step_time");
    fprintf(fp, "\n");
  }

  // Data
  for (int state_id = 0; state_id < num_state; state_id++) {
    for (int input_id = 0; input_id < num_input; input_id++) {
      int id = state_id * num_input + input_id;
      fprintf(fp, "%d,", state_id);
      fprintf(fp, "%d,", input_id);
      fprintf(fp, "%1.4lf", step_time[id]);
      fprintf(fp, "\n");
    }
  }

  fclose(fp);
}

/* device function */

__device__ bool inPolygon(Vector2 point, Vector2 *convex, const int max_size, int swf_id){
  int num_vertex = 0;
  for(int i = 0; i < max_size; i++) {
    int convex_id = swf_id * max_size + i;
    if(convex[convex_id].x_ > -1) {
      num_vertex++;
    }
  }

  Vector2 *vertex = new Vector2[num_vertex];
  for(int i = 0; i < num_vertex; i++) {
    int convex_id = swf_id * max_size + i;
    vertex[i] = convex[convex_id];
  }

  bool flag = inPolygon(point, vertex, num_vertex);

  delete vertex;

  return flag;
}

__device__ bool inPolygon(Vector2 point, Vector2 *vertex, int num_vertex){
  bool        flag    = false;
  double      product = 0.0;
  int         sign    = 0, on_line = 0;
  const float epsilon = 0.00001;

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

__device__ Vector2 getClosestPoint(Vector2 point, Vector2* vertex, int num_vertex) {
  Vector2 closest;
  Vector2 v1, v2, v3, v4; // vector
  Vector2 n1, n2;         // normal vector

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
        float k = v1 * v2 / ( v2.norm() * v2.norm() );
        closest = vertex[i] + k * v2;
      }
    }
  }

  return closest;
}

__device__ State step(State state, Input input, Vector2 cop, double step_time, Physics *physics) {
  // LIPM
  Vector2 icp;
  icp = ( state.icp - cop ) * exp(physics->omega * step_time) + cop;

  // 状態変換
  State state_;
  state_.icp.setCartesian(-input.swf.x() + icp.x(), input.swf.y() - icp.y() );
  state_.swf.setCartesian(-input.swf.x(), input.swf.y() );

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

__device__ bool existState(State state, GridCartesian *grid) {
  bool flag_icp_x = false, flag_icp_y = false;
  bool flag_swf_x = false, flag_swf_y = false;

  // icp_x
  if (state.icp.x_ >= grid->icp_x_min - grid->icp_x_step / 2.0 &&
      state.icp.x_ < grid->icp_x_max + grid->icp_x_step / 2.0) {
    flag_icp_x = true;
  }
  // icp_y
  if (state.icp.y_ >= grid->icp_y_min - grid->icp_y_step / 2.0 &&
      state.icp.y_ < grid->icp_y_max + grid->icp_y_step / 2.0) {
    flag_icp_y = true;
  }
  // swf_x
  if (state.swf.x_ >= grid->swf_x_min - grid->swf_x_step / 2.0 &&
      state.swf.x_ < grid->swf_x_max + grid->swf_x_step / 2.0) {
    flag_swf_x = true;
  }
  // swf_y
  if (state.swf.y_ >= grid->swf_y_min - grid->swf_y_step / 2.0 &&
      state.swf.y_ < grid->swf_y_max + grid->swf_y_step / 2.0) {
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

  icp_x_id = roundValue( ( state.icp.x() - grid->icp_x_min ) / grid->icp_x_step);
  icp_y_id = roundValue( ( state.icp.y() - grid->icp_y_min ) / grid->icp_y_step);
  swf_x_id = roundValue( ( state.swf.x() - grid->swf_x_min ) / grid->swf_x_step);
  swf_y_id = roundValue( ( state.swf.y() - grid->swf_y_min ) / grid->swf_y_step);

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

__global__ void calcCop(State *state, GridCartesian *grid, Vector2 *foot_r, Vector2 *cop){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < grid->num_state) {
    cop[tid] = getClosestPoint(state[tid].icp, foot_r, grid->num_foot_r );

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void calcStepTime(State *state, Input *input, GridCartesian *grid, double *step_time, Physics *physics){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  Vector2 foot_dist;
  while (tid < grid->num_state * grid->num_input) {
    int state_id = tid / grid->num_input;
    int input_id = tid % grid->num_input;

    foot_dist      = state[state_id].swf - input[input_id].swf;
    step_time[tid] = foot_dist.norm() / physics->v + physics->dt;

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void calcBasin(State *state, int *basin, GridCartesian *grid, Vector2 *foot_r, Vector2 *foot_l, Vector2 *convex){
  int       tid      = threadIdx.x + blockIdx.x * blockDim.x;
  const int max_size = grid->num_foot_r + grid->num_foot_l;

  if(enableDoubleSupport) {
    while (tid < grid->num_state) {
      int swf_id = tid % grid->num_input;

      if(inPolygon(state[tid].icp, convex, max_size, swf_id) )
        basin[tid] = 0;
      tid += blockDim.x * gridDim.x;
    }
  }else{
    while (tid < grid->num_state) {
      if(inPolygon(state[tid].icp, foot_r, grid->num_foot_r) )
        basin[tid] = 0;
      tid += blockDim.x * gridDim.x;
    }
  }
}

__global__ void calcTrans(State *state, Input *input, int *trans, GridCartesian *grid,
                          Vector2 *cop, double *step_time, Physics *physics){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < grid->num_state * grid->num_input) {
    int state_id = tid / grid->num_input;
    int input_id = tid % grid->num_input;

    State state_ = step(state[state_id], input[input_id], cop[state_id], step_time[tid], physics);
    if (existState(state_, grid) )
      trans[tid] = getStateIndex(state_, grid);
    else
      trans[tid] = -1;

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void calcTrans(State *state, Input *input, int *trans, GridPolar *grid,
                          Vector2 *cop, double *step_time, Physics *physics){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < grid->num_state * grid->num_input) {
    int state_id = tid / grid->num_input;
    int input_id = tid % grid->num_input;

    State state_ = step(state[state_id], input[input_id], cop[state_id], step_time[tid], physics);
    if (existState(state_, grid) )
      trans[tid] = getStateIndex(state_, grid);
    else
      trans[tid] = -1;

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void exeNstep(int N, int *basin,
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

__global__ void exeNstep(int N, int *basin,
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