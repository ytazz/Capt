#include "cuda_analysis.cuh"

namespace Cuda {

/* struct */

__device__ void State::operator=(const State &state) {
  this->icp = state.icp;
  this->swf = state.swf;
}

__device__ void Input::operator=(const Input &input) {
  this->swf = input.swf;
}

/* host function */

__host__ void initState(State *state, int *next_id, Condition cond) {
  const int num_state = cond.grid->getNumState();
  const int num_input = cond.grid->getNumInput();

  for (int i = 0; i < cond.grid->getNumState(); i++) {
    state[i].icp.x_  = cond.grid->getState(i).icp.x;
    state[i].icp.y_  = cond.grid->getState(i).icp.y;
    state[i].icp.r_  = cond.grid->getState(i).icp.r;
    state[i].icp.th_ = cond.grid->getState(i).icp.th;
    state[i].swf.x_  = cond.grid->getState(i).swft.x;
    state[i].swf.y_  = cond.grid->getState(i).swft.y;
    state[i].swf.r_  = cond.grid->getState(i).swft.r;
    state[i].swf.th_ = cond.grid->getState(i).swft.th;
  }

  for (int state_id = 0; state_id < num_state; state_id++) {
    for (int input_id = 0; input_id < num_input; input_id++) {
      int id = state_id * num_input + input_id;
      next_id[id] = -1;
    }
  }
}

__host__ void initInput(Input *input, Condition cond) {
  for (int i = 0; i < cond.grid->getNumInput(); i++) {
    input[i].swf.x_  = cond.grid->getInput(i).swft.x;
    input[i].swf.y_  = cond.grid->getInput(i).swft.y;
    input[i].swf.r_  = cond.grid->getInput(i).swft.r;
    input[i].swf.th_ = cond.grid->getInput(i).swft.th;
  }
}

__host__ void initNstep(int *zero_step, int *n_step, Condition cond) {
  for (int i = 0; i < cond.grid->getNumState(); i++) {
    zero_step[i] = -1;
  }

  for (int i = 0; i < cond.grid->getNumState() * cond.grid->getNumInput(); i++) {
    n_step[i] = -1;
  }
}

__host__ void initGrid(Grid *cgrid, Condition cond) {
  cgrid->num_state  = cond.grid->getNumState();
  cgrid->num_input  = cond.grid->getNumInput();
  cgrid->num_n_step = cond.grid->getNumState() * cond.grid->getNumInput();

  cgrid->icp_r_min  = cond.param->getVal("icp_r", "min");
  cgrid->icp_r_max  = cond.param->getVal("icp_r", "max");
  cgrid->icp_r_step = cond.param->getVal("icp_r", "step");
  cgrid->icp_r_num  = cond.param->getVal("icp_r", "num");

  cgrid->icp_th_min  = cond.param->getVal("icp_th", "min");
  cgrid->icp_th_max  = cond.param->getVal("icp_th", "max");
  cgrid->icp_th_step = cond.param->getVal("icp_th", "step");
  cgrid->icp_th_num  = cond.param->getVal("icp_th", "num");

  cgrid->swf_r_min  = cond.param->getVal("swft_r", "min");
  cgrid->swf_r_max  = cond.param->getVal("swft_r", "max");
  cgrid->swf_r_step = cond.param->getVal("swft_r", "step");
  cgrid->swf_r_num  = cond.param->getVal("swft_r", "num");

  cgrid->swf_th_min  = cond.param->getVal("swft_th", "min");
  cgrid->swf_th_max  = cond.param->getVal("swft_th", "max");
  cgrid->swf_th_step = cond.param->getVal("swft_th", "step");
  cgrid->swf_th_num  = cond.param->getVal("swft_th", "num");
}

__host__ void initCop(Vector2 *cop, Condition cond) {
  Capt::State                state;
  Capt::Polygon              polygon;
  std::vector<Capt::Vector2> region = cond.model->getVec("foot", "foot_r_convex");
  Capt::Vector2              cop_;

  for (int state_id = 0; state_id < cond.grid->getNumState(); state_id++) {
    state = cond.grid->getState(state_id);
    cop_  = polygon.getClosestPoint(state.icp, region);

    cop[state_id].x_  = cop_.x;
    cop[state_id].y_  = cop_.y;
    cop[state_id].r_  = cop_.r;
    cop[state_id].th_ = cop_.th;
  }
}

__host__ void initPhysics(Physics *physics, Condition cond) {
  physics->g     = cond.model->getVal("environment", "gravity");
  physics->h     = cond.model->getVal("physics", "com_height");
  physics->v     = cond.model->getVal("physics", "foot_vel_max");
  physics->dt    = cond.model->getVal("physics", "step_time_min");
  physics->omega = sqrt(physics->g / physics->h);
}

__host__ void outputZeroStep(std::string file_name, bool header, Condition cond,
                             int *zero_step) {
  FILE     *fp        = fopen(file_name.c_str(), "w");
  const int num_state = cond.grid->getNumState();

  // Header
  if (header) {
    fprintf(fp, "%s,", "state_id");
    fprintf(fp, "%s", "0_step");
    fprintf(fp, "\n");
  }

  // Data
  for (int state_id = 0; state_id < num_state; state_id++) {
    fprintf(fp, "%d,", state_id);
    fprintf(fp, "%d", zero_step[state_id]);
    fprintf(fp, "\n");
  }

  fclose(fp);
}

__host__ void outputNStep(std::string file_name, bool header, Condition cond,
                          int *n_step, int *next_id) {
  FILE     *fp        = fopen(file_name.c_str(), "w");
  const int num_state = cond.grid->getNumState();
  const int num_input = cond.grid->getNumInput();

  // Header
  if (header) {
    fprintf(fp, "%s,", "state_id");
    fprintf(fp, "%s,", "input_id");
    fprintf(fp, "%s,", "next_id");
    fprintf(fp, "%s", "n_step");
    fprintf(fp, "\n");
  }

  // Data
  for (int state_id = 0; state_id < num_state; state_id++) {
    for (int input_id = 0; input_id < num_input; input_id++) {
      int id = state_id * num_input + input_id;
      fprintf(fp, "%d,", state_id);
      fprintf(fp, "%d,", input_id);
      fprintf(fp, "%d,", next_id[id]);
      fprintf(fp, "%d", n_step[id]);
      fprintf(fp, "\n");
    }
  }

  fclose(fp);
}

__host__ void exeZeroStep(Capt::Grid grid, Capt::Model model, int *zero_step) {
  for (int state_id = 0; state_id < grid.getNumState(); state_id++) {
    Capt::State state = grid.getState(state_id);

    Capt::Polygon polygon;
    polygon.setVertex(model.getVec("foot", "foot_r_convex"));
    polygon.setVertex(model.getVec("foot", "foot_l_convex", state.swft));

    bool flag = false;
    flag = polygon.inPolygon(state.icp, polygon.getConvexHull());

    if (flag) {
      zero_step[state_id] = 0;
    }
  }
}

/* device function */

__device__ State step(State state, Input input) {
  State state_;
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

__device__ bool existState(State state, Grid grid) {
  bool flag_icp_r  = false, flag_icp_th = false;
  bool flag_swft_r = false, flag_swft_th = false;

  // icp_r
  if (state.icp.r_ >= grid.icp_r_min - grid.icp_r_step / 2.0 &&
      state.icp.r_ < grid.icp_r_max + grid.icp_r_step / 2.0) {
    flag_icp_r = true;
  }
  // icp_th
  if (state.icp.th_ >= grid.icp_th_min - grid.icp_th_step / 2.0 &&
      state.icp.th_ < grid.icp_th_max + grid.icp_th_step / 2.0) {
    flag_icp_th = true;
  }
  // swft_r
  if (state.swf.r_ >= grid.swf_r_min - grid.swf_r_step / 2.0 &&
      state.swf.r_ < grid.swf_r_max + grid.swf_r_step / 2.0) {
    flag_swft_r = true;
  }
  // swft_th
  if (state.swf.th_ >= grid.swf_th_min - grid.swf_th_step / 2.0 &&
      state.swf.th_ < grid.swf_th_max + grid.swf_th_step / 2.0) {
    flag_swft_th = true;
  }

  bool flag = flag_icp_r * flag_icp_th * flag_swft_r * flag_swft_th;
  return flag;
}

__device__ int getStateIndex(State state, Grid grid) {
  int icp_r_id = 0, icp_th_id = 0;
  int swf_r_id = 0, swf_th_id = 0;

  icp_r_id  = roundValue((state.icp.r() - grid.icp_r_min) / grid.icp_r_step);
  icp_th_id = roundValue((state.icp.th() - grid.icp_th_min) / grid.icp_th_step);
  swf_r_id  = roundValue((state.swf.r() - grid.swf_r_min) / grid.swf_r_step);
  swf_th_id = roundValue((state.swf.th() - grid.swf_th_min) / grid.swf_th_step);

  int state_id = 0;
  state_id = grid.swf_th_num * grid.swf_r_num * grid.icp_th_num * icp_r_id +
             grid.swf_th_num * grid.swf_r_num * icp_th_id +
             grid.swf_th_num * swf_r_id + swf_th_id;

  return state_id;
}

/* global function */

__global__ void exeNStep(State *state, Input *input, int *n_step,
                         int *next_id, Grid *grid, Vector2 *cop,
                         Physics *physics) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < grid->num_state * grid->num_input) {
    int state_id = tid / grid->num_input;
    int input_id = tid % grid->num_input;

    bool flag = false;
    // flag = polygon.inPolygon(state[state_id].icp, foot_convex);

    if (flag)
      n_step[tid] = 100;

    tid += blockDim.x * gridDim.x;
  }
}

} // namespace Cuda