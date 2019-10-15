#include "cuda_analysis.cuh"

namespace Cuda {

/* struct */

void GridCartesian::operator=(const GridCartesian &grid){
  this->num_state = grid.num_state;
  this->num_input = grid.num_input;
  this->num_grid  = grid.num_grid;

  this->icp_x_num = grid.icp_x_num;
  this->icp_y_num = grid.icp_y_num;
  this->swf_x_num = grid.swf_x_num;
  this->swf_y_num = grid.swf_y_num;

  this->icp_x_min  = grid.icp_x_min;
  this->icp_x_max  = grid.icp_x_max;
  this->icp_x_step = grid.icp_x_step;
  this->icp_y_min  = grid.icp_y_min;
  this->icp_y_max  = grid.icp_y_max;
  this->icp_y_step = grid.icp_y_step;
  this->swf_x_min  = grid.swf_x_min;
  this->swf_x_max  = grid.swf_x_max;
  this->swf_x_step = grid.swf_x_step;
  this->swf_y_min  = grid.swf_y_min;
  this->swf_y_max  = grid.swf_y_max;
  this->swf_y_step = grid.swf_y_step;
}

void GridPolar::operator=(const GridPolar &grid){
  this->num_state = grid.num_state;
  this->num_input = grid.num_input;
  this->num_grid  = grid.num_grid;

  this->icp_r_num  = grid.icp_r_num;
  this->icp_th_num = grid.icp_th_num;
  this->swf_r_num  = grid.swf_r_num;
  this->swf_th_num = grid.swf_th_num;

  this->icp_r_min   = grid.icp_r_min;
  this->icp_r_max   = grid.icp_r_max;
  this->icp_r_step  = grid.icp_r_step;
  this->icp_th_min  = grid.icp_th_min;
  this->icp_th_max  = grid.icp_th_max;
  this->icp_th_step = grid.icp_th_step;
  this->swf_r_min   = grid.swf_r_min;
  this->swf_r_max   = grid.swf_r_max;
  this->swf_r_step  = grid.swf_r_step;
  this->swf_th_min  = grid.swf_th_min;
  this->swf_th_max  = grid.swf_th_max;
  this->swf_th_step = grid.swf_th_step;
}

__device__ void State::operator=(const State &state) {
  this->icp = state.icp;
  this->swf = state.swf;
}

__device__ void Input::operator=(const Input &input) {
  this->swf = input.swf;
}

/* host function */

void MemoryManager::setGrid(GridCartesian* grid){
  this->grid.num_state = grid->num_state;
  this->grid.num_input = grid->num_input;
  this->grid.num_grid  = grid->num_grid;
}

void MemoryManager::setGrid(GridPolar* grid){
  this->grid.num_state = grid->num_state;
  this->grid.num_input = grid->num_input;
  this->grid.num_grid  = grid->num_grid;
}

__host__ void MemoryManager::initHostState(State *state, Condition cond) {
  const int num_state = cond.grid->getNumState();

  // state = (Cuda::State*)malloc(sizeof( Cuda::State ) * num_state);

  for (int state_id = 0; state_id < num_state; state_id++) {
    state[state_id].icp.x_  = cond.grid->getState(state_id).icp.x;
    state[state_id].icp.y_  = cond.grid->getState(state_id).icp.y;
    state[state_id].icp.r_  = cond.grid->getState(state_id).icp.r;
    state[state_id].icp.th_ = cond.grid->getState(state_id).icp.th;
    state[state_id].swf.x_  = cond.grid->getState(state_id).swf.x;
    state[state_id].swf.y_  = cond.grid->getState(state_id).swf.y;
    state[state_id].swf.r_  = cond.grid->getState(state_id).swf.r;
    state[state_id].swf.th_ = cond.grid->getState(state_id).swf.th;
  }
}

__host__ void MemoryManager::initHostInput(Input *input, Condition cond) {
  const int num_input = cond.grid->getNumInput();

  // input = (Cuda::Input*)malloc(sizeof( Cuda::Input ) * num_input );

  for (int input_id = 0; input_id < num_input; input_id++) {
    input[input_id].swf.x_  = cond.grid->getInput(input_id).swf.x;
    input[input_id].swf.y_  = cond.grid->getInput(input_id).swf.y;
    input[input_id].swf.r_  = cond.grid->getInput(input_id).swf.r;
    input[input_id].swf.th_ = cond.grid->getInput(input_id).swf.th;
  }
}

__host__ void MemoryManager::initHostTrans(int *trans, Condition cond) {
  const int num_state = cond.grid->getNumState();
  const int num_input = cond.grid->getNumInput();

  // trans = (int*)malloc(sizeof( int ) * num_state * num_input );

  for (int grid_id = 0; grid_id < num_state * num_input; grid_id++) {
    trans[grid_id] = -1;
  }
}

__host__ void MemoryManager::initHostBasin(int *basin, Condition cond) {
  const int num_state = cond.grid->getNumState();

  // basin = (int*)malloc(sizeof( int ) * num_state );

  for (int state_id = 0; state_id < num_state; state_id++) {
    basin[state_id] = -1;
  }
}

__host__ void MemoryManager::initHostNstep(int *nstep, Condition cond) {
  const int num_state = cond.grid->getNumState();
  const int num_input = cond.grid->getNumInput();
  const int num_grid  = num_state * num_input;

  // nstep = (int*)malloc(sizeof( int ) * num_grid );

  for (int grid_id = 0; grid_id < num_grid; grid_id++) {
    nstep[grid_id] = -1;
  }
}

__host__ void MemoryManager::initHostGrid(GridCartesian *grid, Condition cond) {
  // grid = new Cuda::GridCartesian;

  grid->num_state = cond.grid->getNumState();
  grid->num_input = cond.grid->getNumInput();
  grid->num_grid  = cond.grid->getNumState() * cond.grid->getNumInput();

  grid->icp_x_min  = cond.param->getVal("icp_x", "min");
  grid->icp_x_max  = cond.param->getVal("icp_x", "max");
  grid->icp_x_step = cond.param->getVal("icp_x", "step");
  grid->icp_x_num  = cond.param->getVal("icp_x", "num");

  grid->icp_y_min  = cond.param->getVal("icp_y", "min");
  grid->icp_y_max  = cond.param->getVal("icp_y", "max");
  grid->icp_y_step = cond.param->getVal("icp_y", "step");
  grid->icp_y_num  = cond.param->getVal("icp_y", "num");

  grid->swf_x_min  = cond.param->getVal("swf_x", "min");
  grid->swf_x_max  = cond.param->getVal("swf_x", "max");
  grid->swf_x_step = cond.param->getVal("swf_x", "step");
  grid->swf_x_num  = cond.param->getVal("swf_x", "num");

  grid->swf_y_min  = cond.param->getVal("swf_y", "min");
  grid->swf_y_max  = cond.param->getVal("swf_y", "max");
  grid->swf_y_step = cond.param->getVal("swf_y", "step");
  grid->swf_y_num  = cond.param->getVal("swf_y", "num");
}

__host__ void MemoryManager::initHostGrid(GridPolar *grid, Condition cond) {
  // grid = new Cuda::GridPolar;

  grid->num_state = cond.grid->getNumState();
  grid->num_input = cond.grid->getNumInput();
  grid->num_grid  = cond.grid->getNumState() * cond.grid->getNumInput();

  grid->icp_r_min  = cond.param->getVal("icp_r", "min");
  grid->icp_r_max  = cond.param->getVal("icp_r", "max");
  grid->icp_r_step = cond.param->getVal("icp_r", "step");
  grid->icp_r_num  = cond.param->getVal("icp_r", "num");

  grid->icp_th_min  = cond.param->getVal("icp_th", "min");
  grid->icp_th_max  = cond.param->getVal("icp_th", "max");
  grid->icp_th_step = cond.param->getVal("icp_th", "step");
  grid->icp_th_num  = cond.param->getVal("icp_th", "num");

  grid->swf_r_min  = cond.param->getVal("swf_r", "min");
  grid->swf_r_max  = cond.param->getVal("swf_r", "max");
  grid->swf_r_step = cond.param->getVal("swf_r", "step");
  grid->swf_r_num  = cond.param->getVal("swf_r", "num");

  grid->swf_th_min  = cond.param->getVal("swf_th", "min");
  grid->swf_th_max  = cond.param->getVal("swf_th", "max");
  grid->swf_th_step = cond.param->getVal("swf_th", "step");
  grid->swf_th_num  = cond.param->getVal("swf_th", "num");
}

__host__ void MemoryManager::initCop(Vector2 *cop, Condition cond){
  // cop = new Cuda::Vector2[grid.num_state];

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

__host__ void MemoryManager::initPhysics(Physics *physics, Condition cond){
  // physics = new Cuda::Physics;

  physics->g     = cond.model->getVal("environment", "gravity");
  physics->h     = cond.model->getVal("physics", "com_height");
  physics->v     = cond.model->getVal("physics", "foot_vel_max");
  physics->dt    = cond.model->getVal("physics", "step_time_min");
  physics->omega = sqrt(physics->g / physics->h);
}

__host__ void MemoryManager::initDevState(Cuda::State *dev_state){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_state, grid.num_state * sizeof( Cuda::State ) ) );
}

__host__ void MemoryManager::initDevInput(Cuda::Input *dev_input){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_input, grid.num_input * sizeof( Cuda::Input ) ) );
}

__host__ void MemoryManager::initDevTrans(int *dev_trans){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_trans, grid.num_grid * sizeof( int ) ) );
}

__host__ void MemoryManager::initDevBasin(int *dev_basin){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_basin, grid.num_state * sizeof( int ) ) );
}

__host__ void MemoryManager::initDevNstep(int *dev_nstep){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_nstep, grid.num_grid * sizeof( int ) ) );
}

__host__ void MemoryManager::initDevGrid(Cuda::GridCartesian *dev_grid){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_grid, sizeof( Cuda::GridCartesian ) ) );
}

__host__ void MemoryManager::initDevGrid(Cuda::GridPolar *dev_grid){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_grid, sizeof( Cuda::GridPolar ) ) );
}

__host__ void MemoryManager::initDevCop(Vector2 *dev_cop){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_cop, grid.num_state * sizeof( Cuda::Vector2 ) ) );
}

__host__ void MemoryManager::initDevPhysics(Physics *dev_physics){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_physics, sizeof( Cuda::Physics ) ) );
}

__host__ void MemoryManager::copyHostToDevState(State *state, Cuda::State *dev_state){
  HANDLE_ERROR(cudaMemcpy(dev_state, state, grid.num_state * sizeof( Cuda::State ), cudaMemcpyHostToDevice) );
}

__host__ void MemoryManager::copyHostToDevInput(Input *input, Cuda::Input *dev_input){
  HANDLE_ERROR(cudaMemcpy(dev_input, input, grid.num_input * sizeof( Cuda::Input ), cudaMemcpyHostToDevice) );
}

__host__ void MemoryManager::copyHostToDevTrans(int *trans, int *dev_trans){
  HANDLE_ERROR(cudaMemcpy(dev_trans, trans, grid.num_grid * sizeof( int ), cudaMemcpyHostToDevice) );
}

__host__ void MemoryManager::copyHostToDevBasin(int *basin, int *dev_basin){
  HANDLE_ERROR(cudaMemcpy(dev_basin, basin, grid.num_state * sizeof( int ), cudaMemcpyHostToDevice) );
}

__host__ void MemoryManager::copyHostToDevNstep(int *nstep, int *dev_nstep){
  HANDLE_ERROR(cudaMemcpy(dev_nstep, nstep, grid.num_grid * sizeof( int ), cudaMemcpyHostToDevice) );
}

__host__ void MemoryManager::copyHostToDevGrid(GridCartesian *grid, Cuda::GridCartesian *dev_grid){
  HANDLE_ERROR(cudaMemcpy(dev_grid, grid, sizeof( Cuda::GridCartesian ), cudaMemcpyHostToDevice) );
}

__host__ void MemoryManager::copyHostToDevGrid(GridPolar *grid, Cuda::GridPolar *dev_grid){
  HANDLE_ERROR(cudaMemcpy(dev_grid, grid, sizeof( Cuda::GridPolar ), cudaMemcpyHostToDevice) );
}

__host__ void MemoryManager::copyHostToDevCop(Vector2 *cop, Vector2 *dev_cop){
  HANDLE_ERROR(cudaMemcpy(dev_cop, cop, grid.num_state * sizeof( Cuda::Vector2 ), cudaMemcpyHostToDevice ) );
}

__host__ void MemoryManager::copyHostToDevPhysics(Physics *physics, Physics *dev_physics){
  HANDLE_ERROR(cudaMemcpy(dev_physics, physics, sizeof( Cuda::Physics ), cudaMemcpyHostToDevice ) );
}

__host__ void MemoryManager::copyDevToHostState(Cuda::State *dev_state, State *state){
  HANDLE_ERROR(cudaMemcpy(state, dev_state, grid.num_state * sizeof( Cuda::State ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostInput(Cuda::Input *dev_input, Input *input){
  HANDLE_ERROR(cudaMemcpy(input, dev_input, grid.num_input * sizeof( Cuda::Input ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostTrans(int *dev_trans, int *trans){
  HANDLE_ERROR(cudaMemcpy(trans, dev_trans, grid.num_grid * sizeof( int ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostBasin(int *dev_basin, int *basin){
  HANDLE_ERROR(cudaMemcpy(basin, dev_basin, grid.num_state * sizeof( int ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostNstep(int *dev_nstep, int *nstep){
  HANDLE_ERROR(cudaMemcpy(nstep, dev_nstep, grid.num_grid * sizeof( int ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostGrid(Cuda::GridCartesian *dev_grid, GridCartesian *grid){
  HANDLE_ERROR(cudaMemcpy(grid, dev_grid, sizeof( Cuda::GridCartesian ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostGrid(Cuda::GridPolar *dev_grid, GridPolar *grid){
  HANDLE_ERROR(cudaMemcpy(grid, dev_grid, sizeof( Cuda::GridPolar ), cudaMemcpyDeviceToHost) );
}

__host__ void outputBasin(std::string file_name, bool header, Condition cond,
                          int *basin) {
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

__host__ void outputNStep(std::string file_name, bool header, Condition cond,
                          int *nstep, int *trans) {
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