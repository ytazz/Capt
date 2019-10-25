#include "cuda_memory_manager.cuh"

namespace Cuda {

/* struct */

void GridCartesian::operator=(const GridCartesian &grid){
  this->num_state  = grid.num_state;
  this->num_input  = grid.num_input;
  this->num_grid   = grid.num_grid;
  this->num_foot_r = grid.num_foot_r;
  this->num_foot_l = grid.num_foot_l;

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
  this->num_state  = grid.num_state;
  this->num_input  = grid.num_input;
  this->num_grid   = grid.num_grid;
  this->num_foot_r = grid.num_foot_r;
  this->num_foot_l = grid.num_foot_l;

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
  this->grid.num_state  = grid->num_state;
  this->grid.num_input  = grid->num_input;
  this->grid.num_grid   = grid->num_grid;
  this->grid.num_foot_r = grid->num_foot_r;
  this->grid.num_foot_l = grid->num_foot_l;
}

void MemoryManager::setGrid(GridPolar* grid){
  this->grid.num_state  = grid->num_state;
  this->grid.num_input  = grid->num_input;
  this->grid.num_grid   = grid->num_grid;
  this->grid.num_foot_r = grid->num_foot_r;
  this->grid.num_foot_l = grid->num_foot_l;
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

  grid->num_state  = cond.grid->getNumState();
  grid->num_input  = cond.grid->getNumInput();
  grid->num_grid   = cond.grid->getNumState() * cond.grid->getNumInput();
  grid->num_foot_r = (int)cond.model->getVec("foot", "foot_r_convex").size();
  grid->num_foot_l = (int)cond.model->getVec("foot", "foot_l_convex").size();

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

  grid->num_state  = cond.grid->getNumState();
  grid->num_input  = cond.grid->getNumInput();
  grid->num_grid   = cond.grid->getNumState() * cond.grid->getNumInput();
  grid->num_foot_r = (int)cond.model->getVec("foot", "foot_r_convex").size();
  grid->num_foot_l = (int)cond.model->getVec("foot", "foot_l_convex").size();

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

__host__ void MemoryManager::initHostFootR(Vector2 *foot_r, Condition cond){
  Capt::arr2_t cfoot_r    = cond.model->getVec("foot", "foot_r_convex");
  const int    num_foot_r = cfoot_r.size();
  for (int vertex_id = 0; vertex_id < num_foot_r; vertex_id++) {
    foot_r[vertex_id].x_  = cfoot_r[vertex_id].x;
    foot_r[vertex_id].y_  = cfoot_r[vertex_id].y;
    foot_r[vertex_id].r_  = cfoot_r[vertex_id].r;
    foot_r[vertex_id].th_ = cfoot_r[vertex_id].th;
  }
}

__host__ void MemoryManager::initHostFootL(Vector2 *foot_l, Condition cond){
  Capt::arr2_t cfoot_l    = cond.model->getVec("foot", "foot_l_convex");
  const int    num_foot_l = cfoot_l.size();
  for (int vertex_id = 0; vertex_id < num_foot_l; vertex_id++) {
    foot_l[vertex_id].x_  = cfoot_l[vertex_id].x;
    foot_l[vertex_id].y_  = cfoot_l[vertex_id].y;
    foot_l[vertex_id].r_  = cfoot_l[vertex_id].r;
    foot_l[vertex_id].th_ = cfoot_l[vertex_id].th;
  }
}

__host__ void MemoryManager::initHostConvex(Vector2 *convex, Condition cond){
  Capt::arr2_t cfoot_r = cond.model->getVec("foot", "foot_r_convex");
  Capt::arr2_t cfoot_l = cond.model->getVec("foot", "foot_l_convex");
  Capt::arr2_t region;

  const int num_foot_r = cfoot_r.size();
  const int num_foot_l = cfoot_l.size();
  const int num_swf    = cond.grid->getNumInput();
  const int num_vertex = num_foot_r + num_foot_l;

  for (int swf_id = 0; swf_id < num_swf; swf_id++) {
    // equal to following index:
    //  icp_x_id (icp_r_id)  = 0
    //  icp_y_id (icp_th_id) = 0
    //  swf_x_id (swf_r_id)  = * (< num_swf_x)
    //  swf_y_id (swf_th_id) = * (< num_swf_y)
    cfoot_l = cond.model->getVec("foot", "foot_l_convex", cond.grid->getState(swf_id).swf);

    Capt::Polygon polygon;
    polygon.setVertex(cfoot_r);
    polygon.setVertex(cfoot_l);
    region = polygon.getConvexHull();

    for (int vertex_id = 0; vertex_id < num_vertex; vertex_id++) {
      int id = swf_id * num_vertex + vertex_id;
      // printf("swf_id: %d, vertex_id: %d, id: %d\n", swf_id, vertex_id, id);
      if(vertex_id < (int)region.size() ) {
        convex[id].x_  = region[vertex_id].x;
        convex[id].y_  = region[vertex_id].y;
        convex[id].r_  = region[vertex_id].r;
        convex[id].th_ = region[vertex_id].th;
      }else{
        convex[id].x_  = -10;
        convex[id].y_  = -10;
        convex[id].r_  = -10;
        convex[id].th_ = -10;
      }
    }
  }
}

__host__ void MemoryManager::initHostCop(Vector2 *cop, Condition cond){
  // cop = new Cuda::Vector2[grid.num_state];

  for (int state_id = 0; state_id < grid.num_state; state_id++) {
    cop[state_id].x_  = 0.0;
    cop[state_id].y_  = 0.0;
    cop[state_id].r_  = 0.0;
    cop[state_id].th_ = 0.0;
  }
}

__host__ void MemoryManager::initHostStepTime(double *step_time, Condition cond){
  for (int id = 0; id < grid.num_grid; id++) {
    step_time[id] = 0.0;
  }
}

__host__ void MemoryManager::initHostPhysics(Physics *physics, Condition cond){
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

__host__ void MemoryManager::initDevFootR(Vector2 *dev_foot_r){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_foot_r, grid.num_foot_r * sizeof( Cuda::Vector2 ) ) );
}

__host__ void MemoryManager::initDevFootL(Vector2 *dev_foot_l){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_foot_l, grid.num_foot_l * sizeof( Cuda::Vector2 ) ) );
}

__host__ void MemoryManager::initDevConvex(Vector2 *dev_convex){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_convex, grid.num_input * ( grid.num_foot_r + grid.num_foot_l ) * sizeof( Cuda::Vector2 ) ) );
}

__host__ void MemoryManager::initDevCop(Vector2 *dev_cop){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_cop, grid.num_state * sizeof( Cuda::Vector2 ) ) );
}

__host__ void MemoryManager::initDevStepTime(double *dev_step_time){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_step_time, grid.num_grid * sizeof( double ) ) );
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

__host__ void MemoryManager::copyHostToDevFootR(Vector2 *foot_r, Vector2 *dev_foot_r){
  HANDLE_ERROR(cudaMemcpy(dev_foot_r, foot_r, grid.num_foot_r * sizeof( Cuda::Vector2 ), cudaMemcpyHostToDevice ) );
}

__host__ void MemoryManager::copyHostToDevFootL(Vector2 *foot_l, Vector2 *dev_foot_l){
  HANDLE_ERROR(cudaMemcpy(dev_foot_l, foot_l, grid.num_foot_l * sizeof( Cuda::Vector2 ), cudaMemcpyHostToDevice ) );
}

__host__ void MemoryManager::copyHostToDevConvex(Vector2 *convex, Vector2 *dev_convex){
  HANDLE_ERROR(cudaMemcpy(dev_convex, convex, grid.num_input * ( grid.num_foot_r + grid.num_foot_l ) * sizeof( Cuda::Vector2 ), cudaMemcpyHostToDevice ) );
}

__host__ void MemoryManager::copyHostToDevCop(Vector2 *cop, Vector2 *dev_cop){
  HANDLE_ERROR(cudaMemcpy(dev_cop, cop, grid.num_state * sizeof( Cuda::Vector2 ), cudaMemcpyHostToDevice ) );
}

__host__ void MemoryManager::copyHostToDevStepTime(double *step_time, double *dev_step_time){
  HANDLE_ERROR(cudaMemcpy(dev_step_time, step_time, grid.num_grid * sizeof( double ), cudaMemcpyHostToDevice) );
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

__host__ void MemoryManager::copyDevToHostFootR(Vector2 *dev_foot_r, Vector2 *foot_r){
  HANDLE_ERROR(cudaMemcpy(foot_r, dev_foot_r, grid.num_foot_r * sizeof( Cuda::Vector2 ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostFootL(Vector2 *dev_foot_l, Vector2 *foot_l){
  HANDLE_ERROR(cudaMemcpy(foot_l, dev_foot_l, grid.num_foot_l * sizeof( Cuda::Vector2 ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostConvex(Vector2 *dev_convex, Vector2 *convex){
  HANDLE_ERROR(cudaMemcpy(convex, dev_convex, grid.num_input * ( grid.num_foot_r + grid.num_foot_l ) * sizeof( Cuda::Vector2 ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostCop(Vector2 *dev_cop, Vector2 *cop){
  HANDLE_ERROR(cudaMemcpy(cop, dev_cop, grid.num_state * sizeof( Cuda::Vector2 ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostStepTime(double *dev_step_time, double *step_time){
  HANDLE_ERROR(cudaMemcpy(step_time, dev_step_time, grid.num_grid * sizeof( double ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostPhysics(Physics *dev_physics, Physics *physics){
  HANDLE_ERROR(cudaMemcpy(physics, dev_physics, sizeof( Physics ), cudaMemcpyDeviceToHost) );
}

} // namespace Cuda