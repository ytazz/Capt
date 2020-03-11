#include "cuda_memory_manager.cuh"

namespace Cuda {

/* struct */

void Grid::operator=(const Grid &grid){
  this->state_num = grid.state_num;
  this->input_num = grid.input_num;
  this->grid_num  = grid.grid_num;

  this->foot_r_num = grid.foot_r_num;
  this->foot_l_num = grid.foot_l_num;

  this->icp_x_min = grid.icp_x_min;
  this->icp_x_max = grid.icp_x_max;
  this->icp_x_stp = grid.icp_x_stp;
  this->icp_x_num = grid.icp_x_num;

  this->icp_y_min = grid.icp_y_min;
  this->icp_y_max = grid.icp_y_max;
  this->icp_y_stp = grid.icp_y_stp;
  this->icp_y_num = grid.icp_y_num;

  this->swf_x_min = grid.swf_x_min;
  this->swf_x_max = grid.swf_x_max;
  this->swf_x_stp = grid.swf_x_stp;
  this->swf_x_num = grid.swf_x_num;

  this->swf_y_min = grid.swf_y_min;
  this->swf_y_max = grid.swf_y_max;
  this->swf_y_stp = grid.swf_y_stp;
  this->swf_y_num = grid.swf_y_num;

  this->swf_z_min = grid.swf_z_min;
  this->swf_z_max = grid.swf_z_max;
  this->swf_z_stp = grid.swf_z_stp;
  this->swf_z_num = grid.swf_z_num;

  this->cop_x_min = grid.cop_x_min;
  this->cop_x_max = grid.cop_x_max;
  this->cop_x_stp = grid.cop_x_stp;
  this->cop_x_num = grid.cop_x_num;

  this->cop_y_min = grid.cop_y_min;
  this->cop_y_max = grid.cop_y_max;
  this->cop_y_stp = grid.cop_y_stp;
  this->cop_y_num = grid.cop_y_num;
}

__device__ void State::operator=(const State &state) {
  this->icp = state.icp;
  this->swf = state.swf;
}

__device__ void Input::operator=(const Input &input) {
  this->cop = input.cop;
  this->swf = input.swf;
}

/* host function */

__host__ void MemoryManager::set(Capt::Model* model, Capt::Param* param, Capt::Grid* grid){
  this->model = model;
  this->param = param;
  this->cgrid = grid;

  this->grid.state_num = grid->getNumState();
  this->grid.input_num = grid->getNumInput();
  this->grid.grid_num  = grid->getNumGrid();

  model->read(&this->grid.foot_r_num, "foot_r_convex_num");
  model->read(&this->grid.foot_l_num, "foot_l_convex_num");

  param->read(&this->grid.icp_x_min, "icp_x_min");
  param->read(&this->grid.icp_x_max, "icp_x_max");
  param->read(&this->grid.icp_x_stp, "icp_x_stp");
  param->read(&this->grid.icp_x_num, "icp_x_num");

  param->read(&this->grid.icp_y_min, "icp_y_min");
  param->read(&this->grid.icp_y_max, "icp_y_max");
  param->read(&this->grid.icp_y_stp, "icp_y_stp");
  param->read(&this->grid.icp_y_num, "icp_y_num");

  param->read(&this->grid.swf_x_min, "swf_x_min");
  param->read(&this->grid.swf_x_max, "swf_x_max");
  param->read(&this->grid.swf_x_stp, "swf_x_stp");
  param->read(&this->grid.swf_x_num, "swf_x_num");

  param->read(&this->grid.swf_y_min, "swf_y_min");
  param->read(&this->grid.swf_y_max, "swf_y_max");
  param->read(&this->grid.swf_y_stp, "swf_y_stp");
  param->read(&this->grid.swf_y_num, "swf_y_num");

  param->read(&this->grid.swf_z_min, "swf_z_min");
  param->read(&this->grid.swf_z_max, "swf_z_max");
  param->read(&this->grid.swf_z_stp, "swf_z_stp");
  param->read(&this->grid.swf_z_num, "swf_z_num");

  param->read(&this->grid.cop_x_min, "cop_x_min");
  param->read(&this->grid.cop_x_max, "cop_x_max");
  param->read(&this->grid.cop_x_stp, "cop_x_stp");
  param->read(&this->grid.cop_x_num, "cop_x_num");

  param->read(&this->grid.cop_y_min, "cop_y_min");
  param->read(&this->grid.cop_y_max, "cop_y_max");
  param->read(&this->grid.cop_y_stp, "cop_y_stp");
  param->read(&this->grid.cop_y_num, "cop_y_num");
}

__host__ void MemoryManager::initHostState(State *state) {
  for (int state_id = 0; state_id < grid.state_num; state_id++) {
    state[state_id].icp.x = cgrid->getState(state_id).icp.x();
    state[state_id].icp.y = cgrid->getState(state_id).icp.y();
    state[state_id].swf.x = cgrid->getState(state_id).swf.x();
    state[state_id].swf.y = cgrid->getState(state_id).swf.y();
    state[state_id].swf.z = cgrid->getState(state_id).swf.z();
  }
}

__host__ void MemoryManager::initHostInput(Input *input) {
  for (int input_id = 0; input_id < grid.input_num; input_id++) {
    input[input_id].cop.x = cgrid->getInput(input_id).cop.x();
    input[input_id].cop.y = cgrid->getInput(input_id).cop.y();
    input[input_id].swf.x = cgrid->getInput(input_id).swf.x();
    input[input_id].swf.y = cgrid->getInput(input_id).swf.y();
  }
}

__host__ void MemoryManager::initHostTrans(int *trans) {
  for (int id = 0; id < grid.grid_num; id++) {
    trans[id] = -1;
  }
}

__host__ void MemoryManager::initHostBasin(int *basin) {
  for (int state_id = 0; state_id < grid.state_num; state_id++) {
    basin[state_id] = -1;
  }
}

__host__ void MemoryManager::initHostNstep(int *nstep) {
  for (int id = 0; id < grid.grid_num; id++) {
    nstep[id] = -1;
  }
}

__host__ void MemoryManager::initHostGrid(Grid *grid) {
  *grid = this->grid;
}

__host__ void MemoryManager::initHostFootR(vec2_t *foot_r){
  Capt::arr2_t cfoot_r;
  int          foot_r_num;
  model->read(&cfoot_r, "foot_r_convex");
  model->read(&foot_r_num, "foot_r_convex_num");
  for (int vertex_id = 0; vertex_id < foot_r_num; vertex_id++) {
    foot_r[vertex_id].x = cfoot_r[vertex_id].x();
    foot_r[vertex_id].y = cfoot_r[vertex_id].y();
  }
}

__host__ void MemoryManager::initHostFootL(vec2_t *foot_l){
  Capt::arr2_t cfoot_l;
  int          foot_l_num;
  model->read(&cfoot_l, "foot_l_convex");
  model->read(&foot_l_num, "foot_l_convex_num");
  for (int vertex_id = 0; vertex_id < foot_l_num; vertex_id++) {
    foot_l[vertex_id].x = cfoot_l[vertex_id].x();
    foot_l[vertex_id].y = cfoot_l[vertex_id].y();
  }
}

__host__ void MemoryManager::initHostConvex(vec2_t *convex){
  Capt::arr2_t cfoot_r;
  Capt::arr2_t cfoot_l;
  Capt::arr2_t region;
  model->read(&cfoot_r, "foot_r_convex");
  model->read(&cfoot_l, "foot_l_convex");

  const int foot_r_num = cfoot_r.size();
  const int foot_l_num = cfoot_l.size();
  const int num_swf    = cgrid->getNumInput();
  const int num_vertex = foot_r_num + foot_l_num;

  for (int swf_id = 0; swf_id < num_swf; swf_id++) {
    // equal to following index:
    //  icp_x_id (icp_r_id)  = 0
    //  icp_y_id (icp_th_id) = 0
    //  swf_x_id (swf_r_id)  = * (< swf_x_num)
    //  swf_y_id (swf_th_id) = * (< swf_y_num)
    Capt::vec2_t foot_l_pos;
    foot_l_pos.x() = cgrid->getState(swf_id).swf.x();
    foot_l_pos.y() = cgrid->getState(swf_id).swf.y();
    model->read(&cfoot_l, "foot_l_convex", foot_l_pos);

    Capt::Polygon polygon;
    polygon.setVertex(cfoot_r);
    polygon.setVertex(cfoot_l);
    region = polygon.getConvexHull();

    for (int vertex_id = 0; vertex_id < num_vertex; vertex_id++) {
      int id = swf_id * num_vertex + vertex_id;
      if(vertex_id < (int)region.size() ) {
        convex[id].x = region[vertex_id].x();
        convex[id].y = region[vertex_id].y();
      }else{
        convex[id].x = -10;
        convex[id].y = -10;
      }
    }
  }
}

__host__ void MemoryManager::initHostStepTime(double *step_time){
  for (int id = 0; id < grid.grid_num; id++) {
    step_time[id] = 0.0;
  }
}

__host__ void MemoryManager::initHostPhysics(Physics *physics){
  model->read(&physics->g,     "gravity");
  model->read(&physics->h,     "com_height");
  model->read(&physics->omega, "omega");
  model->read(&physics->v_max, "foot_vel_max");
  model->read(&physics->z_max, "swing_height_max");
}

__host__ void MemoryManager::initDevState(State *dev_state){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_state, grid.state_num * sizeof( State ) ) );
}

__host__ void MemoryManager::initDevInput(Input *dev_input){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_input, grid.input_num * sizeof( Input ) ) );
}

__host__ void MemoryManager::initDevTrans(int *dev_trans){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_trans, grid.grid_num * sizeof( int ) ) );
}

__host__ void MemoryManager::initDevBasin(int *dev_basin){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_basin, grid.state_num * sizeof( int ) ) );
}

__host__ void MemoryManager::initDevNstep(int *dev_nstep){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_nstep, grid.grid_num * sizeof( int ) ) );
}

__host__ void MemoryManager::initDevGrid(Grid *dev_grid){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_grid, sizeof( Grid ) ) );
}

__host__ void MemoryManager::initDevFootR(vec2_t *dev_foot_r){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_foot_r, grid.foot_r_num * sizeof( vec2_t ) ) );
}

__host__ void MemoryManager::initDevFootL(vec2_t *dev_foot_l){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_foot_l, grid.foot_l_num * sizeof( vec2_t ) ) );
}

__host__ void MemoryManager::initDevConvex(vec2_t *dev_convex){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_convex, grid.input_num * ( grid.foot_r_num + grid.foot_l_num ) * sizeof( vec2_t ) ) );
}

__host__ void MemoryManager::initDevStepTime(double *dev_step_time){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_step_time, grid.grid_num * sizeof( double ) ) );
}

__host__ void MemoryManager::initDevPhysics(Physics *dev_physics){
  HANDLE_ERROR(cudaMalloc( (void **)&dev_physics, sizeof( Physics ) ) );
}

__host__ void MemoryManager::copyHostToDevState(State *state, State *dev_state){
  HANDLE_ERROR(cudaMemcpy(dev_state, state, grid.state_num * sizeof( State ), cudaMemcpyHostToDevice) );
}

__host__ void MemoryManager::copyHostToDevInput(Input *input, Input *dev_input){
  HANDLE_ERROR(cudaMemcpy(dev_input, input, grid.input_num * sizeof( Input ), cudaMemcpyHostToDevice) );
}

__host__ void MemoryManager::copyHostToDevTrans(int *trans, int *dev_trans){
  HANDLE_ERROR(cudaMemcpy(dev_trans, trans, grid.grid_num * sizeof( int ), cudaMemcpyHostToDevice) );
}

__host__ void MemoryManager::copyHostToDevBasin(int *basin, int *dev_basin){
  HANDLE_ERROR(cudaMemcpy(dev_basin, basin, grid.state_num * sizeof( int ), cudaMemcpyHostToDevice) );
}

__host__ void MemoryManager::copyHostToDevNstep(int *nstep, int *dev_nstep){
  HANDLE_ERROR(cudaMemcpy(dev_nstep, nstep, grid.grid_num * sizeof( int ), cudaMemcpyHostToDevice) );
}

__host__ void MemoryManager::copyHostToDevGrid(Grid *grid, Grid *dev_grid){
  HANDLE_ERROR(cudaMemcpy(dev_grid, grid, sizeof( Grid ), cudaMemcpyHostToDevice) );
}

__host__ void MemoryManager::copyHostToDevFootR(vec2_t *foot_r, vec2_t *dev_foot_r){
  HANDLE_ERROR(cudaMemcpy(dev_foot_r, foot_r, grid.foot_r_num * sizeof( vec2_t ), cudaMemcpyHostToDevice ) );
}

__host__ void MemoryManager::copyHostToDevFootL(vec2_t *foot_l, vec2_t *dev_foot_l){
  HANDLE_ERROR(cudaMemcpy(dev_foot_l, foot_l, grid.foot_l_num * sizeof( vec2_t ), cudaMemcpyHostToDevice ) );
}

__host__ void MemoryManager::copyHostToDevConvex(vec2_t *convex, vec2_t *dev_convex){
  HANDLE_ERROR(cudaMemcpy(dev_convex, convex, grid.input_num * ( grid.foot_r_num + grid.foot_l_num ) * sizeof( vec2_t ), cudaMemcpyHostToDevice ) );
}

__host__ void MemoryManager::copyHostToDevStepTime(double *step_time, double *dev_step_time){
  HANDLE_ERROR(cudaMemcpy(dev_step_time, step_time, grid.grid_num * sizeof( double ), cudaMemcpyHostToDevice) );
}

__host__ void MemoryManager::copyHostToDevPhysics(Physics *physics, Physics *dev_physics){
  HANDLE_ERROR(cudaMemcpy(dev_physics, physics, sizeof( Physics ), cudaMemcpyHostToDevice ) );
}

__host__ void MemoryManager::copyDevToHostState(State *dev_state, State *state){
  HANDLE_ERROR(cudaMemcpy(state, dev_state, grid.state_num * sizeof( State ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostInput(Input *dev_input, Input *input){
  HANDLE_ERROR(cudaMemcpy(input, dev_input, grid.input_num * sizeof( Input ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostTrans(int *dev_trans, int *trans){
  HANDLE_ERROR(cudaMemcpy(trans, dev_trans, grid.grid_num * sizeof( int ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostBasin(int *dev_basin, int *basin){
  HANDLE_ERROR(cudaMemcpy(basin, dev_basin, grid.state_num * sizeof( int ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostNstep(int *dev_nstep, int *nstep){
  HANDLE_ERROR(cudaMemcpy(nstep, dev_nstep, grid.grid_num * sizeof( int ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostGrid(Grid *dev_grid, Grid *grid){
  HANDLE_ERROR(cudaMemcpy(grid, dev_grid, sizeof( Grid ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostFootR(vec2_t *dev_foot_r, vec2_t *foot_r){
  HANDLE_ERROR(cudaMemcpy(foot_r, dev_foot_r, grid.foot_r_num * sizeof( vec2_t ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostFootL(vec2_t *dev_foot_l, vec2_t *foot_l){
  HANDLE_ERROR(cudaMemcpy(foot_l, dev_foot_l, grid.foot_l_num * sizeof( vec2_t ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostConvex(vec2_t *dev_convex, vec2_t *convex){
  HANDLE_ERROR(cudaMemcpy(convex, dev_convex, grid.input_num * ( grid.foot_r_num + grid.foot_l_num ) * sizeof( vec2_t ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostStepTime(double *dev_step_time, double *step_time){
  HANDLE_ERROR(cudaMemcpy(step_time, dev_step_time, grid.grid_num * sizeof( double ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::copyDevToHostPhysics(Physics *dev_physics, Physics *physics){
  HANDLE_ERROR(cudaMemcpy(physics, dev_physics, sizeof( Physics ), cudaMemcpyDeviceToHost) );
}

__host__ void MemoryManager::saveBasin(std::string file_name, int *basin, bool header){
  FILE *fp = fopen(file_name.c_str(), "w");

  // Header
  if (header) {
    // fprintf(fp, "%s,", "state_id");
    fprintf(fp, "%s", "nstep");
    fprintf(fp, "\n");
  }

  // Data
  for (int state_id = 0; state_id < grid.state_num; state_id++) {
    // fprintf(fp, "%d,", state_id);
    fprintf(fp, "%d", basin[state_id]);
    fprintf(fp, "\n");
  }

  fclose(fp);
}

__host__ void MemoryManager::saveNstep(std::string file_name, int *nstep, int *trans, int max_step, bool header){
  FILE *fp = fopen(file_name.c_str(), "w");

  // Header
  if (header) {
    // fprintf(fp, "%s,", "state_id");
    // fprintf(fp, "%s,", "input_id");
    fprintf(fp, "%s,", "trans");
    fprintf(fp, "%s", "nstep");
    fprintf(fp, "\n");
  }

  // Data
  int num_step[max_step + 1]; // 最大踏み出し歩数を10とする
  for(int i = 0; i < max_step + 1; i++) {
    num_step[i] = 0;
  }
  for (int state_id = 0; state_id < grid.state_num; state_id++) {
    for (int input_id = 0; input_id < grid.input_num; input_id++) {
      int id = state_id * grid.input_num + input_id;
      // fprintf(fp, "%d,", state_id);
      // fprintf(fp, "%d,", input_id);
      fprintf(fp, "%d,", trans[id]);
      fprintf(fp, "%d", nstep[id]);
      fprintf(fp, "\n");
      if(nstep[id] > 0)
        num_step[nstep[id]]++;
    }
  }

  printf("*** Result ***\n");
  printf("  Feasible maximum steps: %d\n", max_step);
  for(int i = 1; i <= max_step; i++) {
    printf("  %d-step capture point  : %8d\n", i, num_step[i]);
  }

  fclose(fp);
}

__host__ void MemoryManager::saveStepTime(std::string file_name, double *step_time, bool header){
  FILE *fp = fopen(file_name.c_str(), "w");

  // Header
  if (header) {
    // fprintf(fp, "%s,", "state_id");
    // fprintf(fp, "%s,", "input_id");
    fprintf(fp, "%s", "step_time");
    fprintf(fp, "\n");
  }

  // Data
  for (int state_id = 0; state_id < grid.state_num; state_id++) {
    for (int input_id = 0; input_id < grid.input_num; input_id++) {
      int id = state_id * grid.input_num + input_id;
      // fprintf(fp, "%d,", state_id);
      // fprintf(fp, "%d,", input_id);
      fprintf(fp, "%1.4lf", step_time[id]);
      fprintf(fp, "\n");
    }
  }

  fclose(fp);
}

} // namespace Cuda