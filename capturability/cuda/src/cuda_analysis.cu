#include "cuda_analysis.cuh"

void copyGrid(CA::Grid *grid, CudaGrid *cgrid) {
  cgrid->state = new CudaState[grid->getNumState()];
  for (int i = 0; i < grid->getNumState(); i++) {
    cgrid->state[i].icp.x_ = grid->getState(i).icp.x;
    cgrid->state[i].icp.y_ = grid->getState(i).icp.y;
    cgrid->state[i].icp.r_ = grid->getState(i).icp.r;
    cgrid->state[i].icp.th_ = grid->getState(i).icp.th;
    cgrid->state[i].swf.x_ = grid->getState(i).swft.x;
    cgrid->state[i].swf.y_ = grid->getState(i).swft.y;
    cgrid->state[i].swf.r_ = grid->getState(i).swft.r;
    cgrid->state[i].swf.th_ = grid->getState(i).swft.th;
  }

  cgrid->input = new CudaInput[grid->getNumInput()];
  for (int i = 0; i < grid->getNumInput(); i++) {
    cgrid->input[i].swf.x_ = grid->getInput(i).swft.x;
    cgrid->input[i].swf.y_ = grid->getInput(i).swft.y;
    cgrid->input[i].swf.r_ = grid->getInput(i).swft.r;
    cgrid->input[i].swf.th_ = grid->getInput(i).swft.th;
  }

  cgrid->nstep = new int[grid->getNumState() * grid->getNumInput()];
  for (int i = 0; i < grid->getNumState() * grid->getNumInput(); i++) {
    cgrid->nstep[i] = -1;
  }

  cgrid->num_state = grid->getNumState();
  cgrid->num_input = grid->getNumInput();

  // 後いろいろ代入
}

__global__ void exeZeroStep(CudaGrid *grid) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  grid->nstep[0] = 0;
  grid->num_icp_r = 100;

  while (tid < grid->num_state * grid->num_input) {
    int state_id = tid / grid->num_input;
    int input_id = tid % grid->num_input;

    CudaState state = grid->state[state_id];
    CudaInput input = grid->input[input_id];

    grid->nstep[state_id * grid->num_input + input_id] = 0;

    tid += blockDim.x * gridDim.x;
  }
}