#include "cuda_analysis.cuh"

void initNstep(CA::Grid grid, int *cnstep) {
  for (int i = 0; i < grid.getNumState() * grid.getNumInput(); i++) {
    cnstep[i] = -1;
  }
}

void copyState(CA::Grid grid, CudaState *cstate) {
  for (int i = 0; i < grid.getNumState(); i++) {
    cstate[i].icp.x_ = grid.getState(i).icp.x;
    cstate[i].icp.y_ = grid.getState(i).icp.y;
    cstate[i].icp.r_ = grid.getState(i).icp.r;
    cstate[i].icp.th_ = grid.getState(i).icp.th;
    cstate[i].swf.x_ = grid.getState(i).swft.x;
    cstate[i].swf.y_ = grid.getState(i).swft.y;
    cstate[i].swf.r_ = grid.getState(i).swft.r;
    cstate[i].swf.th_ = grid.getState(i).swft.th;
  }
}

void copyInput(CA::Grid grid, CudaInput *cinput) {
  for (int i = 0; i < grid.getNumInput(); i++) {
    cinput[i].swf.x_ = grid.getInput(i).swft.x;
    cinput[i].swf.y_ = grid.getInput(i).swft.y;
    cinput[i].swf.r_ = grid.getInput(i).swft.r;
    cinput[i].swf.th_ = grid.getInput(i).swft.th;
  }
}

void copyGrid(CA::Grid grid, CudaGrid *cgrid) {
  cgrid->num_state = grid.getNumState();
  cgrid->num_input = grid.getNumInput();
  cgrid->num_nstep = grid.getNumState() * grid.getNumInput();

  // 後いろいろ代入
}

__global__ void exeZeroStep(CudaState *state, CudaInput *input, int *nstep,
                            CudaGrid *grid) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < grid->num_state * grid->num_input) {
    int state_id = tid / grid->num_input;
    int input_id = tid % grid->num_input;

    if ((input_id % 2) == 0)
      nstep[state_id * grid->num_input + input_id] = 2;
    else
      nstep[state_id * grid->num_input + input_id] = 3;

    tid += blockDim.x * gridDim.x;
  }
}