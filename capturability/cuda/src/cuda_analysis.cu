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

void copyGrid(CA::Grid grid, CA::Param param, CudaGrid *cgrid) {
  cgrid->num_state = grid.getNumState();
  cgrid->num_input = grid.getNumInput();
  cgrid->num_nstep = grid.getNumState() * grid.getNumInput();

  cgrid->icp_r_min = param.getVal("icp_r", "min");
  cgrid->icp_r_max = param.getVal("icp_r", "max");
  cgrid->icp_r_step = param.getVal("icp_r", "step");
  cgrid->icp_r_num = param.getVal("icp_r", "num");

  cgrid->icp_th_min = param.getVal("icp_th", "min");
  cgrid->icp_th_max = param.getVal("icp_th", "max");
  cgrid->icp_th_step = param.getVal("icp_th", "step");
  cgrid->icp_th_num = param.getVal("icp_th", "num");

  cgrid->swf_r_min = param.getVal("swft_r", "min");
  cgrid->swf_r_max = param.getVal("swft_r", "max");
  cgrid->swf_r_step = param.getVal("swft_r", "step");
  cgrid->swf_r_num = param.getVal("swft_r", "num");

  cgrid->swf_th_min = param.getVal("swft_th", "min");
  cgrid->swf_th_max = param.getVal("swft_th", "max");
  cgrid->swf_th_step = param.getVal("swft_th", "step");
  cgrid->swf_th_num = param.getVal("swft_th", "num");
}

__global__ void exeZeroStep(CudaState *state, CudaInput *input, int *nstep,
                            CudaGrid *grid) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < grid->num_state * grid->num_input) {
    int state_id = tid / grid->num_input;
    int input_id = tid % grid->num_input;

    nstep[0] = grid->icp_r_num;
    nstep[1] = grid->icp_th_num;
    nstep[2] = grid->swf_r_num;
    nstep[3] = grid->swf_th_num;

    tid += blockDim.x * gridDim.x;
  }
}