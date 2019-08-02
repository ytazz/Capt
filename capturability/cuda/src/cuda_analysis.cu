#include "cuda_analysis.cuh"

/* struct */

__device__ void CudaState::operator=(const CudaState &state) {
  this->icp = state.icp;
  this->swf = state.swf;
}

__device__ void CudaInput::operator=(const CudaInput &input) {
  this->swf = input.swf;
}

/* host function */

void setNstep(CA::Grid grid, int *cnstep) {
  for (int i = 0; i < grid.getNumState() * grid.getNumInput(); i++) {
    cnstep[i] = -1;
  }
}

void setState(CA::Grid grid, CudaState *cstate) {
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

void setInput(CA::Grid grid, CudaInput *cinput) {
  for (int i = 0; i < grid.getNumInput(); i++) {
    cinput[i].swf.x_ = grid.getInput(i).swft.x;
    cinput[i].swf.y_ = grid.getInput(i).swft.y;
    cinput[i].swf.r_ = grid.getInput(i).swft.r;
    cinput[i].swf.th_ = grid.getInput(i).swft.th;
  }
}

void setGrid(CA::Grid grid, CA::Model model, CA::Param param, CudaGrid *cgrid) {
  cgrid->num_state = grid.getNumState();
  cgrid->num_input = grid.getNumInput();
  cgrid->num_nstep = grid.getNumState() * grid.getNumInput();
  cgrid->num_foot = model.getVec("foot", "foot_r").size();

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

void setFoot(CudaVector2 *cfoot, CudaVector2 *cfoot_r, CudaVector2 *cfoot_l,
             const int num_foot) {
  for (int i = 0; i < num_foot; i++) {
    cfoot[i] = cfoot_r[i];
  }
  for (int i = 0; i < num_foot; i++) {
    cfoot[i + num_foot] = cfoot_r[i];
  }
}

void init(int *next_state_id, int size) {
  for (int i = 0; i < size; i++) {
    next_state_id[i] = -1;
  }
}

__host__ void exeZeroStep(CA::Grid grid, CA::Model model, int *nstep,
                          int *next_state_id) {
  for (int state_id = 0; state_id < grid.getNumState(); state_id++) {
    bool flag = false;

    CA::Polygon polygon;
    flag = polygon.inPolygon(grid.getState(state_id).icp,
                             model.getVec("foot", "foot_r_convex"));

    if (flag) {
      for (int input_id = 0; input_id < grid.getNumInput(); input_id++) {
        int id = state_id * grid.getNumInput() + input_id;
        nstep[id] = 0;
        next_state_id[id] = 0;
      }
    }
  }
}

/* device function */

__device__ int roundValue(double value) {
  int result = (int)value;

  double decimal = value - (int)value;
  if (decimal >= 0.5) {
    result += 1;
  }

  return result;
}

__device__ int getStateIndex(CudaState state, CudaGrid grid) {
  int icp_r_id = 0, icp_th_id = 0;
  int swf_r_id = 0, swf_th_id = 0;

  icp_r_id = roundValue((state.icp.r() - grid.icp_r_min) / grid.icp_r_step);
  icp_th_id = roundValue((state.icp.th() - grid.icp_th_min) / grid.icp_th_step);
  swf_r_id = roundValue((state.swf.r() - grid.swf_r_min) / grid.swf_r_step);
  swf_th_id = roundValue((state.swf.th() - grid.swf_th_min) / grid.swf_th_step);

  int state_id = 0;
  if (icp_r_id < 0 || icp_th_id < 0 || swf_r_id < 0 || swf_th_id < 0) {
    state_id = -1;
  } else {
    state_id = grid.swf_th_num * grid.swf_r_num * grid.icp_th_num * icp_r_id +
               grid.swf_th_num * grid.swf_r_num * icp_th_id +
               grid.swf_th_num * swf_r_id + swf_th_id;
  }

  return state_id;
}

/* global function */