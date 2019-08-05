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

void initState(CudaState *cstate, int *next_state_id, Condition cond) {
  const int num_state = cond.grid->getNumState();
  const int num_input = cond.grid->getNumInput();

  for (int i = 0; i < cond.grid->getNumState(); i++) {
    cstate[i].icp.x_ = cond.grid->getState(i).icp.x;
    cstate[i].icp.y_ = cond.grid->getState(i).icp.y;
    cstate[i].icp.r_ = cond.grid->getState(i).icp.r;
    cstate[i].icp.th_ = cond.grid->getState(i).icp.th;
    cstate[i].swf.x_ = cond.grid->getState(i).swft.x;
    cstate[i].swf.y_ = cond.grid->getState(i).swft.y;
    cstate[i].swf.r_ = cond.grid->getState(i).swft.r;
    cstate[i].swf.th_ = cond.grid->getState(i).swft.th;
  }

  for (int state_id = 0; state_id < num_state; state_id++) {
    for (int input_id = 0; input_id < num_input; input_id++) {
      int id = state_id * num_input + input_id;
      next_state_id[id] = -1;
    }
  }
}

void initInput(CudaInput *cinput, Condition cond) {
  for (int i = 0; i < cond.grid->getNumInput(); i++) {
    cinput[i].swf.x_ = cond.grid->getInput(i).swft.x;
    cinput[i].swf.y_ = cond.grid->getInput(i).swft.y;
    cinput[i].swf.r_ = cond.grid->getInput(i).swft.r;
    cinput[i].swf.th_ = cond.grid->getInput(i).swft.th;
  }
}

void initNstep(int *cnstep, Condition cond) {
  for (int i = 0; i < cond.grid->getNumState() * cond.grid->getNumInput();
       i++) {
    cnstep[i] = -1;
  }
}

void initGrid(CudaGrid *cgrid, Condition cond) {
  cgrid->num_state = cond.grid->getNumState();
  cgrid->num_input = cond.grid->getNumInput();
  cgrid->num_nstep = cond.grid->getNumState() * cond.grid->getNumInput();
  cgrid->num_foot = cond.model->getVec("foot", "foot_r").size();

  cgrid->icp_r_min = cond.param->getVal("icp_r", "min");
  cgrid->icp_r_max = cond.param->getVal("icp_r", "max");
  cgrid->icp_r_step = cond.param->getVal("icp_r", "step");
  cgrid->icp_r_num = cond.param->getVal("icp_r", "num");

  cgrid->icp_th_min = cond.param->getVal("icp_th", "min");
  cgrid->icp_th_max = cond.param->getVal("icp_th", "max");
  cgrid->icp_th_step = cond.param->getVal("icp_th", "step");
  cgrid->icp_th_num = cond.param->getVal("icp_th", "num");

  cgrid->swf_r_min = cond.param->getVal("swft_r", "min");
  cgrid->swf_r_max = cond.param->getVal("swft_r", "max");
  cgrid->swf_r_step = cond.param->getVal("swft_r", "step");
  cgrid->swf_r_num = cond.param->getVal("swft_r", "num");

  cgrid->swf_th_min = cond.param->getVal("swft_th", "min");
  cgrid->swf_th_max = cond.param->getVal("swft_th", "max");
  cgrid->swf_th_step = cond.param->getVal("swft_th", "step");
  cgrid->swf_th_num = cond.param->getVal("swft_th", "num");
}

void initCop(CudaVector2 *cop, Condition cond) {
  CA::State state;
  CA::Polygon polygon;
  std::vector<CA::Vector2> region = cond.model->getVec("foot", "foot_r_convex");
  CA::Vector2 cop_;

  for (int state_id = 0; state_id < cond.grid->getNumState(); state_id++) {
    state = cond.grid->getState(state_id);
    cop_ = polygon.getClosestPoint(state.icp, region);

    cop[state_id].x_ = cop_.x;
    cop[state_id].y_ = cop_.y;
    cop[state_id].r_ = cop_.r;
    cop[state_id].th_ = cop_.th;
  }
}

void output(std::string file_name, Condition cond, int *cnstep,
            int *next_state_id) {
  FILE *fp = fopen("result.csv", "w");
  const int num_state = cond.grid->getNumState();
  const int num_input = cond.grid->getNumInput();

  fprintf(fp, "%s,", "state_id");
  fprintf(fp, "%s,", "input_id");
  fprintf(fp, "%s,", "next_state_id");
  fprintf(fp, "%s,", "nstep");
  fprintf(fp, "\n");
  for (int state_id = 0; state_id < num_state; state_id++) {
    for (int input_id = 0; input_id < num_input; input_id++) {
      int id = state_id * num_input + input_id;
      fprintf(fp, "%d,", state_id);
      fprintf(fp, "%d,", input_id);
      fprintf(fp, "%d,", next_state_id[id]);
      fprintf(fp, "%d,", cnstep[id]);
      fprintf(fp, "\n");
    }
  }

  fclose(fp);
}

__host__ void exeZeroStep(CA::Grid grid, CA::Model model, int *nstep,
                          int *next_state_id) {
  for (int state_id = 0; state_id < grid.getNumState(); state_id++) {
    bool flag = false;

    CA::State state = grid.getState(state_id);

    CA::Polygon polygon;
    polygon.setVertex(model.getVec("foot", "foot_r_convex"));
    polygon.setVertex(model.getVec("foot", "foot_l_convex", state.swft));
    flag = polygon.inPolygon(state.icp, polygon.getConvexHull());

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