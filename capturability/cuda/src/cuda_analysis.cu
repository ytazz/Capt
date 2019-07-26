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

__device__ int size(CudaVector2 *array) {
  int size = (sizeof(array) / sizeof(array[0]));
  return size;
}

__device__ void getConvexHull(CudaVector2 *vertex, CudaVector2 *convex) {
  CudaVector2 tmp;
  bool flag_continue = true;
  while (flag_continue) {
    flag_continue = false;
    for (int i = 0; i < 22 - 1; i++) {
      if ((vertex[i + 1]).y() < (vertex[i]).y()) {
        tmp = vertex[i];
        vertex[i] = vertex[i + 1];
        vertex[i + 1] = tmp;
        flag_continue = true;
      }
    }
  }

  bool in_convex[22];
  for (int i = 0; i < 22; i++) {
    in_convex[i] = false;
  }

  int convex_size = 0;
  convex[convex_size] = vertex[0];
  in_convex[0] = true;
  flag_continue = true;
  int back = 0;
  // while (flag_continue) {
  flag_continue = false;
  for (int i = 0; i < 22; i++) {
    int product = 0;
    if (!in_convex[i]) {
      product = 1;
      for (int j = 0; j < 22; j++) {
        if (i != j && !in_convex[i]) {
          if ((vertex[i] - vertex[back]) % (vertex[j] - vertex[i]) < 0.0) {
            product *= 0;
          }
        }
      }
    }
    if (product) {
      if (!in_convex[i]) {
        convex_size++;
        convex[convex_size] = vertex[i];
        in_convex[i] = true;
        flag_continue = true;
        back = i;
      }
      break;
    }
  }
  // }
  convex_size++;
  convex[convex_size] = vertex[0];

  for (int i = convex_size; i < size(vertex); i++) {
    convex[i] = vertex[0];
  }
  // for (int i = 0; i < 22; i++) {
  //   convex[i] = vertex[i];
  // }
}

__device__ CudaVector2 getClosestPoint(CudaVector2 point, CudaVector2 *vertex) {
  CudaVector2 closest;
  CudaVector2 v1, v2, v3, v4; // vector
  CudaVector2 n1, n2;         // normal vector

  if (inPolygon(point, vertex)) {
    closest = point;
  } else {
    for (int i = 0; i < size(vertex) - 1; i++) {
      //最近点が角にあるとき
      if (i == 0) {
        n1 = ((CudaVector2)vertex[1] - (CudaVector2)vertex[i]).normal();
        n2 = ((CudaVector2)vertex[i] - (CudaVector2)vertex[size(vertex) - 2])
                 .normal();
      } else {
        n1 = ((CudaVector2)vertex[i + 1] - (CudaVector2)vertex[i]).normal();
        n2 = ((CudaVector2)vertex[i] - (CudaVector2)vertex[i - 1]).normal();
      }
      v1 = (CudaVector2)point - (CudaVector2)vertex[i];
      if ((n1 % v1) < 0 && (n2 % v1) > 0) {
        closest = vertex[i];
      }
      // 最近点が辺にあるとき
      n1 = ((CudaVector2)vertex[i + 1] - (CudaVector2)vertex[i]).normal();
      v1 = point - (CudaVector2)vertex[i];
      v2 = (CudaVector2)vertex[i + 1] - (CudaVector2)vertex[i];
      v3 = point - (CudaVector2)vertex[i + 1];
      v4 = (CudaVector2)vertex[i] - (CudaVector2)vertex[i + 1];
      if ((n1 % v1) > 0 && (v2 % v1) < 0 && (n1 % v3) < 0 && (v4 % v3) > 0) {
        float k = v1 * v2 / (v2.norm() * v2.norm());
        closest = (CudaVector2)vertex[i] + k * v2;
      }
    }
  }

  return closest;
}

__device__ bool inPolygon(CudaVector2 point, CudaVector2 *vertex) {
  bool flag = false;
  double product = 0.0;
  int sign = 0, on_line = 0;
  const float epsilon = 0.00001;

  for (size_t i = 0; i < size(vertex) - 1; i++) {
    product = (point - (CudaVector2)vertex[i]) %
              ((CudaVector2)vertex[i + 1] - (CudaVector2)vertex[i]);
    if (-epsilon <= product && product <= epsilon) {
      on_line += 1;
    } else if (product > 0) {
      sign += 1;
    } else if (product < 0) {
      sign -= 1;
    }
  }

  if (sign == int(size(vertex) - 1 - on_line) ||
      sign == -int(size(vertex) - 1 - on_line)) {
    flag = true;
  }

  return flag;
}

/* global function */

__global__ void exeZeroStep(CudaState *state, CudaInput *input, int *nstep,
                            CudaVector2 *foot, CudaGrid *grid) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < grid->num_state * grid->num_input) {
    int state_id = tid / grid->num_input;
    int input_id = tid % grid->num_input;

    for (int i = grid->num_foot; i < 2 * grid->num_foot; i++) {
      foot[i].setCartesian(0.0 + foot[i].x(), 0.1 + foot[i].y());
    }

    CudaVector2 foot_convex[22];
    getConvexHull(foot, foot_convex);
    // bool flag = polygon.inPolygon(state[state_id].icp, foot_convex);

    // if (flag)
    //   nstep[state_id * grid->num_input + input_id] = 100;

    // delete foot;
    // delete foot_convex;

    tid += blockDim.x * gridDim.x;
  }
}