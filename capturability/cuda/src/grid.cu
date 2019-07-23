#include "grid.cuh"

__device__ Grid::Grid() {}

__device__ Grid::~Grid() {}

__device__ int Grid::round(double value) {
  int result = (int)value;

  double decimal = value - (int)value;
  if (decimal >= 0.5) {
    result += 1;
  }

  return result;
}

__device__ int Grid::getStateIndex(State state, StateTable table) {
  int icp_r_id = 0, icp_th_id = 0;
  int swf_r_id = 0, swf_th_id = 0;

  icp_r_id = round((state.icp.r() - table.icp_r_min) / table.icp_r_step);
  icp_th_id = round((state.icp.th() - table.icp_th_min) / table.icp_th_step);
  swf_r_id = round((state.swf.r() - table.swf_r_min) / table.swf_r_step);
  swf_th_id = round((state.swf.th() - table.swf_th_min) / table.swf_th_step);

  int state_id = 0;
  if (icp_r_id < 0 || icp_th_id < 0 || swf_r_id < 0 || swf_th_id < 0) {
    state_id = -1;
  } else {
    state_id =
        table.num_swf_th * table.num_swf_r * table.num_icp_th * icp_r_id +
        table.num_swf_th * table.num_swf_r * icp_th_id +
        table.num_swf_th * swf_r_id + swf_th_id;
  }

  return state_id;
}