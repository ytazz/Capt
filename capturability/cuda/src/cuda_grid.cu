#include "cuda_grid.cuh"

__device__ int CudaGrid::round(double value) {
  int result = (int)value;

  double decimal = value - (int)value;
  if (decimal >= 0.5) {
    result += 1;
  }

  return result;
}

__device__ int CudaGrid::getStateIndex(CudaState state) {
  int icp_r_id = 0, icp_th_id = 0;
  int swf_r_id = 0, swf_th_id = 0;

  icp_r_id = round((state.icp.r() - icp_r_min) / icp_r_step);
  icp_th_id = round((state.icp.th() - icp_th_min) / icp_th_step);
  swf_r_id = round((state.swf.r() - swf_r_min) / swf_r_step);
  swf_th_id = round((state.swf.th() - swf_th_min) / swf_th_step);

  int state_id = 0;
  if (icp_r_id < 0 || icp_th_id < 0 || swf_r_id < 0 || swf_th_id < 0) {
    state_id = -1;
  } else {
    state_id = swf_th_num * swf_r_num * icp_th_num * icp_r_id +
               swf_th_num * swf_r_num * icp_th_id + swf_th_num * swf_r_id +
               swf_th_id;
  }

  return state_id;
}