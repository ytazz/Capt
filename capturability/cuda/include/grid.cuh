#ifndef __GRID_CUH__
#define __GRID_CUH__

#include "input.cuh"
#include "state.cuh"
#include "vector.cuh"
#include <iostream>
#include <string>
#include <vector>

struct StateTable {
  State *state;
  Input *input;

  int num_state;
  int num_input;

  int num_icp_r;
  int num_icp_th;
  int num_swf_r;
  int num_swf_th;

  double icp_r_min, icp_r_max, icp_r_step;
  double icp_th_min, icp_th_max, icp_th_step;
  double swf_r_min, swf_r_max, swf_r_step;
  double swf_th_min, swf_th_max, swf_th_step;
};

class Grid {
public:
  __device__ Grid();
  __device__ ~Grid();

  __device__ int getStateIndex(State state, StateTable table);

public:
  __device__ int round(double value);
};

#endif // __GRID_CUH__