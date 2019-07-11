#ifndef __GRID_CUH__
#define __GRID_CUH__

#include "input.cuh"
#include "param.cuh"
#include "state.cuh"
#include "vector.cuh"
#include <iostream>
#include <string>
#include <vector>

namespace GPGPU {

namespace GridSpace {
enum Element { MIN, MAX, STEP, NUMELEMENT };
}

struct GridState {
  int id;
  State state;

  __device__ void operator=(const GridState &grid_state) {
    this->state = grid_state.state;
    this->id = grid_state.id;
  }
};

struct GridInput {
  int id;
  Input input;

  __device__ void operator=(const GridInput &grid_input) {
    this->input = grid_input.input;
    this->id = grid_input.id;
  }
};

class Grid {
public:
  __device__ Grid(Param param);
  __device__ ~Grid();

  __device__ State getState(int index);
  __device__ State getState(int icp_r_id, int icp_th_id, int swft_r_id,
                            int swft_th_id);
  __device__ Input getInput(int index);
  __device__ Input getInput(int swft_r_id, int swft_th_id);

  __device__ int getStateIndex(int icp_r_id, int icp_th_id, int swft_r_id,
                               int swft_th_id);
  __device__ int getStateIndex(State state_);

  __device__ bool existState(int state_id);
  __device__ bool existState(State state_);
  __device__ bool existInput(int input_id);

  __device__ int getNumState();
  __device__ int getNumInput();

  __device__ GridState roundState(State state_);

public:
  __device__ void create();

  __device__ void setStatePolar(float icp_r, float icp_th, float swft_r,
                                float swft_th);
  __device__ void setStateCartesian(float icp_x, float icp_y, float swft_x,
                                    float swft_y);
  __device__ void setInputPolar(float swft_r, float swft_th);
  __device__ void setInputCartesian(float swft_x, float swft_y);

  __device__ int round(float value);
  __device__ int max(int val1, int val2);
  __device__ int max(int val1, int val2, int val3, int val4);

  Param param;

  std::vector<State> state;
  std::vector<Input> input;

  int num_state;
  int num_input;

  int num_icp_r, num_icp_th;
  int num_swft_r, num_swft_th;

  int num_icp_x, num_icp_y;
  int num_swft_x, num_swft_y;

  float icp_r[GridSpace::NUMELEMENT];
  float icp_th[GridSpace::NUMELEMENT];
  float swft_r[GridSpace::NUMELEMENT];
  float swft_th[GridSpace::NUMELEMENT];
};

} // namespace GPGPU

#endif // __GRID_CUH__