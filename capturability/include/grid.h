#ifndef __GRID_H__
#define __GRID_H__

#include "input.h"
#include "param.h"
#include "state.h"
#include "vector.h"
#include <iostream>
#include <string>
#include <vector>

namespace Capt {

namespace GridSpace {
enum Element { MIN, MAX, STEP, NUMELEMENT };
}

struct GridState {
  int id;
  State state;

  void operator=(const GridState &grid_state) {
    this->state = grid_state.state;
    this->id = grid_state.id;
  }
};

struct GridInput {
  int id;
  Input input;

  void operator=(const GridInput &grid_input) {
    this->input = grid_input.input;
    this->id = grid_input.id;
  }
};

class Grid {
public:
  Grid(Param param);
  ~Grid();

  State getState(int index);
  State getState(int icp_r_id, int icp_th_id, int swft_r_id, int swft_th_id);
  Input getInput(int index);
  Input getInput(int swft_r_id, int swft_th_id);

  int getStateIndex(int icp_r_id, int icp_th_id, int swft_r_id, int swft_th_id);
  int getStateIndex(State state_);

  bool existState(int state_id);
  bool existState(State state_);
  bool existInput(int input_id);

  int getNumState();
  int getNumInput();

  GridState roundState(State state_);

private:
  void create();

  void setStatePolar(float icp_r, float icp_th, float swft_r, float swft_th);
  void setStateCartesian(float icp_x, float icp_y, float swft_x, float swft_y);
  void setInputPolar(float swft_r, float swft_th);
  void setInputCartesian(float swft_x, float swft_y);

  int round(float value);
  int max(int val1, int val2);
  int max(int val1, int val2, int val3, int val4);

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

} // namespace Capt

#endif // __GRID_H__