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
  int   id;
  State state;

  void operator=(const GridState &grid_state) {
    this->state = grid_state.state;
    this->id    = grid_state.id;
  }
};

struct GridInput {
  int   id;
  Input input;

  void operator=(const GridInput &grid_input) {
    this->input = grid_input.input;
    this->id    = grid_input.id;
  }
};

class Grid {
public:
  Grid(Param param);
  ~Grid();

  State getState(int index);
  Input getInput(int index);

  int getStateIndex(State state_);

  bool existState(int state_id);
  bool existState(State state_);
  bool existInput(int input_id);

  int getNumState();
  int getNumInput();
  int getNumGrid();

  GridState roundState(State state_);

private:
  void create();

  void setStatePolar(float icp_r, float icp_th, float swf_r, float swf_th);
  void setStateCartesian(float icp_x, float icp_y, float swf_x, float swf_y);
  void setInputPolar(float swf_r, float swf_th);
  void setInputCartesian(float swf_x, float swf_y);

  int getStateIndexCartesian(int icp_x_id, int icp_y_id, int swf_x_id, int swf_y_id);
  int getStateIndexPolar(int icp_r_id, int icp_th_id, int swf_r_id, int swf_th_id);

  int round(float value);
  int max(int val1, int val2);
  int max(int val1, int val2, int val3, int val4);

  Param             param;
  const std::string coord;

  std::vector<State> state;
  std::vector<Input> input;

  int num_state;
  int num_input;

  int num_icp_r, num_icp_th;
  int num_swf_r, num_swf_th;

  int num_icp_x, num_icp_y;
  int num_swf_x, num_swf_y;

  float icp_r[GridSpace::NUMELEMENT];
  float icp_th[GridSpace::NUMELEMENT];
  float swf_r[GridSpace::NUMELEMENT];
  float swf_th[GridSpace::NUMELEMENT];

  float icp_x[GridSpace::NUMELEMENT];
  float icp_y[GridSpace::NUMELEMENT];
  float swf_x[GridSpace::NUMELEMENT];
  float swf_y[GridSpace::NUMELEMENT];
};

} // namespace Capt

#endif // __GRID_H__