#ifndef __GRID_H__
#define __GRID_H__

#include "input.h"
#include "param.h"
#include "state.h"
#include "base.h"
#include <iostream>
#include <string>
#include <vector>

namespace Capt {

namespace CaptEnum {
enum Element { MIN, MAX, STP, NUMELEMENT };
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
  Grid(Param *param);
  ~Grid();

  State getState(int index);
  Input getInput(int index);

  vec2_t getIcp(int index);
  vec2_t getSwf(int index);
  vec2_t getCop(int index);

  bool existState(int state_id);
  bool existState(State state_);
  bool existInput(int input_id);

  int getNumState();
  int getNumInput();
  int getNumGrid();

  int getStateIndex(State state_);

  vec2_t    roundVec(vec2_t vec);
  GridState roundState(State state_);

  int indexIcp(vec2_t icp);
  int indexSwf(vec2_t swf);
  int indexCop(vec2_t cop);

  void print();

private:
  void create();

  void setState(double icp_x, double icp_y, double swf_x, double swf_y);
  void setInput(double cop_x, double cop_y, double swf_x, double swf_y);

  int getStateIndex(int icp_x_id, int icp_y_id, int swf_x_id, int swf_y_id);

  int max(int val1, int val2);
  int max(int val1, int val2, int val3, int val4);

  Param *param;

  std::vector<State> state;
  std::vector<Input> input;

  int state_num;
  int input_num;

  int icp_x_num, icp_y_num;
  int swf_x_num, swf_y_num;
  int cop_x_num, cop_y_num;

  double icp_x[CaptEnum::NUMELEMENT];
  double icp_y[CaptEnum::NUMELEMENT];
  double swf_x[CaptEnum::NUMELEMENT];
  double swf_y[CaptEnum::NUMELEMENT];
  double cop_x[CaptEnum::NUMELEMENT];
  double cop_y[CaptEnum::NUMELEMENT];
};

} // namespace Capt

#endif // __GRID_H__