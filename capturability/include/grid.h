#ifndef __GRID_H__
#define __GRID_H__

#include "param.h"
#include "vector.h"
#include <iostream>
#include <string>
#include <vector>

namespace CA {

namespace GridSpace {
enum Element { MIN, MAX, STEP, NUMELEMENT };
}

struct State {
  Vector2 icp;
  Vector2 swft;
};

struct Input {
  Vector2 step;
};

class Grid {
public:
  Grid(Param param);
  ~Grid();

private:
  void create();

  void setStatePolar(float icp_r, float icp_th, float swft_r, float swft_th);
  void setStateCartesian(float icp_x, float icp_y, float swft_x, float swft_y);
  void setInputPolar(float swft_r, float swft_th);
  void setInputCartesian(float swft_x, float swft_y);

  State getState(int index);
  State getState(int icp_r_id, int icp_th_id, int swft_r_id, int swft_th_id);
  Input getInput(int index);

  int getNumState();
  int getNumInput();

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

} // namespace CA

#endif // __GRID_H____