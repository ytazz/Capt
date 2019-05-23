#ifndef __ANALYSIS_H__
#define __ANALYSIS_H__

#include "grid.h"
#include "input.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "state.h"
#include "swing_foot.h"
#include "vector.h"
#include <iostream>
#include <stdio.h>
#include <vector>

namespace CA {

struct CaptureState {
  int n_capturable;
  GridState grid_state;
  GridInput grid_input;

  void operator=(const CaptureState &capture_state) {
    this->grid_state = capture_state.grid_state;
    this->grid_input = capture_state.grid_input;
  }
};

class Analysis {

public:
  Analysis(Model model, Param param);
  ~Analysis();

  void exe();
  void save(const char *file_name);

private:
  bool capturable(const State state, const Input input);
  void setCaptureState(const GridState grid_state, const GridInput grid_input,
                       const int n_step_capturable);

  Grid grid;
  Pendulum pendulum;
  SwingFoot swing_foot;

  std::vector<CaptureState> capture_state;

  FILE *debug;
};

} // namespace CA

#endif // __ANALYSIS_H__