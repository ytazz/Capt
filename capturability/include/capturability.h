#ifndef __CAPTURABILITY_H__
#define __CAPTURABILITY_H__

#include "grid.h"
#include "input.h"
#include "model.h"
#include "param.h"
#include "polygon.h"
#include "state.h"
#include <iostream>
#include <stdio.h>
#include <vector>

namespace CA {

struct CaptureSet {
  int state_id;
  int input_id;
  int next_state_id;
  int n; // N-step capturable

  void operator=(const CaptureSet &capture_set) {
    this->state_id = capture_set.state_id;
    this->input_id = capture_set.input_id;
    this->next_state_id = capture_set.next_state_id;
    this->n = capture_set.n;
  }
};

class Capturability {
public:
  Capturability(Model model, Param param);
  ~Capturability();

  void load(const char *file_name);
  void save(const char *file_name, int n_step_capture_set);

  void setCaptureSet(const int state_id, const int input_id,
                     const int next_state_id, const int n_step_capturable);
  std::vector<Input> getCaptureRegion(const int state_id,
                                      const int n_step_capturable);

  bool capturable(State state, int n_step_capture_region);

private:
  Grid grid;
  Model model;
  std::vector<CaptureSet> capture_set;
};

} // namespace CA

#endif // __CAPTURABILITY_H__