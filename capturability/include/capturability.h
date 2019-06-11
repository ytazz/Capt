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
  int n_step; // N-step capturable

  vec2_t swft; // next landing position
  vec2_t cop;
  float step_time;

  void operator=(const CaptureSet &capture_set) {
    this->state_id = capture_set.state_id;
    this->input_id = capture_set.input_id;
    this->next_state_id = capture_set.next_state_id;
    this->n_step = capture_set.n_step;
    this->swft = capture_set.swft;
    this->cop = capture_set.cop;
    this->step_time = capture_set.step_time;
  }
};

class Capturability {
public:
  Capturability(Model model, Param param);
  ~Capturability();

  void load(const char *file_name);
  void save(const char *file_name, int n_step);

  void setCaptureSet(const int state_id, const int input_id,
                     const int next_state_id, const int n_step,
                     const vec2_t cop, const float step_time);
  std::vector<CaptureSet> getCaptureRegion(const int state_id,
                                           const int n_step);
  std::vector<CaptureSet> getCaptureRegion(const State state, const int n_step);

  bool capturable(State state, int n_step_capture_region);
  bool capturable(int state_id, int n_step_capture_region);

private:
  Grid grid;
  Model model;
  std::vector<CaptureSet> capture_set;
};

} // namespace CA

#endif // __CAPTURABILITY_H__