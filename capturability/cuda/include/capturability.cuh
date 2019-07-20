#ifndef __CAPTURABILITY_CUH__
#define __CAPTURABILITY_CUH__

#include "grid.cuh"
#include "input.cuh"
#include "model.cuh"
#include "param.cuh"
#include "polygon.cuh"
#include "state.cuh"
#include <iostream>
#include <stdio.h>
#include <vector>

struct CaptureSet {
  int state_id;
  int input_id;
  int next_state_id;
  int n_step; // N-step capturable

  vec2_t swft; // next landing position
  vec2_t cop;
  float step_time;

  __device__ void operator=(const CaptureSet &capture_set) {
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
  __device__ Capturability(Model model, Param param);
  __device__ ~Capturability();

  __device__ void load(const char *file_name);
  __device__ void save(const char *file_name, int n_step);

  __device__ void setCaptureSet(const int state_id, const int input_id,
                                const int next_state_id, const int n_step,
                                const vec2_t cop, const float step_time);
  __device__ void setCaptureSet(const int state_id, const int input_id,
                                const int next_state_id, const int n_step);
  __device__ std::vector<CaptureSet> getCaptureRegion(const int state_id,
                                                      const int n_step);
  __device__ std::vector<CaptureSet> getCaptureRegion(const State state,
                                                      const int n_step);

  __device__ bool capturable(State state, int n_step_capture_region);
  __device__ bool capturable(int state_id, int n_step_capture_region);

private:
  Grid grid;
  Model model;
  std::vector<CaptureSet> capture_set;
};

#endif // __CAPTURABILITY_CUH__