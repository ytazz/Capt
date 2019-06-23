#include "analysis.h"

namespace CA {

Analysis::Analysis(Model model, Param param)
    : grid(param), model(model), pendulum(model), swing_foot(model),
      capturability(model, param) {}

Analysis::~Analysis() {}

void Analysis::exe(int n_step) {
  State state, state_;
  Input input;

  printf("-----------------------------------------\n");
  printf("Start %d-step capturability analysis\n", n_step);
  printf("state:%d, input:%d\n", grid.getNumState(), grid.getNumInput());

  for (int state_id = 0; state_id < grid.getNumState(); state_id++) {
    state = grid.getState(state_id);
    for (int input_id = 0; input_id < grid.getNumInput(); input_id++) {
      input = grid.getInput(input_id);
      state_ = step(state, input);
      if (grid.existState(state_)) {
        if (capturability.capturable(state_, n_step - 1)) {
          capturability.setCaptureSet(state_id, input_id,
                                      grid.getStateIndex(state_), n_step, cop,
                                      step_time);
        }
      }
    }
    if ((state_id % 1000) == 0) {
      float percentage = (float)state_id / grid.getNumState() * 100;
      printf("%d \t/ %d \t(%lf %%)\n", state_id, grid.getNumState(),
             percentage);
    }
  }
}

State Analysis::step(const State state, const Input input) {
  Vector2 icp;

  pendulum.setIcp(state.icp);
  Polygon polygon;
  std::vector<vec2_t> region = model.getVec("foot", "foot_r_convex");
  cop = polygon.getClosestPoint(state.icp, region);
  Vector2 cop_;
  cop_.setCartesian(cop.x, cop.y);
  pendulum.setCop(cop);

  swing_foot.set(state.swft, input.swft);
  step_time = swing_foot.getTime();

  icp = pendulum.getIcp(step_time);

  Vector2 icp_, swft_;
  icp_.setCartesian(-input.swft.x + icp.x, input.swft.y - icp.y);
  swft_.setCartesian(-input.swft.x, input.swft.y);

  State state_;
  state_.icp = icp_;
  state_.swft = swft_;

  return state_;
}

void Analysis::save(const char *file_name, const int n_step_capturable) {
  capturability.save(file_name, n_step_capturable);
}

} // namespace CA