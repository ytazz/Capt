#include "analysis.h"

namespace CA {

Analysis::Analysis(Model model, Param param)
    : grid(param), pendulum(model), swing_foot(model),
      capturability(model, param) {}

Analysis::~Analysis() {}

void Analysis::exe(int n_step) {
  State state, state_;
  Input input;
  int state_id = 0, input_id = 0;

  state_id = 0, input_id = 0;
  while (grid.existState(state_id)) {
    state = grid.getState(state_id);
    input_id = 0;
    while (grid.existInput(input_id)) {
      input = grid.getInput(input_id);
      state_ = step(state, input);
      if (capturability.capturable(state_, 0)) {
        capturability.setCaptureSet(state_id, input_id,
                                    grid.getStateIndex(state_), 1);
      }
      input_id++;
    }
    state_id++;
    printf("%d / %d\n", state_id, grid.getNumState());
  }
}

// bool Analysis::capturable(const State state, const Input input) {
//   bool flag = false;
//
//   pendulum.setIcp(state.icp);
//   Vector2 cop;
//   cop.setPolar(0.04, state.icp.th);
//   pendulum.setCop(cop);
//
//   swing_foot.set(state.swft, input.swft);
//   float dt = swing_foot.getTime();
//
//   Vector2 icp = pendulum.getIcp(dt);
//   Vector2 swft = swing_foot.getTraj(dt);
//
//   float dist = (icp - swft).norm();
//   if (dist <= 0.04) {
//     flag = true;
//   }
//
//   return flag;
// }

State Analysis::step(const State state, const Input input) {
  Vector2 icp;
  float dt;

  pendulum.setIcp(state.icp);
  Vector2 cop;
  cop.setPolar(0.04, state.icp.th);
  pendulum.setCop(cop);

  swing_foot.set(state.swft, input.swft);
  dt = swing_foot.getTime();

  icp = pendulum.getIcp(dt);

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