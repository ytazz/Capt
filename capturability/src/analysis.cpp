#include "analysis.h"

namespace CA {

Analysis::Analysis(Model model, Param param)
    : grid(param), pendulum(model), swing_foot(model) {
  debug = fopen("debug.csv", "w");
  fprintf(
      debug,
      "input_id, input_x, input_y, dt, swft_x, swft_y, icp_x, icp_y, dist\n");
  capture_state.clear();
}

Analysis::~Analysis() {}

void Analysis::exe() {
  State state;
  Input input;
  GridState grid_state;
  GridInput grid_input;
  int state_id = 0, input_id = 0;
  std::vector<bool> one_step;

  state_id = 0, input_id = 0;
  while (grid.existState(state_id)) {
    state = grid.getState(state_id);
    printf("1step:\t%d / %d\n", state_id, grid.getNumState());
    input_id = 0;
    bool flag = false;
    while (grid.existInput(input_id)) {
      input = grid.getInput(input_id);
      input_id++;
      if (capturable(state, input)) {
        grid_state.state = state;
        grid_state.id = state_id;
        grid_input.input = input;
        grid_input.id = input_id;
        setCaptureState(grid_state, grid_input, 1);
        flag = true;
      }
    }
    if (flag == true) {
      one_step.push_back(true);
    } else {
      one_step.push_back(false);
    }
    state_id++;
  }

  for (size_t i = 0; i < one_step.size(); i++) {
    if (one_step[i]) {
      printf("%d\n", (int)i);
    }
  }

  state_id = 95153, input_id = 0;
  // while (grid.existState(state_id)) {
  while (state_id == 95153) {
    state = grid.getState(state_id);
    printf("2step:\t%d / %d\n", state_id, grid.getNumState());
    input_id = 0;
    while (grid.existInput(input_id)) {
      input = grid.getInput(input_id);
      input_id++;
      state = step(state, input);
      if (grid.existState(state)) {
        GridState gs = grid.roundState(state);
        if (one_step[gs.id]) {
          grid_state.state = state;
          grid_state.id = state_id;
          grid_input.input = input;
          grid_input.id = input_id;
          setCaptureState(grid_state, grid_input, 2);
        }
      }
    }
    state_id++;
  }
}

bool Analysis::capturable(const State state, const Input input) {
  Vector2 icp, swft;
  float dt;
  bool flag = false;

  pendulum.setIcp(state.icp);
  Vector2 cop;
  cop.setPolar(0.04, state.icp.th);
  pendulum.setCop(cop);

  swing_foot.set(state.swft, input.swft);
  dt = swing_foot.getTime();

  icp = pendulum.getIcp(dt);
  swft = swing_foot.getTraj(dt);

  float dist = (icp - swft).norm();
  if (dist <= 0.04) {
    flag = true;
  }

  fprintf(debug, "%lf, %lf,", input.swft.x, input.swft.y);
  fprintf(debug, "%lf,", dt);
  fprintf(debug, "%lf, %lf,", cop.x, cop.y);
  fprintf(debug, "%lf, %lf,", icp.x, icp.y);
  fprintf(debug, "%lf\n", dist);

  return flag;
}

void Analysis::setCaptureState(const GridState grid_state,
                               const GridInput grid_input,
                               const int n_step_capturable) {
  CaptureState cs;
  cs.grid_state = grid_state;
  cs.grid_input = grid_input;
  cs.n_capturable = n_step_capturable;

  capture_state.push_back(cs);
}

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
  FILE *fp = fopen(file_name, "w");
  fprintf(fp, "state_id, state_icp_x, state_icp_y, state_swft_x, state_swft_y,"
              "input_id, input_swft_x, input_swft_y\n");
  for (size_t i = 0; i < capture_state.size(); i++) {
    if (capture_state[i].n_capturable == n_step_capturable) {
      fprintf(fp, "%d,", capture_state[i].grid_state.id);
      fprintf(fp, "%lf, %lf,", capture_state[i].grid_state.state.icp.x,
              capture_state[i].grid_state.state.icp.y);
      fprintf(fp, "%lf, %lf,", capture_state[i].grid_state.state.swft.x,
              capture_state[i].grid_state.state.swft.y);
      fprintf(fp, "%d,", capture_state[i].grid_input.id);
      fprintf(fp, "%lf, %lf,", capture_state[i].grid_input.input.swft.x,
              capture_state[i].grid_input.input.swft.y);
      fprintf(fp, "\n");
    }
  }
}

} // namespace CA