#include "analysis.h"

namespace CA {

Analysis::Analysis(Model model, Param param)
    : grid(param), pendulum(model), swing_foot(model) {
  debug = fopen("debug.csv", "w");
  fprintf(
      debug,
      "input_id, input_x, input_y, dt, swft_x, swft_y, icp_x, icp_y, dist\n");
}

Analysis::~Analysis() {}

void Analysis::exe() {
  State state;
  Input input;
  GridState grid_state;
  GridInput grid_input;

  int state_id = 462641, input_id = 0;
  // while (grid.existState(state_id)) {
  while (state_id <= 462641) {
    state = grid.getState(state_id);
    printf("%d / %d\n", state_id, grid.getNumState());
    input_id = 0;
    while (grid.existInput(input_id)) {
      input = grid.getInput(input_id);
      input_id++;
      fprintf(debug, "%d,", input_id);
      if (capturable(state, input)) {
        grid_state.state = state;
        grid_state.id = state_id;
        grid_input.input = input;
        grid_input.id = input_id;
        setCaptureState(grid_state, grid_input, 1);
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

void Analysis::save(const char *file_name) {
  FILE *fp = fopen(file_name, "w");
  fprintf(fp, "state_id, state_icp_x, state_icp_y, state_swft_x, state_swft_y, "
              "input_id, input_swft_x, input_swft_y\n");
  for (size_t i = 0; i < capture_state.size(); i++) {
    if (capture_state[i].grid_state.id == 462641) {
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