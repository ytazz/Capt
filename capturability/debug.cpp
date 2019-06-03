#include "analysis.h"
#include "grid.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "vector.h"
#include <iostream>

using namespace std;
using namespace CA;

int main(int argc, char const *argv[]) {
  Model model("nao.xml");
  model.parse();

  Param param("analysis.xml");
  param.parse();

  Grid grid(param);

  State state, state_;
  Input input;
  GridState grid_state;
  GridInput grid_input;

  Analysis analysis(model, param);

  FILE *fp = fopen("test.csv", "w");
  fprintf(fp,
          "state_id,input_id,state.icp.r,state.icp.th,state.swft.r,state.swft."
          "th\n");

  int state_id = 0;
  int input_id = 0;
  state_id = 0, input_id = 0;
  // while (grid.existState(state_id)) {
  state = grid.getState(state_id);
  // printf("2step:\t%d / %d\n", state_id, grid.getNumState());
  input_id = 0;
  while (grid.existInput(input_id)) {
    input = grid.getInput(input_id);
    state_ = analysis.step(state, input);
    fprintf(fp, "%d,%d,", state_id, input_id);
    fprintf(fp, "%lf,%lf,", state_.icp.r, state_.icp.th);
    fprintf(fp, "%lf,%lf\n", state_.swft.r, state_.swft.th);
    // if (grid.existState(state)) {
    //   fprintf(fp, "%d\n", input_id);
    //   GridState gs = grid.roundState(state);
    //   // if (one_step[gs.id]) {
    //   // grid_state.state = state;
    //   // grid_state.id = state_id;
    //   // grid_input.input = input;
    //   // grid_input.id = input_id;
    //   // setCaptureState(grid_state, grid_input, 2);
    //   // }
    // }
    input_id++;
  }
  state_id++;
  // }

  printf("\t%d\n", grid.existState(state));
  // Analysis analysis(model, param);
  // State state_ = analysis.step(state, input);
  // state_.printCartesian();

  return 0;
}