#include "analysis.h"
#include "capturability.h"
#include "grid.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "polygon.h"
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
  Capturability capturability(model, param);

  Analysis analysis(model, param);
  analysis.exe(1);
  analysis.save("1step.csv", 1);

  // State state, state_;
  // Input input;
  // int state_id = 0, input_id = 0;
  //
  // state_id = 0, input_id = 0;
  // while (grid.existState(state_id)) {
  //   state = grid.getState(state_id);
  //   input_id = 0;
  //   while (grid.existInput(input_id)) {
  //     input = grid.getInput(input_id);
  //     state_ = step(state, input);
  //     if (capturability.capturable(state_, 0)) {
  //       capturability.setCaptureSet(state_id, input_id,
  //                                   grid.getStateIndex(state_), 1);
  //     }
  //     input_id++;
  //   }
  //   state_id++;
  //   printf("%d / %d\n", state_id, grid.getNumState());
  // }

  return 0;
}