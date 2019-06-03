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

  int state_id = 95153;
  int input_id = 223;
  State state = grid.getState(state_id);
  state.printCartesian();
  // state.printPolar();
  Input input = grid.getInput(input_id);
  input.printCartesian();
  // input.printPolar();

  Analysis analysis(model, param);
  State state_ = analysis.step(state, input);
  state_.printCartesian();

  return 0;
}