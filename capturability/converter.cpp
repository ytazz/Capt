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

  int mode = 0;
  int state_id = 0, input_id = 0;
  State state;
  Input input;

  while (true) {
    std::cout << "1: stateからstate_id, 2: state_idからstate" << '\n';
    std::cout << "3: inputからinput_id, 4: input_idからinput" << '\n';
    std::cout << "モードを入力してください";
    std::cin >> mode;

    switch (mode) {
    case 2:
      std::cout << "state_idを入力してください";
      std::cin >> state_id;
      state = grid.getState(state_id);
      state.printCartesian();
      break;
    case 4:
      std::cout << "input_idを入力してください";
      std::cin >> input_id;
      input = grid.getInput(input_id);
      input.printCartesian();
      break;
    default:
      break;
    }

    std::cout << '\n';
  }

  return 0;
}