#include "capturability.h"
#include "grid.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "polygon.h"
#include "base.h"
#include <iostream>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Param *param = new Param("data/valkyrie_xy.xml");
  Grid  *grid  = new Grid(param);

  int       mode     = 0;
  int       state_id = 0, input_id = 0;
  State     state;
  Input     input;
  double    tmp1, tmp2;
  GridState gstate;

  while (true) {
    std::cout << "モードを入力してください" << '\n';
    std::cout << " 1: stateからstate_id, 2: state_idからstate" << '\n';
    std::cout << " 3: inputからinput_id, 4: input_idからinput" << "  mode: ";
    std::cin >> mode;

    switch (mode) {
    case 1:
      std::cout << "icp_xを入力してください ";
      std::cin >> state.icp.x();
      std::cout << "icp_yを入力してください ";
      std::cin >> state.icp.y();

      std::cout << "swf_xを入力してください ";
      std::cin >> state.swf.x();
      std::cout << "swf_yを入力してください ";
      std::cin >> state.swf.y();

      std::cout << "elapsedを入力してください ";
      std::cin >> state.elp;

      gstate = grid->roundState(state);
      gstate.state.print();
      std::cout << "state_id: " << gstate.id << '\n';
      break;
    case 2:
      std::cout << "state_idを入力してください ";
      std::cin >> state_id;
      state = grid->getState(state_id);
      state.print();
      break;
    case 4:
      std::cout << "input_idを入力してください ";
      std::cin >> input_id;
      input = grid->getInput(input_id);
      input.print();
      break;
    default:
      break;
    }

    std::cout << '\n';
  }

  return 0;
}