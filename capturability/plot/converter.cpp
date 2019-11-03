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
  Model model("data/nao.xml");
  Param param("data/nao_xy.xml");
  Grid  grid(param);

  int       mode     = 0;
  int       coord    = 0;
  int       state_id = 0, input_id = 0;
  State     state, state_;
  Input     input;
  double     tmp1, tmp2;
  GridState gstate;

  while (true) {
    std::cout << "モードを入力してください" << '\n';
    std::cout << " 1: stateからstate_id, 2: state_idからstate" << '\n';
    std::cout << " 3: inputからinput_id, 4: input_idからinput" << "  mode: ";
    std::cin >> mode;

    switch (mode) {
    case 1:
      std::cout << "座標系を入力してください ";
      std::cout << "1: polar, 2: cartesian " << '\n';
      std::cin >> coord;
      if (coord == 1) {
        std::cout << "icp_rを入力してください ";
        std::cin >> tmp1;
        std::cout << "icp_thを入力してください ";
        std::cin >> tmp2;
        state.icp.setPolar(tmp1, tmp2);
      } else if (coord == 2) {
        std::cout << "icp_xを入力してください ";
        std::cin >> tmp1;
        std::cout << "icp_yを入力してください ";
        std::cin >> tmp2;
        state.icp.setCartesian(tmp1, tmp2);
      } else {
        std::cout << "error" << '\n';
        break;
      }
      std::cout << "座標系を入力してください ";
      std::cout << "1: polar, 2: cartesian " << '\n';
      std::cin >> coord;
      if (coord == 1) {
        std::cout << "swf_rを入力してください ";
        std::cin >> tmp1;
        std::cout << "swf_thを入力してください ";
        std::cin >> tmp2;
        state.swf.setPolar(tmp1, tmp2);
      } else if (coord == 2) {
        std::cout << "swf_xを入力してください ";
        std::cin >> tmp1;
        std::cout << "swf_yを入力してください ";
        std::cin >> tmp2;
        state.swf.setCartesian(tmp1, tmp2);
      } else {
        std::cout << "error" << '\n';
        break;
      }
      gstate = grid.roundState(state);
      gstate.state.printCartesian();
      std::cout << "state_id: " << gstate.id << '\n';
      break;
    case 2:
      std::cout << "state_idを入力してください ";
      std::cin >> state_id;
      state = grid.getState(state_id);
      printf("cartesian:\n");
      state.printCartesian();
      printf("polar:\n");
      state.printPolar();
      break;
    case 4:
      std::cout << "input_idを入力してください ";
      std::cin >> input_id;
      input = grid.getInput(input_id);
      printf("cartesian:\n");
      input.printCartesian();
      printf("polar:\n");
      input.printPolar();
      break;
    default:
      break;
    }

    std::cout << '\n';
  }

  return 0;
}