#include "CA.h"
#include <iostream>

using namespace std;
using namespace CA;

int main(int argc, char const *argv[]) {
  Model model("nao.xml");
  Param param("analysis.xml");

  Grid grid(param);

  Capturability capturability(model, param);
  capturability.load("1step_dsp.csv");

  State state;
  GridState gstate;
  state.icp.setCartesian(-0.013626, 0.08);
  state.swft.setPolar(0.11, 3.14159 / 2);
  gstate = grid.roundState(state);

  FILE *fp;
  fp = fopen("cap.csv", "w");
  for (int i = 0; i < 40; i++) {
    for (int j = 0; j < 40; j++) {
      vec2_t icp;
      icp.setCartesian(-0.2 + 0.01 * j, -0.2 + 0.01 * i);
      state.icp = icp;
      state.swft.setCartesian(0.05, 0.11);
      if (grid.existState(state)) {
        gstate = grid.roundState(state);
        bool is_capt = capturability.capturable(gstate.id, 1);
        if (is_capt) {
          fprintf(fp, "%f,%f\n", icp.x, icp.y);
        }
      }
    }
  }

  return 0;
}