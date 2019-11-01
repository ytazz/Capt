#include "grid.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "base.h"
#include "cr_plot.h"
#include <iostream>

#define PLOT_0STEP_CAPTURE_REGION false
#define PLOT_NSTEP_CAPTURE_REGION true
#define PLOT_RESOLUTION 50

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model model("data/valkyrie.xml");
  Param param("data/valkyrie_xy.xml");
  Grid  grid(param);

  Capturability capturability(model, param);
  capturability.load("cpu/Basin.csv", DataType::BASIN);
  capturability.load("cpu/Nstep.csv", DataType::NSTEP);

  CRPlot cr_plot(model, param);

  // nao
  // double icp_x = 0.04;
  // double icp_y = 0.05;
  // double swf_x = -0.08;
  // double swf_y = 0.10;
  // val
  double icp_x = 0.0;
  double icp_y = 0.15;
  double swf_x = 0;
  double swf_y = 0.4;
  State  state;
  state.icp.setCartesian(icp_x, icp_y);
  state.swf.setCartesian(swf_x, swf_y);
  state = grid.roundState(state).state;

  // 0-step capture region
  if(PLOT_0STEP_CAPTURE_REGION) {
    cr_plot.initCaptureMap();
    cr_plot.setFoot(state.swf);
    cr_plot.setIcp(state.icp);
    for(int i = 0; i < grid.getNumState(); i++) {
      State state_ = grid.getState(i);
      state_.swf = state.swf;
      if(capturability.capturable(state_, 0) ) {
        cr_plot.setCaptureMap(state_.icp.x, state_.icp.y, 3);
      }
    }
    cr_plot.plot();
  }else if(PLOT_NSTEP_CAPTURE_REGION) {
    cr_plot.initCaptureMap();
    cr_plot.setFoot(state.swf);
    cr_plot.setIcp(state.icp);
    for(int N = 1; N <= 4; N++) {
      if(capturability.capturable(state, N) ) {
        std::vector<CaptureSet> region = capturability.getCaptureRegion(state, N);
        for(size_t i = 0; i < region.size(); i++) {
          Input input = grid.getInput(region[i].input_id);
          cr_plot.setCaptureMap(input.swf.x, input.swf.y, N);
        }
      }
    }
    cr_plot.plot();
  }

  return 0;
}