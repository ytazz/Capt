#include "grid.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "vector.h"
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
  capturability.load("BasinGpu.csv", DataType::BASIN);
  capturability.load("NstepGpu.csv", DataType::NSTEP);

  CRPlot cr_plot(model, param);

  // 0-step capture region
  if(PLOT_0STEP_CAPTURE_REGION) {
    State state;
    state.icp.setCartesian(0.0, 0.06);
    state.swf.setCartesian(-0.08, 0.10);
    state = grid.roundState(state).state;
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
    // cr_plot.setOutput("gif");
    double omega  = 2 * 3.141 / PLOT_RESOLUTION;
    double icp_r  = 0.14;
    double icp_th = 0.0;
    double icp_x  = 0.0;
    double icp_y  = 0.0;
    double swf_x  = -0.1;
    double swf_y  = 0.3;
    State  state;
    for(int i = 0; i < PLOT_RESOLUTION; i++) {
      icp_th = omega * i;
      icp_x  = icp_r * cos(icp_th);
      icp_y  = icp_r * sin(icp_th);
      state.icp.setCartesian(icp_x, icp_y);
      state.swf.setCartesian(swf_x, swf_y);
      State state_ = grid.roundState(state).state;
      cr_plot.initCaptureMap();
      cr_plot.setFoot(state_.swf);
      cr_plot.setIcp(state_.icp);
      for(int N = 1; N <= NUM_STEP_MAX; N++) {
        if(capturability.capturable(state_, N) ) {
          std::vector<CaptureSet> region = capturability.getCaptureRegion(state_, N);
          for(size_t i = 0; i < region.size(); i++) {
            Input input = grid.getInput(region[i].input_id);
            cr_plot.setCaptureMap(input.swf.x, input.swf.y, N);
          }
        }
      }
      cr_plot.plot();
      usleep(0.5 * 1000000);
    }
  }


  return 0;
}