#include "grid.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "base.h"
#include "cr_plot.h"
#include <iostream>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model *model = new Model("data/valkyrie.xml");
  Param *param = new Param("data/valkyrie_xy.xml");
  Grid  *grid  = new Grid(param);

  Capturability *capturability = new Capturability(grid);
  capturability->loadBasin("gpu/Basin.csv");
  capturability->loadNstep("cpu/1step.csv", 1);
  capturability->loadNstep("cpu/2step.csv", 2);
  capturability->loadNstep("cpu/3step.csv", 3);
  capturability->loadNstep("cpu/4step.csv", 4);
  capturability->loadNstep("cpu/5step.csv", 5);
  capturability->loadNstep("cpu/6step.csv", 6);

  CRPlot *cr_plot = new CRPlot(model, param, grid);
  // cr_plot->setOutput("gif");

  // nao
  // double icp_x = 0.04;
  // double icp_y = 0.05;
  // double swf_x = -0.08;
  // double swf_y = 0.10;
  // val
  double icp_x = 0.00;
  double icp_y = 0.10;
  double swf_x = -0.25;
  double swf_y = 0.4;
  double swf_z = 0.0;

  // N-step capture region
  double icp_x_min, icp_x_max, icp_x_stp;
  int    icp_x_num;
  param->read(&icp_x_min, "icp_x_min");
  param->read(&icp_x_max, "icp_x_max");
  param->read(&icp_x_stp, "icp_x_stp");
  param->read(&icp_x_num, "icp_x_num");

  while(true) {
    for(int i = 0; i < icp_x_num; i++) {
      // set state
      State state;
      state.icp.x() = icp_x_min + icp_x_stp * i;
      state.icp.y() = icp_y;
      state.swf.x() = swf_x;
      state.swf.y() = swf_y;
      state.swf.z() = swf_z;
      state         = grid->roundState(state).state;

      cr_plot->initCaptureMap();
      cr_plot->setSwf(vec3Tovec2(state.swf) );
      cr_plot->setIcp(state.icp);
      for(int N = 1; N <= 4; N++) {
        if(capturability->capturable(state, N) ) {
          std::vector<CaptureSet*> region = capturability->getCaptureRegion(state, N);
          for(size_t i = 0; i < region.size(); i++) {
            Input input = grid->getInput(region[i]->input_id);
            cr_plot->setCaptureMap(input.swf.x(), input.swf.y(), N);
          }
        }
      }
      cr_plot->plot();
      usleep(0.5 * 1000000);
    }
  }

  return 0;
}