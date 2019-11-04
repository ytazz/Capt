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
  capturability->load("gpu/Basin.csv", DataType::BASIN);
  capturability->load("gpu/Nstep.csv", DataType::NSTEP);

  CRPlot *cr_plot = new CRPlot(model, param, grid);

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
  state.icp << icp_x, icp_y;
  state.swf << swf_x, swf_y;
  state = grid->roundState(state).state;

  // N-step capture region
  cr_plot->initCaptureMap();
  cr_plot->setFoot(state.swf);
  cr_plot->setIcp(state.icp);
  for(int N = 1; N <= 4; N++) {
    if(capturability->capturable(state, N) ) {
      std::vector<CaptureSet> region = capturability->getCaptureRegion(state, N);
      for(size_t i = 0; i < region.size(); i++) {
        Input input = grid->getInput(region[i].input_id);
        cr_plot->setCaptureMap(input.swf.x(), input.swf.y(), N);
      }
    }
  }
  cr_plot->plot();

  return 0;
}