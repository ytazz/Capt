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

  // paper plot for nao
  // double icp_x        = 0.00;
  // double icp_y        = 0.05;
  // double swf_x        = -0.08;
  // double swf_y        = 0.10;
  // double icp_x_offset = 0.05;
  // double icp_y_offset = 0.02;
  // paper plot for val
  double icp_x        = 0.00;
  double icp_y        = 0.10;
  double swf_x        = -0.25;
  double swf_y        = 0.4;
  double icp_x_offset = 0.1;
  double icp_y_offset = 0.05;
  // walk val
  // double icp_x = 0.0;
  // double icp_y = 0.15;
  // double swf_x = 0;
  // double swf_y = 0.4;

  State state;

  CRPlot *cr_plot;
  // case1
  printf("plot case 1\n");
  state.icp << icp_x, icp_y;
  state.swf << swf_x, swf_y;
  state = grid->roundState(state).state;

  cr_plot = new CRPlot(model, param, grid);
  cr_plot->initCaptureMap();
  cr_plot->setFoot(state.swf);
  cr_plot->setIcp(state.icp);
  for(int N = 1; N <= 5; N++) {
    if(capturability->capturable(state, N) ) {
      std::vector<CaptureSet> region = capturability->getCaptureRegion(state, N);
      printf("%d-step capture points %5d\n", N, (int)region.size() );
      for(size_t i = 0; i < region.size(); i++) {
        Input input = grid->getInput(region[i].input_id);
        cr_plot->setCaptureMap(input.swf.x(), input.swf.y(), N);
      }
    }
  }
  cr_plot->plot();
  delete cr_plot;

  // case2
  printf("plot case 2\n");
  icp_x += icp_x_offset;
  state.icp << icp_x, icp_y;
  state.swf << swf_x, swf_y;
  state = grid->roundState(state).state;

  cr_plot = new CRPlot(model, param, grid);
  cr_plot->initCaptureMap();
  cr_plot->setFoot(state.swf);
  cr_plot->setIcp(state.icp);
  for(int N = 1; N <= 5; N++) {
    if(capturability->capturable(state, N) ) {
      std::vector<CaptureSet> region = capturability->getCaptureRegion(state, N);
      printf("%d-step capture points %5d\n", N, (int)region.size() );
      for(size_t i = 0; i < region.size(); i++) {
        Input input = grid->getInput(region[i].input_id);
        cr_plot->setCaptureMap(input.swf.x(), input.swf.y(), N);
      }
    }
  }
  cr_plot->plot();
  delete cr_plot;

  // case3
  printf("plot case 3\n");
  icp_x -= icp_x_offset;
  icp_y += icp_y_offset;
  state.icp << icp_x, icp_y;
  state.swf << swf_x, swf_y;
  state = grid->roundState(state).state;

  cr_plot = new CRPlot(model, param, grid);
  cr_plot->initCaptureMap();
  cr_plot->setFoot(state.swf);
  cr_plot->setIcp(state.icp);
  for(int N = 1; N <= 5; N++) {
    if(capturability->capturable(state, N) ) {
      std::vector<CaptureSet> region = capturability->getCaptureRegion(state, N);
      printf("%d-step capture points %5d\n", N, (int)region.size() );
      for(size_t i = 0; i < region.size(); i++) {
        Input input = grid->getInput(region[i].input_id);
        cr_plot->setCaptureMap(input.swf.x(), input.swf.y(), N);
      }
    }
  }
  cr_plot->plot();
  delete cr_plot;

  return 0;
}