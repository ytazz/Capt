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
  capturability->loadBasin("cpu/Basin.csv");
  capturability->loadNstep("cpu/Nstep.csv");

  // paper plot for nao
  // double icp_x        = 0.00;
  // double icp_y        = 0.05;
  // double swf_x        = -0.08;
  // double swf_y        = 0.10;
  // double icp_x_offset = 0.05;
  // double icp_y_offset = 0.02;
  // paper plot for val
  // double icp_x        = 0.00;
  // double icp_y        = 0.0;
  // double swf_x        = -0.25;
  // double swf_y        = 0.4;
  // double icp_x_offset = 0.1;
  // double icp_y_offset = 0.05;
  // walk val
  double icp_x = 0.040819;
  double icp_y = 0.042048;
  double swf_x = -0.2;
  double swf_y = 0.3;

  State state;
  Input input;
  // int   cop_id = 6;

  CRPlot *cr_plot;
  // case1
  state.icp << icp_x, icp_y;
  state.swf << swf_x, swf_y;
  state.elp = 0;
  state     = grid->roundState(state).state;
  int stateId = grid->roundState(state).id;

  int count = 0;
  // for(int cop_id = 0; cop_id < 15; cop_id++) {
  cr_plot = new CRPlot(model, param, grid);
  cr_plot->initCaptureMap();
  cr_plot->setState(state);
  // cr_plot->setCop(grid->getCop(cop_id) );
  // for(int N = 1; N <= 4; N++) {
  // if(capturability->capturable(state, N) ) {
  std::vector<CaptureSet*> region = capturability->getCaptureRegion(stateId);
  // printf("%d-step capture points %5d\n", N, (int)region.size() );
  // count += (int)region.size();
  for(size_t i = 0; i < region.size(); i++) {
    Input input = grid->getInput(region[i]->input_id);
    // if(grid->indexCop(input.cop) == cop_id)
    cr_plot->setCaptureMap(input.swf.x(), input.swf.y(), region[i]->nstep);
  }
  // }
  // }
  cr_plot->plot();
  delete cr_plot;
  // }

  // printf("%d\n", count);

  return 0;
}