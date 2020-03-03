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

  CRPlot *cr_plot = new CRPlot(model, param, grid);

  Capturability *capturability = new Capturability(grid);
  capturability->loadBasin("cpu/Basin.csv");
  capturability->loadNstep("cpu/1step.csv", 1);
  capturability->loadNstep("cpu/2step.csv", 2);
  capturability->loadNstep("cpu/3step.csv", 3);
  capturability->loadNstep("cpu/4step.csv", 4);
  capturability->loadNstep("cpu/5step.csv", 5);
  capturability->loadNstep("cpu/6step.csv", 6);

  double icp_x = -0.15;
  double icp_y = +0.05;
  double swf_x = -0.1;
  double swf_y = +0.3;
  double swf_z = +0.0;

  State state;
  Input input;

  state.icp << icp_x, icp_y;
  state.swf << swf_x, swf_y, swf_z;
  state = grid->roundState(state).state;
  state.print();
  int stateId = grid->roundState(state).id;

  cr_plot->initCaptureMap();
  cr_plot->setState(state);

  std::vector<CaptureSet*> region = capturability->getCaptureRegion(stateId);
  for(size_t i = 0; i < region.size(); i++) {
    Input input = grid->getInput(region[i]->input_id);
    cr_plot->setCaptureMap(input.swf.x(), input.swf.y(), region[i]->nstep);
  }

  cr_plot->plot();

  delete cr_plot;

  return 0;
}