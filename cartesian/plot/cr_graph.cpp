#include "grid.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "base.h"
#include "cr_plot.h"
#include <iostream>

using namespace std;
using namespace Capt;

double icp_x = +0.0;
double icp_y = +0.1;
double swf_x = +0.0;
double swf_y = +0.25;
double swf_z = +0.0;

const int nmax = 5;

int main(int argc, char const *argv[]) {
  Model *model = new Model("data/valkyrie.xml");
  Param *param = new Param("data/valkyrie_xy.xml");
  Grid  *grid  = new Grid(param);
  Swing *swing = new Swing(model, param);

  CRPlot *cr_plot = new CRPlot(model, param, grid);

  Capturability *cap = new Capturability(model, param, grid, swing);
  cap->loadTrans("cpu/trans0.bin", 0, true);
  cap->loadTrans("cpu/trans1.bin", 1, true);
  cap->loadTrans("cpu/trans2.bin", 2, true);
  cap->loadTrans("cpu/trans3.bin", 3, true);
  cap->loadTrans("cpu/trans4.bin", 4, true);
  cap->loadTrans("cpu/trans5.bin", 5, true);
  printf("load done\n");

  // retrieve state from commandline arguments
  if(argc == 6){
    icp_x = atof(argv[1]);
    icp_y = atof(argv[2]);
    swf_x = atof(argv[3]);
    swf_y = atof(argv[4]);
    swf_z = atof(argv[5]);
  }

  State st;
  st.icp << icp_x, icp_y;
  st.swf << swf_x, swf_y, swf_z;
  int state_id = grid->roundState(st);
  st = grid->state[state_id];
  st.print();

  cr_plot->setState(st);

  printf("get cap regions\n");
  std::vector<CaptureBasin> basin;
  basin.resize(nmax+1);
  for(int n = 0; n <= nmax; n++){
    cap->getCaptureBasin(st, n, basin[n]);
    printf("%d-step cap regions: %d\n", n, (int)basin[n].size());

    for(CaptureState& ct : basin[n]){
      State stnext = grid->state[ct.state_id];
      Input in     = cap->calcInput(st, stnext);
      cr_plot->setCaptureInput(in, n);
    }
  }

  cr_plot->plot();

  delete cr_plot;

  return 0;
}
