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
  Model *model = new Model("/home/dl-box/Capturability/cartesian/data/valkyrie.xml");
  Param *param = new Param("/home/dl-box/Capturability/cartesian/data/valkyrie_xy.xml");

  CRPlot *cr_plot = new CRPlot(model, param);

  Capturability *cap = new Capturability(model, param);
  cap->loadTrans("/home/dl-box/Capturability/cartesian/cpu/trans0.bin", 0, true);
  cap->loadTrans("/home/dl-box/Capturability/cartesian/cpu/trans1.bin", 1, true);
  cap->loadTrans("/home/dl-box/Capturability/cartesian/cpu/trans2.bin", 2, true);
  cap->loadTrans("/home/dl-box/Capturability/cartesian/cpu/trans3.bin", 3, true);
  cap->loadTrans("/home/dl-box/Capturability/cartesian/cpu/trans4.bin", 4, true);
  cap->loadTrans("/home/dl-box/Capturability/cartesian/cpu/trans5.bin", 5, true);
  printf("load done\n");

  // retrieve state from commandline arguments
  if(argc == 6){
    icp_x = atof(argv[1]);
    icp_y = atof(argv[2]);
    swf_x = atof(argv[3]);
    swf_y = atof(argv[4]);
    swf_z = atof(argv[5]);
  }

  State st, stnext;
  st.icp << icp_x, icp_y;
  st.swf << swf_x, swf_y, swf_z;
  int state_id = cap->grid->roundState(st);
  st = cap->grid->state[state_id];
  st.print();

  cr_plot->setState(st);

  printf("get cap regions\n");
  std::vector<CaptureBasin> basin;
  basin.resize(nmax+1);
  for(int n = 0; n <= nmax; n++){
    cap->getCaptureBasin(st, n, basin[n]);
    printf("%d-step cap regions: %d\n", n, (int)basin[n].size());

    for(CaptureState& cs : basin[n]){
      stnext.swf = cap->grid->swf[cs.swf_id];
      stnext.icp = cap->grid->icp[cs.icp_id];
      Input in     = cap->calcInput(st, stnext);
      cr_plot->setCaptureInput(in, n);
    }
  }

  cr_plot->plot();

  delete cr_plot;

  return 0;
}
