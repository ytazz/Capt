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
  cap->load("/home/dl-box/Capturability/cartesian/cpu/");
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
  st.icp = vec2_t(icp_x, icp_y);
  st.swf = vec3_t(swf_x, swf_y, swf_z);

  cr_plot->setState(st);

  printf("get cap regions\n");
  CaptureBasin basin;
  cap->getCaptureBasin(st, -1, basin);
  printf("get done: %d\n", basin.size());
  for(CaptureState& cs : basin){
    stnext.swf = cap->grid->xyz[cap->swf_to_xyz[cs.swf_id]];
    stnext.icp = cap->grid->xy [cs.icp_id];
    Input in   = cap->calcInput(st, stnext);
    cr_plot->setCaptureInput(in, cs.nstep);
  }

  cr_plot->plot();

  delete cr_plot;

  return 0;
}
