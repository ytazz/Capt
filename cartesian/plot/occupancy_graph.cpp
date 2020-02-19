#include "occupancy_plot.h"
#include <iostream>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Param *param = new Param("data/valkyrie_xy.xml");

  OccupancyPlot *plot = new OccupancyPlot(param);

  plot->setOccupancy(0.0, 0.0, OccupancyType::EXIST);
  plot->setOccupancy(0.5, 0.5, OccupancyType::EXIST);

  plot->plot();

  delete param;
  delete plot;

  return 0;
}