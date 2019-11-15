#include "occupancy_plot.h"
#include <iostream>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Param *param = new Param("data/footstep.xml");

  OccupancyPlot *plot = new OccupancyPlot(param);

  for(int i = 0; i <= 5; i++) {
    for(int j = 0; j <= 5; j++) {
      plot->setOccupancy(0.5 + 0.05 * i, -0.5 + 0.05 * j, OccupancyType::OBSTACLE);
    }
  }

  for(int i = 0; i <= 10; i++) {
    for(int j = 0; j <= 10; j++) {
      plot->setOccupancy(0.05 * i, 0.05 * j, OccupancyType::OPEN);
    }
  }

  plot->setOccupancy(0.0, 0.0, OccupancyType::CLOSED);
  plot->setOccupancy(0.5, 0.5, OccupancyType::CLOSED);

  plot->plot();

  delete param;
  delete plot;

  return 0;
}