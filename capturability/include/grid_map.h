#ifndef __GRID_MAP_H__
#define __GRID_MAP_H__

#include "base.h"
#include "param.h"
#include "node.h"
#include <iostream>
#include <vector>

namespace Capt {

class GridMap {
public:
  GridMap(Param *param);
  ~GridMap();

  void setNode(int idx, int idy, Node node);

  void setObstacle(int idx, int idy);

  Node* getNode(int idx, int idy);

  void plot();

private:
  Param     *param;
  Grid       grid;
  RegionPlot plotter;
};

} // namespace Capt

#endif // __GRID_MAP_H__