#ifndef __GRID_MAP_H__
#define __GRID_MAP_H__

#include "base.h"
#include "param.h"
#include "node.h"
// #include "occupancy_plot.h"
#include <iostream>
#include <vector>

namespace Capt {

struct Cell {
  Cell(){
  }

  vec2_t        pos;
  Node          node;
  OccupancyType type;
};

class GridMap {
public:
  GridMap(Param *param);
  ~GridMap();

  void setObstacle(vec2_t pos);
  void setObstacle(vec2i_t id);

  void setNode(vec2_t pos, Node node);
  void setNode(vec2i_t id, Node node);

  void setOccupancy(vec2_t pos, OccupancyType type);
  void setOccupancy(vec2i_t id, OccupancyType type);

  Node* getNode(vec2_t pos);
  Node* getNode(vec2i_t id);

  Cell* findMinCostCell();

  OccupancyType getOccupancy(vec2_t pos);
  OccupancyType getOccupancy(vec2i_t id);

  void plot();

private:
  vec2i_t posToId(vec2_t pos);

  Param *param;
  // OccupancyPlot *plt;

  double x_min, x_max, x_stp;
  double y_min, y_max, y_stp;
  int    x_num, y_num;

  std::vector<std::vector<Cell> > grid;
};

} // namespace Capt

#endif // __GRID_MAP_H__