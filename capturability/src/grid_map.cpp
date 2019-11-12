#include "grid_map.h"

namespace Capt {

GridMap::GridMap(Param *param) : param(param){
  param->read(&x_min, "map_x_min");
  param->read(&x_max, "map_x_max");
  param->read(&x_stp, "map_x_stp");
  param->read(&x_num, "map_x_num");
  param->read(&y_min, "map_y_min");
  param->read(&y_max, "map_y_max");
  param->read(&y_stp, "map_y_stp");
  param->read(&y_num, "map_y_num");

  grid.clear();
  grid.resize(x_num);
  vec2_t pos;
  for (int i = 0; i < x_num; i++) {
    pos.x() = x_min + x_stp * i;
    grid[i].resize(y_num);
    for (int j = 0; j < y_num; j++) {
      pos.y() = y_min + y_stp * j;

      grid[i][j].pos  = pos;
      grid[i][j].type = OccupancyType::NONE;
    }
  }

  plt = new OccupancyPlot(param);
}

GridMap::~GridMap(){
}

void GridMap::setObstacle(vec2_t pos){
  setObstacle(posToId(pos) );
}

void GridMap::setObstacle(vec2i_t id){
  grid[id.x()][id.y()].type = OccupancyType::OBSTACLE;
}

vec2i_t GridMap::posToId(vec2_t pos){
  int     idx = round( ( pos.x() - x_min ) / x_stp);
  int     idy = round( ( pos.y() - y_min ) / y_stp);
  vec2i_t id(idx, idy);
  return id;
}

void GridMap::plot(){
  plt->initOccupancy();
  for (int i = 0; i < x_num; i++) {
    for (int j = 0; j < y_num; j++) {
      plt->setOccupancy(grid[i][j].pos, grid[i][j].type);
    }
  }
  plt->plot();
}

} // namespace Capt