#include "grid_map.h"

namespace Capt {

GridMap::GridMap(Param *param){
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
      pos.y()         = y_min + y_stp * j;
      grid[i][j].node = NULL;
      grid[i][j].type = OccupancyType::EMPTY;
    }
  }

  plt = new OccupancyPlot(param);
}

GridMap::~GridMap(){
}

void GridMap::setOccupancy(vec2_t pos, OccupancyType type){
  vec2i_t id = pos2id(pos);
  grid[id.x()][id.y()].type = type;
}

void GridMap::setNode(vec2_t pos, Node *node){
  vec2i_t id = pos2id(pos);
  grid[id.x()][id.y()].node = node;
  grid[id.x()][id.y()].type = OccupancyType::EXIST;
}

OccupancyType GridMap::getOccupancy(vec2_t pos){
  vec2i_t id = pos2id(pos);
  if(0 <= id.x() && id.x() < x_num && 0 <= id.y() && id.y() < y_num)
    return ( grid[id.x()][id.y()].type );
  else
    return OccupancyType::NONE;
}

Node* GridMap::getNode(vec2_t pos){
  Node   *node_ = NULL;
  vec2i_t id    = pos2id(pos);
  if(0 <= id.x() && id.x() < x_num && 0 <= id.y() && id.y() < y_num)
    node_ = grid[id.x()][id.y()].node;
  return node_;
}

vec2i_t GridMap::pos2id(vec2_t pos){
  int     idx = round( ( pos.x() - x_min ) / x_stp);
  int     idy = round( ( pos.y() - y_min ) / y_stp);
  vec2i_t id(idx, idy);
  return id;
}

void GridMap::plot(){
  plt->initOccupancy();
  for (int i = 0; i < x_num; i++) {
    for (int j = 0; j < y_num; j++) {
      plt->setOccupancy(vec2_t(x_min + x_stp * i, y_min + y_stp * j), grid[i][j].type);
    }
  }
  plt->plot();
}

} // namespace Capt