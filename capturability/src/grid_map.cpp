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
      grid[i][j].type = OccupancyType::EMPTY;
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

void GridMap::setNode(vec2_t pos, Node node){
  setNode(posToId(pos), node);
}

void GridMap::setNode(vec2i_t id, Node node){
  grid[id.x()][id.y()].node = node;
  grid[id.x()][id.y()].type = OccupancyType::OPEN;
}

Node* GridMap::getNode(vec2_t pos){
  return getNode(posToId(pos) );
}

Node* GridMap::getNode(vec2i_t id){
  return &( grid[id.x()][id.y()].node );
}

Cell* GridMap::findMinCostCell(){
  double min_cost = 100;
  Cell * cell     = NULL;
  for (int i = 0; i < x_num; i++) {
    for (int j = 0; j < y_num; j++) {
      if(grid[i][j].type == OPEN) {
        if(grid[i][j].node.cost < min_cost) {
          cell = &grid[i][j];
        }
      }
    }
  }
  return cell;
}

OccupancyType GridMap::getOccupancy(vec2_t pos){
  return getOccupancy(posToId(pos) );
}

OccupancyType GridMap::getOccupancy(vec2i_t id){
  if(0 <= id.x() && id.x() < x_num && 0 <= id.y() && id.y() < y_num)
    return ( grid[id.x()][id.y()].type );
  else
    return ( OccupancyType::NONE );
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