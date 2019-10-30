#include "grid_map.h"

namespace Capt {

void List::add(Node* node){
  nodes.push_back(node);
}

Node* List::findMinCostNode(){
  double min_cost  = nodes[0]->cost;
  int    min_index = 0;
  for(size_t i = 0; i < nodes.size(); i++) {
    if(nodes[i]->cost < cost) {
      min_cost  = nodes[i]->cost;
      min_index = i;
    }
  }
  Node* min_node = nodes[min_index];
  nodes.erase(nodes.begin() + min_index);

  return min_node;
}

GridMap::GridMap(){
}

GridMap::~GridMap() {
}

} // namespace Capt