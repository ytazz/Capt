#include "grid_map.h"

namespace Capt {

Search::Search(){
}

Search::~Search() {
}

void Search::clear(){
  nodes.clear();
  opens.clear();
}

void Search::addNode(Node node){
  nodes.push_back(node);
}

void Search::addOpen(Node* node){
  opens.push_back(node);
}

bool Search::existNode(){
  bool flag = false;
  if(nodes.size() > 0)
    flag = true;
  return flag;
}

bool Search::existOpen(){
  bool flag = false;
  if(opens.size() > 0)
    flag = true;
  return flag;
}

Node* Search::findMinCostNode(){
  double min_cost  = opens[0]->cost;
  int    min_index = 0;
  for(size_t i = 0; i < opens.size(); i++) {
    if(opens[i]->cost < cost) {
      min_cost  = opens[i]->cost;
      min_index = i;
    }
  }
  Node* min_node = opens[min_index];
  opens.erase(opens.begin() + min_index);

  return min_node;
}

void Search::exe(){
  while(existOpen() ) {
    openNode(findMinCostNode() );
  }
}

} // namespace Capt