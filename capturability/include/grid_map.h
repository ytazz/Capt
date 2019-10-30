#ifndef __GRID_MAP_H__
#define __GRID_MAP_H__

#include "param.h"
#include <vector>

namespace Capt {

enum Foot { RIGHT, LEFT };

struct Node {
  Node * parent;
  int    state_id;
  Foot   suf;
  double cost;
  int    step;
};

struct List {
  void  add(Node* node);
  Node* findMinCostNode();

  std::vector<Node*> nodes;
};

class GridMap {
public:
  GridMap();
  ~GridMap();

  void init(Param param);

  void clear();

  void setNode(int i, int j, Node node);
  std::vector<Node*> getNode(i, j);

private:
  Grid grid;
  List open_list;
};

} // namespace Capt

#endif // __GRID_MAP_H__