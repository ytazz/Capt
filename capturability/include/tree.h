#ifndef __TREE_H__
#define __TREE_H__

#include "capturability.h"
#include "grid_map.h"
#include "node.h"

#define MAX_NODE_SIZE 1000000

namespace Capt {

class Tree {
public:
  Tree(Param *param, Grid *grid, Capturability* capturability);
  ~Tree();

  void clear();

  // generate tree
  Node* search(int state_id, Foot suf, vec2_t g_rfoot, vec2_t g_lfoot);

private:
  Grid          *grid;
  Capturability *capturability;
  GridMap       *gridMap;

  int                num_node;
  Node               nodes[MAX_NODE_SIZE];
  std::vector<Node*> opens;
};

} // namespace Capt

#endif // __TREE_H__