#ifndef __TREE_H__
#define __TREE_H__

#include "capturability.h"
#include "grid_map.h"
#include "node.h"

#define MAX_NODE_SIZE 10000

namespace Capt {

class Tree {
public:
  Tree(Capturability* capturability, Grid *grid, Param *param);
  ~Tree();

  // set maximum feasible steps
  void setPreviewStep(int stepMax);

  // generate tree
  void generate();

  // getter
  Node* getReafNode(int state_id, vec2_t pos);
  int   getPreviewStep();

private:
  Capturability *capturability;
  Grid          *grid;
  GridMap       *gridMap;

  int captMax, stepMax;
  int state_num;

  int                num_node;
  Node               nodes[MAX_NODE_SIZE];
  std::vector<Node*> opens;
};

} // namespace Capt

#endif // __TREE_H__