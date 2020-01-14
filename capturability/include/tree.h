#ifndef __TREE_H__
#define __TREE_H__

#include "capturability.h"
#include "node.h"

#define MAX_NODE_SIZE 100000

namespace Capt {

struct CaptData {
  vec3_t pos;
  int    nstep;
};

class Tree {
public:
  Tree(Grid *grid, Capturability* capturability);
  ~Tree();

  void clear();

  // generate tree
  Node* search(int state_id, Foot s_suf, arr2_t g_foot, int preview);

  std::vector<CaptData> getCaptureRegion(int state_id, int input_id, Foot suf, vec2_t p_suf);

private:
  Grid          *grid;
  Capturability *capturability;

  int          num_node, opened;
  Node         nodes[MAX_NODE_SIZE];
  const double epsilon;
};

} // namespace Capt

#endif // __TREE_H__