#ifndef __TREE_H__
#define __TREE_H__

#include "capturability.h"
#include "node.h"

#define MAX_NODE_SIZE 1000000

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
  Node* search(int state_id, Foot suf, vec2_t g_rfoot, vec2_t g_lfoot);

  std::vector<CaptData> getCaptureRegion(int state_id, int input_id, Foot suf, vec2_t p_suf);

private:
  Grid          *grid;
  Capturability *capturability;

  int                num_node;
  Node               nodes[MAX_NODE_SIZE];
  std::vector<Node*> opens;
};

} // namespace Capt

#endif // __TREE_H__