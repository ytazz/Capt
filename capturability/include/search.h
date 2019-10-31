#ifndef __SEARCH_H__
#define __SEARCH_H__

#include "param.h"
#include <vector>

namespace Capt {

enum Foot { RIGHT, LEFT };

struct Node {
  Node * parent;
  int    state_id;
  double dist;
  double cost;
  int    step;
};

class Search {
public:
  Search();
  ~Search();

  void clear();

  // world frame coord.
  void setStart(vec2_t rfoot, vec2_t lfoot, vec2_t icp, Foot suf);
  void setGoal(vec2_t rfoot, vec2_t lfoot);

  void addNode(Node node);
  void addOpen(Node* node);

  void openNode(Node* node);

  bool existNode();
  bool existOpen();

  Node* findMinCostNode();

private:
  Grid          grid;
  Capturability capturability;

  std::vector<Node>  nodes;
  std::vector<Node*> opens;
};

} // namespace Capt

#endif // __SEARCH_H__