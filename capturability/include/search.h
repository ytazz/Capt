#ifndef __SEARCH_H__
#define __SEARCH_H__

#include "param.h"
#include "vector.h"
#include "grid.h"
#include "capturability.h"
#include <vector>

namespace Capt {

enum Foot { RFOOT = 0, LFOOT = 1 };

struct Node {
  Node * parent;
  int    state_id;
  int    step;
  vec2_t pos; // 後で計算
};

class Search {
public:
  Search(Grid *grid, Capturability *capturability);
  ~Search();

  void clear();

  // world frame coord.
  void setStart(vec2_t rfoot, vec2_t lfoot, vec2_t icp, Foot suf);
  void setGoal(vec2_t rfoot, vec2_t lfoot);

  void addNode(Node node);
  void addOpen(int node_id);

  void openNode(int node_id);

  bool existNode();
  bool existOpen();

  void exe();

  void step();

private:
  Grid          *grid;
  Capturability *capturability;

  int max_step;

  std::vector<Node> nodes;
  std::vector<int>  opens;

  Foot   s_suf;
  vec2_t s_rfoot, s_lfoot, s_icp; // start
  vec2_t g_rfoot, g_lfoot;        // goal
};

} // namespace Capt

#endif // __SEARCH_H__