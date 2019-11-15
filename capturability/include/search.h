#ifndef __SEARCH_H__
#define __SEARCH_H__

#include "param.h"
#include "base.h"
#include "grid.h"
#include "node.h"
#include "capturability.h"
#include "grid_map.h"
#include <vector>

namespace Capt {

class Search {
public:
  Search(GridMap *gridmap, Grid *grid, Capturability *capturability);
  ~Search();

  void clear();

  // set foot width when standing upright
  // stance = abs(rfoot - lfoot)
  void setStanceWidth(double stance);

  // world frame coord.
  void setStart(vec2_t rfoot, vec2_t lfoot, vec2_t icp, Foot suf);
  void setGoal(vec2_t center);

  bool open(Cell* cell);

  bool existNode();
  bool existOpen();

  void init();

  void exe();

  bool step();

  Node* getGoalNode();

private:
  GridMap       *gridmap;
  Grid          *grid;
  Capturability *capturability;

  int max_step;

  // std::vector<Node*> opens;

  // 直立した時の足幅
  double stance;

  Foot   s_suf;
  vec2_t s_rfoot, s_lfoot, s_icp; // start
  vec2_t g_rfoot, g_lfoot;        // goal

  int    yaxis[2];
  vec2_t s_arr[2];
  vec2_t g_arr[2];

  Node *g_node;
};

} // namespace Capt

#endif // __SEARCH_H__