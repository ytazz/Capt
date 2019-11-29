#ifndef __SEARCH_H__
#define __SEARCH_H__

#include "param.h"
#include "base.h"
#include "grid.h"
#include "node.h"
#include "capturability.h"
#include "grid_map.h"
#include "tree.h"
#include <vector>

namespace Capt {

struct Trans {
  int                size;
  std::vector<State> states;
  std::vector<Input> inputs;

  void clear(){
    states.clear();
    inputs.clear();
  }
};

class Search {
public:
  Search(Grid *grid, Tree *tree);
  ~Search();

  void clear();

  // world frame coord.
  void setStart(vec2_t rfoot, vec2_t lfoot, vec2_t icp, Foot suf);
  void setGoal(vec2_t center);

  void calc();

  Trans getTrans();

  void                  calcFootstep();
  std::vector<Footstep> getFootstep();
  std::vector<vec3_t>   getFootstepR();
  std::vector<vec3_t>   getFootstepL();

private:
  Grid *grid;
  Tree *tree;

  int max_step;

  // 現在の状態
  vec2_t rfoot, lfoot;

  Foot   s_suf, g_suf;
  vec2_t s_rfoot, s_lfoot, s_icp;  // start
  vec2_t g_rfoot, g_lfoot, g_foot; // goal

  Node *g_node;

  // 結果保存用変数
  std::vector<Footstep> footstep;
};

} // namespace Capt

#endif // __SEARCH_H__