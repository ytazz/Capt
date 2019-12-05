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
  void setGoal(vec2_t center, double stance);

  bool calc();

  Trans getTrans();
  State getState();
  Input getInput();

  void                  calcFootstep();
  std::vector<Footstep> getFootstep();
  arr3_t                getFootstepR();
  arr3_t                getFootstepL();

  std::vector<CaptData> getCaptureRegion();

private:
  Grid *grid;
  Tree *tree;

  int max_step;

  // 現在の状態
  vec2_t rfoot, lfoot;

  Foot   s_suf, g_suf;
  State  s_state;
  int    s_state_id, s_input_id;
  vec2_t s_rfoot, s_lfoot, s_icp; // start
  vec2_t g_rfoot, g_lfoot;        // goal

  Node *g_node;

  // 結果保存用変数
  std::vector<Footstep> footstep;
  std::vector<CaptData> region;

  State ini_state;
  Input ini_input;
};

} // namespace Capt

#endif // __SEARCH_H__