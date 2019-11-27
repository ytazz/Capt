#ifndef __SEARCH_H__
#define __SEARCH_H__

#include "param.h"
#include "base.h"
#include "grid.h"
#include "node.h"
#include "capturability.h"
#include "grid_map.h"
#include <vector>

#define MAX_NODE_SIZE 1000000

namespace Capt {

enum { INI_SUP, INI_SWF };

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
  Search(Grid *grid, Capturability *capturability);
  ~Search();

  void clear();

  // set foot width when standing upright
  // stance = abs(rfoot - lfoot)
  void setStanceWidth(double stance);

  // world frame coord.
  void setStart(vec2_t rfoot, vec2_t lfoot, vec2_t icp, Foot suf);
  void setGoal(vec2_t center);

  Node* findMinCostNode();
  bool  open(Node *node);

  void init();
  void exe();
  bool step();

  Trans getTrans();

  void                  calcFootstep();
  std::vector<Footstep> getFootstep();
  std::vector<vec3_t>   getFootstepR();
  std::vector<vec3_t>   getFootstepL();

private:
  Grid          *grid;
  Capturability *capturability;

  int max_step;
  int num_node;

  Node               nodes[MAX_NODE_SIZE];
  std::vector<Node*> opens;

  // 直立した時の足幅
  double stance;

  // 現在の状態
  vec2_t rfoot, lfoot;

  Foot   s_suf;
  vec2_t s_rfoot, s_lfoot, s_icp; // start
  vec2_t g_rfoot, g_lfoot;        // goal

  int    yaxis[2];
  vec2_t s_arr[2];
  vec2_t g_arr[2];

  Node *g_node;

  // 結果保存用変数
  std::vector<Footstep> footstep;
};

} // namespace Capt

#endif // __SEARCH_H__