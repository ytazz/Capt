#include "search.h"

namespace Capt {

Search::Search(Grid *grid, Capturability *capturability) :
  grid(grid), capturability(capturability){
  max_step = capturability->getMaxStep();
}

Search::~Search() {
}

void Search::clear(){
  nodes.clear();
  opens.clear();
}

void Search::setStart(vec2_t rfoot, vec2_t lfoot, vec2_t icp, Foot suf){
  s_rfoot = rfoot;
  s_lfoot = lfoot;
  s_icp   = icp;
  s_suf   = suf;
}

void Search::setGoal(vec2_t rfoot, vec2_t lfoot){
  g_rfoot = rfoot;
  g_lfoot = lfoot;
}

void Search::addNode(Node node){
  nodes.push_back(node);
}

void Search::addOpen(int node_id){
  opens.push_back(node_id);
}

void Search::openNode(int node_id){
  for(int n = 2; n < max_step; n++) {
    std::vector<CaptureSet> region = capturability->getCaptureRegion(nodes[node_id].state_id, n);
    for(size_t i = 0; i < region.size(); i++) {
      Node node_;
      node_.parent   = &nodes[node_id];
      node_.state_id = region[i].next_id;
      node_.step     = node_.parent->step + 1;

      addNode(node_);
      addOpen(nodes.size() - 1);
    }
  }

  opens.erase(opens.begin() );
}

bool Search::existNode(){
  bool flag = false;
  if(nodes.size() > 0)
    flag = true;
  return flag;
}

bool Search::existOpen(){
  bool flag = false;
  if(opens.size() > 0)
    flag = true;
  return flag;
}

void Search::exe(){
  // Calculate initial state
  State state;
  if(s_suf == RFOOT) {
    state.swf = s_lfoot - s_rfoot;
    state.icp = s_icp - s_rfoot;
  }else{
    state.swf = s_rfoot - s_lfoot;
    state.icp = s_icp - s_lfoot;
  }

  // Calculate initial node
  Node node;
  node.parent   = NULL;
  node.state_id = grid->getStateIndex(state);
  node.step     = 0;
  nodes.push_back(node);

  // Add open list
  addOpen(0);

  // Search
  // while(existOpen() ) {
  //   openNode(opens[0]);
  // }
}

void Search::step(){
  if(existOpen() ) {
    openNode(opens[0]);
  }

  printf("-----------------------------\n");
  printf("nodes: %d\n", (int)nodes.size() );
  printf("\t| state_id | step |\n");
  for(size_t i = 0; i < nodes.size(); i++)
    printf("\t| %8d | %4d |\n", nodes[i].state_id, nodes[i].step);
}

} // namespace Capt